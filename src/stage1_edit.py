"""
Stage 1: edit each base photo with FLUX.1 Kontext. Runs once, on one GPU.

Uses sequential CPU offload: the 23.8GB transformer does not fit in the A10's
~21.4GiB of free memory as one piece. Diffusion output is not reproducible
across library versions even at a fixed seed, so the edits are generated once
and published, and every run writes data/bases/stage1_provenance.json with the
model revision and library versions.

    python -m src.stage1_edit --preflight   # checks everything, downloads nothing
    python -m scripts.smoke_edit            # one real edit on a synthetic image

After this runs:
    cp out/edit_drift.csv data/bases/          # if it has been computed
    tar czf bases.tar.gz -C data bases
    python -c "from huggingface_hub import HfApi; HfApi().upload_file(
        path_or_fileobj='bases.tar.gz', path_in_repo='bases.tar.gz',
        repo_id='mcv-spatial-audit/mcv-spatial-audit', repo_type='dataset')"
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import shutil
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# torch is imported inside functions so --preflight's input checks run on a
# machine without it.

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# FLUX.1-Kontext-dev is ~34GB on the Hub; the download needs ~40GB at peak.
MODEL_GB = 34
NEED_GB = 45


def driver_cuda_version() -> str | None:
    """The highest CUDA version the driver supports, per nvidia-smi.

    Used to explain a torch build that is newer than the driver allows.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def load_editor(model_id: str, offload: str = "sequential"):
    """Load FLUX Kontext under CPU offload.

    "model" offload moves one whole component to the GPU at a time and does
    not fit the transformer on an A10; "sequential" streams submodules and
    fits, at a few minutes per image.
    """
    import torch
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unknown offload mode {offload!r}")
    _enable_vae_slicing(pipe)
    return pipe


def largest_component_gb(model_id: str) -> tuple[float, str]:
    """Size of the biggest pipeline component, from Hub metadata (no download)."""
    from collections import defaultdict
    from huggingface_hub import HfApi
    info = HfApi().model_info(model_id, files_metadata=True)
    by_dir: dict[str, int] = defaultdict(int)
    for f in info.siblings or []:
        if f.rfilename.endswith(".safetensors") and f.size:
            by_dir[f.rfilename.split("/")[0]] += f.size
    if not by_dir:
        return 0.0, "unknown"
    name = max(by_dir, key=by_dir.get)
    return by_dir[name] / 2**30, name


def _enable_vae_slicing(pipe) -> str:
    """Enable VAE slicing on the pipeline or the VAE, whichever this diffusers
    version supports, and say which."""
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        return "pipe.enable_vae_slicing()"
    if hasattr(getattr(pipe, "vae", None), "enable_slicing"):
        pipe.vae.enable_slicing()
        return "pipe.vae.enable_slicing()"
    return "UNAVAILABLE - neither API present"


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def edit_state(root: Path, spec: dict) -> str:
    """"fresh" | "stale" | "missing" for one base's edit.png.

    "stale": the edit was made from a different instruction than the current
    one, or has no edit.json to say which.
    """
    d = root / spec["base_id"]
    if not (d / "edit.png").exists():
        return "missing"
    meta_path = d / "edit.json"
    if not meta_path.exists():
        return "stale"
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return "stale"
    return "fresh" if meta.get("instruction_sha") == _sha(spec["instruction"])         else "stale"


def provenance(model_id: str, a) -> dict:
    """Model revision, settings and library versions behind the edits."""
    import diffusers
    import torch
    import transformers
    rev = None
    try:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(model_id).sha
    except Exception as e:                       # offline, or not authorised
        rev = f"unavailable: {type(e).__name__}"
    return {
        "model": model_id, "revision": rev,
        "steps": a.steps, "guidance": a.guidance, "max_side": a.max_side,
        "offload": getattr(a, "offload", "sequential"), "seed": 0,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def preflight(root: Path, model_id: str, a) -> int:
    """Check the API, GPU, HF access, disk and inputs before downloading."""
    print("=== PREFLIGHT: no weights are fetched ===")
    problems: list[str] = []

    print("\n--- 1. diffusers API surface ---")
    try:
        from diffusers import FluxKontextPipeline
        print(f"  OK   from diffusers import FluxKontextPipeline")
        sig = inspect.signature(FluxKontextPipeline.__call__).parameters
        for p in ("image", "prompt", "num_inference_steps", "guidance_scale",
                  "generator"):
            if p in sig:
                print(f"  OK   __call__ accepts {p}")
            else:
                problems.append(f"FluxKontextPipeline.__call__ has no '{p}'")
                print(f"  FAIL __call__ has NO parameter '{p}'")
        extra = [p for p in ("max_area", "_auto_resize", "height", "width")
                 if p in sig]
        if extra:
            print(f"  note __call__ also accepts {extra} - Kontext may resize "
                  f"internally; we resize the output back to source size.")
        api = f"enable_{a.offload}_cpu_offload"
        if not hasattr(FluxKontextPipeline, api):
            problems.append(f"no {api}")
            print(f"  FAIL no {api} (--offload {a.offload})")
        else:
            print(f"  OK   {api} present")
        print(f"  note vae slicing will use: "
              f"{'pipe.enable_vae_slicing()' if hasattr(FluxKontextPipeline, 'enable_vae_slicing') else 'pipe.vae.enable_slicing()'}")
    except ImportError as e:
        problems.append(f"cannot import FluxKontextPipeline: {e}")
        print(f"  FAIL {e}")

    print("\n--- 2. GPU ---")
    try:
        import torch
    except ImportError:
        torch = None
        print("  SKIP torch is not installed here. Everything below except "
              "this check is still meaningful on the laptop; run --preflight "
              "again on the editor VM for the GPU line.")
    free_vram = None
    if torch is not None and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_vram = free / 2**30
        print(f"  OK   {torch.cuda.get_device_name(0)}: "
              f"{free_vram:.2f}GiB free of {total / 2**30:.2f}GiB")
    elif torch is not None:
        built = getattr(torch.version, "cuda", None)
        drv = driver_cuda_version()
        print("  FAIL torch.cuda.is_available() is False")
        print(f"       torch {torch.__version__} was built for CUDA {built}; "
              f"this driver supports CUDA {drv or 'unknown'}")
        if drv and built and int(built.split(".")[0]) > int(drv.split(".")[0]):
            tag = "cu" + drv.replace(".", "")
            problems.append(
                f"torch is built for CUDA {built} but the driver caps at {drv}")
            print("       This is a wheel/driver mismatch, not a missing GPU. "
                  "The driver")
            print("       cannot be updated without sudo, so install a matching "
                  "torch:")
            print(f"         pip install --force-reinstall torch "
                  f"--index-url https://download.pytorch.org/whl/{tag}")
        else:
            problems.append("no CUDA device")

    print("\n--- 3. HF auth and gated-repo access ---")
    # FLUX.1-Kontext-dev is gated: the token's account must accept the licence.
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()
        print(f"  OK   logged in as {who.get('name')}")
        try:
            info = api.model_info(model_id)
            print(f"  OK   {model_id} accessible, revision {info.sha[:12]}")

            # Model-level offload needs the largest component resident at once.
            gb, comp = largest_component_gb(model_id)
            if gb:
                print(f"  note largest component is {comp}/ at {gb:.1f}GiB")
                if free_vram is not None:
                    fits = gb < free_vram * 0.95
                    print(f"  {'OK  ' if fits else 'NOTE'} "
                          f"offload=model needs {gb:.1f}GiB resident vs "
                          f"{free_vram:.1f}GiB free -> "
                          f"{'fits' if fits else 'DOES NOT FIT'}")
                    if not fits and a.offload == "model":
                        problems.append(
                            f"offload=model cannot work: {comp} is {gb:.1f}GiB "
                            f"but only {free_vram:.1f}GiB of VRAM is free")
                        print("       Use --offload sequential (the default). "
                              "This is a VRAM limit,")
                        print("       not a disk limit - freeing disk or "
                              "unloading a judge changes nothing.")
                    elif not fits:
                        print("       offload=sequential streams submodules, "
                              "so it fits. Slower per image.")
        except Exception as e:
            problems.append(f"cannot access {model_id}: {type(e).__name__}")
            print(f"  FAIL cannot access {model_id}: {type(e).__name__}: "
                  f"{str(e)[:200]}")
            print(f"       Accept the licence at "
                  f"https://huggingface.co/{model_id}")
    except Exception as e:
        problems.append(f"not logged in to HF: {type(e).__name__}")
        print(f"  FAIL {type(e).__name__}: {str(e)[:200]}")
        print("       Run: hf auth login   (older hub: huggingface-cli login)")

    print("\n--- 4. disk for the weights ---")
    import os
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    probe = hf_home if hf_home.exists() else hf_home.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 2**30

    # Cached weights need no download. Count real files only: the snapshot
    # tree is symlinks into blobs/.
    repo_dir = hf_home / "hub" / ("models--" + model_id.replace("/", "--"))
    cached_gb = 0.0
    if repo_dir.exists():
        cached_gb = sum(f.stat().st_size for f in repo_dir.rglob("*")
                        if f.is_file() and not f.is_symlink()) / 2**30
    have_it = cached_gb >= MODEL_GB * 0.9

    if have_it:
        print(f"  OK   {model_id} already cached ({cached_gb:.0f}GiB in "
              f"{repo_dir.name}); no download needed")
        print(f"       {free_gb:.0f}GiB free for outputs")
    else:
        print(f"  {'OK  ' if free_gb >= NEED_GB else 'FAIL'} HF_HOME={hf_home}: "
              f"{free_gb:.0f}GiB free; the model is ~{MODEL_GB}GB, "
              f"~{NEED_GB}GB wanted with transfer overhead")
        if cached_gb:
            print(f"       ({cached_gb:.0f}GiB is already cached - a partial "
                  f"download will resume)")
    if not have_it and free_gb < NEED_GB:
        problems.append(f"only {free_gb:.0f}GiB free for a ~{MODEL_GB}GB download")
        print("       This VM is role-specialised: it must NOT also hold a "
              "judge checkpoint. Clear ~/hf_cache of any Qwen weights.")

    # Stage-0 inputs block the real run but not scripts/smoke_edit.py.
    inputs: list[str] = []
    print("\n--- 5. inputs from stage 0 (not needed for smoke_edit) ---")
    bj = root / "bases.json"
    if not bj.exists():
        inputs.append(f"missing {bj}")
        print(f"  MISS {bj} - run stage 0 first")
    else:
        specs = json.loads(bj.read_text())
        if a.limit:
            specs = specs[: a.limit]
        missing_src = [s["base_id"] for s in specs
                       if not (root / s["base_id"] / "source.png").exists()]
        no_instr = [s["base_id"] for s in specs if not s.get("instruction")]
        states = [edit_state(root, s) for s in specs]
        n_fresh = states.count("fresh")
        n_stale = states.count("stale")
        print(f"  OK   {len(specs)} base specs")
        print(f"       {n_fresh} already edited and up to date; "
              f"{states.count('missing')} never edited")
        if n_stale:
            print(f"  WARN {n_stale} edit.png are STALE - generated from a "
                  f"different instruction")
            print(f"       (or predate edit.json). Stage 1 will regenerate "
                  f"them; {n_stale} x ~190s.")
        if missing_src:
            inputs.append(f"{len(missing_src)} bases missing source.png")
            print(f"  FAIL {len(missing_src)} missing source.png, "
                  f"e.g. {missing_src[0]}")
        else:
            print("  OK   every base has source.png")
        if no_instr:
            inputs.append(f"{len(no_instr)} bases have an empty instruction")
            print(f"  FAIL {len(no_instr)} empty instructions, e.g. {no_instr[0]}")
        else:
            print("  OK   every base has a non-empty instruction")
        if specs:
            s0 = specs[0]
            im = Image.open(root / s0["base_id"] / "source.png")
            print(f"\n  first job: {s0['base_id']}  {im.size[0]}x{im.size[1]}")
            print(f"    instruction: {s0['instruction']}")
            print(f"    regions:     {len(s0['regions'])}")
            if max(im.size) > a.max_side:
                print(f"    note source exceeds --max-side {a.max_side}; it "
                      f"will be edited smaller and resized back to "
                      f"{im.size[0]}x{im.size[1]}, so masks stay aligned but "
                      f"fine detail is lost.")

    print("\n--- would run ---")
    print(f"  model={model_id} dtype=bfloat16 offload={a.offload}")
    print(f"  steps={a.steps} guidance={a.guidance} max_side={a.max_side} seed=0")

    print()
    if problems:
        print(f"BLOCKED - {len(problems)} problem(s) stop everything, "
              f"including smoke_edit:")
        for p in problems:
            print(f"  - {p}")
    if inputs:
        print(f"\nSTAGE 0 NOT READY - {len(inputs)} item(s) stop the real "
              f"stage-1 run, but NOT smoke_edit:")
        for p in inputs:
            print(f"  - {p}")
    if problems:
        return 1
    if inputs:
        print("\nThe editor itself is ready. Run "
              "`python -m scripts.smoke_edit` now;\nstage 0 only has to exist "
              "before the real run.")
        return 0
    print("PREFLIGHT OK - safe to start the download.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offload", default="sequential",
                    choices=["sequential", "model"],
                    help="sequential streams submodules and is the only mode "
                         "that fits FLUX's 23.8GiB transformer on a 24GB A10; "
                         "model is faster but needs the whole component "
                         "resident")
    ap.add_argument("--preflight", action="store_true",
                    help="check the API, auth, gating, disk and inputs without "
                         "downloading any weights, then exit")
    a = ap.parse_args()

    root = Path(a.bases)
    if a.preflight:
        raise SystemExit(preflight(root, a.model, a))

    import torch

    specs = json.loads((root / "bases.json").read_text())
    if a.limit:
        specs = specs[: a.limit]

    fresh, stale, missing = [], [], []
    for spec in specs:
        state = edit_state(root, spec)
        (missing if state == "missing" else
         stale if state == "stale" else fresh).append(spec)
    todo = missing + stale
    print(f"{len(fresh)} up to date, {len(missing)} never edited, "
          f"{len(stale)} STALE (instruction changed since the edit)")
    if stale:
        print("  re-editing stale bases. Their edit.png was generated from a "
              "different instruction:")
        for spec in stale[:3]:
            print(f"    {spec['base_id']}: now {spec['instruction'][:60]!r}")
    print(f"{len(todo)} of {len(specs)} bases need editing")
    if not todo:
        return

    pipe = load_editor(a.model, offload=a.offload)

    # Written before the loop and after every image, so an interrupted run
    # still has provenance. n_edited counts this run only.
    prov_path = root / "stage1_provenance.json"
    prov = provenance(a.model, a)
    prov["n_bases"] = len(specs)
    prov["n_todo"] = len(todo)
    prov["n_edited"] = 0
    prov_path.write_text(json.dumps(prov, indent=2))

    t0, n = time.time(), 0
    for s in tqdm(todo, desc="editing"):
        d = root / s["base_id"]
        src = Image.open(d / "source.png").convert("RGB")
        before = src.size
        src.thumbnail((a.max_side, a.max_side), Image.LANCZOS)

        out = pipe(
            image=src,
            prompt=s["instruction"],
            num_inference_steps=a.steps,
            guidance_scale=a.guidance,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # Back to the exact source size, so the source-resolution masks align.
        out = out.resize(before, Image.LANCZOS)
        out.save(d / "edit.png")
        # Record which instruction this edit was made from (see edit_state).
        (d / "edit.json").write_text(json.dumps({
            "instruction": s["instruction"],
            "instruction_sha": _sha(s["instruction"]),
            "model": a.model, "steps": a.steps, "guidance": a.guidance,
        }, indent=2))
        n += 1
        prov["n_edited"] = n
        prov_path.write_text(json.dumps(prov, indent=2))
        gc.collect()
        torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"edited {n} images in {dt / 60:.1f} min ({dt / max(n, 1):.1f}s each)")

    print(f"provenance -> {prov_path}")
    print("done - now tar data/bases and upload once")


if __name__ == "__main__":
    main()
