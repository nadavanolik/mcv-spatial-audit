"""
Stage 1 - generate the base edits. Runs ONCE, on the editor VM only.

Two properties make this stage different from every other:

1. Latency-insensitive. ~200 images, one time. So CPU offload is entirely
   acceptable: we have 440GB of RAM to stage weights in and only need this to
   finish overnight, once.

   It must be SEQUENTIAL offload, not model-level. FLUX Kontext's transformer
   is 23.8GB in bf16 and an A10-24Q leaves only ~21.37GiB free, so
   `enable_model_cpu_offload` - which makes one whole component resident -
   OOMs before step 1 of 28. Confirmed on mcvgpu2025s-0050, 2026-08-26.
   `enable_sequential_cpu_offload` streams submodules and fits; it is much
   slower per image, so measure with scripts/smoke_edit.py before fixing the
   base count.

2. NOT reproducible from a seed. Diffusion sampling drifts across library
   versions, attention backends and kernel selection even at fixed seed. So
   unlike stage 2, the output here is an IMMUTABLE ARTEFACT: generate once,
   tar it, ship it, and treat that tarball as ground truth - including for the
   reproducibility claim in the report. Do not regenerate it per-VM.

   Because it cannot be reproduced, it must be *described*: every run writes
   `data/bases/stage1_provenance.json` with the model revision and the library
   versions that produced the edits. That file is the reproducibility claim.

TEST IT BEFORE YOU DOWNLOAD 34GB:

    python -m src.stage1_edit --preflight

checks the pipeline class name, its call signature, the offload API, HF auth,
gated-repo access, free disk, and whether the largest component can fit the
offload mode you asked for - all without fetching a single weight. Then:

    python -m scripts.smoke_edit

does one real edit on a synthetic image and reports s/image and peak VRAM.

After this runs:
    tar czf bases.tar.gz -C data bases
    python -c "from huggingface_hub import HfApi; HfApi().upload_file(
        path_or_fileobj='bases.tar.gz', path_in_repo='bases.tar.gz',
        repo_id='YOUR_ORG/mcv-probe-set', repo_type='dataset')"
    # ~300MB. That is the only large transfer in the entire project.
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

# torch is NOT imported at module scope. --preflight's input checks (bases.json,
# source.png, instructions, resolution vs --max-side) are pure CPU and are
# precisely what you want to run on the laptop BEFORE pushing to the editor VM.
# A top-level torch import made that impossible, which defeated the point --
# same reasoning as stage3_judge's --dry-run not importing vLLM.

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# FLUX.1-Kontext-dev is ~34GB on the Hub: the transformer is ~23.8GB in bf16
# (12B params), T5-XXL another ~9.5GB, plus CLIP and the VAE. The xet chunk
# cache adds a few GB of transient overhead during the transfer, so ~40GB is
# the real peak. 45 leaves working room without blocking a machine that would
# actually have fitted it - the previous 60 was padding on a guess, and it
# failed a VM with 59GB free that had ~25GB to spare.
MODEL_GB = 34
NEED_GB = 45


def driver_cuda_version() -> str | None:
    """The maximum CUDA version this machine's driver supports, per nvidia-smi.

    torch reports the CUDA it was BUILT against; the driver caps what it can
    actually run. When a cu13 wheel lands on a 12.x driver, torch simply says
    `cuda.is_available() is False` and buries the reason in a UserWarning, so
    the preflight reads as "no GPU" on a machine with a perfectly good A10.
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

    OFFLOAD MODE IS NOT A TUNING KNOB ON THIS HARDWARE - it decides whether the
    thing runs at all.

    `enable_model_cpu_offload` moves one whole COMPONENT to the GPU at a time.
    FLUX Kontext's transformer is 11.9B params in bf16 = 23.8GB, and the
    A10-24Q reports 23.72GiB total with only ~21.37GiB free (the vGPU layer
    keeps ~2.4GiB permanently). 23.8 > 21.37, so model-level offload OOMs while
    moving the transformer in, before step 1 of 28. Nothing else is on the card;
    freeing disk or unloading a judge does not change this.

    `enable_sequential_cpu_offload` moves individual SUBMODULES instead, so the
    resident set is a few layers rather than the whole transformer. It fits with
    room to spare. It is much slower - every forward pass streams weights over
    PCIe, every step, so expect minutes per image rather than seconds - but
    stage 1 runs once, over ~200 images, and we have 440GB of RAM to stage it
    in. Measure the real number with scripts/smoke_edit.py before committing to
    a base count.
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
    """Size of the biggest single pipeline component, from the Hub metadata.

    No download: `model_info(files_metadata=True)` returns per-file sizes. This
    is what tells us whether model-level offload can work BEFORE we spend six
    minutes fetching 34GB and then OOM on the first step.
    """
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
    """diffusers moved this from the pipeline onto the VAE. Take whichever
    exists and say which, rather than crashing on one version or silently
    skipping the memory saving on the other."""
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

    "stale" means an edit exists but was generated from a different
    instruction. That happens whenever stage 0 is re-run with changed
    templates: it overwrites regions.json and instruction.txt in place and
    leaves edit.png alone. The result is an edited image paired with an
    instruction it never saw - which produces plausible-looking scores that
    mean nothing, and which no downstream stage can detect.

    An edit with no edit.json predates this fingerprint and cannot be
    verified, so it is treated as stale rather than trusted.
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
    """What it would take to explain these bytes to a reviewer. Diffusion
    output is not seed-reproducible across versions, so this is the closest
    thing to a hash the artefact can have."""
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
    """Everything that can fail, checked before anything is downloaded.

    Deliberately mirrors stage3_judge's --dry-run: the expensive, gated,
    hardware-bound part of the stage should be the LAST thing that can go
    wrong, not the first.
    """
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
        if not hasattr(FluxKontextPipeline, "enable_model_cpu_offload"):
            problems.append("no enable_model_cpu_offload")
            print("  FAIL no enable_model_cpu_offload")
        else:
            print("  OK   enable_model_cpu_offload present")
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
            # pip resolves diffusers/accelerate's bare `torch>=2.0.0` to the
            # newest wheel, which is now a cu13 build. There is no sudo here to
            # move the driver, so the wheel has to move instead.
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
    # FLUX.1-Kontext-dev is gated: the licence must be accepted by the account
    # whose token this is. That failure otherwise appears only after the
    # download starts, and reads as a 403 with no explanation.
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()
        print(f"  OK   logged in as {who.get('name')}")
        try:
            info = api.model_info(model_id)
            print(f"  OK   {model_id} accessible, revision {info.sha[:12]}")

            # THE CHECK THAT MATTERS, and the one whose absence let a
            # structurally impossible run download 34GB and then OOM on step 0.
            # Model-level offload needs the LARGEST single component resident,
            # not the average one. FLUX Kontext's transformer is 23.8GB in bf16
            # against 21.37GiB free on an A10-24Q.
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
        # `huggingface-cli login` was renamed; recent hub versions want
        # `hf auth login` and only mention the new spelling in the error.
        print("       Run: hf auth login   (older hub: huggingface-cli login)")

    print("\n--- 4. disk for the weights ---")
    import os
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    probe = hf_home if hf_home.exists() else hf_home.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 2**30

    # Already-cached weights need no transfer, so the download budget does not
    # apply. Without this the check fails the moment the model is in place --
    # exactly when the run is finally ready to go -- and demands 45GB to fetch
    # something already on disk. Count blobs only: the snapshot/ tree is
    # symlinks into blobs/ and following them double-counts every file.
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

    # Stage-0 inputs are tracked separately: they block the real stage-1 run
    # but NOT scripts/smoke_edit.py, which builds its own synthetic image. One
    # undifferentiated failure list would send you off to fetch COCO before you
    # have any evidence the editor loads at all.
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
            # Not a blocker: stage 1 re-edits these automatically. It is called
            # out because it means stage 0 was re-run with changed
            # instructions, and anyone who copied data/bases elsewhere in the
            # meantime is holding edits that no longer match their prompts.
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
                      f"will be thumbnailed, and masks from stage 0 are at "
                      f"SOURCE resolution. Keep --max-side >= the largest "
                      f"source or stage 2 will misalign.")

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

    t0, n = time.time(), 0
    for s in tqdm(todo, desc="editing"):
        d = root / s["base_id"]
        src = Image.open(d / "source.png").convert("RGB")
        before = src.size
        src.thumbnail((a.max_side, a.max_side), Image.LANCZOS)

        import torch
        out = pipe(
            image=src,
            prompt=s["instruction"],
            num_inference_steps=a.steps,
            guidance_scale=a.guidance,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # Masks were computed at SOURCE resolution and stage 2 indexes them
        # straight into edit.png, so the edit must come back at exactly the
        # source's size - not the thumbnailed size, and not whatever internal
        # resolution Kontext chose. Misalignment here corrupts the wrong pixels
        # and silently invalidates every downstream number.
        out = out.resize(before, Image.LANCZOS)
        out.save(d / "edit.png")
        # Fingerprint the instruction this edit was actually made from. Stage 0
        # rewrites regions.json and instruction.txt in place but never deletes
        # edit.png, so a re-run of stage 0 with changed instructions leaves
        # every existing edit silently mismatched against the instruction the
        # judge will be shown. Nothing downstream could detect that.
        (d / "edit.json").write_text(json.dumps({
            "instruction": s["instruction"],
            "instruction_sha": _sha(s["instruction"]),
            "model": a.model, "steps": a.steps, "guidance": a.guidance,
        }, indent=2))
        n += 1
        gc.collect()
        torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"edited {n} images in {dt / 60:.1f} min ({dt / max(n, 1):.1f}s each)")

    prov = provenance(a.model, a)
    prov["n_edited"] = n
    (root / "stage1_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"provenance -> {root / 'stage1_provenance.json'}")
    print("done - now tar data/bases and upload once")


if __name__ == "__main__":
    main()
