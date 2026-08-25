"""
Stage 1 — generate the base edits. Runs ONCE, on the editor VM only.

Two properties make this stage different from every other:

1. Latency-insensitive. ~200 images, one time. So CPU offload is entirely
   acceptable: with 440GB of RAM, enable_model_cpu_offload puts a 12B editor
   on a 24GB A10 at ~15-30s/image. Two hours, overnight, done forever.

2. NOT reproducible from a seed. Diffusion sampling drifts across library
   versions, attention backends and kernel selection even at fixed seed. So
   unlike stage 2, the output here is an IMMUTABLE ARTEFACT: generate once,
   tar it, ship it, and treat that tarball as ground truth — including for the
   reproducibility claim in the report. Do not regenerate it per-VM.

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
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def load_editor(model_id: str):
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # Model-level offload: keeps one submodule on GPU at a time. Sequential
    # offload is ~5x slower still but survives even tighter budgets.
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    root = Path(a.bases)
    specs = json.loads((root / "bases.json").read_text())
    if a.limit:
        specs = specs[: a.limit]

    todo = [s for s in specs if not (root / s["base_id"] / "edit.png").exists()]
    print(f"{len(todo)} of {len(specs)} bases still need editing")
    if not todo:
        return

    pipe = load_editor(a.model)

    for s in tqdm(todo, desc="editing"):
        d = root / s["base_id"]
        src = Image.open(d / "source.png").convert("RGB")
        src.thumbnail((a.max_side, a.max_side), Image.LANCZOS)

        out = pipe(
            image=src,
            prompt=s["instruction"],
            num_inference_steps=a.steps,
            guidance_scale=a.guidance,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # Masks were computed at source resolution; keep the edit aligned to it.
        out = out.resize(src.size, Image.LANCZOS)
        out.save(d / "edit.png")
        gc.collect()
        torch.cuda.empty_cache()

    print("done — now tar data/bases and upload once")


if __name__ == "__main__":
    main()
