"""
One real FLUX Kontext edit on a synthetic image. Editor VM only.

The stage-1 counterpart to scripts/smoke_judge.py: needs a loaded model but NO
project data, so it runs before stage 0 has produced anything. Run
`python -m src.stage1_edit --preflight` first - that settles the API surface
without downloading 34GB. This settles whether the thing actually generates.

What it answers, in order of how expensive each is to learn late:

  1. Does the pipeline load and RUN under sequential CPU offload on a 24GB
     A10 at all? (Model-level offload provably cannot: the transformer is
     23.8GB against ~21.37GiB free.)
  2. Does the edit come back at the SOURCE resolution? Stage 0's masks are at
     source resolution and stage 2 indexes them straight into edit.png, so a
     size mismatch corrupts the wrong pixels and silently invalidates every
     number downstream. This is the single most important check here.
  3. Did it edit the region it was told to, and leave the other one alone?
     Measured as a colour shift, not a judgement - the judge is a separate
     stage with its own smoke test.
  4. How many seconds per image, and how much VRAM? 200 bases have to fit in
     an overnight run on one VM.

Usage (editor VM):
    python -m scripts.smoke_edit
    python -m scripts.smoke_edit --steps 20 --out /dev/shm/smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage1_edit import MODEL_ID, load_editor  # noqa: E402

# Two squares, well separated. R0 is the one the instruction targets; R1 is the
# control that must survive untouched.
R0 = (60, 60, 200, 200)      # x1,y1,x2,y2 - blue, to become red
R1 = (250, 250, 390, 390)    # x1,y1,x2,y2 - green, must stay green
INSTRUCTION = "change the blue square to red"


def synthetic_source(size: int = 448) -> Image.Image:
    im = Image.new("RGB", (size, size), (240, 240, 240))
    d = ImageDraw.Draw(im)
    d.rectangle(R0, fill=(30, 30, 200))       # blue
    d.rectangle(R1, fill=(30, 160, 30))       # green
    return im


def mean_rgb(im: Image.Image, box) -> np.ndarray:
    a = np.asarray(im.convert("RGB"), dtype=np.float64)
    x1, y1, x2, y2 = box
    return a[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--n-bases", type=int, default=200,
                    help="extrapolate the wall clock to this many bases")
    ap.add_argument("--offload", default="sequential",
                    choices=["sequential", "model"],
                    help="sequential is the only mode that fits on a 24GB A10")
    ap.add_argument("--out", default="out/smoke_edit",
                    help="where to write source/edit PNGs for eyeballing")
    a = ap.parse_args()

    import torch

    ok = True
    src = synthetic_source(a.size)

    print("=" * 70)
    print("1. LOADING (CPU offload; the download happens once)")
    print("=" * 70)
    # Goes through load_editor, not a private copy of it, so the smoke test
    # exercises the loader stage 1 actually uses -- offload mode included.
    t0 = time.time()
    print(f"  offload={a.offload}")
    pipe = load_editor(a.model, offload=a.offload)
    print(f"  loaded in {time.time() - t0:.0f}s")

    print("\n" + "=" * 70)
    print("2. ONE EDIT")
    print("=" * 70)
    print(f"  instruction: {INSTRUCTION!r}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        image=src,
        prompt=INSTRUCTION,
        num_inference_steps=a.steps,
        guidance_scale=a.guidance,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else float("nan"))
    print(f"  {dt:.1f}s at {a.steps} steps; peak VRAM {peak:.2f}GiB")
    print(f"  raw pipeline output size: {out.size} (source was {src.size})")

    print("\n" + "=" * 70)
    print("3. RESOLUTION - the check that protects every downstream number")
    print("=" * 70)
    # Kontext resizes internally to a preferred resolution, so the raw output
    # is NOT guaranteed to match the input. stage1_edit resizes back to the
    # SOURCE size (the size before any thumbnailing) because stage 0's masks
    # are at source resolution and stage 2 indexes them straight into edit.png.
    if out.size != src.size:
        print(f"  note pipeline returned {out.size}, not {src.size}. That is "
              f"expected - Kontext picks its own resolution. stage1_edit "
              f"resizes back, which is exactly why that line exists.")
    fixed = out.resize(src.size, Image.LANCZOS)
    ok &= check("edit aligns to source resolution after the resize",
                fixed.size == src.size, f"{fixed.size} vs {src.size}")

    print("\n" + "=" * 70)
    print("4. DID IT EDIT THE RIGHT SQUARE?")
    print("=" * 70)
    s0, s1 = mean_rgb(src, R0), mean_rgb(src, R1)
    e0, e1 = mean_rgb(fixed, R0), mean_rgb(fixed, R1)
    print(f"  region 0 (target): {s0.round(0)} -> {e0.round(0)}")
    print(f"  region 1 (control): {s1.round(0)} -> {e1.round(0)}")
    moved0 = float(np.abs(e0 - s0).mean())
    moved1 = float(np.abs(e1 - s1).mean())
    print(f"  mean abs channel shift: target {moved0:.1f}, control {moved1:.1f}")

    ok &= check("the image changed at all", moved0 > 5,
                f"target shift {moved0:.1f}")
    # Redness: R - (G+B)/2. Going blue -> red should raise it substantially.
    redness = lambda v: v[0] - (v[1] + v[2]) / 2
    print(f"  redness of region 0: {redness(s0):+.0f} -> {redness(e0):+.0f}")
    ok &= check("target square became redder", redness(e0) > redness(s0) + 30,
                f"{redness(s0):+.0f} -> {redness(e0):+.0f}")
    ok &= check("control square moved less than the target",
                moved1 < moved0, f"{moved1:.1f} < {moved0:.1f}")
    if moved1 > 30:
        print("  WARNING: the untouched square moved a lot. FLUX Kontext is "
              "supposed to preserve unedited content; heavy global drift would "
              "mean the 'clean' base edits are already noisy everywhere, which "
              "raises the noise floor stage 4 measures everything against.")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    src.save(outdir / "source.png")
    fixed.save(outdir / "edit.png")
    print(f"\n  wrote {outdir / 'source.png'} and {outdir / 'edit.png'} - "
          f"look at them before trusting any of the above")

    print("\n" + "=" * 70)
    print("5. BUDGET")
    print("=" * 70)
    total_h = dt * a.n_bases / 3600
    print(f"  {dt:.1f}s/image x {a.n_bases} bases = {total_h:.1f}h "
          f"on this one VM, single pass")
    print(f"  peak VRAM {peak:.2f}GiB of 24GB")
    if total_h > 8:
        print(f"  WARNING: {total_h:.1f}h does not fit an overnight run. Lower "
              f"--steps (28 -> 20 costs little on Kontext), or cut n_bases.")
    else:
        print("  Fits an overnight run.")

    print(f"\n{'SMOKE PASSED' if ok else 'SMOKE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
