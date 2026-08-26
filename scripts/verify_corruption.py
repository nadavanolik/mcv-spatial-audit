"""
Did the corruption actually damage the image, and only inside the mask?

CPU only, no GPU, no judge. Run this before concluding anything about judge
sensitivity: "the judge did not react" is only a finding if there was something
to react to. Nobody had checked, and a subtle corruption on a textured
photograph would explain a flat result without implicating the judge at all.

Reports, per corruption and severity:

  inside      mean |pixel difference| within the target region's mask
  outside     the same outside it -- MUST be ~0, or stage 2 is damaging pixels
              it was never asked to touch and every leakage number is wrong
  changed%    share of masked pixels moved by more than 5 intensity levels
  contrast    inside / outside, i.e. how localised the damage is

The `inside` column is also the number to quote in the report when stating how
visible the stimulus was.

Usage:
    python -m scripts.verify_corruption --manifest out/manifest.parquet \\
        --bases data/bases --variants /dev/shm/mcv/variants
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VISIBLE_LEVELS = 5      # a difference below this is invisible in an 8-bit image


def load_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if img is None else img.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--variants", default="/dev/shm/mcv/variants")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap variants examined (default: all)")
    a = ap.parse_args()

    bases, variants = Path(a.bases), Path(a.variants)
    df = pd.read_parquet(a.manifest)
    df = df[~df.is_control]
    if a.limit:
        df = df.head(a.limit)

    rows, missing = [], 0
    for r in df.itertuples():
        edit = load_gray(bases / r.base_id / "edit.png")
        var = load_gray(variants / f"{r.variant_id}.png")
        mask = cv2.imread(
            str(bases / r.base_id / "masks" / f"r{r.target_region_id}.png"),
            cv2.IMREAD_GRAYSCALE)
        if edit is None or var is None or mask is None:
            missing += 1
            continue
        if edit.shape[:2] != var.shape[:2] or mask.shape[:2] != edit.shape[:2]:
            print(f"SHAPE MISMATCH {r.variant_id}: edit {edit.shape[:2]} "
                  f"var {var.shape[:2]} mask {mask.shape[:2]}")
            continue

        diff = np.abs(edit - var).mean(axis=2)      # mean over BGR
        m = mask > 127
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        rows.append(dict(
            corruption=r.corruption, severity=r.severity,
            inside=float(diff[m].mean()),
            outside=float(diff[~m].mean()),
            changed=float((diff[m] > VISIBLE_LEVELS).mean()),
            area_frac=float(m.mean()),
        ))

    if missing:
        print(f"NOTE: {missing} variants missing on disk; re-run stage 2.")
    if not rows:
        raise SystemExit("nothing measured - check --bases and --variants")

    out = pd.DataFrame(rows)
    g = out.groupby(["corruption", "severity"]).agg(
        n=("inside", "size"),
        inside=("inside", "mean"),
        outside=("outside", "mean"),
        changed=("changed", "mean"),
        area=("area_frac", "mean"),
    ).reset_index()
    g["contrast"] = g.inside / g.outside.replace(0, np.nan)

    print("\n=== corruption strength (8-bit levels, 0-255) ===")
    print(g.round(3).to_string(index=False))

    print("\n--- reading this ---")
    weak = g[g.inside < VISIBLE_LEVELS]
    if len(weak):
        print(f"  WARNING: {len(weak)} cell(s) change the masked region by less")
        print(f"  than {VISIBLE_LEVELS} levels on average. A judge that ignores")
        print("  those is not wrong. Raise the severities before reporting any")
        print("  insensitivity result:")
        print(weak[["corruption", "severity", "inside"]].round(2)
              .to_string(index=False))
    else:
        print(f"  OK: every corruption moves the masked region by more than")
        print(f"  {VISIBLE_LEVELS} levels, so the damage is visible and a flat")
        print("  judge response is about the judge.")

    leaky = g[g.outside > 1.0]
    if len(leaky):
        print("\n  WARNING: pixels OUTSIDE the target mask changed. Stage 2 is")
        print("  meant to be perfectly localised, so this would invalidate the")
        print("  leakage analysis - every 'the judge penalised the wrong region'")
        print("  reading assumes the wrong region was untouched.")
        print(leaky[["corruption", "severity", "outside"]].round(3)
              .to_string(index=False))
    else:
        print("\n  OK: outside-mask difference is ~0, so damage is confined to")
        print("  the region we targeted. Leakage in the scores is the judge's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
