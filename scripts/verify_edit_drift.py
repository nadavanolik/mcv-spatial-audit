"""Did the edit move the objects the masks describe?  [CPU]

Stage 0 computes masks on `source.png`. Stage 2 applies them to `edit.png`, and
stage 3 shows the judge bounding boxes in SOURCE coordinates. That chain assumes
FLUX Kontext preserved the layout. It does not always: on some bases it re-poses
people, re-frames the shot, or re-composes an interior entirely, and then the
mask named "person" covers whatever moved into that spot.

The failure mode is dangerous because it is not neutral. A mask that no longer
covers its object makes the judge look like it failed to localise -- which is
the finding this project is testing. The confound MIMICS the result, so it has
to be quantified rather than assumed away.

Two numbers per base, and the difference between them matters:

  bg_absdiff   mean |source - edit| OUTSIDE every mask. Mostly harmless: stage
               2's baseline is edit.png, not source.png, so a global recolour
               cancels out. Reported because it is what a reader expects to see
               and because it explains the grayscale-source cases.

  edge_iou     Canny edges, dilated, intersection-over-union. Colour-invariant,
               so it isolates the part that does matter: whether structure
               stayed put. 1.0 is an identical layout.

`edge_iou` is an UPPER BOUND on displacement, not proof of it: "change the chair
to glass" is supposed to destroy that object's edges. Treat a low score as
"cannot vouch for this base", not as "this base moved".

    python -m scripts.verify_edit_drift
    python -m scripts.verify_edit_drift --min-iou 0.4 --out out/edit_drift.csv

Feed the csv to stage 4 to report every headline number on both the full set and
the subset that keeps its geometry:

    python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' \
        --drift-csv out/edit_drift.csv --min-edge-iou 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 3x3 dilated twice: a one- or two-pixel resampling jitter is not a moved
# object, and undilated Canny IoU punishes it as if it were.
KERNEL = np.ones((3, 3), np.uint8)
GRAY_SAT = 12.0          # mean HSV saturation below this: a b/w photograph


def edges(img: np.ndarray) -> np.ndarray:
    return cv2.dilate(cv2.Canny(img, 50, 150), KERNEL, iterations=2) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def measure(d: Path, spec: dict) -> dict:
    src = cv2.imread(str(d / "source.png"))
    edt = cv2.imread(str(d / "edit.png"))
    if src is None or edt is None:
        raise FileNotFoundError(f"{d}: need both source.png and edit.png")
    if src.shape != edt.shape:
        raise ValueError(f"{d}: {src.shape} vs {edt.shape} - stage 1's "
                         f"resize-back did not run, everything downstream is "
                         f"misaligned")

    union = np.zeros(src.shape[:2], bool)
    for r in spec["regions"]:
        union |= cv2.imread(str(d / r["mask_file"]), cv2.IMREAD_GRAYSCALE) > 127

    diff = np.abs(src.astype(np.int16) - edt.astype(np.int16)).mean(axis=2)
    e_src, e_edt = edges(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)), \
        edges(cv2.cvtColor(edt, cv2.COLOR_BGR2GRAY))

    # Per region as well as whole-frame: one object walking out of its mask is
    # invisible in a frame-wide average of a busy photograph.
    per = {r["label"]: iou(e_src & m, e_edt & m)
           for r in spec["regions"]
           for m in [cv2.imread(str(d / r["mask_file"]),
                                cv2.IMREAD_GRAYSCALE) > 127]}
    worst = min(per, key=per.get)

    return dict(
        base_id=spec["base_id"],
        edge_iou=round(iou(e_src, e_edt), 4),
        bg_absdiff=round(float(diff[~union].mean()), 2),
        fg_absdiff=round(float(diff[union].mean()), 2) if union.any() else None,
        worst_region_iou=round(per[worst], 4),
        worst_region=worst,
        source_saturation=round(
            float(cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[:, :, 1].mean()), 1),
        n_regions=len(spec["regions"]),
        instruction=spec["instruction"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--out", default="out/edit_drift.csv")
    ap.add_argument("--min-iou", type=float, default=0.4,
                    help="report how many bases fall below this. It is a "
                         "reporting threshold, not a filter: nothing is "
                         "deleted here, and stage 4 decides what to exclude.")
    a = ap.parse_args()

    root = Path(a.bases)
    specs = json.loads((root / "bases.json").read_text())
    df = pd.DataFrame([measure(root / s["base_id"], s) for s in specs])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("edge_iou").to_csv(out, index=False)

    print(f"{len(df)} bases -> {out}")
    print("\nedge IoU (1.0 = layout identical; colour-invariant)")
    for p in (1, 5, 10, 25, 50, 75, 90):
        print(f"  p{p:<3} {df.edge_iou.quantile(p / 100):5.2f}")
    print(f"  min  {df.edge_iou.min():5.2f}   max {df.edge_iou.max():5.2f}")
    print(f"\nbelow --min-iou {a.min_iou}: {(df.edge_iou < a.min_iou).sum()} "
          f"of {len(df)} bases")
    print(f"grayscale sources (saturation < {GRAY_SAT}): "
          f"{(df.source_saturation < GRAY_SAT).sum()} - these come back "
          f"colorized, which inflates bg_absdiff without moving anything")
    print(f"\nbg_absdiff (0-255, outside every mask): "
          f"median {df.bg_absdiff.median():.1f}, max {df.bg_absdiff.max():.1f}")

    cols = ["edge_iou", "worst_region_iou", "worst_region", "bg_absdiff",
            "base_id"]
    print("\nworst 10 by edge IoU:")
    print(df.nsmallest(10, "edge_iou")[cols].to_string(index=False))
    print("\nbest 3:")
    print(df.nlargest(3, "edge_iou")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
