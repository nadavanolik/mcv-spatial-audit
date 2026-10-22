"""
Does damage to one region land inside ANOTHER region's box?  [CPU]

The judge is told about regions as boxes (bbox_2d), and stage 0 never checked
that boxes do not overlap (its docstring promised "low mask overlap"; no code
enforced it). On `main` a person sits on a motorcycle, a cup on a table. When
we damage region A and region B's box covers part of A, B's score moving is not
clearly a leak: some of what the judge was told is region B really changed.
Such pairs are contaminated negatives, and they pull AUROC toward 0.5 and
inflate coupling for a reason that has nothing to do with the judge.

This measures it on the RENDERED bytes, not from the masks: a pixel counts as
damaged when the variant differs from the clean edit by more than 5 levels
(scripts/verify_corruption.py's visibility threshold), which includes the
feather band around the mask.

One row per (corrupted variant, non-target region):

  other_box_damaged    share of the other region's box that is visibly damaged.
                       The contamination measure stage 4's --overlap-csv uses.
  damage_share_in_box  share of all visible damage that falls in that box
  damage_px            visible damaged pixels in the whole variant
  box_iou, box_contain IoU of the target and other boxes, and intersection over
                       the SMALLER box -- the geometry alone, render-independent

Usage:
    python -m scripts.region_overlap --manifest out/manifest_main.parquet \\
        --bases data/bases --variants /dev/shm/mcv/variants --out out/overlap.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VISIBLE_LEVELS = 5      # as in verify_corruption.py


def box_slice(bbox, shape) -> tuple[slice, slice]:
    """COCO (x, y, w, h) -> the pixel rows/cols it covers, clipped to the image.
    Rounded outward: a box that touches a pixel covers it."""
    x, y, w, h = bbox
    H, W = shape[:2]
    return (slice(max(0, math.floor(y)), min(H, math.ceil(y + h))),
            slice(max(0, math.floor(x)), min(W, math.ceil(x + w))))


def box_geometry(a, b) -> tuple[float, float]:
    """(IoU, intersection / smaller area) of two COCO boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    iw = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    ih = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    small = min(aw * ah, bw * bh)
    return (inter / union if union > 0 else 0.0,
            inter / small if small > 0 else 0.0)


def measure(manifest: pd.DataFrame, bases: Path, variants: Path) -> tuple[pd.DataFrame, int]:
    rows, missing = [], 0
    edit_cache: dict[str, np.ndarray] = {}
    regions_cache: dict[str, list] = {}
    todo = manifest[~manifest.is_control]
    for i, r in enumerate(todo.itertuples()):
        if r.base_id not in edit_cache:
            edit_cache.clear()      # manifest is grouped by base; keep one
            img = cv2.imread(str(bases / r.base_id / "edit.png"), cv2.IMREAD_COLOR)
            edit_cache[r.base_id] = None if img is None else img.astype(np.int16)
            regions_cache[r.base_id] = json.loads(
                (bases / r.base_id / "regions.json").read_text())
        edit = edit_cache[r.base_id]
        var = cv2.imread(str(variants / f"{r.variant_id}.png"), cv2.IMREAD_COLOR)
        if edit is None or var is None or var.shape != edit.shape:
            missing += 1
            continue
        changed = np.abs(var.astype(np.int16) - edit).mean(axis=2) > VISIBLE_LEVELS
        total = int(changed.sum())
        regions = regions_cache[r.base_id]
        tgt = [g for g in regions if str(g["region_id"]) == str(r.target_region_id)]
        for o in regions:
            if str(o["region_id"]) == str(r.target_region_id):
                continue
            sl = box_slice(o["bbox"], changed.shape)
            inside = changed[sl]
            iou, contain = box_geometry(tgt[0]["bbox"], o["bbox"]) if tgt else (np.nan, np.nan)
            rows.append(dict(
                variant_id=r.variant_id, base_id=r.base_id,
                corruption=r.corruption, severity=r.severity,
                target_region_id=str(r.target_region_id),
                other_region_id=str(o["region_id"]),
                other_box_damaged=float(inside.mean()) if inside.size else 0.0,
                damage_share_in_box=float(inside.sum() / total) if total else 0.0,
                damage_px=total, box_iou=iou, box_contain=contain,
            ))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(todo)} variants")
    return pd.DataFrame(rows), missing


def summarise(ov: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in [("all", ov)] + list(ov.groupby("corruption")):
        per_photo = g.groupby("base_id").other_box_damaged.max()
        out.append(dict(
            corruption=name, n_pairs=len(g),
            pairs_any=float((g.other_box_damaged > 0).mean()),
            pairs_over_10pct=float((g.other_box_damaged > 0.1).mean()),
            mean_box_damaged=float(g.other_box_damaged.mean()),
            n_photos=len(per_photo),
            photos_any=int((per_photo > 0).sum()),
            photos_over_10pct=int((per_photo > 0.1).sum()),
        ))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--variants", default="/dev/shm/mcv/variants")
    ap.add_argument("--out", default="out/overlap.csv")
    a = ap.parse_args()

    man = pd.read_parquet(a.manifest).sort_values(["base_id", "variant_id"])
    ov, missing = measure(man, Path(a.bases), Path(a.variants))
    if missing:
        print(f"NOTE: {missing} variants missing or mis-shaped; re-run stage 2.")
    if ov.empty:
        raise SystemExit("nothing measured - check --bases and --variants")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ov.to_csv(a.out, index=False)
    s = summarise(ov)
    print("\n=== visible damage inside OTHER regions' boxes ===")
    print(s.round(3).to_string(index=False))
    s.to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"), index=False)
    geo = ov.drop_duplicates(["base_id", "target_region_id", "other_region_id"])
    print(f"\nbox geometry: {float((geo.box_iou > 0).mean()):.1%} of region pairs' "
          f"boxes intersect; {float((geo.box_contain > 0.5).mean()):.1%} have the "
          f"smaller box more than half covered")
    print(f"wrote {a.out} ({len(ov)} rows)")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
