"""
In which coordinate frame did the judge read the region boxes? [CPU]

The prompt's output format asks the judge to echo each region's bbox_2d.
Qwen3-VL and InternVL3 ground boxes in a 0-1000 frame; if they read pixel boxes
in that frame, they look at the wrong part of the image. For each echoed box,
against the true region box:

  iou_norm   echo vs the true box in 0-1000: the judge located the region.
  iou_pixel  echo vs the pixel numbers read as 0-1000: it copied the numbers
             and looked elsewhere.
  copy       the echo equals the box that was sent.

Only photos where the two readings differ (IoU < 0.5) separate the cases; the
summary reports them separately.

Usage:
    python -m scripts.bbox_frame_check \
        --scores 'results/main8b_2026-10-22/main_qwen_pixel/scores_shard*.parquet' \
        --bases data/bases --out out/bbox_frame.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.judge_prompt import _extract_json, bbox_in_frame  # noqa: E402

DISTINCT = 0.5      # IoU between the two readings below which they are separable


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def echoes(df: pd.DataFrame, bases: Path) -> pd.DataFrame:
    """One row per (response, echoed region)."""
    keys = ["judge", "variant_id", "sample_idx"]
    resp = df.drop_duplicates(keys)
    frame_col = "bbox_frame" in resp.columns
    regions, sizes = {}, {}
    rows = []
    for r in resp.itertuples():
        if r.base_id not in regions:
            regions[r.base_id] = {str(g["region_id"]): g["bbox"] for g in json.loads(
                (bases / r.base_id / "regions.json").read_text())}
            with Image.open(bases / r.base_id / "source.png") as im:
                sizes[r.base_id] = im.size
        sent_frame = r.bbox_frame if frame_col else "pixel"
        d = _extract_json(r.raw or "")
        if not isinstance(d, dict):
            continue
        for item in d.get("edit_region") or []:
            if not isinstance(item, dict):
                continue
            rid, e = str(item.get("id")), item.get("bbox_2d")
            if rid not in regions[r.base_id] or not (
                    isinstance(e, list) and len(e) == 4
                    and all(isinstance(v, (int, float)) for v in e)):
                continue
            bbox, size = regions[r.base_id][rid], sizes[r.base_id]
            px = bbox_in_frame(bbox, "pixel")
            nm = bbox_in_frame(bbox, "norm1000", size)
            sent = px if sent_frame == "pixel" else nm
            rows.append(dict(
                judge=r.judge, variant_id=r.variant_id, base_id=r.base_id,
                presentation=getattr(r, "presentation", "baseline"),
                sent_frame=sent_frame, region_id=rid,
                width=size[0], height=size[1],
                echo=e, iou_norm=iou(e, nm), iou_pixel=iou(e, px),
                readings_iou=iou(px, nm), is_copy=list(e) == list(sent),
                echo_max=max(e)))
    return pd.DataFrame(rows)


def summary(ec: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (judge, frame), g in ec.groupby(["judge", "sent_frame"]):
        for subset, s in (("all", g), ("separable", g[g.readings_iou < DISTINCT])):
            if s.empty:
                continue
            out.append(dict(
                judge=judge, sent_frame=frame, subset=subset, n=len(s),
                n_photos=s.base_id.nunique(),
                iou_norm=s.iou_norm.mean(), iou_pixel=s.iou_pixel.mean(),
                norm_wins=float((s.iou_norm > s.iou_pixel).mean()),
                located=float((s.iou_norm >= 0.5).mean()),
                copy=float(s.is_copy.mean()),
                echo_over_width=float((s.echo_max > s.width).mean())))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--out", default="out/analysis/bbox_frame.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(a.scores))
    if not files:
        raise SystemExit(f"no score files match {a.scores}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ec = echoes(df, Path(a.bases))
    if ec.empty:
        raise SystemExit("no echoed bbox_2d found in `raw`")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ec.to_csv(a.out, index=False)
    s = summary(ec)
    s.to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"), index=False)
    print("=== echoed boxes vs the true region, read in each frame ===")
    print(s.round(3).to_string(index=False))
    print("\n--- reading this ---")
    print("  iou_norm high: the judge places the region where it is.")
    print("  iou_pixel high with sent_frame=pixel: it took our pixel numbers as")
    print("  0-1000 coordinates, so it scored a box shrunk toward the top-left.")
    print("  Read the `separable` rows; on the rest the two readings coincide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
