"""
Stage 0 — build base specs from COCO instance segmentation.

This is what removes the "no public benchmark has multi-region annotations"
risk from the proposal: COCO ships named, non-overlapping, pixel-accurate
instance masks, so both the regions AND the instruction templates come for
free, with zero manual annotation and a provenance no reviewer will argue with.

Selection criteria (tune in config.yaml):
  - 3-5 instances per image
  - each instance 2-25% of image area (big enough to corrupt visibly, small
    enough that "region" means something)
  - distinct categories, so instructions are unambiguous about their target
  - low mask overlap, so a penalty landing on region A vs B is interpretable

Usage:
    python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
        --images data/coco/val2017 --out data/bases --n 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

# Instruction templates keyed by what we can plausibly ask an editor to do.
COLOR_TEMPLATES = ["make the {label} {color}", "change the {label} to {color}"]
ATTR_TEMPLATES = ["add sunglasses to the {label}", "make the {label} look older"]
REMOVE_TEMPLATES = ["remove the {label}", "erase the {label}"]
COLORS = ["red", "blue", "green", "yellow", "purple"]

# Categories that tolerate a recolour instruction sensibly.
RECOLORABLE = {"car", "bus", "truck", "bicycle", "motorcycle", "boat", "umbrella",
               "chair", "couch", "bed", "backpack", "handbag", "suitcase", "vase",
               "bench", "tie", "kite", "surfboard"}
PERSONLIKE = {"person"}
REMOVABLE = {"trash can", "bottle", "cup", "bowl", "book", "clock", "potted plant",
             "fire hydrant", "parking meter", "stop sign", "traffic light"}


def instruction_for(label: str, rng: random.Random) -> str | None:
    if label in RECOLORABLE:
        return rng.choice(COLOR_TEMPLATES).format(label=label, color=rng.choice(COLORS))
    if label in PERSONLIKE:
        return rng.choice(ATTR_TEMPLATES).format(label=label)
    if label in REMOVABLE:
        return rng.choice(REMOVE_TEMPLATES).format(label=label)
    return None


def select(coco, cfg, rng):
    """Yield (img_info, [ann,...]) for images meeting the region criteria."""
    from pycocotools.coco import COCO
    assert isinstance(coco, COCO)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        area_img = info["width"] * info["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))

        keep, seen = [], set()
        for a in anns:
            label = cats[a["category_id"]]
            frac = a["area"] / area_img
            if not (cfg["min_area_frac"] <= frac <= cfg["max_area_frac"]):
                continue
            if label in seen:                     # distinct categories only
                continue
            if instruction_for(label, rng) is None:
                continue
            seen.add(label)
            keep.append((a, label, frac))

        if cfg["min_regions"] <= len(keep) <= cfg["max_regions"]:
            yield info, keep


def write_base(out_dir: Path, info, keep, coco, rng, images_dir: Path):
    base_id = f"{info['id']:012d}"
    d = out_dir / base_id
    (d / "masks").mkdir(parents=True, exist_ok=True)

    src = cv2.imread(str(images_dir / info["file_name"]), cv2.IMREAD_COLOR)
    if src is None:
        return None
    cv2.imwrite(str(d / "source.png"), src)

    regions, instrs = [], []
    for rid, (a, label, frac) in enumerate(keep):
        m = (coco.annToMask(a) * 255).astype(np.uint8)
        cv2.imwrite(str(d / "masks" / f"r{rid}.png"), m)
        x, y, w, h = [int(v) for v in a["bbox"]]
        regions.append(dict(region_id=rid, label=label, bbox=[x, y, w, h],
                            mask_file=f"masks/r{rid}.png", area_frac=round(frac, 4)))
        instrs.append(instruction_for(label, rng))

    (d / "regions.json").write_text(json.dumps(regions, indent=2))
    instruction = ", ".join(instrs)
    (d / "instruction.txt").write_text(instruction)
    return dict(base_id=base_id, source_file=info["file_name"],
                instruction=instruction, regions=regions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="instances_*.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/bases")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from pycocotools.coco import COCO
    cfg = dict(min_regions=3, max_regions=5, min_area_frac=0.02, max_area_frac=0.25)
    rng = random.Random(a.seed)
    coco = COCO(a.coco)

    out, specs = Path(a.out), []
    for info, keep in select(coco, cfg, rng):
        s = write_base(out, info, keep, coco, rng, Path(a.images))
        if s:
            specs.append(s)
        if len(specs) >= a.n:
            break

    (out / "bases.json").write_text(json.dumps(specs, indent=2))
    print(f"wrote {len(specs)} base specs to {out}")
    print(f"regions/base: {np.mean([len(s['regions']) for s in specs]):.2f}")


if __name__ == "__main__":
    main()
