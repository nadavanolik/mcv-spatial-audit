"""
Stage 0: select photographs, regions and instructions from COCO instance masks.

Criteria (config.yaml, `selection:`):
  - 3-5 regions per image, each 2-25% of the image area
  - distinct categories, with no other instance of a region's category anywhere
    in the image, so an instruction like "make the car red" is unambiguous
  - categories we can write an instruction for (INSTRUCTABLE)

`select` uses only the methods in `CocoLike`, so tests can drive it with a stub
instead of pycocotools (tests/test_stage0.py).

Usage:
    python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
        --images data/coco/val2017 --out data/bases --n 200
    python -m src.stage0_coco --coco ... --images ... --survey
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
import yaml


class CocoLike(Protocol):
    """The part of pycocotools' COCO API that stage 0 uses."""
    def getImgIds(self) -> list: ...
    def loadImgs(self, ids) -> list: ...
    # Called without iscrowd: crowd annotations count as duplicates too.
    def getAnnIds(self, imgIds=..., iscrowd=...) -> list: ...
    def loadAnns(self, ids) -> list: ...
    def getCatIds(self) -> list: ...
    def loadCats(self, ids) -> list: ...
    def annToMask(self, ann) -> np.ndarray: ...


# Instruction templates keyed by what we can plausibly ask an editor to do.
COLOR_TEMPLATES = ["make the {label} {color}", "change the {label} to {color}"]
MATERIAL_TEMPLATES = ["make the {label} look like it is made of {material}",
                      "change the {label} to {material}"]
ATTR_TEMPLATES = ["add sunglasses to the {label}", "make the {label} look older",
                  "add a beard to the {label}", "add a moustache to the {label}"]
REMOVE_TEMPLATES = ["remove the {label}", "erase the {label}"]
COLORS = ["red", "blue", "green", "yellow", "purple"]
MATERIALS = ["wood", "metal", "glass", "marble", "leather"]

# Chance that a region that can take either a colour or a material gets a
# material instruction (drawn per region).
MATERIAL_SHARE = 0.5

# At most one removal per image; further removable regions get a material
# instruction instead, so the region count is unchanged.
MAX_REMOVALS_PER_BASE = 1

# Categories by instruction type. Every name must be one of COCO's 80
# (checked by tests/test_stage0.py).
RECOLORABLE = {"car", "bus", "truck", "bicycle", "motorcycle", "boat", "umbrella",
               "chair", "couch", "bed", "backpack", "handbag", "suitcase", "vase",
               "bench", "tie", "kite", "surfboard"}
PERSONLIKE = {"person"}
REMOVABLE = {"bottle", "cup", "bowl", "book", "clock", "potted plant",
             "fire hydrant", "parking meter", "stop sign", "traffic light"}

# Solid objects a material swap makes sense on. Contains every REMOVABLE
# category, so a removal over the cap always has a fallback.
MATERIALIZABLE = {"chair", "couch", "bed", "bench", "vase", "suitcase",
                  "handbag", "backpack", "umbrella", "car", "boat", "bicycle",
                  "surfboard", "tie", "truck", "bus", "motorcycle", "kite",
                  "bottle", "cup", "bowl", "book", "clock", "potted plant",
                  "fire hydrant", "parking meter", "stop sign", "traffic light"}

INSTRUCTABLE = RECOLORABLE | PERSONLIKE | REMOVABLE | MATERIALIZABLE


def _draw_unique(rng: random.Random, pool: list, used: Optional[set]) -> str:
    """Pick from `pool`, preferring values this image has not used yet."""
    free = [v for v in pool if v not in (used or ())]
    v = rng.choice(free or pool)
    if used is not None:
        used.add(v)
    return v


def instruction_for(label: str, rng: random.Random,
                    used_colors: Optional[set] = None,
                    used_materials: Optional[set] = None,
                    allow_remove: bool = True) -> Optional[str]:
    """Draw one instruction for a label, or None if it cannot be instructed.

    Consumes RNG state, so it is called exactly once per region.
    `used_colors` / `used_materials` are per-image sets that keep two regions
    from getting the same colour or material. `allow_remove=False` once the
    image has used its removal (MAX_REMOVALS_PER_BASE).
    """
    if label in REMOVABLE and allow_remove:
        return rng.choice(REMOVE_TEMPLATES).format(label=label)

    can_color = label in RECOLORABLE
    can_material = label in MATERIALIZABLE
    if can_color or can_material:
        use_material = (rng.random() < MATERIAL_SHARE
                        if can_color and can_material else can_material)
        if use_material:
            material = _draw_unique(rng, MATERIALS, used_materials)
            return rng.choice(MATERIAL_TEMPLATES).format(label=label,
                                                         material=material)
        color = _draw_unique(rng, COLORS, used_colors)
        return rng.choice(COLOR_TEMPLATES).format(label=label, color=color)

    if label in PERSONLIKE:
        return rng.choice(ATTR_TEMPLATES).format(label=label)
    return None


def load_selection_cfg(path: str = "config.yaml") -> dict:
    """Read config.yaml's `selection:` block, with defaults if it is absent."""
    defaults = dict(min_regions=3, max_regions=5,
                    min_area_frac=0.02, max_area_frac=0.25,
                    duplicate_area_frac=0.01)
    try:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
    except FileNotFoundError:
        return defaults
    return {**defaults, **(cfg.get("selection") or {})}


def duplicate_categories(anns: list, area_img: int, cats: dict,
                         cfg: dict) -> set:
    """Categories that appear more than once in the image; not usable as regions.

    `anns` is every annotation of the image, crowds included. Instances smaller
    than `duplicate_area_frac` of the image are ignored.
    """
    floor = float(cfg.get("duplicate_area_frac", 0.01))
    counts = Counter(cats[a["category_id"]] for a in anns
                     if a["area"] / area_img >= floor)
    return {label for label, n in counts.items() if n > 1}


def candidates(all_anns: list, area_img: int, cats: dict, cfg: dict) -> list:
    """[(ann, label, area_frac), ...] for the regions selection would keep.

    Shared by `select` and `survey`; draws no instructions.
    """
    dup = duplicate_categories(all_anns, area_img, cats, cfg)

    out, seen = [], set()
    for a in all_anns:
        if a.get("iscrowd"):                      # crowds are never a region
            continue
        label = cats[a["category_id"]]
        if label in dup:                          # another instance elsewhere
            continue
        frac = a["area"] / area_img
        if not (cfg["min_area_frac"] <= frac <= cfg["max_area_frac"]):
            continue
        if label in seen:                         # distinct categories only
            continue
        if label not in INSTRUCTABLE:
            continue
        seen.add(label)
        out.append((a, label, frac))
    return out


def select(coco: CocoLike, cfg: dict, rng: random.Random):
    """Yield (img_info, [(ann, label, area_frac, instruction), ...])."""
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        area_img = info["width"] * info["height"]
        all_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        cand = candidates(all_anns, area_img, cats, cfg)

        # Reject before drawing instructions, so skipped images do not advance
        # the RNG.
        if not (cfg["min_regions"] <= len(cand) <= cfg["max_regions"]):
            continue

        keep, colors, materials = [], set(), set()
        removals = 0
        for a, label, frac in cand:
            # The first removable region in annotation order uses the removal.
            allow = removals < MAX_REMOVALS_PER_BASE
            instr = instruction_for(label, rng, colors, materials,
                                    allow_remove=allow)
            if allow and label in REMOVABLE:
                removals += 1
            keep.append((a, label, frac, instr))
        yield info, keep


def survey(coco: CocoLike, cfg: dict, rng: Optional[random.Random] = None) -> dict:
    """How many images the filter would accept, from the annotations alone.

    Returns a histogram of usable regions per image. `rng` is unused.
    """
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    hist: dict[int, int] = {}
    n_img = 0
    for img_id in coco.getImgIds():
        n_img += 1
        info = coco.loadImgs(img_id)[0]
        area_img = info["width"] * info["height"]
        all_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        k = len(candidates(all_anns, area_img, cats, cfg))
        hist[k] = hist.get(k, 0) + 1
    usable = sum(v for k, v in hist.items()
                 if cfg["min_regions"] <= k <= cfg["max_regions"])
    return {"n_images": n_img, "hist": dict(sorted(hist.items())), "usable": usable}


def image_url(info: dict, coco_json: str) -> str:
    """Download URL of one COCO image: `coco_url`, or rebuilt from the split
    name in the annotations filename."""
    url = info.get("coco_url")
    if url:
        return url
    split = Path(coco_json).stem.split("_")[-1]
    return f"http://images.cocodataset.org/{split}/{info['file_name']}"


def write_base(out_dir: Path, info, keep, coco: CocoLike, images_dir: Path):
    base_id = f"{info['id']:012d}"
    d = out_dir / base_id

    # Read the source first, so a missing image leaves no empty directory.
    src = cv2.imread(str(images_dir / info["file_name"]), cv2.IMREAD_COLOR)
    if src is None:
        return None
    (d / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "source.png"), src)

    regions, instrs = [], []
    for rid, (a, label, frac, instr) in enumerate(keep):
        m = (coco.annToMask(a) * 255).astype(np.uint8)
        cv2.imwrite(str(d / "masks" / f"r{rid}.png"), m)
        x, y, w, h = [int(v) for v in a["bbox"]]
        regions.append(dict(region_id=rid, label=label, bbox=[x, y, w, h],
                            mask_file=f"masks/r{rid}.png", area_frac=round(frac, 4)))
        instrs.append(instr)

    (d / "regions.json").write_text(json.dumps(regions, indent=2))
    instruction = ", ".join(instrs)
    (d / "instruction.txt").write_text(instruction)
    return dict(base_id=base_id, source_file=info["file_name"],
                instruction=instruction, regions=regions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="instances_*.json")
    ap.add_argument("--images", help="image dir; not needed with --survey")
    ap.add_argument("--out", default="data/bases")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--survey", action="store_true",
                    help="report how many bases the filter would yield, and "
                         "the distribution of usable regions per image, then "
                         "exit. Needs only the annotations file - no images, "
                         "no writes. Run this BEFORE downloading any images.")
    ap.add_argument("--list-urls", action="store_true",
                    help="print the download URL of each image selection would "
                         "use, most-preferred first, then exit. Feed to "
                         "`wget -P <dir> -i -`. train2017 qualifies about 1 "
                         "image in 100, so fetching the whole 18GB split to "
                         "keep 150 files is 700x more download than the job "
                         "needs, and does not fit beside FLUX on a 90G root.")
    a = ap.parse_args()

    from pycocotools.coco import COCO

    # Under --list-urls stdout must hold only URLs, so progress output
    # (including pycocotools') goes to stderr.
    chatter = sys.stderr if a.list_urls else sys.stdout

    cfg = load_selection_cfg(a.config)
    print(f"selection: {cfg}", file=chatter)
    rng = random.Random(a.seed)
    with redirect_stdout(chatter):
        coco = COCO(a.coco)

    if a.list_urls:
        for i, (info, _) in enumerate(select(coco, cfg, rng)):
            if i >= a.n:
                break
            print(image_url(info, a.coco))
        return

    if a.survey:
        s = survey(coco, cfg, rng)
        print(f"\n{s['n_images']} images scanned")
        print("usable regions per image -> image count:")
        for k, v in s["hist"].items():
            mark = "  <- selectable" if cfg["min_regions"] <= k <= cfg["max_regions"] else ""
            print(f"  {k}: {v}{mark}")
        print(f"\n{s['usable']} images meet the {cfg['min_regions']}-"
              f"{cfg['max_regions']} region criterion")
        if s["usable"] < 200:
            print("WARNING: fewer than 200 usable bases. Either lower "
                  "min_regions, widen the area band, or add categories to "
                  "RECOLORABLE / REMOVABLE in this file.")
        return

    if not a.images:
        ap.error("--images is required unless --survey is given")

    out, specs = Path(a.out), []
    for info, keep in select(coco, cfg, rng):
        s = write_base(out, info, keep, coco, Path(a.images))
        if s:
            specs.append(s)
        if len(specs) >= a.n:
            break

    out.mkdir(parents=True, exist_ok=True)
    (out / "bases.json").write_text(json.dumps(specs, indent=2))
    print(f"wrote {len(specs)} base specs to {out}")
    if not specs:
        raise SystemExit("no bases selected - run --survey to see why")
    print(f"regions/base: {np.mean([len(s['regions']) for s in specs]):.2f}")
    if len(specs) < a.n:
        print(f"WARNING: asked for {a.n}, got {len(specs)}. The filter ran out "
              f"of qualifying images; run --survey for the breakdown.")


if __name__ == "__main__":
    main()
