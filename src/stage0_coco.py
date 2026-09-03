"""
Stage 0 - build base specs from COCO instance segmentation.

This is what removes the "no public benchmark has multi-region annotations"
risk from the proposal: COCO ships named, non-overlapping, pixel-accurate
instance masks, so both the regions AND the instruction templates come for
free, with zero manual annotation and a provenance no reviewer will argue with.

Selection criteria come from config.yaml's `selection:` block (they used to be
hardcoded here, so tuning the config silently did nothing).

  - 3-5 instances per image
  - each instance 2-25% of image area (big enough to corrupt visibly, small
    enough that "region" means something)
  - distinct categories, so instructions are unambiguous about their target
  - UNIQUE categories: no *second* instance of a region's category anywhere in
    the image, not merely none among the regions we kept. Without this,
    "make the car red" gets issued against a photo containing three cars, only
    one of which is a region -- the editor may recolour any of them and the
    judge is asked whether the instruction succeeded inside one box. See
    `duplicate_categories`.
  - low mask overlap, so a penalty landing on region A vs B is interpretable

NOTE ON TESTABILITY: `select` deliberately does not import or type-check
pycocotools. It needs only the six methods listed in `CocoLike`, so the
selection logic - the part with the real risk in it, namely whether the filter
yields enough bases at all - can be exercised on a machine that cannot build
pycocotools installed. See tests/test_stage0.py.

Usage:
    python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
        --images data/coco/val2017 --out data/bases --n 200
    python -m src.stage0_coco --coco ... --images ... --survey
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
import yaml


class CocoLike(Protocol):
    """The entire pycocotools surface stage 0 uses.

    Written down as a Protocol so the selection logic can be driven by a stub
    in tests. pycocotools is an extra install that only stage 0 needs, and
    binding `select` to it would make the highest-risk part of this stage
    untestable on the four VMs that never run stage 0 at all.
    """
    def getImgIds(self) -> list: ...
    def loadImgs(self, ids) -> list: ...
    # Called WITHOUT iscrowd: we need the crowd annotations too, because a
    # "crowd of cars" blob makes "make the car red" just as ambiguous as a
    # second individual car does. Crowds are filtered out of the region
    # candidates in Python instead, which also drops our dependence on
    # pycocotools' own `iscrowd=` comparison semantics.
    def getAnnIds(self, imgIds=..., iscrowd=...) -> list: ...
    def loadAnns(self, ids) -> list: ...
    def getCatIds(self) -> list: ...
    def loadCats(self, ids) -> list: ...
    def annToMask(self, ann) -> np.ndarray: ...


# Instruction templates keyed by what we can plausibly ask an editor to do.
COLOR_TEMPLATES = ["make the {label} {color}", "change the {label} to {color}"]
ATTR_TEMPLATES = ["add sunglasses to the {label}", "make the {label} look older"]
REMOVE_TEMPLATES = ["remove the {label}", "erase the {label}"]
COLORS = ["red", "blue", "green", "yellow", "purple"]

# Categories that tolerate a recolour instruction sensibly.
# Every name here must be one of COCO's 80; tests/test_stage0.py enforces that,
# because a typo silently removes a category from selection rather than failing.
RECOLORABLE = {"car", "bus", "truck", "bicycle", "motorcycle", "boat", "umbrella",
               "chair", "couch", "bed", "backpack", "handbag", "suitcase", "vase",
               "bench", "tie", "kite", "surfboard"}
PERSONLIKE = {"person"}
# "trash can" used to be in here and is NOT a COCO category - it matched
# nothing, for free, forever. COCO's bin-like classes do not exist.
REMOVABLE = {"bottle", "cup", "bowl", "book", "clock", "potted plant",
             "fire hydrant", "parking meter", "stop sign", "traffic light"}

INSTRUCTABLE = RECOLORABLE | PERSONLIKE | REMOVABLE


def instruction_for(label: str, rng: random.Random,
                    used_colors: Optional[set] = None) -> Optional[str]:
    """Draw one instruction for a label, or None if we cannot instruct it.

    CONSUMES RNG STATE. It used to be called twice per region - once in
    `select` merely to test for None, once in `write_base` to get the string -
    which meant the instruction actually written was a *different* draw from
    the one that passed the filter, and depended on how many images had been
    scanned first. `select` now carries the validated instruction through, so
    this is drawn exactly once per region.

    `used_colors` keeps two regions of the SAME image from being sent to the
    same colour. Independent draws produced real instructions like
    "change the car to yellow, change the motorcycle to yellow", which is a bad
    stimulus for a spatial-credit audit: the two regions end up visually
    interchangeable, so a judge that confuses them is indistinguishable from a
    judge that is merely colour-blind to position. Pass a per-image set.
    """
    if label in RECOLORABLE:
        free = [c for c in COLORS if c not in (used_colors or ())]
        # More recolourable regions than colours: fall back rather than fail.
        # max_regions is 5 and so is len(COLORS), so this is unreachable today.
        color = rng.choice(free or COLORS)
        if used_colors is not None:
            used_colors.add(color)
        return rng.choice(COLOR_TEMPLATES).format(label=label, color=color)
    if label in PERSONLIKE:
        return rng.choice(ATTR_TEMPLATES).format(label=label)
    if label in REMOVABLE:
        return rng.choice(REMOVE_TEMPLATES).format(label=label)
    return None


def load_selection_cfg(path: str = "config.yaml") -> dict:
    """Read the `selection:` block. Falls back to the historical defaults if
    the file or the block is absent, so the module still imports standalone."""
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
    """Categories appearing more than once in this image - not usable as regions.

    `anns` must be EVERY annotation of the image, crowds included, not just the
    region candidates: the whole point is the instances selection would
    otherwise never look at. A crowd annotation counts as one duplicate rather
    than the N instances it contains, which is enough - it only ever needs to
    push a count from 1 to 2.

    Only annotations at or above `duplicate_area_frac` count. A second car 40
    pixels wide in the far background will not confuse the editor or the judge,
    and discarding the image for it is pure yield lost. Default 0.01, half of
    min_area_frac: big enough to see, too small to be a region.
    """
    floor = float(cfg.get("duplicate_area_frac", 0.01))
    counts = Counter(cats[a["category_id"]] for a in anns
                     if a["area"] / area_img >= floor)
    return {label for label, n in counts.items() if n > 1}


def candidates(all_anns: list, area_img: int, cats: dict, cfg: dict) -> list:
    """[(ann, label, area_frac), ...] for the regions selection would keep.

    Everything except the instruction, which `select` draws separately because
    drawing consumes RNG state and `survey` must not perturb it.

    `select` and `survey` BOTH go through here. They used to carry two copies
    of this filter - `select` testing `instruction_for(...) is None`, `survey`
    testing `label not in INSTRUCTABLE` - equivalent by accident rather than by
    construction. A survey reporting a yield the selector cannot deliver is
    worse than no survey.
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
    """Yield (img_info, [(ann, label, area_frac, instruction), ...]).

    The instruction is drawn here and carried out, rather than re-drawn later.
    """
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        area_img = info["width"] * info["height"]
        all_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        cand = candidates(all_anns, area_img, cats, cfg)

        # Reject on count BEFORE drawing instructions, so an unusable image
        # does not advance the RNG stream.
        if not (cfg["min_regions"] <= len(cand) <= cfg["max_regions"]):
            continue

        keep, colors = [], set()
        for a, label, frac in cand:
            keep.append((a, label, frac, instruction_for(label, rng, colors)))
        yield info, keep


def survey(coco: CocoLike, cfg: dict, rng: Optional[random.Random] = None) -> dict:
    """How many bases would selection actually yield, and why are the rest
    rejected?

    Worth its own entry point: the filter wants 3-5 *distinct instructable*
    categories at 2-25% area in one image, with no other instance of any of
    them anywhere in the frame, and there is no guarantee COCO val2017 contains
    200 such images. Discovering that after downloading 20GB and burning
    editor-VM hours would be an expensive way to learn it.

    `rng` is accepted and unused: surveying draws no instructions.
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


def write_base(out_dir: Path, info, keep, coco: CocoLike, images_dir: Path):
    base_id = f"{info['id']:012d}"
    d = out_dir / base_id

    # Read the source BEFORE creating anything. The old order left an empty
    # <base_id>/masks/ behind for every image whose file was missing.
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
                         "no writes. Run this BEFORE downloading val2017.")
    a = ap.parse_args()

    from pycocotools.coco import COCO
    cfg = load_selection_cfg(a.config)
    print(f"selection: {cfg}")
    rng = random.Random(a.seed)
    coco = COCO(a.coco)

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
