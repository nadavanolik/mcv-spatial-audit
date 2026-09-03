"""
Stage 0 selection logic, driven by a stub COCO. Laptop-runnable.

`src.stage0_coco.select` is duck-typed against the six methods in `CocoLike`
rather than importing pycocotools, which only stage 0 needs. That makes the
part of stage 0 with the real risk in it testable anywhere.

What is checked here is OUR logic: the area band, the distinct-category rule,
the region-count window, instruction/region alignment, and end-to-end writing
of a base directory. What is NOT checked is whether pycocotools' own API calls
(`annToMask`, `getAnnIds(iscrowd=...)`) behave as assumed - that needs the real
library and the real annotations file:

    python -m src.stage0_coco --coco .../instances_val2017.json --survey

Run:  ./.venv/Scripts/python.exe tests/test_stage0.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import stage0_coco as S  # noqa: E402

# COCO's 80 category names, spelled exactly as the dataset spells them.
# RECOLORABLE / PERSONLIKE / REMOVABLE are matched against `cats[...]` by
# string equality, so a typo does not raise - it just silently excludes the
# category from selection forever. "trash can" sat in REMOVABLE doing exactly
# that until this test was written.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
CAT_ID = {name: i for i, name in enumerate(COCO80)}

W = H = 400
AREA = W * H


class StubCoco:
    """Implements exactly src.stage0_coco.CocoLike, nothing more.

    `images` is {img_id: [(category_name, area_fraction[, iscrowd]), ...]}.
    Masks are solid rectangles sized to the requested area fraction, which is
    enough for write_base: it only ever multiplies the mask by 255 and writes
    it. Crowd annotations are returned like any other -- stage 0 asks for the
    full annotation list and filters them out itself.
    """

    def __init__(self, images: dict):
        self._images = images
        self._anns: dict[int, list] = {}
        ann_id = 0
        for img_id, specs in images.items():
            rows = []
            for spec in specs:
                name, frac = spec[0], spec[1]
                crowd = spec[2] if len(spec) > 2 else 0
                side = int(np.sqrt(frac * AREA))
                rows.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": CAT_ID[name], "iscrowd": int(crowd),
                    "area": frac * AREA,
                    "bbox": [10.0, 10.0, float(side), float(side)],
                    "_side": side,
                })
                ann_id += 1
            self._anns[img_id] = rows

    def getImgIds(self):
        return list(self._images)

    def loadImgs(self, ids):
        ids = [ids] if isinstance(ids, int) else ids
        return [{"id": i, "width": W, "height": H, "file_name": f"{i:012d}.jpg"}
                for i in ids]

    def getAnnIds(self, imgIds=None, iscrowd=None):
        return [a["id"] for a in self._anns[imgIds]]

    def loadAnns(self, ids):
        flat = {a["id"]: a for rows in self._anns.values() for a in rows}
        return [flat[i] for i in ids]

    def getCatIds(self):
        return list(CAT_ID.values())

    def loadCats(self, ids):
        return [{"id": CAT_ID[n], "name": n} for n in COCO80]

    def annToMask(self, ann):
        m = np.zeros((H, W), np.uint8)
        s = ann["_side"]
        m[10:10 + s, 10:10 + s] = 1
        return m


CFG = dict(min_regions=3, max_regions=5, min_area_frac=0.02, max_area_frac=0.25,
           duplicate_area_frac=0.01)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def main() -> int:
    ok = True

    print("=" * 68)
    print("1. CATEGORY NAMES all exist in COCO's 80")
    print("=" * 68)
    for setname in ("RECOLORABLE", "PERSONLIKE", "REMOVABLE"):
        s = getattr(S, setname)
        bad = sorted(x for x in s if x not in CAT_ID)
        ok &= check(f"{setname} ({len(s)} entries) all real COCO names",
                    not bad, f"unknown: {bad}" if bad else "")
    ok &= check("every instructable category yields an instruction",
                all(S.instruction_for(c, random.Random(0)) is not None
                    for c in S.INSTRUCTABLE))
    ok &= check("a non-instructable category yields None",
                S.instruction_for("giraffe", random.Random(0)) is None)

    print("\n" + "=" * 68)
    print("2. SELECTION honours the area band, distinctness and count window")
    print("=" * 68)
    imgs = {
        1: [("car", 0.10), ("bottle", 0.05), ("chair", 0.08)],          # 3 -> keep
        2: [("car", 0.10), ("bottle", 0.05)],                            # 2 -> reject
        3: [("car", 0.001), ("bottle", 0.05), ("chair", 0.08),
            ("cup", 0.06)],                                              # tiny dropped -> 3
        4: [("car", 0.90), ("bottle", 0.05), ("chair", 0.08)],           # huge dropped -> 2
        5: [("car", 0.10), ("car", 0.09), ("bottle", 0.05),
            ("chair", 0.08)],                                            # dup car -> 2
        6: [("giraffe", 0.10), ("zebra", 0.09), ("bottle", 0.05),
            ("chair", 0.08)],                                            # 2 instructable
        7: [("car", .05), ("bottle", .05), ("chair", .05), ("cup", .05),
            ("bowl", .05), ("book", .05)],                               # 6 -> too many
        # The uniqueness rule's three interesting cases.
        8: [("car", 0.10), ("car", 0.005), ("bottle", 0.05),
            ("chair", 0.08)],                                # 2nd car invisible -> 3
        9: [("car", 0.10), ("car", 0.09, 1), ("bottle", 0.05),
            ("chair", 0.08)],                                # crowd of cars -> 2
        10: [("person", 0.10, 1), ("car", 0.10), ("bottle", 0.05),
             ("chair", 0.08)],                               # crowd is not a region -> 3
    }
    coco = StubCoco(imgs)
    got = {info["id"]: keep for info, keep in S.select(coco, CFG, random.Random(0))}
    ok &= check("selected exactly the qualifying images",
                sorted(got) == [1, 3, 8, 10], f"got {sorted(got)}")
    ok &= check("image 3 dropped the 0.1%-area region",
                len(got.get(3, [])) == 3 and
                all(f >= CFG["min_area_frac"] for _, _, f, _ in got[3]))
    ok &= check("image 7 rejected for having too many regions", 7 not in got)
    ok &= check("image 6 rejected: only 2 instructable categories", 6 not in got)

    print("\n" + "=" * 68)
    print("2b. UNIQUENESS: a region's category appears exactly once in the image")
    print("=" * 68)
    # Not "distinct among the regions we kept" - distinct in the whole frame.
    # Otherwise "make the car red" is issued against a photo with three cars,
    # only one of which is a region, and neither the editor nor the judge has
    # any way to know which one was meant.
    ok &= check("image 5 rejected: a second in-band car makes 'the car' ambiguous",
                5 not in got)
    ok &= check("image 8 kept: the second car is below duplicate_area_frac",
                8 in got and [lab for _, lab, _, _ in got.get(8, [])] ==
                ["car", "bottle", "chair"], f"{[l for _, l, _, _ in got.get(8, [])]}")
    ok &= check("image 9 rejected: a CROWD of cars counts as a duplicate",
                9 not in got)
    ok &= check("image 10 kept, and the crowd annotation is not itself a region",
                10 in got and "person" not in
                [lab for _, lab, _, _ in got.get(10, [])])

    # duplicate_categories directly, so a failure above localises.
    cats = {CAT_ID[n]: n for n in COCO80}
    dups = lambda i: S.duplicate_categories(coco.loadAnns(coco.getAnnIds(imgIds=i)),
                                            AREA, cats, CFG)
    ok &= check("duplicate_categories flags the doubled category", dups(5) == {"car"})
    ok &= check("an invisible duplicate is not flagged", dups(8) == set(), f"{dups(8)}")
    ok &= check("a crowd annotation is counted", dups(9) == {"car"})
    ok &= check("a lone category is never flagged", dups(1) == set())
    ok &= check("the floor is configurable",
                S.duplicate_categories(coco.loadAnns(coco.getAnnIds(imgIds=8)), AREA,
                                       cats, {**CFG, "duplicate_area_frac": 0.001})
                == {"car"})

    print("\n" + "=" * 68)
    print("3. INSTRUCTIONS are drawn once and match their region, in order")
    print("=" * 68)
    for img_id, keep in got.items():
        for _, label, _, instr in keep:
            ok &= check(f"img {img_id}: '{instr}' names '{label}'",
                        label in instr)
    # No two regions of one image may be sent to the same colour. Independent
    # per-region draws produced "change the car to yellow, change the
    # motorcycle to yellow" on a real COCO image, which makes the two regions
    # visually interchangeable - the worst possible stimulus for an audit whose
    # whole question is whether the judge can tell regions apart.
    COLOR_WORDS = set(S.COLORS)
    for img_id, keep in got.items():
        used = [w for _, _, _, instr in keep
                for w in instr.split() if w in COLOR_WORDS]
        ok &= check(f"img {img_id}: colours are distinct ({used})",
                    len(used) == len(set(used)))
    many = {99: [("car", .05), ("bus", .05), ("truck", .05), ("boat", .05),
                 ("chair", .05)]}
    k = next(iter(S.select(StubCoco(many), CFG, random.Random(3))))[1]
    used5 = [w for _, _, _, i in k for w in i.split() if w in COLOR_WORDS]
    ok &= check(f"5 recolourable regions get 5 distinct colours ({used5})",
                len(used5) == 5 and len(set(used5)) == 5)

    # The old code drew the instruction twice - once to test for None in
    # select(), once for real in write_base() - so the string written could
    # name a different colour than the one that passed the filter.
    a = list(S.select(coco, CFG, random.Random(7)))
    b = list(S.select(coco, CFG, random.Random(7)))
    ok &= check("selection is deterministic given a seed",
                [[x[3] for x in k] for _, k in a] == [[x[3] for x in k] for _, k in b])
    c = list(S.select(coco, CFG, random.Random(8)))
    ok &= check("a different seed gives different instructions",
                [[x[3] for x in k] for _, k in a] != [[x[3] for x in k] for _, k in c])

    print("\n" + "=" * 68)
    print("4. WRITE_BASE produces what stages 1-3 actually open")
    print("=" * 68)
    tmp = Path(tempfile.mkdtemp(prefix="mcv_stage0_"))
    imgdir = tmp / "images"
    imgdir.mkdir()
    for i in got:
        cv2.imwrite(str(imgdir / f"{i:012d}.jpg"),
                    np.full((H, W, 3), 128, np.uint8))
    out = tmp / "bases"
    specs = []
    for info, keep in S.select(coco, CFG, random.Random(0)):
        s = S.write_base(out, info, keep, coco, imgdir)
        if s:
            specs.append(s)
    ok &= check("wrote a spec per selected image", len(specs) == len(got),
                f"{len(specs)} vs {len(got)}")

    spec = specs[0]
    d = out / spec["base_id"]
    # These are the exact paths stage 1, 2 and 3 open. stage3.preflight looks
    # for regions.json and source.png; stage2 opens masks/r{i}.png.
    ok &= check("source.png exists", (d / "source.png").exists())
    ok &= check("regions.json exists", (d / "regions.json").exists())
    ok &= check("instruction.txt exists", (d / "instruction.txt").exists())
    regions = json.loads((d / "regions.json").read_text())
    ok &= check("every region's mask file exists",
                all((d / r["mask_file"]).exists() for r in regions))
    ok &= check("region_ids are 0..n-1 in order",
                [r["region_id"] for r in regions] == list(range(len(regions))))
    ok &= check("regions.json carries the keys judge_prompt reads",
                all({"region_id", "label", "bbox"} <= set(r) for r in regions))
    ok &= check("instruction names every region's label",
                all(r["label"] in spec["instruction"] for r in regions))

    m = cv2.imread(str(d / regions[0]["mask_file"]), cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(str(d / "source.png"), cv2.IMREAD_COLOR)
    # stage2_corrupt indexes the mask against edit.png, which stage1 resizes
    # back to source.png's size. If these ever disagree, corruption lands in
    # the wrong place and every downstream number is meaningless.
    ok &= check("mask resolution == source resolution",
                m.shape[:2] == src.shape[:2], f"{m.shape[:2]} vs {src.shape[:2]}")
    ok &= check("mask is binary 0/255", set(np.unique(m)) <= {0, 255},
                f"{np.unique(m)[:5]}")
    ok &= check("mask is non-empty", int(m.sum()) > 0)

    bbox = regions[0]["bbox"]
    ys, xs = np.nonzero(m)
    ok &= check("bbox (x,y,w,h) contains the mask",
                bbox[0] <= xs.min() and bbox[1] <= ys.min()
                and bbox[0] + bbox[2] >= xs.max()
                and bbox[1] + bbox[3] >= ys.max(),
                f"bbox={bbox} mask x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    print("\n" + "=" * 68)
    print("5. A MISSING IMAGE FILE is skipped without leaving debris")
    print("=" * 68)
    out2 = tmp / "bases2"
    info, keep = next(iter(S.select(coco, CFG, random.Random(0))))
    r = S.write_base(out2, info, keep, coco, tmp / "does_not_exist")
    ok &= check("returns None", r is None)
    ok &= check("left no half-built base directory",
                not (out2 / f"{info['id']:012d}").exists())

    print("\n" + "=" * 68)
    print("6. CONFIG: the selection block is read, not hardcoded")
    print("=" * 68)
    root = Path(__file__).resolve().parents[1]
    cfg = S.load_selection_cfg(str(root / "config.yaml"))
    ok &= check("config.yaml selection block is picked up",
                cfg == CFG, f"{cfg}")
    custom = tmp / "custom.yaml"
    custom.write_text("selection:\n  min_regions: 2\n  max_regions: 9\n")
    c2 = S.load_selection_cfg(str(custom))
    ok &= check("a changed config actually changes the criteria",
                c2["min_regions"] == 2 and c2["max_regions"] == 9, f"{c2}")
    ok &= check("unset keys keep their defaults", c2["min_area_frac"] == 0.02)
    ok &= check("a missing config file falls back to defaults",
                S.load_selection_cfg(str(tmp / "nope.yaml")) == CFG)
    n2 = len(list(S.select(coco, c2, random.Random(0))))
    ok &= check("the widened window really selects more images",
                n2 > len(got), f"{n2} vs {len(got)}")

    print("\n" + "=" * 68)
    print("7. SURVEY counts what selection would yield")
    print("=" * 68)
    s = S.survey(coco, CFG, random.Random(0))
    print(f"  {s}")
    ok &= check("survey scanned every image", s["n_images"] == len(imgs))
    ok &= check("survey's usable count matches select()",
                s["usable"] == len(got), f"{s['usable']} vs {len(got)}")

    print("\n" + "=" * 68)
    print("ALL PASS" if ok else "FAILURES ABOVE")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
