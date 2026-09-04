"""Guard the nuisance and exploitability presentation axes.

These axes are unusually easy to get silently wrong. A permutation that never
permutes, a subset that quietly drops the region we damaged, or a prompt order
shuffled while the grammar still pins the answers to the canonical order -- all
three produce a full parquet of plausible scores and a "no effect" conclusion
that is really a bug. Nothing downstream would reveal any of them.

Also builds a synthetic 2-base fixture, which is what makes
`stage3_judge --dry-run` runnable on a laptop with no stage-1 output.

Run:  ./.venv/Scripts/python.exe tests/test_nuisance.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.judge_prompt import region_reward, sc_json_schema      # noqa: E402
from src.presentation import (PRESENTATIONS, draw_boxes, enhance,  # noqa: E402
                              present)
from src.schema import variant_id                              # noqa: E402
from src.stage3_judge import build_requests                    # noqa: E402

REGIONS = [
    {"region_id": 0, "label": "cat", "bbox": [10, 10, 60, 60],
     "mask_file": "masks/r0.png", "area_frac": 0.09},
    {"region_id": 1, "label": "car", "bbox": [90, 20, 50, 50],
     "mask_file": "masks/r1.png", "area_frac": 0.06},
    {"region_id": 2, "label": "tree", "bbox": [30, 100, 70, 60],
     "mask_file": "masks/r2.png", "area_frac": 0.10},
]
IDS = [r["region_id"] for r in REGIONS]


def _img(seed=0, size=(200, 200)):
    """Textured, not flat: enhance() sharpens edges, and a flat field would let
    a broken implementation pass by changing nothing."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3),
                                        dtype=np.uint8), "RGB")


# --- the text axes -----------------------------------------------------------

def test_shuffle_is_a_permutation():
    for vid in (variant_id("b0", 0, "blur", 1, "full"),
                variant_id("b1", 2, "none", 0, "full")):
        got = present(REGIONS, 0, vid, "shuffle")
        assert sorted(r["region_id"] for r in got) == sorted(IDS), got
        assert len(got) == len(REGIONS)


def test_shuffle_is_deterministic():
    vid = variant_id("b0", 0, "blur", 1, "full")
    orders = {tuple(r["region_id"] for r in present(REGIONS, 0, vid, "shuffle"))
              for _ in range(5)}
    assert len(orders) == 1, "shuffle is non-deterministic: %s" % (orders,)


def test_shuffle_varies_across_variants():
    """A single global permutation would be perfectly confounded with region
    id: every variant would ask about region 2 first, and 'the score follows
    slot 0' and 'the score follows region 2' would be the same statement."""
    orders = {tuple(r["region_id"] for r in
                    present(REGIONS, 0,
                            variant_id("b%d" % i, 0, "blur", 1, "full"), "shuffle"))
              for i in range(40)}
    assert len(orders) > 1, "shuffle applies one fixed order to every variant"
    assert any(o != tuple(IDS) for o in orders), "shuffle never changes the order"


def test_subset_keeps_the_target_and_drops_one():
    for target in IDS:
        vid = variant_id("b0", target, "blur", 1, "full")
        ids = [r["region_id"] for r in present(REGIONS, target, vid, "subset")]
        assert len(ids) == len(IDS) - 1, ids
        assert target in ids, "subset dropped the damaged region %s: %s" % (target, ids)


def test_subset_handles_a_single_region_base():
    """selection.min_regions is 3 today, but that lives in config.yaml and is
    being edited on another branch. Degrade to a no-op, not an IndexError."""
    got = present(REGIONS[:1], 0, variant_id("b0", 0, "blur", 1, "full"), "subset")
    assert [r["region_id"] for r in got] == [0]


def test_subset_accepts_a_string_target():
    """target_region_id is an int in the manifest and a str everywhere
    downstream. A silent int/str mismatch here would drop the target."""
    vid = variant_id("b0", 1, "blur", 1, "full")
    ids = [r["region_id"] for r in present(REGIONS, "1", vid, "subset")]
    assert 1 in ids, ids


def test_baseline_is_the_identity():
    vid = variant_id("b0", 0, "blur", 1, "full")
    assert present(REGIONS, 0, vid, "baseline") == REGIONS
    for mode in ("noimg", "enhance", "box"):
        assert [r["region_id"] for r in present(REGIONS, 0, vid, mode)] == IDS


# --- the axis that would silently do nothing ---------------------------------

def test_schema_slots_follow_the_presented_order():
    """THE test. judge_prompt._build_sc_schema fills prefixItems positionally
    from the id list, so the grammar pins slot k to ids[k]. Permute the prompt
    but not the schema and the model is forced back into canonical order: a
    nuisance axis that runs, costs GPU time, and measures nothing."""
    vid = variant_id("b0", 0, "blur", 1, "full")
    shown = [r["region_id"] for r in present(REGIONS, 0, vid, "shuffle")]
    slots = [it["properties"]["id"]["const"] for it in
             sc_json_schema(shown)["properties"]["edit_region"]["prefixItems"]]
    assert slots == shown, \
        "grammar slots %s do not follow prompt order %s" % (slots, shown)

    sub = [r["region_id"] for r in present(REGIONS, 0, vid, "subset")]
    sc = sc_json_schema(sub)["properties"]["edit_region"]
    assert sc["minItems"] == sc["maxItems"] == len(sub) == len(IDS) - 1


# --- the image axes ----------------------------------------------------------

def test_enhance_is_deterministic_and_changes_pixels():
    im = _img(1)
    a, b = enhance(im), enhance(im)
    assert a.tobytes() == b.tobytes(), "enhance is non-deterministic"
    assert a.size == im.size, "enhance resized the image"
    assert a.tobytes() != im.tobytes(), "enhance changed nothing"


def test_enhance_is_global():
    """The exploit's whole claim is that it carries no per-region information.
    If it changed only part of the frame it would be a corruption, not a
    cosmetic lift, and a region score reacting to it would prove nothing."""
    im = _img(2)
    d = np.abs(np.asarray(enhance(im), int) - np.asarray(im, int)).sum(2)
    h, w = d.shape
    quads = [d[:h // 2, :w // 2], d[:h // 2, w // 2:],
             d[h // 2:, :w // 2], d[h // 2:, w // 2:]]
    touched = [float((q > 0).mean()) for q in quads]
    assert min(touched) > 0.5, "enhance is not global: %s" % (touched,)


def test_draw_boxes_marks_only_the_presented_regions():
    im = _img(3)
    out = draw_boxes(im, REGIONS[:1])
    d = np.abs(np.asarray(out, int) - np.asarray(im, int)).sum(2)
    assert d[10, 10] > 0, "no box drawn at the first region's corner"
    assert d[90, 20] == 0 and d[30, 100] == 0, "drew a box for an unlisted region"
    assert out.size == im.size


# --- the arithmetic the exploit rests on -------------------------------------

def test_reward_rises_with_pq_at_fixed_phi():
    """Equation (3): sqrt(phi * min(PQ)) / C, with min(PQ) a single IMAGE-level
    term multiplying every region. A global cosmetic change that raises PQ
    therefore raises every region's reward with no edit improved. That is the
    hypothesis `enhance` tests, and this is its arithmetic."""
    sc = {"regions": {0: [20.0, 18.0], 1: [10.0, 25.0]}, "background": 22.0}
    lo = [region_reward(sc, r, [12.0, 12.0]) for r in (0, 1, "bg")]
    hi = [region_reward(sc, r, [24.0, 24.0]) for r in (0, 1, "bg")]
    assert all(h > l for h, l in zip(hi, lo)), (lo, hi)
    # phi is untouched by PQ, so the gain is entirely the image-level factor.
    assert all(abs(h / l - 2.0 ** 0.5) < 1e-9 for h, l in zip(hi, lo))


# --- fixture + end-to-end request assembly -----------------------------------

def build_fixture(root: Path, n_bases: int = 2) -> Path:
    """A synthetic stand-in for stage 0 + stage 1 output.

    The laptop has no data/bases (stage 1 is editor-VM only), so without this
    there is no way to exercise build_requests or --dry-run here at all.
    """
    bases = root / "bases"
    specs = []
    for i in range(n_bases):
        bid = "fix%03d" % i
        d = bases / bid
        (d / "masks").mkdir(parents=True, exist_ok=True)
        _img(i).save(d / "source.png")
        _img(i + 100).save(d / "edit.png")
        for r in REGIONS:
            m = np.zeros((200, 200), np.uint8)
            x, y, w, h = r["bbox"]
            m[y:y + h, x:x + w] = 255
            Image.fromarray(m, "L").save(d / "masks" / ("r%d.png" % r["region_id"]))
        (d / "regions.json").write_text(json.dumps(REGIONS))
        specs.append({"base_id": bid, "source_file": bid + ".jpg",
                      "instruction": "Make the cat orange and remove the car.",
                      "regions": REGIONS})
    (bases / "bases.json").write_text(json.dumps(specs))
    return bases


def fixture_manifest(bases: Path):
    from src.schema import BaseSpec, Region, build_manifest
    specs = json.loads((bases / "bases.json").read_text())
    b = [BaseSpec(base_id=s["base_id"], source_file=s["source_file"],
                  instruction=s["instruction"],
                  regions=tuple(Region(r["region_id"], r["label"], tuple(r["bbox"]),
                                       r["mask_file"], r["area_frac"])
                                for r in s["regions"]))
         for s in specs]
    return build_manifest(b, dict(corruptions=["none", "blur"], severities=[1],
                                  area_bins=["full"]))


def render_variants(bases: Path, df, out: Path) -> None:
    """Stage 2 for real, so the variant files are the ones stage 3 would open."""
    import cv2
    from src.corruptions import apply_corruption
    out.mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        edit = cv2.cvtColor(cv2.imread(str(bases / row.base_id / "edit.png")),
                            cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(bases / row.base_id / "masks" /
                              ("r%d.png" % row.target_region_id)),
                          cv2.IMREAD_GRAYSCALE)
        got = apply_corruption(edit, mask, row.corruption, row.severity,
                               row.area_bin, int(row.seed))
        cv2.imwrite(str(out / (row.variant_id + ".png")),
                    cv2.cvtColor(got, cv2.COLOR_RGB2BGR))


def _listed_ids(msg) -> list:
    """The region ids actually written into the prompt text, parsed back out of
    it. Checking meta against meta would prove nothing -- the prompt is what
    the judge sees."""
    text = [c for c in msg[0]["content"] if c["type"] == "text"][0]["text"]
    tail = text[text.index("Editing regions:") + len("Editing regions:"):]
    return [d["id"] for d in json.loads(tail)]


def test_build_requests_under_every_presentation():
    """End to end through the real stage-3 path: manifest -> regions.json ->
    presentation -> prompt -> message list, for all six axes."""
    tmp = Path(tempfile.mkdtemp())
    try:
        bases = build_fixture(tmp)
        df = fixture_manifest(bases)
        variants = tmp / "variants"
        render_variants(bases, df, variants)

        for mode in PRESENTATIONS:
            msgs, meta = build_requests(df, bases, variants, mode)
            assert len(msgs) == len(meta) == 2 * len(df), mode

            for m, x in zip(meta, msgs):
                n_img = sum(1 for c in x[0]["content"] if c["type"] == "image_pil")
                want = 0 if mode == "noimg" else (2 if m["kind"] == "sc" else 1)
                assert n_img == want, (mode, m["kind"], n_img, want)
                assert m["presentation"] == mode
                assert m["n_shown"] == len(m["region_ids"])
                assert m["n_shown"] == (len(IDS) - 1 if mode == "subset" else len(IDS))
                if mode == "subset":
                    assert int(m["target_region_id"]) in m["region_ids"]
                if m["kind"] == "sc":
                    assert _listed_ids(x) == list(m["region_ids"]), \
                        (mode, _listed_ids(x), m["region_ids"])

        one = df.head(1)
        b_msgs, _ = build_requests(one, bases, variants, "baseline")
        e_msgs, _ = build_requests(one, bases, variants, "enhance")
        # enhance must reach BOTH requests: PQ is where AES comes from, and the
        # exploit is entirely an AES effect.
        for i, kind in ((0, "SC"), (1, "PQ")):
            b = [c for c in b_msgs[i][0]["content"] if c["type"] == "image_pil"]
            e = [c for c in e_msgs[i][0]["content"] if c["type"] == "image_pil"]
            assert b[-1]["image_pil"].tobytes() != e[-1]["image_pil"].tobytes(), \
                "enhance did not reach the %s request" % kind
        # ...and must NOT touch the source, which is the real photograph.
        b_src = [c for c in b_msgs[0][0]["content"] if c["type"] == "image_pil"][0]
        e_src = [c for c in e_msgs[0][0]["content"] if c["type"] == "image_pil"][0]
        assert b_src["image_pil"].tobytes() == e_src["image_pil"].tobytes(), \
            "enhance modified the source; only the edit is the editor's output"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_shuffle_is_a_permutation, test_shuffle_is_deterministic,
         test_shuffle_varies_across_variants,
         test_subset_keeps_the_target_and_drops_one,
         test_subset_handles_a_single_region_base,
         test_subset_accepts_a_string_target, test_baseline_is_the_identity,
         test_schema_slots_follow_the_presented_order,
         test_enhance_is_deterministic_and_changes_pixels, test_enhance_is_global,
         test_draw_boxes_marks_only_the_presented_regions,
         test_reward_rises_with_pq_at_fixed_phi,
         test_build_requests_under_every_presentation]


def main() -> int:
    ok = True
    for fn in TESTS:
        try:
            fn()
            print("PASS " + fn.__name__)
        except AssertionError as e:
            ok = False
            print("FAIL " + fn.__name__ + ": " + str(e)[:400])
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
