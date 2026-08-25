"""Guard the property the whole multi-VM plan rests on: identical inputs must
produce byte-identical corrupted variants, on any machine, in any order."""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions import apply_corruption           # noqa: E402
from src.schema import seed_for, variant_id            # noqa: E402


def _fixture():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    mask[64:160, 64:160] = 255
    return img, mask


def _digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_repeatable():
    img, mask = _fixture()
    for c in ["blur", "saturate", "noise", "jpeg", "remove"]:
        s = seed_for(variant_id("b0", 0, c, 2, "full"))
        d = {_digest(apply_corruption(img, mask, c, 2, "full", s)) for _ in range(3)}
        assert len(d) == 1, f"{c} is non-deterministic"


def test_seed_sensitive():
    img, mask = _fixture()
    a = apply_corruption(img, mask, "noise", 2, "full", 111)
    b = apply_corruption(img, mask, "noise", 2, "full", 222)
    assert _digest(a) != _digest(b), "seed has no effect — check rng plumbing"


def test_order_independent():
    """Rendering in a different order must not change any output."""
    img, mask = _fixture()
    ids = [variant_id("b0", 0, c, 2, "full") for c in ["blur", "noise", "jpeg"]]
    fwd = {i: _digest(apply_corruption(img, mask, c, 2, "full", seed_for(i)))
           for i, c in zip(ids, ["blur", "noise", "jpeg"])}
    rev = {i: _digest(apply_corruption(img, mask, c, 2, "full", seed_for(i)))
           for i, c in reversed(list(zip(ids, ["blur", "noise", "jpeg"])))}
    assert fwd == rev


def test_localized():
    """Corruption must not touch pixels far outside the (feathered) mask."""
    img, mask = _fixture()
    out = apply_corruption(img, mask, "blur", 3, "full", 7)
    far = np.s_[0:32, 0:32]
    assert np.array_equal(img[far], out[far]), "corruption leaked outside the region"


def test_area_bins_shrink():
    img, mask = _fixture()
    areas = []
    for ab in ["full", "half", "quarter"]:
        o = apply_corruption(img, mask, "saturate", 3, ab, 7)
        areas.append(int((np.abs(o.astype(int) - img.astype(int)).sum(2) > 4).sum()))
    assert areas[0] > areas[1] > areas[2], f"area bins not monotone: {areas}"


if __name__ == "__main__":
    for fn in [test_repeatable, test_seed_sensitive, test_order_independent,
               test_localized, test_area_bins_shrink]:
        fn(); print(f"PASS {fn.__name__}")
    print("\nall determinism guarantees hold")
