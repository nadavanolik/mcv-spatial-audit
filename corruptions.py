"""
Localized corruption engine.

Every function here MUST be a pure function of (image, mask, severity, rng).
No global state, no time, no unseeded randomness, no thread-dependent order.
If this property breaks, scores from different shards stop being comparable
and the whole audit is invalid. tests/test_determinism.py guards it.

cv2 is imported from opencv-python-headless: the GUI build needs libGL.so.1,
which requires apt and therefore root, which we do not have.
"""
from __future__ import annotations

import cv2
import numpy as np

# Severity ladders. Index by severity 1..3.
_BLUR_SIGMA = {1: 1.5, 2: 3.5, 3: 7.0}
_SAT_GAIN = {1: 1.4, 2: 2.0, 3: 3.0}
_NOISE_STD = {1: 6.0, 2: 14.0, 3: 28.0}
_JPEG_Q = {1: 40, 2: 18, 3: 7}
_REMOVE_RADIUS = {1: 3, 2: 6, 3: 10}


def feather(mask: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Binary mask -> float32 alpha in [0,1] with soft edges.

    Un-feathered pastes leave a hard seam, and a judge can plausibly detect the
    seam rather than the degradation — which would be a confound, not a finding.
    """
    m = (mask > 127).astype(np.float32)
    k = int(max(3, round(sigma * 3)) | 1)          # odd kernel
    a = cv2.GaussianBlur(m, (k, k), sigma)
    return np.clip(a, 0.0, 1.0)[..., None]


def resize_mask_area(mask: np.ndarray, area_bin: str, rng: np.random.Generator) -> np.ndarray:
    """Shrink the corrupted footprint within the region.

    'full'    = the whole region mask
    'half'    = ~50% of region area, 'quarter' = ~25%, taken as a centered
                erosion so the corruption stays inside the semantic region.
    """
    if area_bin == "full":
        return mask
    target = {"half": 0.5, "quarter": 0.25}[area_bin]
    m = (mask > 127).astype(np.uint8)
    a0 = m.sum()
    if a0 == 0:
        return mask
    cur, it = m.copy(), 0
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while cur.sum() > target * a0 and it < 200:
        cur = cv2.erode(cur, kern, iterations=1)
        it += 1
        if cur.sum() == 0:
            cur = cv2.erode(m, kern, iterations=max(0, it - 1))
            break
    return (cur * 255).astype(np.uint8)


# --- the five degradations ---------------------------------------------------

def _blur(img, sev, rng):
    s = _BLUR_SIGMA[sev]
    k = int(max(3, round(s * 4)) | 1)
    return cv2.GaussianBlur(img, (k, k), s)


def _saturate(img, sev, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * _SAT_GAIN[sev], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _noise(img, sev, rng):
    n = rng.normal(0.0, _NOISE_STD[sev], img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _jpeg(img, sev, rng):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_Q[sev]])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _remove(img, sev, rng, mask=None):
    """Small-object removal via inpainting. Unlike the others this needs the
    mask, since it paints the region away rather than degrading it in place."""
    m = (mask > 127).astype(np.uint8) if mask is not None else np.ones(img.shape[:2], np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, m, _REMOVE_RADIUS[sev], cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


_OPS = {"blur": _blur, "saturate": _saturate, "noise": _noise,
        "jpeg": _jpeg, "remove": _remove}


def apply_corruption(img: np.ndarray, mask: np.ndarray, corruption: str,
                     severity: int, area_bin: str, seed: int,
                     feather_sigma: float = 6.0) -> np.ndarray:
    """Composite a localized degradation. img is RGB uint8 HxWx3.

    Deterministic given (img, mask, corruption, severity, area_bin, seed).
    """
    if corruption == "none":
        return img.copy()
    if corruption not in _OPS:
        raise ValueError(f"unknown corruption {corruption!r}")

    rng = np.random.default_rng(seed)
    m = resize_mask_area(mask, area_bin, rng)

    if corruption == "remove":
        full = _remove(img, severity, rng, mask=m)
    else:
        full = _OPS[corruption](img, severity, rng)

    a = feather(m, feather_sigma)
    out = img.astype(np.float32) * (1 - a) + full.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
