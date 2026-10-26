"""
Presentation variants for the nuisance and exploitability analyses.

These re-package a request without changing the rendered variant: they alter
how the regions are listed, or apply a change that carries no information about
edit quality, and the analysis asks whether the score moves anyway.

Presentation is a stage-3 flag, not a manifest column, because adding a field
to variant_id would change every seed and therefore every rendered image.
The text axes choose deterministically by hashing (variant_id, mode, region_id);
the image axes use deterministic PIL operations.
"""
from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

#   baseline        the request as normally built.
#   shuffle         reorder the region list (prompt and schema slots together).
#   subset          drop one non-target region from the list.
#   box             draw the listed boxes on both images.
#   noimg           send the request with no images.
#   enhance         sharpen and lift contrast/saturation over the whole edit.
#   enhance_target  the same lift, inside the target region's box only.
PRESENTATIONS = ("baseline", "shuffle", "subset", "box", "noimg", "enhance",
                 "enhance_target")

# Nuisance axes should not move the score; exploit axes are predicted to raise
# it without improving the edit.
TEXT_AXES = ("shuffle", "subset")                   # no pixels change
IMAGE_AXES = ("box", "enhance", "enhance_target")   # pixels change
NUISANCE_AXES = ("shuffle", "subset", "box")
EXPLOIT_AXES = ("noimg", "enhance", "enhance_target")
NO_IMAGES = "noimg"

# Mild on purpose: the result should look like a nicer edit, not a corruption.
_UNSHARP = dict(radius=2, percent=150, threshold=3)
_CONTRAST = 1.15
_SATURATION = 1.15

_BOX_OUTLINE = (255, 0, 0)
_BOX_WIDTH = 3


def _key(variant_id: str, mode: str, region_id) -> str:
    """Deterministic sort key for one region under one variant and mode."""
    return hashlib.sha256(f"{variant_id}|{mode}|{region_id}".encode()).hexdigest()


def present(regions: list, target_region_id, variant_id: str,
            mode: str) -> list:
    """The region list as shown to the judge, in order.

    The JSON schema's answer slots are built from this same order, so a
    reordering applies to the prompt and the grammar together.
    """
    if mode not in PRESENTATIONS:
        raise ValueError(f"presentation must be one of {PRESENTATIONS}")
    if mode == "shuffle":
        return sorted(regions, key=lambda r: _key(variant_id, mode, r["region_id"]))
    if mode == "subset":
        # Compare as strings: the manifest stores the target id as an int.
        others = [r for r in regions
                  if str(r["region_id"]) != str(target_region_id)]
        if not others:
            return regions          # single-region base: nothing to drop
        drop = max(others, key=lambda r: _key(variant_id, mode, r["region_id"]))
        return [r for r in regions if r["region_id"] != drop["region_id"]]
    return regions


def enhance(img: Image.Image) -> Image.Image:
    """Global cosmetic lift: sharpen, contrast, saturation."""
    out = img.filter(ImageFilter.UnsharpMask(**_UNSHARP))
    out = ImageEnhance.Contrast(out).enhance(_CONTRAST)
    return ImageEnhance.Color(out).enhance(_SATURATION)


def enhance_region(img: Image.Image, bbox) -> Image.Image:
    """enhance() inside one COCO (x, y, w, h) box, clamped to the frame.

    The whole frame is enhanced and then cropped, so the unsharp mask sees real
    neighbours at the box edge.
    """
    x, y, w, h = (int(v) for v in bbox)
    box = (max(x, 0), max(y, 0), min(x + w, img.width), min(y + h, img.height))
    out = img.copy()
    if box[0] < box[2] and box[1] < box[3]:
        out.paste(enhance(img).crop(box), box[:2])
    return out


def draw_boxes(img: Image.Image, regions: list) -> Image.Image:
    """Draw the regions' COCO (x, y, w, h) boxes."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for r in regions:
        x, y, w, h = r["bbox"]
        d.rectangle([int(x), int(y), int(x + w), int(y + h)],
                    outline=_BOX_OUTLINE, width=_BOX_WIDTH)
    return out
