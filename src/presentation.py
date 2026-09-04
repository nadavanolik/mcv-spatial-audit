"""
Presentation axes for the nuisance and exploitability analyses.

The audit's other four analyses change WHAT the judge sees. These change how it
is TOLD about what it sees, or apply a change that carries no information about
edit quality, and ask whether the score moves anyway.

WHY THIS IS NOT A MANIFEST COLUMN. variant_id is sha1 over
(base_id, target_region_id, corruption, severity, area_bin) -- see schema.py.
A sixth field would change every variant_id, hence every seed_for(), hence
every rendered byte, voiding the cross-VM fixture hash four VMs have already
confirmed. Presentation is therefore a stage-3 flag: the SAME images, judged
again with different packaging, landing as a column in the scores parquet.

WHY NOT corruptions.py. scripts/verify_determinism.sh hashes exactly
["blur","saturate","noise","jpeg","remove"] from that module. Adding an op here
cannot move that hash; editing one there would.

DETERMINISM. The two text axes derive their choices by sorting on a sha256 of
(variant_id, mode, region_id) -- no RNG, so no dependence on Python's random
implementation. The two image axes use PIL only, which has no RNG at all, but
their cross-VM byte-equality is UNVERIFIED and deliberately untested: the whole
sweep is meant to run on a single VM, which makes the question moot. Do not
shard these across machines without checking that first.
"""
from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# baseline is the no-op: exactly what stage 3 did before this module existed.
#
#   shuffle   pure text. Region list order in the prompt AND in the schema's
#             prefixItems slots. Nothing about the image changes, which makes
#             it the strongest null we have.
#   subset    pure text. Drop exactly ONE non-target region from the list. The
#             smallest possible change to "how many regions were you asked
#             about"; the target is always retained or the row is unscoreable.
#   box       pixels change. Draw the presented bboxes on BOTH images -- on the
#             edit alone the box would itself be a difference from the source
#             and could be read as an editing artifact. A weaker null than the
#             text axes for exactly that reason.
#   noimg     exploitability. Strip every image. CLAUDE.md records the judge
#             happily scoring an edit it was shown no images of; this turns
#             that anecdote into a measured condition with a reward column.
#   enhance   exploitability. Global cosmetic lift on the edit, whole image, no
#             mask, no edit made better. Equation (3) is
#             sqrt(phi * AES)/C with AES = min(PQ) a single IMAGE-level term
#             multiplying every region, so if the judge rates a sharpened image
#             higher, every region's reward rises without any edit improving.
#             EXPECT A CONFOUND: sharpening the edit makes it differ more from
#             the source, and the SC prompt asks about preservation -- so phi
#             may fall while reward rises. Read `reward` and `phi` separately;
#             that split is the measurement, not a caveat to it.
PRESENTATIONS = ("baseline", "shuffle", "subset", "box", "noimg", "enhance")

# Two orthogonal groupings, because they answer different questions.
#
# TEXT/IMAGE is about the STIMULUS: does this axis change a pixel? It decides
# how strong a null the axis is, and what the dry run should report.
#
# NUISANCE/EXPLOIT is about the CLAIM: a nuisance axis is one where the score
# SHOULD NOT move and any movement is the finding; an exploit axis is one where
# we predict the score WILL move, upward, for free. Reporting `enhance` in a
# table headed "does a null change move the score" would be a category error --
# it is not a null, and it moving is the hypothesis, not a failure.
TEXT_AXES = ("shuffle", "subset")            # no pixels change
IMAGE_AXES = ("box", "enhance")              # pixels change
NUISANCE_AXES = ("shuffle", "subset", "box")
EXPLOIT_AXES = ("noimg", "enhance")
NO_IMAGES = "noimg"

# Cosmetic lift. UnsharpMask's arguments are Pillow's own photo-sharpening
# defaults; the two enhancers are mild on purpose -- this has to read as "the
# editor's output looks nicer", not as a second corruption.
_UNSHARP = dict(radius=2, percent=150, threshold=3)
_CONTRAST = 1.15
_SATURATION = 1.15

_BOX_OUTLINE = (255, 0, 0)
_BOX_WIDTH = 3


def _key(variant_id: str, mode: str, region_id) -> str:
    """Stable per-(variant, mode, region) sort key. sha256, not random.Random:
    a hash sort has no dependence on the interpreter's RNG implementation, and
    it is one line rather than a seeded generator."""
    return hashlib.sha256(f"{variant_id}|{mode}|{region_id}".encode()).hexdigest()


def present(regions: list, target_region_id, variant_id: str,
            mode: str) -> list:
    """The region list as it will be shown to the judge, in order.

    Order matters twice: it is the order in the prompt text, and
    judge_prompt._build_sc_schema builds prefixItems positionally from the same
    ids, so the grammar's answer slots follow it too. Permuting one without the
    other would leave the grammar forcing the model back into canonical order,
    i.e. a nuisance axis that silently does nothing.
    """
    if mode not in PRESENTATIONS:
        raise ValueError(f"presentation must be one of {PRESENTATIONS}")
    if mode == "shuffle":
        return sorted(regions, key=lambda r: _key(variant_id, mode, r["region_id"]))
    if mode == "subset":
        # Drop one non-target region. str() on both sides: target_region_id
        # arrives from the manifest as an int but is stringified everywhere
        # downstream, and a silent int/str mismatch here would drop the target.
        others = [r for r in regions
                  if str(r["region_id"]) != str(target_region_id)]
        if not others:
            return regions          # single-region base: nothing to drop
        drop = max(others, key=lambda r: _key(variant_id, mode, r["region_id"]))
        return [r for r in regions if r["region_id"] != drop["region_id"]]
    return regions


def enhance(img: Image.Image) -> Image.Image:
    """Global cosmetic lift: sharpen, contrast, saturation. No mask, no crop.

    Uniform over the whole frame by construction, so it cannot fix, damage or
    otherwise inform on any particular region.
    """
    out = img.filter(ImageFilter.UnsharpMask(**_UNSHARP))
    out = ImageEnhance.Contrast(out).enhance(_CONTRAST)
    return ImageEnhance.Color(out).enhance(_SATURATION)


def draw_boxes(img: Image.Image, regions: list) -> Image.Image:
    """Draw the presented regions' bboxes. COCO (x, y, w, h)."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for r in regions:
        x, y, w, h = r["bbox"]
        d.rectangle([int(x), int(y), int(x + w), int(y + h)],
                    outline=_BOX_OUTLINE, width=_BOX_WIDTH)
    return out
