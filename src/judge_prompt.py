"""
Judge prompts, JSON schemas, response parsing and the region reward.

The region-scoring (SC) prompt is SpatialFlow-GRPO's Appendix A.4.3
(arXiv:2606.26872), copied verbatim. Two parts are ours because the paper does
not print them: how the instruction and region list are appended to that
prompt, and the image-quality (PQ) prompt, written to produce the output shape
of its Appendix A.4.4.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Optional

# A.4.3 scores every region on 0-25, not 1-5.
SCORE_MIN, SCORE_MAX = 0, 25

# Verbatim, arXiv:2606.26872 Appendix A.4.3.
SFREWARD_SC_VERBATIM = """You are a professional digital artist. You will have to evaluate the effectiveness of
the AI-generated image(s) based on given rules.
All the input images are AI-generated.
OUTPUT FORMAT:
{
"edit_region": [
{"id": 0, "label": "region label",
"bbox_2d": [x1, y1, x2, y2],
"score": [score_success, score_preserve],
"reasoning": "brief reason"},
...
],
"background": {"score": value, "reasoning": "..."},
"overall_score": [overall_success, overall_preserve]
}
RULES:
Two images will be provided: original and edited version.
The objective is to evaluate how successfully the editing instruction has been executed.
You will be provided with pre-identified editing regions (bounding boxes with labels).
Score each region separately.
SCORING (per region, 0-25):
1) score_success: how well the edit follows the instruction (0=no change, 25=perfect).
2) score_preserve: degree of preservation within the region (0=completely different,
25=minimal effective edit).
BACKGROUND (0-25):
Rate how well non-edited areas are preserved. Penalize unexpected edits, layout changes,
artifacts outside editing regions.
OVERALL (0-25):
Overall success score and overall overediting score."""

# Ours: the paper does not show how the instruction and regions are appended.
_SC_SUFFIX = """

Editing instruction: {instruction}

Editing regions:
{regions_json}"""

# Ours: the paper shows the PQ output (A.4.4) but not its prompt.
SFREWARD_PQ_RECONSTRUCTED = """The image is an AI-generated image. Evaluate the generation quality.
SCORING (0-25):
naturalness: 0 means the image does not look natural; 25 means it looks natural.
artifacts: 0 means severe distortion, watermark, blur, unusual body parts, or
disharmonized subjects; 25 means no artifacts.
Output: {"score": [naturalness, artifacts], "reasoning": "..."}"""


def xywh_to_xyxy(bbox) -> list:
    """COCO stores (x, y, w, h); A.4.3's bbox_2d is [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


# Coordinate frame of bbox_2d in the prompt.
#   pixel     source-image pixels.
#   norm1000  relative, 0-1000 per axis: the frame Qwen3-VL and InternVL3
#             ground in (the boxes they echo back are in it).
BBOX_FRAMES = ("pixel", "norm1000")


def bbox_in_frame(bbox, frame: str = "pixel", size=None) -> list:
    """COCO (x, y, w, h) in source pixels -> bbox_2d [x1, y1, x2, y2] in `frame`.

    `size` is the image's (width, height), required for norm1000.
    """
    xyxy = xywh_to_xyxy(bbox)
    if frame == "pixel":
        return xyxy
    if frame != "norm1000":
        raise ValueError(f"frame must be one of {BBOX_FRAMES}, got {frame!r}")
    if not size:
        raise ValueError("norm1000 needs the image size (width, height)")
    w, h = size
    x, y, bw, bh = bbox
    return [min(1000, max(0, round(v))) for v in
            (x / w * 1000, y / h * 1000, (x + bw) / w * 1000, (y + bh) / h * 1000)]


def build_sc_prompt(instruction: str, regions: list, frame: str = "pixel",
                    size=None) -> str:
    """The SC prompt for one image, listing every region to be scored.

    As in A.4.3, all regions are scored in one request. `frame`/`size`: see
    BBOX_FRAMES.
    """
    listed = [{"id": r["region_id"], "label": r["label"],
               "bbox_2d": bbox_in_frame(r["bbox"], frame, size)} for r in regions]
    return SFREWARD_SC_VERBATIM + _SC_SUFFIX.format(
        instruction=instruction,
        regions_json=json.dumps(listed, indent=2),
    )


def build_pq_prompt() -> str:
    return SFREWARD_PQ_RECONSTRUCTED


# JSON schemas for constrained decoding. They fix the format only: every
# requested region must be scored and every score in 0-25 stays reachable.
# Without them the judge drops regions and loops inside `reasoning`.
REASONING_MAXLEN = 120

_SCORE_PAIR = {
    "type": "array",
    "items": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
    "minItems": 2, "maxItems": 2,
}


# How the `reasoning` field appears in the schema (grammar cost dominates
# judging time).
#   bounded  string with maxLength. 7.5x slower than `free` (xgrammar tracks a
#            character counter); do not use.
#   free     plain string (default).
#   none     no reasoning field; the protocol asks for one, so this changes it.
REASONING_MODES = ("bounded", "free", "none")


def _reasoning_prop(mode: str):
    if mode == "none":
        return None
    if mode == "free":
        return {"type": "string"}
    return {"type": "string", "maxLength": REASONING_MAXLEN}


def _with_reasoning(props: dict, required: list, mode: str) -> tuple:
    r = _reasoning_prop(mode)
    if r is not None:
        props["reasoning"] = r
        required = required + ["reasoning"]
    return props, required


def _region_item(region_id: int, mode: str = "free") -> dict:
    """One `edit_region` slot, pinned to a single region id."""
    props = {
        "id": {"type": "integer", "const": int(region_id)},
        "label": {"type": "string", "maxLength": 60},
        "bbox_2d": {"type": "array", "items": {"type": "integer"},
                    "minItems": 4, "maxItems": 4},
        "score": _SCORE_PAIR,
    }
    props, required = _with_reasoning(props, ["id", "score"], mode)
    return {"type": "object", "properties": props, "required": required}


@lru_cache(maxsize=64)
def _sc_schema_cached(ids: tuple, mode: str) -> dict:
    """Cached so identical region lists reuse one schema object (and xgrammar
    its compiled grammar). Callers must not modify the returned dict."""
    return _build_sc_schema(list(ids), mode)


def sc_json_schema(region_ids, reasoning: str = "free") -> dict:
    """Schema for one SC response scoring exactly `region_ids`."""
    if reasoning not in REASONING_MODES:
        raise ValueError(f"reasoning must be one of {REASONING_MODES}")
    return _sc_schema_cached(tuple(int(r) for r in region_ids), reasoning)


def _build_sc_schema(ids: list, mode: str) -> dict:
    return {
        "type": "object",
        "properties": {
            # prefixItems pins slot k to region ids[k], so each answer slot
            # belongs to a known region.
            "edit_region": {
                "type": "array",
                "minItems": len(ids), "maxItems": len(ids),
                "prefixItems": [_region_item(i, mode) for i in ids],
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": ids},
                        "label": {"type": "string", "maxLength": 60},
                        # Requested by A.4.3's format; optional and not read.
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4, "maxItems": 4,
                        },
                        "score": _SCORE_PAIR,
                        "reasoning": {"type": "string",
                                      "maxLength": REASONING_MAXLEN},
                    },
                    "required": ["id", "score", "reasoning"],
                },
            },
            "background": _bg_item(mode),
            "overall_score": _SCORE_PAIR,
        },
        "required": ["edit_region", "background", "overall_score"],
    }


def _bg_item(mode: str) -> dict:
    props = {"score": {"type": "integer",
                       "minimum": SCORE_MIN, "maximum": SCORE_MAX}}
    props, required = _with_reasoning(props, ["score"], mode)
    return {"type": "object", "properties": props, "required": required}


def pq_json_schema(reasoning: str = "free") -> dict:
    props = {"score": _SCORE_PAIR}
    props, required = _with_reasoning(props, ["score"], reasoning)
    return {"type": "object", "properties": props, "required": required}


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced {...} out of a response that may carry prose."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _valid(v) -> bool:
    return isinstance(v, (int, float)) and SCORE_MIN <= v <= SCORE_MAX


def parse_sc(text: str) -> dict:
    """SC response -> {regions: {id: [succ, pres]}, background, overall}.

    Missing or invalid pieces are omitted, not defaulted."""
    d = _extract_json(text)
    if not isinstance(d, dict):
        return {}

    out: dict = {"regions": {}}
    for r in d.get("edit_region") or []:
        if not isinstance(r, dict):
            continue
        sc = r.get("score")
        if isinstance(sc, list) and len(sc) == 2 and all(_valid(v) for v in sc):
            try:
                out["regions"][int(r["id"])] = [float(sc[0]), float(sc[1])]
            except (KeyError, TypeError, ValueError):
                pass

    bg = (d.get("background") or {}).get("score") if isinstance(d.get("background"), dict) else None
    if _valid(bg):
        out["background"] = float(bg)

    ov = d.get("overall_score")
    if isinstance(ov, list) and len(ov) == 2 and all(_valid(v) for v in ov):
        out["overall"] = [float(ov[0]), float(ov[1])]

    return out


def parse_pq(text: str) -> Optional[list]:
    """PQ output -> [naturalness, artifacts]."""
    d = _extract_json(text)
    if not isinstance(d, dict):
        return None
    sc = d.get("score")
    if isinstance(sc, list) and len(sc) == 2 and all(_valid(v) for v in sc):
        return [float(sc[0]), float(sc[1])]
    return None


def region_reward(sc: dict, region_id, pq: Optional[list], C: float = 25.0) -> Optional[float]:
    """Equation (3): R_{i,r} = sqrt( phi(IF_{i,r}) * AES_i ) / C.

    phi = min(score_success, score_preserve) for a region, or the background
    score for r == "bg". AES_i = min(PQ) is image-level, shared by every region.
    """
    if region_id == "bg":
        phi = sc.get("background")
    else:
        pair = sc.get("regions", {}).get(region_id)
        phi = min(pair) if pair else None
    if phi is None or not pq:
        return None
    return math.sqrt(phi * min(pq)) / C


def expected_score_from_logprobs(output) -> dict:
    """Not implemented, deliberately.

    A continuous readout would turn "the score did not move" into a small
    non-zero change, hiding the ties the analysis reports. Must keep raising.
    """
    raise NotImplementedError(
        "expected_score_from_logprobs has not been ported to the A.4.3 output "
        "format (two-digit 0-25 scores nested after variable-length reasoning). "
        "See the docstring; use the sampled readout until this is designed."
    )
