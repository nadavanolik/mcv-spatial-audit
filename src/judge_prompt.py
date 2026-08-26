"""
Judge prompting and score extraction — SpatialFlow-GRPO's published protocol.

SOURCE: arXiv:2606.26872, Appendix A.4.3, "Scoring prompt for the region-aware
reward model". SFREWARD_SC_VERBATIM below is copied from the paper. Do not
reword it. The audit's whole claim is that it evaluates *the published
protocol*, so paraphrase would quietly invalidate the comparison.

PROVENANCE — read before trusting any of this:

  VERBATIM from A.4.3:  SFREWARD_SC_VERBATIM, in full.
  OURS, necessarily:    how the instruction and the region list are appended.
                        A.4.3 says only "You will be provided with
                        pre-identified editing regions (bounding boxes with
                        labels)" and never shows the injection format. We follow
                        A.5.2's convention ("Editing instruction: {instruction}")
                        and pass regions as the JSON the model must echo back.
  RECONSTRUCTED:        SFREWARD_PQ_RECONSTRUCTED. The paper gives SFReward's PQ
                        *output* (A.4.4) but never its PQ *prompt*. A.5.2's PQ
                        prompt is the MultiEditBench/VIEScore one on a 0-10
                        scale for GPT-4.1 — a different purpose. Ours matches
                        A.4.4's output shape. FLAG THIS IN THE REPORT.

Two further gaps worth stating in the write-up rather than papering over:

  1. SFReward is a *fine-tuned* model (Qwen3-VL-8B-Instruct + SFReward-14K).
     A.4.3's prompt is what labelled that data with a Gemini-3-Pro teacher. We
     apply it to base Qwen3-VL-8B, so we audit the prompt-based protocol, not
     the released reward model.
  2. Equation (3) multiplies every region reward by AES_i = min(PQ), a single
     image-level factor shared across all regions of an image. Part of each
     "region score" is therefore global by construction. Within one image it
     cancels out of region-to-region comparisons; across variants it does not.
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

# A.4.3 scores every region on 0-25, not 1-5.
SCORE_MIN, SCORE_MAX = 0, 25

# --- verbatim, arXiv:2606.26872 Appendix A.4.3 -------------------------------
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

# --- ours: the paper never shows how these are appended ----------------------
_SC_SUFFIX = """

Editing instruction: {instruction}

Editing regions:
{regions_json}"""

# --- reconstructed: A.4.4 shows the output, never the prompt -----------------
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


def build_sc_prompt(instruction: str, regions: list) -> str:
    """One prompt scoring EVERY region of one image, per A.4.3.

    Note this is one request per *variant*, not per region — the protocol hands
    the judge all regions at once and asks it to score them separately. Which
    means cross-region leakage is measurable exactly as the paper's judge would
    exhibit it, rather than as an artefact of how we chose to slice requests.
    """
    listed = [{"id": r["region_id"], "label": r["label"],
               "bbox_2d": xywh_to_xyxy(r["bbox"])} for r in regions]
    return SFREWARD_SC_VERBATIM + _SC_SUFFIX.format(
        instruction=instruction,
        regions_json=json.dumps(listed, indent=2),
    )


def build_pq_prompt() -> str:
    return SFREWARD_PQ_RECONSTRUCTED


# Grammar-constrained decoding. Free-running `reasoning` text is what breaks
# this protocol in practice: on real COCO edits the judge fell into verbatim
# loops ("a black and white motorcycle with a rider wearing a helmet, " x110)
# that ran to max_tokens mid-JSON, and separately dropped regions and omitted
# background/overall_score. 30% of responses parsed as nothing and coverage of
# the rest was 60%.
#
# A schema fixes all three at once: maxLength bounds the loop, minItems/maxItems
# force every requested region, and `required` forces the two top-level keys.
#
# NOTE FOR THE REPORT: this constrains the FORMAT, never the VALUES. Any score
# in 0-25 remains reachable, so what the judge thinks is unaffected -- only its
# ability to wander off mid-sentence. A.4.3 asks for a "brief reason" already;
# this enforces what the prompt requests rather than adding a new demand.
# 120, not 200. Under a grammar the model spends its budget differently, and
# reasoning is the one field we never read -- only `score` matters.
REASONING_MAXLEN = 120

_SCORE_PAIR = {
    "type": "array",
    "items": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
    "minItems": 2, "maxItems": 2,
}


def _region_item(region_id: int) -> dict:
    """One `edit_region` slot, pinned to a single region id."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "const": int(region_id)},
            "label": {"type": "string", "maxLength": 60},
            "bbox_2d": {"type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4},
            "score": _SCORE_PAIR,
            "reasoning": {"type": "string", "maxLength": REASONING_MAXLEN},
        },
        "required": ["id", "score", "reasoning"],
    }


def sc_json_schema(region_ids) -> dict:
    """Schema for one SC response scoring exactly `region_ids`."""
    ids = [int(r) for r in region_ids]
    return {
        "type": "object",
        "properties": {
            # prefixItems pins slot k to region ids[k], rather than letting
            # any slot hold any id. With a bare enum + minItems the model
            # scored region 0, was then forced to open a second object with
            # nothing telling it which region that was, and produced filler
            # ("label": "AI generated"). Forcing a count without forcing an
            # identity asks the model to invent regions; pinning the identity
            # turns each slot into a well-posed question.
            "edit_region": {
                "type": "array",
                "minItems": len(ids), "maxItems": len(ids),
                "prefixItems": [_region_item(i) for i in ids],
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": ids},
                        "label": {"type": "string", "maxLength": 60},
                        # Kept optional and unread: A.4.3's output format asks
                        # for it, but the model answers in its own normalised
                        # 0-1000 space rather than our pixels, so it is not a
                        # usable echo of the region we specified.
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
            "background": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer",
                              "minimum": SCORE_MIN, "maximum": SCORE_MAX},
                    "reasoning": {"type": "string",
                                  "maxLength": REASONING_MAXLEN},
                },
                "required": ["score", "reasoning"],
            },
            "overall_score": _SCORE_PAIR,
        },
        "required": ["edit_region", "background", "overall_score"],
    }


def pq_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "score": _SCORE_PAIR,
            "reasoning": {"type": "string", "maxLength": REASONING_MAXLEN},
        },
        "required": ["score", "reasoning"],
    }


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
    """A.4.3's structured SC output -> {regions: {id: [succ, pres]},
    background: float, overall: [succ, pres]}. Missing pieces are omitted
    rather than defaulted, so a malformed response is visible as a gap."""
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

    phi = min(score_success, score_preserve) for a foreground region, or the
    background score for r == "bg". AES_i = min(PQ), an image-level term shared
    by every region of the image — so it cannot contribute anything spatially
    resolved, and within a single image it cancels out of any region-to-region
    comparison. That is worth saying out loud in the report: part of the
    "region" reward is global by construction, before any judge behaviour is
    measured.
    """
    if region_id == "bg":
        phi = sc.get("background")
    else:
        pair = sc.get("regions", {}).get(region_id)
        phi = min(pair) if pair else None
    if phi is None or not pq:
        return None
    return math.sqrt(phi * min(pq)) / C


# --- legacy 1-5 helpers, kept only until stage3/stage4 are migrated ----------

def parse_scores(text: str) -> dict:
    """DEPRECATED. Reads the old flat {"SC": n, "PQ": n} placeholder format,
    which the real protocol does not produce. Use parse_sc / parse_pq."""
    out = {}
    for key in ("SC", "PQ"):
        m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if m:
            out[key] = int(m.group(1))
    return out


def expected_score_from_logprobs(output) -> dict:
    """DEPRECATED for the real protocol, and not a trivial port.

    The old readout worked because a score was a single digit token immediately
    after `"SC":`. Under A.4.3 a score is a two-digit number on 0-25, nested in
    a list, inside a per-region object, after a free-text `reasoning` field
    whose length is not fixed. Summing p(k)*k over digit tokens is no longer
    well defined: the tens and units digits are separate tokens, and the
    position of the score varies with how long the model's reasoning ran.

    That matters because architectural decision 6 (two readouts, sampled and
    expected) rests on this. The continuous readout is what keeps AUROC off a
    pile of ties. Porting it needs a deliberate design pass — locate the score
    tokens structurally rather than by regex on a running prefix — so it is
    left failing loudly rather than silently returning wrong numbers.
    """
    raise NotImplementedError(
        "expected_score_from_logprobs has not been ported to the A.4.3 output "
        "format (two-digit 0-25 scores nested after variable-length reasoning). "
        "See the docstring; use the sampled readout until this is designed."
    )
