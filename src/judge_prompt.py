"""
Judge prompting and score extraction.

!!! ACTION REQUIRED !!!
SFREWARD_TEMPLATE below is a PLACEHOLDER with the right shape but not the real
wording. Replace it verbatim from the SpatialFlow-GRPO paper's appendix
(arXiv:2606.26872) before generating any results you intend to report. The
audit's whole claim is that it evaluates *the published protocol*, so an
invented prompt would quietly invalidate the paper comparison.

Keep the replacement in this file and nowhere else, so there is exactly one
place to check when a reviewer asks "is this really their prompt?".
"""
from __future__ import annotations

import math
import re
from typing import Optional

SCORE_MIN, SCORE_MAX = 1, 5

SFREWARD_TEMPLATE = """You are evaluating an instruction-based image edit.

Instruction: {instruction}

You are shown the source image and the edited image. Evaluate the region
described as: {region_desc} (bounding box {bbox}).

Rate two axes on a {smin}-{smax} integer scale:
- SC (semantic consistency): does this region reflect the instruction?
- PQ (perceptual quality): is this region free of artifacts and degradation?

Respond with exactly this JSON and nothing else:
{{"SC": <int>, "PQ": <int>}}
"""


def build_prompt(instruction: str, region_desc: str, bbox) -> str:
    return SFREWARD_TEMPLATE.format(
        instruction=instruction, region_desc=region_desc, bbox=list(bbox),
        smin=SCORE_MIN, smax=SCORE_MAX,
    )


def parse_scores(text: str) -> dict:
    """Extract sampled integer SC/PQ. Tolerant of stray prose around the JSON."""
    out = {}
    for key in ("SC", "PQ"):
        m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if m:
            v = int(m.group(1))
            if SCORE_MIN <= v <= SCORE_MAX:
                out[key] = v
    return out


def expected_score_from_logprobs(output) -> dict:
    """Continuous score = sum_k p(k) * k over the digit tokens at the score
    position, renormalised over valid digits.

    Why this matters: a 1-5 integer gives ∆score a granularity of 1, so most
    corrupted/clean pairs tie at 0 and AUROC degenerates into a step function
    on a pile of ties. The logprob expectation is continuous, far lower
    variance, and — since it needs no repeats — is also ~4x cheaper than the
    sampled noise floor it complements. Report both.

    `output` is one vllm CompletionOutput with .token_ids and .logprobs.
    """
    if not output.logprobs:
        return {}

    valid = {str(d) for d in range(SCORE_MIN, SCORE_MAX + 1)}
    text_so_far = ""
    pending: Optional[str] = None
    result: dict = {}

    for tok_id, lp_dict in zip(output.token_ids, output.logprobs):
        chosen = lp_dict[tok_id].decoded_token or ""
        text_so_far += chosen

        # Which key are we about to score? Look at the most recent key seen.
        m = re.findall(r'"(SC|PQ)"\s*:\s*$', text_so_far)
        if m:
            pending = m[-1]
            continue

        if pending and chosen.strip() in valid:
            mass, acc = 0.0, 0.0
            for cand_id, lp in lp_dict.items():
                d = (lp.decoded_token or "").strip()
                if d in valid:
                    p = math.exp(lp.logprob)
                    mass += p
                    acc += p * int(d)
            if mass > 0:
                result[pending] = acc / mass
            pending = None

    return result
