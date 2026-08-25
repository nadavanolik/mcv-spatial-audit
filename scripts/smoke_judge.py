"""
One real judge call against synthetic images. Judge VM only.

Answers the last two stage-3 unknowns, which need a loaded model but do NOT
need any project data — so this can run before stage 1 has produced anything:

  1. Does chat_template_content_format="auto" preserve our custom "image_pil"
     content parts for Qwen3-VL, or silently drop the images? If the judge
     scores two blank images identically to two obviously-different ones, the
     images are not reaching the model and every downstream number is noise.
  2. What is the actual logprob structure? expected_score_from_logprobs walks
     output.logprobs as a list of {token_id: Logprob} and reads .decoded_token.
     If that shape is wrong, sc_expected comes back empty and we silently lose
     the continuous readout that AUROC depends on.

Deliberately reuses load_engine() and the same SamplingParams as stage 3, so
this exercises the real code path rather than a parallel one that could drift.

Usage (judge VM):
    python -m scripts.smoke_judge
    python -m scripts.smoke_judge --model Qwen/Qwen3-VL-8B-Instruct

Downloads the model (~17.6GB) on first run. Nothing else touches disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.judge_prompt import (  # noqa: E402
    build_prompt, parse_scores, expected_score_from_logprobs,
)
from src.stage3_judge import load_engine  # noqa: E402


def _pair() -> tuple[Image.Image, Image.Image]:
    """A source and an 'edit' that differ blatantly in one region.

    Blatant on purpose: this is a plumbing check, not a measurement. If the
    model cannot tell these apart, the images are not reaching it.
    """
    src = Image.new("RGB", (448, 448), (240, 240, 240))
    d = ImageDraw.Draw(src)
    d.rectangle([60, 60, 200, 200], fill=(30, 30, 200))
    d.rectangle([250, 250, 390, 390], fill=(30, 160, 30))

    edit = src.copy()
    ImageDraw.Draw(edit).rectangle([60, 60, 200, 200], fill=(200, 30, 30))
    return src, edit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n-samples", type=int, default=2)
    a = ap.parse_args()

    from vllm import SamplingParams

    src, edit = _pair()
    prompt = build_prompt("make the blue square red", "the blue square", [60, 60, 140, 140])
    msgs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": src},
        {"type": "image_pil", "image_pil": edit},
        {"type": "text", "text": prompt},
    ]}]]

    llm = load_engine(a.model)
    sp = SamplingParams(n=a.n_samples, temperature=0.7, top_p=0.95,
                        max_tokens=32, logprobs=20, seed=1234)
    outs = llm.chat(msgs, sp)
    cand = outs[0].outputs[0]

    print("\n" + "=" * 62)
    print("1. DID THE CALL SURVIVE THE CHAT TEMPLATE?")
    print("=" * 62)
    print(f"raw text:      {cand.text!r}")
    print(f"parse_scores:  {parse_scores(cand.text)}")
    if not parse_scores(cand.text):
        print("  FAIL: nothing parseable. Either the prompt needs work or the")
        print("  images never reached the model. Check the prompt_token_ids")
        print("  count below — a text-only prompt is far shorter.")
    print(f"prompt tokens: {len(outs[0].prompt_token_ids)}")
    print("  Two 448x448 images should contribute hundreds of vision tokens.")
    print("  A number in the low hundreds means the images were dropped and")
    print("  chat_template_content_format needs to be set explicitly.")

    print("\n" + "=" * 62)
    print("2. IS THE LOGPROB STRUCTURE WHAT WE ASSUME?")
    print("=" * 62)
    lps = cand.logprobs
    print(f"type(cand.logprobs):   {type(lps)}")
    if not lps:
        print("  FAIL: empty. sc_expected will be null for every row.")
        return 1
    print(f"len (one per token):   {len(lps)}")
    first = lps[0]
    print(f"type of one position:  {type(first)}")
    print(f"keys are token ids:    {list(first)[:5]}")
    probe = first[list(first)[0]]
    print(f"type of one entry:     {type(probe)}")
    print(f"has .decoded_token:    {hasattr(probe, 'decoded_token')}")
    print(f"has .logprob:          {hasattr(probe, 'logprob')}")
    if hasattr(probe, "decoded_token"):
        print(f"sample decoded_token:  {probe.decoded_token!r}")

    print("\n" + "=" * 62)
    print("3. DOES THE CONTINUOUS READOUT ACTUALLY COME OUT?")
    print("=" * 62)
    exp = expected_score_from_logprobs(cand)
    print(f"expected_score_from_logprobs: {exp}")
    if not exp:
        print("  FAIL: empty. The integer readout would still work, but the")
        print("  continuous one is what keeps AUROC out of a pile of ties.")
        print("  Send the section-2 output back — the walk needs rewriting.")

    print("\n--- all samples ---")
    for i, c in enumerate(outs[0].outputs):
        print(f"  [{i}] {c.text!r} -> {parse_scores(c.text)}")

    ok = bool(parse_scores(cand.text)) and bool(exp)
    print(f"\n{'PASS' if ok else 'INCOMPLETE'}: "
          f"sampled={'ok' if parse_scores(cand.text) else 'FAILED'}, "
          f"expected={'ok' if exp else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
