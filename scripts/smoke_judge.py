"""
Judge plumbing check against synthetic images. Judge VM only.

Needs a loaded model but NO project data, so it runs before stage 1 has
produced anything. Three requests in one engine load:

  A  followed    source + a genuinely edited image (instruction obeyed)
  B  not-followed source + an unchanged copy   (instruction ignored)
  C  text-only   the same prompt with no images at all

What each one buys:

  - A vs C token counts prove the images actually reach the model. vLLM can
    resolve chat_template_content_format to 'string', which silently drops
    custom content parts; scores would then be text-only noise and every
    downstream number meaningless.
  - A vs B is the discriminative test. A high score on A alone proves nothing —
    a judge that answers 5 regardless would also produce it. If B scores as
    high as A, the judge is not reading the images and the whole audit has no
    signal to measure. This is the single most important line of output here.
  - The logprob dump confirms expected_score_from_logprobs walks the real
    structure; a silent empty there costs us the continuous readout that keeps
    AUROC off a pile of ties.

Reuses load_engine() and stage 3's SamplingParams so this exercises the real
code path rather than a parallel one that could drift.

Usage (judge VM):
    python -m scripts.smoke_judge
    python -m scripts.smoke_judge --gpu-util 0.89
    python -m scripts.smoke_judge --model Qwen/Qwen3-VL-4B-Instruct

Downloads the model on first run. Nothing else touches disk.
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
from src.stage3_judge import load_engine, DEFAULT_GPU_UTIL  # noqa: E402

INSTRUCTION = "make the blue square red"
REGION_DESC = "the blue square"
REGION_BBOX = [60, 60, 140, 140]


def _images() -> tuple[Image.Image, Image.Image]:
    """A source, and an edit that blatantly obeys the instruction.

    Blatant on purpose: this is a plumbing check, not a measurement. If the
    model cannot tell these apart, the images are not reaching it.
    """
    src = Image.new("RGB", (448, 448), (240, 240, 240))
    d = ImageDraw.Draw(src)
    d.rectangle([60, 60, 200, 200], fill=(30, 30, 200))      # the blue square
    d.rectangle([250, 250, 390, 390], fill=(30, 160, 30))    # a distractor

    edit = src.copy()
    ImageDraw.Draw(edit).rectangle([60, 60, 200, 200], fill=(200, 30, 30))
    return src, edit


def _msg(prompt: str, *images: Image.Image) -> list:
    content = [{"type": "image_pil", "image_pil": im} for im in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                    help=f"gpu_memory_utilization (default {DEFAULT_GPU_UTIL})")
    a = ap.parse_args()

    from vllm import SamplingParams

    src, edit = _images()
    prompt = build_prompt(INSTRUCTION, REGION_DESC, REGION_BBOX)
    msgs = [
        _msg(prompt, src, edit),   # A: instruction followed
        _msg(prompt, src, src),    # B: instruction ignored
        _msg(prompt),              # C: no images at all
    ]

    llm = load_engine(a.model, util=a.gpu_util)
    sp = SamplingParams(n=a.n_samples, temperature=0.7, top_p=0.95,
                        max_tokens=32, logprobs=20, seed=1234)
    outs = llm.chat(msgs, sp)
    A, B, C = outs
    labels = ["A followed (src + edited)", "B ignored (src + src)", "C text-only"]

    print("\n" + "=" * 66)
    print("1. DO THE IMAGES REACH THE MODEL?")
    print("=" * 66)
    for lbl, o in zip(labels, outs):
        print(f"  {lbl:<28} prompt tokens: {len(o.prompt_token_ids)}")
    delta = len(A.prompt_token_ids) - len(C.prompt_token_ids)
    print(f"\n  two images add {delta} tokens")
    if delta < 50:
        print("  FAIL: images contribute almost nothing. chat_template_content_format")
        print("  resolved to something that drops custom content parts. Every score")
        print("  downstream would be text-only noise.")
    else:
        print("  OK: the vision tokens are really in the prompt.")

    print("\n" + "=" * 66)
    print("2. IS THE JUDGE ACTUALLY READING THEM?  (the one that matters)")
    print("=" * 66)
    for lbl, o in zip(labels, outs):
        sampled = [parse_scores(c.text) for c in o.outputs]
        exp = expected_score_from_logprobs(o.outputs[0])
        sc = [s.get("SC") for s in sampled]
        print(f"  {lbl:<28} SC sampled={sc}  SC expected={exp.get('SC')}")

    c_sc = expected_score_from_logprobs(C.outputs[0]).get("SC")
    if c_sc is not None and c_sc >= 4.5:
        print(f"\n  NOTE: the text-only request scored SC={c_sc:.3f} with NO IMAGES.")
        print("  Whatever the score is measuring, it is not the pixels.")

    a_sc = expected_score_from_logprobs(A.outputs[0]).get("SC")
    b_sc = expected_score_from_logprobs(B.outputs[0]).get("SC")
    if a_sc is not None and b_sc is not None:
        gap = a_sc - b_sc
        print(f"\n  A - B = {gap:+.3f}")
        if gap < 0.5:
            print("  FAIL: an obeyed instruction scores no better than an ignored one.")
            print("  Either the images are not informing the score, or this judge is")
            print("  degenerate on this prompt. There is no signal to audit until")
            print("  this gap is real — fix it before generating any data.")
        else:
            print("  OK: the judge separates an obeyed instruction from an ignored one.")

    print("\n" + "=" * 66)
    print("2b. CAN THE MODEL SEE AT ALL?  (only if 2 failed)")
    print("=" * 66)
    print("  Same images, but a plain question instead of the scoring prompt.")
    print("  If these answers are correct, the vision path is fine and the")
    print("  scoring PROMPT is what is degenerate — a very different problem")
    print("  from the model being unable to use the images.")
    probe = ("Look at the SECOND image. What colour is the large square in the "
             "upper-left area? Answer with one word.")
    psp = SamplingParams(n=1, temperature=0.0, max_tokens=8)
    pouts = llm.chat([_msg(probe, src, edit), _msg(probe, src, src)], psp)
    print(f"\n  second image is the RED edit  -> {pouts[0].outputs[0].text.strip()!r}"
          "   (expect: red)")
    print(f"  second image is the BLUE copy -> {pouts[1].outputs[0].text.strip()!r}"
          "   (expect: blue)")
    sees = ("red" in pouts[0].outputs[0].text.lower()
            and "blue" in pouts[1].outputs[0].text.lower())
    print(f"\n  vision path: {'WORKS — the scoring prompt is the problem' if sees else 'SUSPECT — the model is not using the images'}")

    print("\n" + "=" * 66)
    print("3. LOGPROB STRUCTURE AND THE CONTINUOUS READOUT")
    print("=" * 66)
    cand = A.outputs[0]
    print(f"  raw text:            {cand.text!r}")
    lps = cand.logprobs
    if not lps:
        print("  FAIL: cand.logprobs is empty; sc_expected will be null everywhere.")
        return 1
    probe = lps[0][list(lps[0])[0]]
    print(f"  logprobs:            {type(lps).__name__} of {len(lps)} "
          f"{type(lps[0]).__name__}, entries {type(probe).__name__}")
    print(f"  .decoded_token:      {hasattr(probe, 'decoded_token')} "
          f"({probe.decoded_token!r})")
    print(f"  expected_score(A):   {expected_score_from_logprobs(cand)}")

    images_ok = delta >= 50
    signal_ok = a_sc is not None and b_sc is not None and (a_sc - b_sc) >= 0.5
    parse_ok = bool(parse_scores(cand.text)) and bool(expected_score_from_logprobs(cand))
    print(f"\nimages={'ok' if images_ok else 'FAILED'} "
          f"signal={'ok' if signal_ok else 'FAILED'} "
          f"readout={'ok' if parse_ok else 'FAILED'}")
    return 0 if (images_ok and signal_ok and parse_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
