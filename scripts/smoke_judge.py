"""
Judge plumbing check against synthetic images, using SpatialFlow-GRPO's real
A.4.3 prompt. Judge VM only.

Needs a loaded model but NO project data, so it runs before stage 1 has
produced anything.

Two regions in every image:
    region 0  a square the instruction tells the editor to recolour
    region 1  a distractor square the instruction never mentions

Three SC requests, one engine load:
    A  followed     source + an image where region 0 really was recoloured
    B  ignored      source + an unchanged copy
    C  text-only    the same prompt with no images at all

What each buys:
  - A vs C token counts prove the pixels actually reach the model. vLLM can
    resolve chat_template_content_format to 'string', silently dropping custom
    content parts, and every score downstream would then be text-only noise.
  - A vs B on REGION 0 is the discriminative test. A high score on A alone
    proves nothing; a judge that answers 25 regardless would produce it too.
    If B scores as high as A there is no signal for the audit to measure.
  - REGION 1 is a free preview of the actual research question. It is untouched
    in both A and B, so a judge with spatially resolved scores should hold it
    roughly constant while region 0 moves. If region 1 tracks region 0, that is
    leakage — the thing this project exists to quantify.

Usage (judge VM):
    python -m scripts.smoke_judge --gpu-util 0.89
    python -m scripts.smoke_judge --model Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.judge_prompt import (  # noqa: E402
    build_sc_prompt, build_pq_prompt, parse_sc, parse_pq, region_reward,
)
from src.stage3_judge import load_engine, DEFAULT_GPU_UTIL  # noqa: E402

INSTRUCTION = "Change the blue square to red."
REGIONS = [
    {"region_id": 0, "label": "blue square", "bbox": (60, 60, 140, 140)},
    {"region_id": 1, "label": "green square", "bbox": (250, 250, 140, 140)},
]


def _images() -> tuple[Image.Image, Image.Image]:
    src = Image.new("RGB", (448, 448), (240, 240, 240))
    d = ImageDraw.Draw(src)
    d.rectangle([60, 60, 200, 200], fill=(30, 30, 200))      # region 0
    d.rectangle([250, 250, 390, 390], fill=(30, 160, 30))    # region 1
    edit = src.copy()
    ImageDraw.Draw(edit).rectangle([60, 60, 200, 200], fill=(200, 30, 30))
    return src, edit


def _msg(prompt: str, *images: Image.Image) -> list:
    content = [{"type": "image_pil", "image_pil": im} for im in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _phi(sc: dict, rid: int):
    pair = sc.get("regions", {}).get(rid)
    return min(pair) if pair else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL)
    a = ap.parse_args()

    from vllm import SamplingParams

    src, edit = _images()
    sc_prompt = build_sc_prompt(INSTRUCTION, REGIONS)
    msgs = [
        _msg(sc_prompt, src, edit),   # A
        _msg(sc_prompt, src, src),    # B
        _msg(sc_prompt),              # C
        _msg(build_pq_prompt(), edit),
    ]

    llm = load_engine(a.model, util=a.gpu_util)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1024, seed=1234)
    outs = llm.chat(msgs, sp)
    A, B, C, PQ = outs
    labels = ["A followed (src+edited)", "B ignored (src+src)", "C text-only"]

    print("\n" + "=" * 68)
    print("0. DOES THE REAL A.4.3 PROMPT PARSE AT ALL?")
    print("=" * 68)
    print(f"  response length: {len(A.outputs[0].text)} chars, "
          f"{len(A.outputs[0].token_ids)} tokens")
    scA = parse_sc(A.outputs[0].text)
    if not scA:
        print("  FAIL: could not parse. Raw response follows:")
        print("  " + A.outputs[0].text[:1200].replace("\n", "\n  "))
        return 1
    print(f"  parsed: regions={sorted(scA.get('regions', {}))} "
          f"background={'yes' if 'background' in scA else 'NO'} "
          f"overall={'yes' if 'overall' in scA else 'NO'}")
    pq = parse_pq(PQ.outputs[0].text)
    print(f"  PQ: {pq}")
    if len(A.outputs[0].token_ids) >= 1020:
        print("  WARNING: response hit the 1024-token cap; raise max_tokens.")

    print("\n" + "=" * 68)
    print("1. DO THE IMAGES REACH THE MODEL?")
    print("=" * 68)
    for lbl, o in zip(labels, outs[:3]):
        print(f"  {lbl:<26} prompt tokens: {len(o.prompt_token_ids)}")
    delta = len(A.prompt_token_ids) - len(C.prompt_token_ids)
    print(f"\n  two images add {delta} tokens -> "
          f"{'OK' if delta >= 50 else 'FAIL: content parts were dropped'}")

    print("\n" + "=" * 68)
    print("2. IS THE JUDGE READING THEM?  (the one that matters)")
    print("=" * 68)
    print(f"  {'':26} {'region0 phi':>12} {'region1 phi':>12} {'bg':>8}")
    parsed = []
    for lbl, o in zip(labels, outs[:3]):
        sc = parse_sc(o.outputs[0].text)
        parsed.append(sc)
        print(f"  {lbl:<26} {str(_phi(sc, 0)):>12} {str(_phi(sc, 1)):>12} "
              f"{str(sc.get('background')):>8}")

    a0, b0 = _phi(parsed[0], 0), _phi(parsed[1], 0)
    a1, b1 = _phi(parsed[0], 1), _phi(parsed[1], 1)
    c0 = _phi(parsed[2], 0)
    if c0 is not None:
        print(f"\n  NOTE: text-only, with NO IMAGES, still scored region 0 at {c0}.")
        print("  Whatever that number is measuring, it is not the pixels.")

    ok_signal = False
    if a0 is not None and b0 is not None:
        gap0 = a0 - b0
        print(f"\n  region 0 (edited):   A - B = {gap0:+.1f} of 25")
        ok_signal = gap0 >= 3
        if not ok_signal:
            print("  FAIL: an obeyed instruction scores no better than an ignored")
            print("  one. No signal to audit. Do not generate data yet.")
        else:
            print("  OK: the judge separates obeyed from ignored.")
    if a1 is not None and b1 is not None:
        print(f"  region 1 (untouched): A - B = {a1 - b1:+.1f} of 25")
        print("  (should be ~0 — it is untouched in both. Movement here is")
        print("   leakage, which is exactly what the audit measures.)")

    print("\n" + "=" * 68)
    print("3. EQUATION (3) END TO END")
    print("=" * 68)
    for rid in (0, 1, "bg"):
        print(f"  R(region={rid!r}) = {region_reward(scA, rid, pq)}")

    print(f"\nparse={'ok' if scA else 'FAILED'} "
          f"images={'ok' if delta >= 50 else 'FAILED'} "
          f"signal={'ok' if ok_signal else 'FAILED'}")
    return 0 if (scA and delta >= 50 and ok_signal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
