"""
The whole audit in miniature, on synthetic images. Judge VM only.

Needs a loaded model but NO project data, so it runs before stage 1 exists.
Uses the real A.4.3 prompt and the real corruption engine, so it exercises the
actual pipeline rather than an approximation of it.

Setup mirrors the experiment. Two regions, and — as stage0_coco.py really does
— the instruction names BOTH, so both are legitimate `edit_region` entries:

    instruction: "Change the blue square to red, and the green square to yellow."
    edit:        both changes correctly applied          (the clean control)
    variant_k:   that edit, with REGION 0 ONLY corrupted at severity k

Then score every region of each variant and watch where the number moves.

    region 0 should FALL as severity rises      -> the judge is spatially resolved
    region 1 should HOLD (it is untouched)      -> movement here is leakage
    the severity ladder should be MONOTONE      -> the score is graded, not binary

That last one matters as much as the first. A judge that only ever emits 0 or
25 makes delta-score all-or-nothing, the severity ladder carries no information,
and AUROC collapses to a step function. Blatant synthetic inputs will saturate;
what we need to see is whether anything in between exists.

A text-only request is included as a nuisance probe: the judge will happily
score an edit it was shown no images of, which is worth a line in the report.

Usage (judge VM):
    python -m scripts.smoke_judge --gpu-util 0.89
    python -m scripts.smoke_judge --model Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruptions import apply_corruption  # noqa: E402
from src.judge_prompt import (  # noqa: E402
    build_sc_prompt, build_pq_prompt, parse_sc, parse_pq,
)
from src.stage3_judge import load_engine, DEFAULT_GPU_UTIL  # noqa: E402

INSTRUCTION = "Change the blue square to red, and the green square to yellow."
R0 = (60, 60, 140, 140)      # xywh — the region we corrupt
R1 = (250, 250, 140, 140)    # xywh — untouched, the leakage probe
REGIONS = [
    {"region_id": 0, "label": "the square in the upper left", "bbox": R0},
    {"region_id": 1, "label": "the square in the lower right", "bbox": R1},
]


def _base_pair() -> tuple[Image.Image, Image.Image]:
    """Source, and an edit where BOTH instructed changes were made correctly."""
    src = Image.new("RGB", (448, 448), (240, 240, 240))
    d = ImageDraw.Draw(src)
    d.rectangle([60, 60, 200, 200], fill=(30, 30, 200))      # blue
    d.rectangle([250, 250, 390, 390], fill=(30, 160, 30))    # green

    edit = Image.new("RGB", (448, 448), (240, 240, 240))
    d = ImageDraw.Draw(edit)
    d.rectangle([60, 60, 200, 200], fill=(200, 30, 30))      # -> red
    d.rectangle([250, 250, 390, 390], fill=(220, 210, 40))   # -> yellow
    return src, edit


def _mask_r0() -> np.ndarray:
    m = np.zeros((448, 448), np.uint8)
    m[60:200, 60:200] = 255
    return m


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
    ap.add_argument("--corruption", default="noise",
                    choices=["blur", "saturate", "noise", "jpeg", "remove"])
    a = ap.parse_args()

    from vllm import SamplingParams

    src, edit = _base_pair()
    mask, arr = _mask_r0(), np.asarray(edit)
    sc_prompt = build_sc_prompt(INSTRUCTION, REGIONS)

    # The clean control, then the same edit corrupted in region 0 only.
    variants = [("clean", edit)]
    for sev in (1, 2, 3):
        out = apply_corruption(arr, mask, a.corruption, sev, "full", 1234 + sev)
        variants.append((f"{a.corruption} s{sev}", Image.fromarray(out)))

    msgs = [_msg(sc_prompt, src, im) for _, im in variants]
    msgs.append(_msg(sc_prompt))                    # text-only nuisance probe
    msgs.append(_msg(build_pq_prompt(), edit))      # PQ on the clean edit

    llm = load_engine(a.model, util=a.gpu_util)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1024, seed=1234)
    outs = llm.chat(msgs, sp)
    scs = [parse_sc(o.outputs[0].text) for o in outs[:len(variants)]]
    text_only = parse_sc(outs[len(variants)].outputs[0].text)
    pq = parse_pq(outs[-1].outputs[0].text)

    print("\n" + "=" * 70)
    print("0. PARSING")
    print("=" * 70)
    bad = [i for i, s in enumerate(scs) if not s.get("regions")]
    print(f"  {len(scs) - len(bad)}/{len(scs)} responses parsed; PQ={pq}")
    for i, o in enumerate(outs[:len(variants)]):
        if len(o.outputs[0].token_ids) >= 1020:
            print(f"  WARNING: variant {i} hit the 1024-token cap.")
    if bad:
        print(f"  FAIL: unparsed responses at {bad}. First raw response:")
        print("  " + outs[bad[0]].outputs[0].text[:1000].replace("\n", "\n  "))
        return 1

    print("\n" + "=" * 70)
    print("1. DOES THE SCORE TRACK DAMAGE, AND ONLY WHERE THE DAMAGE IS?")
    print("=" * 70)
    print(f"  {'variant':<16} {'region0 phi':>12} {'region1 phi':>12} {'bg':>8}")
    r0s, r1s = [], []
    for (name, _), sc in zip(variants, scs):
        p0, p1 = _phi(sc, 0), _phi(sc, 1)
        r0s.append(p0)
        r1s.append(p1)
        print(f"  {name:<16} {str(p0):>12} {str(p1):>12} {str(sc.get('background')):>8}")
    print(f"  {'(text-only)':<16} {str(_phi(text_only, 0)):>12} "
          f"{str(_phi(text_only, 1)):>12} {str(text_only.get('background')):>8}")

    print("\n" + "-" * 70)
    if all(v is not None for v in r0s):
        drop = r0s[0] - r0s[-1]
        print(f"  region 0, clean -> worst: {r0s[0]} -> {r0s[-1]}  (drop {drop:+.1f})")
        print(f"  {'OK: corruption moves the targeted region.' if drop >= 3 else 'FAIL: corruption barely moves the region it damaged.'}")
        mono = all(x >= y for x, y in zip(r0s[1:], r0s[2:]))
        distinct = len(set(r0s))
        print(f"  severity ladder: {r0s[1:]}  monotone={mono}  distinct values={distinct}")
        if distinct <= 2:
            print("  WARNING: the judge is emitting only rail values. delta-score")
            print("  becomes all-or-nothing and the severity ladder carries no")
            print("  information. Check this again on real COCO edits.")
    if all(v is not None for v in r1s):
        swing = max(r1s) - min(r1s)
        print(f"\n  region 1 (never touched): {r1s}  swing {swing:.1f}")
        print(f"  {'OK: no leakage into the untouched region.' if swing < 3 else 'LEAKAGE: an untouched region moved when its neighbour was damaged. This is the finding the audit exists to measure.'}")

    print("\n" + "=" * 70)
    print("2. NUISANCE PROBE")
    print("=" * 70)
    t0 = _phi(text_only, 0)
    print(f"  With NO IMAGES at all, the judge scored region 0 at {t0}.")
    print("  It will score an edit it was never shown. Worth a line in the")
    print("  report under exploitability.")

    ok = all(v is not None for v in r0s) and (r0s[0] - r0s[-1]) >= 3
    print(f"\nparse=ok  localisation={'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
