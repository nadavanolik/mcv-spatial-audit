"""
Stage 3 — the hot loop. Batched VLM judging via vLLM.

Config tuned for a full A10 24GB (SM 8.6, Ampere):
  - bf16, NOT fp8: fp8 kernels need SM 8.9+, so the official FP8 checkpoint
    buys us nothing here.
  - --max-model-len 8192: Qwen3-VL defaults to a 256k context and will refuse
    to allocate a KV cache that large next to 17.6GB of weights.
  - max_pixels capped: Qwen3-VL tokenizes by area, so an uncapped 2000px image
    becomes thousands of vision tokens and per-call latency explodes.

Usage:
    python -m src.stage3_judge --manifest out/manifest.parquet \
        --bases data/bases --variants /dev/shm/mcv/variants \
        --shard 0 --of 5 --out out/scores_shard0.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .judge_prompt import build_prompt, parse_scores, expected_score_from_logprobs
from .schema import shard

MAX_PIXELS = 1024 * 1024        # ~1300 vision tokens/image
MIN_PIXELS = 256 * 256


def load_engine(model: str, max_len: int = 8192, util: float = 0.90):
    from vllm import LLM
    return LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=util,
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
        # enforce_eager=True,   # uncomment if CUDA graph capture OOMs on the vGPU
    )


def build_requests(rows: pd.DataFrame, bases: Path, variants: Path) -> tuple[list, list]:
    """One request per (variant, region). Note we score EVERY region of the
    image, not just the corrupted one — leakage into neighbours is the finding."""
    msgs, meta = [], []
    for row in rows.itertuples():
        regions = json.loads((bases / row.base_id / "regions.json").read_text())
        src = Image.open(bases / row.base_id / "source.png").convert("RGB")
        edit = Image.open(variants / f"{row.variant_id}.png").convert("RGB")

        for r in regions:
            prompt = build_prompt(row.instruction, r["label"], r["bbox"])
            msgs.append([{"role": "user", "content": [
                {"type": "image_pil", "image_pil": src},
                {"type": "image_pil", "image_pil": edit},
                {"type": "text", "text": prompt},
            ]}])
            meta.append(dict(variant_id=row.variant_id, base_id=row.base_id,
                             scored_region_id=r["region_id"],
                             target_region_id=row.target_region_id,
                             corruption=row.corruption, severity=row.severity,
                             area_bin=row.area_bin, is_control=row.is_control))
    return msgs, meta


def run(llm, msgs, meta, n_samples: int, temperature: float) -> pd.DataFrame:
    from vllm import SamplingParams

    # n=n_samples shares the prefill across all samples. With images, prefill
    # is nearly the entire cost (outputs are ~15 tokens), so this is close to
    # a free 4-5x versus issuing n_samples separate requests.
    sp = SamplingParams(
        n=n_samples, temperature=temperature, top_p=0.95,
        max_tokens=32, logprobs=20, seed=1234,
    )
    outs = llm.chat(msgs, sp)

    recs = []
    for m, o in zip(meta, outs):
        for i, cand in enumerate(o.outputs):
            sampled = parse_scores(cand.text)
            expected = expected_score_from_logprobs(cand)
            recs.append({**m, "sample_idx": i,
                         "sc_sampled": sampled.get("SC"),
                         "pq_sampled": sampled.get("PQ"),
                         "sc_expected": expected.get("SC"),
                         "pq_expected": expected.get("PQ"),
                         "raw": cand.text})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--variants", default="/dev/shm/mcv/variants")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="pilot mode: cap variants")
    a = ap.parse_args()

    df = shard(pd.read_parquet(a.manifest), a.shard, a.of)
    if a.limit:
        df = df.head(a.limit)
    print(f"judging {len(df)} variants with {a.model}")

    msgs, meta = build_requests(df, Path(a.bases), Path(a.variants))
    print(f"{len(msgs)} requests x n={a.n_samples}")

    llm = load_engine(a.model)
    res = run(llm, msgs, meta, a.n_samples, a.temperature)
    res["judge"] = a.model

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out)

    # Fail loudly on a degenerate judge rather than discovering it in analysis.
    ok = res.sc_sampled.notna().mean()
    print(f"parse rate: {ok:.1%}")
    if ok < 0.95:
        print("WARNING: low parse rate — fix the prompt before scaling up")
    print(res.sc_sampled.value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
