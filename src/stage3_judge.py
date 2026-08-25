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

from . import judge_prompt
from .judge_prompt import build_prompt, parse_scores, expected_score_from_logprobs
from .schema import shard

MAX_PIXELS = 1024 * 1024        # ~1300 vision tokens/image
MIN_PIXELS = 256 * 256


# 0.85, not 0.90. vLLM budgets gpu_memory_utilization as a fraction of TOTAL
# memory but refuses to start unless that much is FREE. The A10-24Q vGPU reports
# 23.72GiB total and only ~21.34GiB free — roughly 2.4GiB is reserved by the
# vGPU layer itself and never comes back — so 0.90 (21.35GiB) misses by 0.01GiB
# and the engine dies before loading a single weight.
#
# All five VMs must use the SAME value. It sizes the KV cache, which changes
# batch composition, which can perturb logits at the numerical margins. Scores
# from differently-configured shards are not cleanly comparable.
DEFAULT_GPU_UTIL = 0.85


def load_engine(model: str, max_len: int = 8192, util: float = DEFAULT_GPU_UTIL):
    from vllm import LLM

    # vLLM's own message for this reports the shortfall but not the fix, and it
    # arrives after a model download. Check first and say what to pass.
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        need = total * util
        print(f"GPU: {free / 2**30:.2f}GiB free of {total / 2**30:.2f}GiB; "
              f"util={util} needs {need / 2**30:.2f}GiB")
        if need > free:
            headroom = (free / total) - 0.02
            raise SystemExit(
                f"gpu_memory_utilization={util} needs {need / 2**30:.2f}GiB but only "
                f"{free / 2**30:.2f}GiB is free.\n"
                f"Retry with --gpu-util {headroom:.2f} (or lower).\n"
                f"If another process is on the GPU, check `nvidia-smi` first — "
                f"otherwise this is the vGPU's own reservation and is permanent.\n"
                f"Whatever you choose, every VM must use the same value."
            )
    except ImportError:
        pass

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


def preflight(rows: pd.DataFrame, bases: Path, variants: Path) -> dict:
    """Stat every file build_requests will open, without decoding any of them.

    Cheap enough to run over a whole shard, and it turns the usual
    "FileNotFoundError on request 4,812, twenty minutes in" into an inventory
    you can read before anything loads.
    """
    missing: dict[str, list[str]] = {"regions.json": [], "source.png": [], "variant": []}
    seen: set[str] = set()
    for row in rows.itertuples():
        b = bases / row.base_id
        if row.base_id not in seen:
            seen.add(row.base_id)
            for f in ("regions.json", "source.png"):
                if not (b / f).exists():
                    missing[f].append(str(b / f))
        v = variants / f"{row.variant_id}.png"
        if not v.exists():
            missing["variant"].append(str(v))
    return {"n_bases": len(seen), "missing": missing}


def describe_message(msg: list) -> str:
    """Render one chat message for reading: images summarised to mode/size,
    prompt text shown in full — the prompt is the thing worth eyeballing."""
    lines = []
    for c in msg[0]["content"]:
        if c["type"] == "image_pil":
            im = c["image_pil"]
            lines.append(f'  {{"type": "image_pil", "image_pil": '
                         f'<PIL.Image {im.mode} {im.width}x{im.height}>}},')
        else:
            lines.append(f'  {{"type": "{c["type"]}", "text": """')
            lines.extend("    " + ln for ln in c["text"].splitlines())
            lines.append('  """}},')
    return "\n".join(lines)


def dry_run(rows: pd.DataFrame, bases: Path, variants: Path, a) -> int:
    """Build the requests and print the first one. Never imports vLLM.

    This exists so message construction and manifest plumbing can be checked on
    a machine with no GPU, leaving only the vLLM API surface itself to be
    debugged on a judge VM.
    """
    print("=== DRY RUN: building requests only, vLLM is never imported ===")

    pf = preflight(rows, bases, variants)
    n_missing = sum(len(v) for v in pf["missing"].values())
    print(f"{len(rows)} variants across {pf['n_bases']} bases")
    if n_missing:
        print(f"\nMISSING INPUTS ({n_missing} paths):")
        for kind, paths in pf["missing"].items():
            if paths:
                print(f"  {kind}: {len(paths)} missing, e.g. {paths[0]}")
        print("\nCannot build requests. Run stage 1 (base edits) and stage 2 "
              "(variants) first, or point --bases/--variants elsewhere.")
        return 1

    msgs, meta = build_requests(rows, bases, variants)
    per_variant = len(msgs) / max(len(rows), 1)
    print(f"{len(msgs)} requests ({per_variant:.2f} regions/variant) "
          f"x n={a.n_samples} = {len(msgs) * a.n_samples} generations")

    print("\n--- first request ---")
    print(f"meta: {meta[0]}")
    print('messages[0] = [{"role": "user", "content": [')
    print(describe_message(msgs[0]))
    print("]}]")

    print("\n--- sampling params that would be used (not constructed) ---")
    print(f"  n={a.n_samples} temperature={a.temperature} top_p=0.95 "
          f"max_tokens=32 logprobs=20 seed=1234")
    print("\n--- engine that would be loaded (not loaded) ---")
    print(f"  model={a.model} dtype=bfloat16 max_model_len=8192 "
          f"gpu_memory_utilization=0.90")
    print(f"  mm_processor_kwargs={{'max_pixels': {MAX_PIXELS}, "
          f"'min_pixels': {MIN_PIXELS}}}")

    if "PLACEHOLDER" in (judge_prompt.__doc__ or ""):
        # ASCII only: this runs on a Windows console, which is not UTF-8.
        print("\nNOTE: src/judge_prompt.py still ships the PLACEHOLDER prompt. "
              "The text above is not SpatialFlow-GRPO's published wording.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--variants", default="/dev/shm/mcv/variants")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", help="output parquet; required unless --dry-run")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="pilot mode: cap variants")
    ap.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                    help=f"gpu_memory_utilization (default {DEFAULT_GPU_UTIL}); "
                         "all five VMs must pass the same value")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the requests and print the first one, without "
                         "importing vLLM — runs on a machine with no GPU")
    a = ap.parse_args()

    df = shard(pd.read_parquet(a.manifest), a.shard, a.of)
    if a.limit:
        df = df.head(a.limit)
    print(f"judging {len(df)} variants with {a.model}")

    if a.dry_run:
        raise SystemExit(dry_run(df, Path(a.bases), Path(a.variants), a))
    if not a.out:
        ap.error("--out is required unless --dry-run is given")

    msgs, meta = build_requests(df, Path(a.bases), Path(a.variants))
    print(f"{len(msgs)} requests x n={a.n_samples}")

    llm = load_engine(a.model, util=a.gpu_util)
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
