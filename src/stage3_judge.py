"""
Stage 3 — the hot loop. Batched VLM judging via vLLM.

Config tuned for a full A10 24GB (SM 8.6, Ampere):
  - bf16, NOT fp8: fp8 kernels need SM 8.9+, so the official FP8 checkpoint
    buys us nothing here.
  - --max-model-len 4096: Qwen3-VL defaults to a 256k context and will refuse
    to allocate a KV cache that large next to 16.6GiB of weights.
  - gpu_memory_utilization has a NARROW workable window on this card, roughly
    (0.87, 0.90). See DEFAULT_GPU_UTIL below; both ends fail.
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
from .judge_prompt import (build_sc_prompt, build_pq_prompt, parse_sc, parse_pq,
                           region_reward, sc_json_schema, pq_json_schema,
                           REASONING_MODES)
from .schema import shard

# 768^2 = 589,824. Chosen to be a no-op on our actual data while shrinking the
# worst case vLLM profiles against: stage 0 writes COCO images at native size
# (typically 640x480 ~ 307k px) and stage1_edit resizes each edit back to
# src.size, so nothing we judge exceeds ~410k px. Nothing is downscaled — but
# vLLM profiles the encoder cache at max_pixels regardless of what we send, and
# at 1024^2 that overhead left the KV cache at -0.40 GiB on the A10.
MAX_PIXELS = 768 * 768
MIN_PIXELS = 256 * 256


# 0.89. This is squeezed between two failures, not chosen for headroom.
#
#   TOO HIGH: vLLM budgets gpu_memory_utilization as a fraction of TOTAL memory
#   but refuses to start unless that much is FREE. The A10-24Q reports 23.72GiB
#   total and only ~21.37GiB free (the vGPU layer keeps ~2.4GiB permanently),
#   so anything above ~0.901 dies before loading a weight.
#
#   TOO LOW: weights (16.64GiB) plus the encoder cache and activation peak come
#   to ~20.41GiB. Below ~0.861 the budget does not cover them and the KV cache
#   is sized NEGATIVE -- vLLM then raises "No available memory for the cache
#   blocks. Try increasing gpu_memory_utilization", which is accurate but reads
#   like a request to raise a value you already lowered on purpose.
#
# Measured on mcvgpu2025s-0050: util 0.85 -> KV -0.25GiB (dies);
# 0.87 -> +0.23GiB; 0.89 -> +0.70GiB (5,072 tokens, max concurrency 1.24x).
# The workable window is (0.8605, 0.9009) and 0.89 sits near its top while
# staying clear of the free-memory ceiling.
#
# This default was 0.85 and had never actually been run: every smoke test
# passed --gpu-util 0.89 explicitly, so the first command that relied on the
# default was the first one to fail.
#
# All five VMs must use the SAME value. It sizes the KV cache, which changes
# batch composition, which can perturb logits at the numerical margins. Scores
# from differently-configured shards are not cleanly comparable.
DEFAULT_GPU_UTIL = 0.89
GPU_UTIL_FLOOR = 0.87        # below this the KV cache goes negative on an A10


def load_engine(model: str, max_len: int = 4096, util: float = DEFAULT_GPU_UTIL):
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

    if util < GPU_UTIL_FLOOR:
        print(f"WARNING: --gpu-util {util} is below {GPU_UTIL_FLOOR}. On an A10 "
              f"the weights and activation peak alone need ~20.4GiB, so the KV "
              f"cache will be sized negative and the engine will refuse to "
              f"start. Raising it is the fix, not lowering it.")

    try:
        return _build_engine(model, max_len, util)
    except (RuntimeError, ValueError) as e:
        # vLLM's own message says "try increasing gpu_memory_utilization",
        # which is correct but sounds like advice to raise a value you lowered
        # deliberately. Say which direction and why, with the numbers.
        if "cache blocks" in str(e) or "Engine core initialization failed" in str(e):
            raise SystemExit(
                f"vLLM could not allocate a KV cache at --gpu-util {util}.\n"
                f"On an A10-24Q the workable window is roughly "
                f"({GPU_UTIL_FLOOR}, 0.90): below it the {util * 23.72:.1f}GiB "
                f"budget does not cover 16.6GiB of weights plus ~3.8GiB of "
                f"encoder cache and activations; above it vLLM cannot start "
                f"because the budget exceeds free memory.\n"
                f"Retry with --gpu-util {DEFAULT_GPU_UTIL} (the default), and "
                f"make sure every VM uses the same value.\n"
                f"Original error: {e}"
            ) from e
        raise


def _whitespace_kwargs() -> dict:
    """Forbid arbitrary whitespace in grammar-constrained output.

    vLLM defaults StructuredOutputsConfig(disable_any_whitespace=False), and
    unrestricted whitespace is ALWAYS grammar-valid -- so a constrained model
    can emit newlines and indentation indefinitely without ever violating the
    schema. On the first schema-constrained run every SC response ran to the
    full 1536-token cap while containing barely 100 tokens of actual JSON, with
    stray commas sitting on their own indented lines. Compact output removes
    the escape hatch.

    The option was renamed across versions, so pick by what EngineArgs
    actually has rather than guessing.
    """
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        return {}
    names: set = set()
    for attr in ("__dataclass_fields__", "__struct_fields__"):
        got = getattr(EngineArgs, attr, None)
        if got:
            names |= set(got)
    # The backend must be named explicitly. vLLM defaults to backend="auto" and
    # then REFUSES the option: "disable_any_whitespace is only supported for
    # xgrammar and guidance backends". auto resolves to xgrammar here anyway
    # (its nanobind objects show up in the shutdown log), so pinning it changes
    # nothing except making the option legal to pass.
    if "structured_outputs_config" in names:
        return {"structured_outputs_config": {
            "backend": "xgrammar", "disable_any_whitespace": True}}
    if "guided_decoding_disable_any_whitespace" in names:
        return {"guided_decoding_backend": "xgrammar",
                "guided_decoding_disable_any_whitespace": True}
    print("WARNING: cannot disable grammar whitespace on this vLLM; "
          "constrained responses may burn their token budget on indentation.")
    return {}


def _build_engine(model: str, max_len: int, util: float):
    from vllm import LLM

    ws = _whitespace_kwargs()
    if ws:
        print(f"grammar whitespace: disabled via {list(ws)[0]}")

    def _make(extra: dict):
        return LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=max_len,
            gpu_memory_utilization=util,
            # "video": 0 is load-bearing, not tidiness. Qwen3-VL accepts video, and
            # if the limit is left unset vLLM sizes the encoder cache for a
            # maximum-length video and profiles with one — a 151250-token budget and
            # a 4.62GiB allocation on top of 16.8GiB of weights, which OOMs the A10
            # during profile_run before a single request is served. We only ever
            # send two images.
            limit_mm_per_prompt={"image": 2, "video": 0},
            mm_processor_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
            # Cap the prefill chunk. vLLM defaults this to max_model_len and sizes
            # the profiling activation peak from it; our prompts are ~1,750 tokens
            # (2 images ~750 each + ~250 of text), so 4096 is ample headroom and
            # halves the peak that was pushing the KV cache negative.
            max_num_batched_tokens=2048,
            # Eager, not CUDA graphs. Graph capture reserves memory this card does
            # not have to spare, and it costs ~37s of torch.compile at every engine
            # start. The throughput it buys is almost all on the decode side, and
            # our outputs are ~15-32 tokens against a multi-image prefill — so
            # prefill dominates and eager costs us very little here.
            enforce_eager=True,
            **extra,
        )

    # A rejected whitespace option must not cost a 4-minute model load and a
    # traceback. Retry once without it and say so -- an unconstrained-whitespace
    # run still produces data, it just wastes tokens on indentation.
    try:
        return _make(ws)
    except ValueError as e:
        if ws and "whitespace" in str(e).lower():
            print(f"WARNING: engine rejected the whitespace option ({e}).\n"
                  f"         Retrying without it; responses may burn their "
                  f"token budget on indentation.")
            return _make({})
        raise


def build_requests(rows: pd.DataFrame, bases: Path, variants: Path) -> tuple[list, list]:
    """Two requests per variant, following A.4.3 / A.4.4:

      SC  source + edited, scoring EVERY region at once (plus background and
          overall) in one structured response.
      PQ  the edited image alone, giving the image-level [naturalness,
          artifacts] that Equation (3) needs.

    This is the protocol's own shape, not ours. The published prompt hands the
    judge all regions together and asks it to score them separately, so
    cross-region leakage shows up the way the paper's judge would exhibit it
    rather than as an artefact of how we sliced requests. It also happens to be
    cheaper: 2 requests per variant instead of one per region.
    """
    msgs, meta = [], []
    for row in rows.itertuples():
        regions = json.loads((bases / row.base_id / "regions.json").read_text())
        src = Image.open(bases / row.base_id / "source.png").convert("RGB")
        edit = Image.open(variants / f"{row.variant_id}.png").convert("RGB")

        common = dict(variant_id=row.variant_id, base_id=row.base_id,
                      target_region_id=str(row.target_region_id),
                      corruption=row.corruption, severity=row.severity,
                      area_bin=row.area_bin, is_control=row.is_control,
                      region_ids=[r["region_id"] for r in regions])

        msgs.append([{"role": "user", "content": [
            {"type": "image_pil", "image_pil": src},
            {"type": "image_pil", "image_pil": edit},
            {"type": "text", "text": build_sc_prompt(row.instruction, regions)},
        ]}])
        meta.append({**common, "kind": "sc"})

        msgs.append([{"role": "user", "content": [
            {"type": "image_pil", "image_pil": edit},
            {"type": "text", "text": build_pq_prompt()},
        ]}])
        meta.append({**common, "kind": "pq"})
    return msgs, meta


def structured_kind() -> str | None:
    """Which structured-output API the installed vLLM exposes, or None.

    vLLM renamed this: `guided_decoding` (older) became `structured_outputs`
    (0.11+). Detect rather than assume, because guessing wrong means either a
    TypeError at request build time or -- worse -- a silently unconstrained run
    that looks fine until the parse rate comes back at 70%.
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        return None
    names: set = set()
    for attr in ("__dataclass_fields__", "__struct_fields__"):
        got = getattr(SamplingParams, attr, None)
        if got:
            names |= set(got)
    if not names:
        import inspect
        try:
            names = set(inspect.signature(SamplingParams).parameters)
        except (TypeError, ValueError):
            return None
    if "structured_outputs" in names:
        return "structured_outputs"
    if "guided_decoding" in names:
        return "guided_decoding"
    return None


def _structured_kwargs(kind: str, schema: dict) -> dict:
    if kind == "structured_outputs":
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    from vllm.sampling_params import GuidedDecodingParams
    return {"guided_decoding": GuidedDecodingParams(json=schema)}


def run(llm, msgs, meta, n_samples: int, temperature: float,
        structured: bool = True, reasoning: str = "free") -> pd.DataFrame:
    from vllm import SamplingParams

    # n=n_samples shares the prefill across all samples. With images, prefill
    # dominates, so this is close to a free 4-5x versus n separate requests.
    #
    # max_tokens is large, not 32: A.4.3 asks for per-region `reasoning`
    # strings before the scores, so a multi-region response is hundreds of
    # tokens. At 32 every response truncates mid-reasoning and parses as
    # nothing.
    # repetition_penalty is load-bearing, not tuning. On the first real run
    # (2026-08-26) 3 of 10 responses collapsed into a loop -- one repeated
    # "The motorcycle is now more visible, and it appears to be a dark color..."
    # about twenty times verbatim until it hit max_tokens mid-JSON and parsed
    # as nothing. That was the ENTIRE 30% parse failure. Free-running reasoning
    # inside a JSON field at temperature 0.7 is exactly the setup that invites
    # it: nothing in the grammar pushes toward closing the string.
    # 1.05 is mild -- enough to break a verbatim loop, small enough that it
    # does not distort the score tokens, which are what we actually measure.
    #
    # max_tokens 1024 -> 1536: a 5-region response with per-region reasoning
    # legitimately runs long, and truncation costs the whole response. The
    # ceiling is max_model_len 4096 minus a ~1,750-token prompt, so 1536 still
    # leaves headroom.
    base = dict(n=n_samples, temperature=temperature, top_p=0.95,
                max_tokens=1536, repetition_penalty=1.1, seed=1234)

    kind = structured_kind() if structured else None
    if kind:
        print(f"structured output: {kind} (JSON schema per request, "
              f"reasoning={reasoning})")
    elif not structured:
        # Distinct from "unavailable": this is the operator's choice. The old
        # message claimed the API was missing either way, which is a lie when
        # --no-structured was passed and sent you looking for the wrong bug.
        print("structured output: DISABLED by --no-structured. Expect dropped "
              "regions and missing\n"
              "         background/overall_score; check coverage with "
              "scripts/diagnose_parse.py.")
    else:
        print("WARNING: this vLLM exposes no structured-output API, so "
              "responses are unconstrained.\n"
              "         Expect dropped regions, missing background/"
              "overall_score, and runaway\n"
              "         `reasoning` loops that truncate mid-JSON. Check the "
              "parse rate and run\n"
              "         scripts/diagnose_parse.py before trusting anything.")

    # One SamplingParams per request: the SC schema pins the exact region ids
    # for that variant, and PQ has a different shape entirely.
    sps = []
    for m in meta:
        kw = dict(base)
        if kind:
            schema = (pq_json_schema(reasoning) if m["kind"] == "pq"
                      else sc_json_schema(m["region_ids"], reasoning))
            kw.update(_structured_kwargs(kind, schema))
        sps.append(SamplingParams(**kw))

    outs = llm.chat(msgs, sps)

    # SC first: one row per (variant, sample, region), plus a background row.
    # PQ is image-level and merged on afterwards.
    recs, pq_by_key = [], {}
    for m, o in zip(meta, outs):
        if m["kind"] != "pq":
            continue
        for i, cand in enumerate(o.outputs):
            pq_by_key[(m["variant_id"], i)] = parse_pq(cand.text)

    for m, o in zip(meta, outs):
        if m["kind"] != "sc":
            continue
        base = {k: v for k, v in m.items() if k not in ("kind", "region_ids")}
        for i, cand in enumerate(o.outputs):
            sc = parse_sc(cand.text)
            pq = pq_by_key.get((m["variant_id"], i))
            for rid in list(m["region_ids"]) + ["bg"]:
                pair = None if rid == "bg" else sc.get("regions", {}).get(rid)
                recs.append({
                    # str(rid), NOT rid. The column carries region ints and the
                    # literal "bg", and pyarrow infers int64 from the ints then
                    # dies on the first "bg" -- which would surface as an
                    # ArrowInvalid in res.to_parquet() AFTER a full shard had
                    # been judged. Stringify here so the parquet write cannot
                    # throw away an hour of A10 time. target_region_id is
                    # stringified alongside it so the two stay comparable.
                    **base, "sample_idx": i, "scored_region_id": str(rid),
                    "sc_success": pair[0] if pair else None,
                    "sc_preserve": pair[1] if pair else None,
                    "sc_background": sc.get("background"),
                    "sc_overall_success": (sc.get("overall") or [None, None])[0],
                    "sc_overall_preserve": (sc.get("overall") or [None, None])[1],
                    "pq_naturalness": pq[0] if pq else None,
                    "pq_artifacts": pq[1] if pq else None,
                    "reward": region_reward(sc, rid, pq),
                    "parsed": bool(sc),
                    # finish_reason and n_tokens distinguish the two ways a
                    # response fails to parse: truncated at max_tokens (fix by
                    # raising the cap or shortening the reasoning) versus
                    # well-terminated but malformed (fix the prompt). Without
                    # them a low parse rate is just a number you cannot act on.
                    "finish_reason": getattr(cand, "finish_reason", None),
                    "n_tokens": len(getattr(cand, "token_ids", ()) or ()),
                    "raw": cand.text,
                })
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
    print(f"{len(msgs)} requests ({per_variant:.2f} per variant: 1 SC + 1 PQ) "
          f"x n={a.n_samples} = {len(msgs) * a.n_samples} generations")

    print("\n--- first request ---")
    print(f"meta: {meta[0]}")
    print('messages[0] = [{"role": "user", "content": [')
    print(describe_message(msgs[0]))
    print("]}]")

    # These mirror run() and load_engine() rather than restating them from
    # memory. The point of a dry run is to show what WOULD happen, so a stale
    # number here is worse than no number at all.
    print("\n--- sampling params that would be used (not constructed) ---")
    print(f"  n={a.n_samples} temperature={a.temperature} top_p=0.95 "
          f"max_tokens=1536 repetition_penalty=1.05 seed=1234")
    print("  (max_tokens is 1024 because A.4.3 asks for per-region reasoning "
          "text before the scores.)")
    print("\n--- engine that would be loaded (not loaded) ---")
    print(f"  model={a.model} dtype=bfloat16 "
          f"max_model_len={a.max_model_len} "
          f"gpu_memory_utilization={a.gpu_util}")
    print("  enforce_eager=True max_num_batched_tokens=2048 "
          "limit_mm_per_prompt={'image': 2, 'video': 0}")
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
    # max_model_len is the concurrency divisor: vLLM's max concurrency is
    # KV-cache-tokens / max_model_len. Our prompts are ~950 tokens and outputs
    # ~300, so 4096 reserves more than double what a request can use and halves
    # how many fit at once. Exposed so it can be measured rather than argued
    # about; it must match across VMs like --gpu-util does.
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="context reserved per request (default 4096). Lower "
                         "raises max concurrency proportionally; must be the "
                         "same on every VM.")
    ap.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                    help=f"gpu_memory_utilization (default {DEFAULT_GPU_UTIL}); "
                         "all five VMs must pass the same value")
    ap.add_argument("--reasoning", default="free", choices=REASONING_MODES,
                    help="how the schema expresses the reasoning field. "
                         "free (default) is 7.5x faster than bounded at "
                         "identical parse and coverage; bounded's maxLength "
                         "makes xgrammar count characters and costs 58.9s per "
                         "request against 7.9s. none drops the field, which "
                         "A.4.3 asks for -- state it in the report if used.")
    ap.add_argument("--no-structured", action="store_true",
                    help="disable JSON-schema-constrained decoding. Only for "
                         "measuring what the constraint is worth -- "
                         "unconstrained runs drop regions and loop.")
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

    llm = load_engine(a.model, max_len=a.max_model_len, util=a.gpu_util)
    res = run(llm, msgs, meta, a.n_samples, a.temperature,
              structured=not a.no_structured, reasoning=a.reasoning)
    res["judge"] = a.model

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out)

    # Fail loudly on a degenerate judge rather than discovering it in analysis.
    ok = res.parsed.mean() if len(res) else 0.0
    print(f"parse rate: {ok:.1%}")
    if ok < 0.95:
        print("WARNING: low parse rate — fix the prompt before scaling up")
    print(res.reward.describe())


if __name__ == "__main__":
    main()
