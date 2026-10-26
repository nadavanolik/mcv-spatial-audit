"""
Stage 3: VLM judging with vLLM, two schema-constrained requests per variant.

Configured for an A10 24GB (SM 8.6): bf16 (fp8 kernels need SM 8.9+), a 4096
context, a capped image size, and a narrow gpu_memory_utilization window (see
DEFAULT_GPU_UTIL).

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
                           REASONING_MODES, BBOX_FRAMES)
from .presentation import (PRESENTATIONS, TEXT_AXES, present, enhance,
                           enhance_region, draw_boxes)
from .schema import shard

# Qwen3-VL pixel budget. Larger than any image we send (~410k px), but vLLM
# sizes the encoder cache from this cap, so a bigger one costs KV-cache memory.
MAX_PIXELS = 768 * 768
MIN_PIXELS = 256 * 256


# The A10-24Q reports 23.72GiB but leaves ~21.37GiB free. Above ~0.90 vLLM
# refuses to start (budget > free memory); below ~0.86 the 16.6GiB of weights
# plus activations leave a negative KV cache. 0.89 gives a 0.70GiB cache.
# Every shard must use the same value: it changes batching and therefore logits.
DEFAULT_GPU_UTIL = 0.89
GPU_UTIL_FLOOR = 0.87        # below this the KV cache goes negative on an A10


# InternVL tiles each image into up to this many 448px crops plus a thumbnail,
# 256 tokens each. InternVL3-8B ran with 6 at --max-model-len 8192.
DEFAULT_MAX_DYNAMIC_PATCH = 2


def load_engine(model: str, max_len: int = 4096, util: float = DEFAULT_GPU_UTIL,
                tiles: int = DEFAULT_MAX_DYNAMIC_PATCH):
    # Check free memory before the model download, and say which value to pass.
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
        return _build_engine(model, max_len, util, tiles)
    except (RuntimeError, ValueError) as e:
        # The engine starts in a child process, so every startup failure arrives
        # as the same generic error. Name the KV-cache cause only when vLLM's own
        # message says so.
        msg = str(e)
        if "cache blocks" in msg or "No available memory" in msg:
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
        if "Engine core initialization failed" in msg:
            raise SystemExit(
                "The vLLM engine failed to start. The real error is in the "
                "child process traceback ABOVE this line; it is not repeated "
                "here.\n"
                "Common causes on this hardware, in the order they have "
                "actually happened:\n"
                "  - disk full while downloading weights "
                "(OSError: No space left on device). Check `df -h ~`.\n"
                f"  - KV cache too small: see --gpu-util (default "
                f"{DEFAULT_GPU_UTIL}, window ~({GPU_UTIL_FLOOR}, 0.90)).\n"
                "  - another process holding the GPU. Check `nvidia-smi`.\n"
                f"Wrapper error: {e}"
            ) from e
        raise


def _whitespace_kwargs() -> dict:
    """Engine kwargs that forbid free whitespace in constrained output.

    Whitespace is always grammar-valid, so without this a constrained model can
    pad its JSON with indentation until it hits max_tokens. The option's name
    differs across vLLM versions, so it is picked from EngineArgs' fields.
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
    # vLLM accepts the option only with an explicit xgrammar or guidance backend.
    if "structured_outputs_config" in names:
        return {"structured_outputs_config": {
            "backend": "xgrammar", "disable_any_whitespace": True}}
    if "guided_decoding_disable_any_whitespace" in names:
        return {"guided_decoding_backend": "xgrammar",
                "guided_decoding_disable_any_whitespace": True}
    print("WARNING: cannot disable grammar whitespace on this vLLM; "
          "constrained responses may burn their token budget on indentation.")
    return {}


def _mm_args(model: str, tiles: int = DEFAULT_MAX_DYNAMIC_PATCH) -> tuple[dict, dict]:
    """(limit_mm_per_prompt, mm_processor_kwargs) for this model family.

    Shared by _build_engine and dry_run so the dry run reports the real config.
    """
    lower = model.lower()
    if "qwen" in lower:
        return ({"image": 2, "video": 0},
                {"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})
    return ({"image": 2},
            {"max_dynamic_patch": tiles} if "internvl" in lower else {})


def _build_engine(model: str, max_len: int, util: float, tiles: int):
    from vllm import LLM

    ws = _whitespace_kwargs()
    if ws:
        print(f"grammar whitespace: disabled via {list(ws)[0]}")

    mm_limits, mm_kwargs = _mm_args(model, tiles)

    # InternVL ships its own modeling code; Qwen keeps vLLM's default (False).
    trust = "qwen" not in model.lower()

    # Sampler warmup at vLLM's default max_num_seqs OOMs the A10 on non-Qwen
    # backbones. Under greedy decoding this cap changes no output.
    seqs = {} if not trust else {"max_num_seqs": 32}

    def _make(extra: dict):
        return LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=max_len,
            gpu_memory_utilization=util,
            # Qwen3-VL needs "video": 0, or vLLM sizes the encoder cache for a
            # maximum-length video and runs out of memory at startup.
            limit_mm_per_prompt=mm_limits,
            mm_processor_kwargs=mm_kwargs,
            trust_remote_code=trust,
            # Prompts are ~1,750 tokens; a smaller prefill chunk lowers the
            # activation peak that vLLM reserves memory for.
            max_num_batched_tokens=2048,
            # CUDA graphs need memory this card does not have, and outputs are
            # short, so eager mode costs little.
            enforce_eager=True,
            **seqs,
            **extra,
        )

    # If this vLLM rejects the whitespace option, retry once without it.
    try:
        return _make(ws)
    except ValueError as e:
        if ws and "whitespace" in str(e).lower():
            print(f"WARNING: engine rejected the whitespace option ({e}).\n"
                  f"         Retrying without it; responses may burn their "
                  f"token budget on indentation.")
            return _make({})
        raise


def build_requests(rows: pd.DataFrame, bases: Path, variants: Path,
                   presentation: str = "baseline",
                   bbox_frame: str = "pixel") -> tuple[list, list]:
    """Two requests per variant (SpatialFlow-GRPO A.4.3 / A.4.4):

      SC  source + edit, scoring every region (plus background and overall)
          in one structured response.
      PQ  the edit alone, giving the image-level [naturalness, artifacts].

    `presentation` re-packages both requests in memory for the nuisance and
    exploitability analyses (src/presentation.py); "baseline" leaves them as
    is. PQ is sent under every presentation so `reward` is always available.

    `bbox_frame` is the coordinate frame of the boxes in the SC prompt
    (judge_prompt.BBOX_FRAMES). norm1000 scales by the source size, which every
    edit shares.
    """
    msgs, meta = [], []
    for row in rows.itertuples():
        regions = json.loads((bases / row.base_id / "regions.json").read_text())
        src = Image.open(bases / row.base_id / "source.png").convert("RGB")
        edit = Image.open(variants / f"{row.variant_id}.png").convert("RGB")
        size = src.size

        shown = present(regions, row.target_region_id, row.variant_id, presentation)
        if presentation == "enhance":
            edit = enhance(edit)          # the edit only: the source is the real photo
        elif presentation == "enhance_target":
            tgt = [r for r in regions
                   if str(r["region_id"]) == str(row.target_region_id)]
            if not tgt:
                raise ValueError(f"{row.base_id}: target region "
                                 f"{row.target_region_id} not in regions.json")
            edit = enhance_region(edit, tgt[0]["bbox"])
        elif presentation == "box":
            src, edit = draw_boxes(src, shown), draw_boxes(edit, shown)
        pics = [] if presentation == "noimg" else [src, edit]

        common = dict(variant_id=row.variant_id, base_id=row.base_id,
                      target_region_id=str(row.target_region_id),
                      corruption=row.corruption, severity=row.severity,
                      area_bin=row.area_bin, is_control=row.is_control,
                      presentation=presentation, n_shown=len(shown),
                      region_ids=[r["region_id"] for r in shown])

        msgs.append([{"role": "user", "content": [
            *({"type": "image_pil", "image_pil": im} for im in pics),
            {"type": "text", "text": build_sc_prompt(row.instruction, shown,
                                                     bbox_frame, size)},
        ]}])
        meta.append({**common, "kind": "sc"})

        msgs.append([{"role": "user", "content": [
            *({"type": "image_pil", "image_pil": im} for im in pics[-1:]),
            {"type": "text", "text": build_pq_prompt()},
        ]}])
        meta.append({**common, "kind": "pq"})
    return msgs, meta


def structured_kind() -> str | None:
    """Which structured-output API this vLLM exposes, or None.

    `guided_decoding` in older versions, `structured_outputs` from 0.11.
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


def sampling_base(n_samples: int, temperature: float) -> dict:
    """Sampling parameters, shared by run() and dry_run().

    max_tokens leaves room for the per-region reasoning text the prompt asks
    for; repetition_penalty stops that text from looping until truncation.
    """
    return dict(n=n_samples, temperature=temperature, top_p=0.95,
                max_tokens=1536, repetition_penalty=1.1, seed=1234)


def run(llm, msgs, meta, n_samples: int, temperature: float,
        structured: bool = True, reasoning: str = "free") -> pd.DataFrame:
    from vllm import SamplingParams

    base = sampling_base(n_samples, temperature)

    kind = structured_kind() if structured else None
    if kind:
        print(f"structured output: {kind} (JSON schema per request, "
              f"reasoning={reasoning})")
    elif not structured:
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

    # One SamplingParams per request: each SC schema names that variant's
    # region ids, and PQ has a different shape.
    sps = []
    for m in meta:
        kw = dict(base)
        if kind:
            schema = (pq_json_schema(reasoning) if m["kind"] == "pq"
                      else sc_json_schema(m["region_ids"], reasoning))
            kw.update(_structured_kwargs(kind, schema))
        sps.append(SamplingParams(**kw))

    outs = llm.chat(msgs, sps)

    # One row per (variant, sample, region) plus a background row; the
    # image-level PQ scores are merged onto every row of their variant.
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
            # slot_idx: the region's position in the list shown to the judge.
            for slot, rid in enumerate(list(m["region_ids"]) + ["bg"]):
                pair = None if rid == "bg" else sc.get("regions", {}).get(rid)
                recs.append({
                    # Region ids are stored as strings so the column can also
                    # hold "bg" when written to parquet.
                    **base, "sample_idx": i, "scored_region_id": str(rid),
                    "slot_idx": -1 if rid == "bg" else slot,
                    "sc_success": pair[0] if pair else None,
                    "sc_preserve": pair[1] if pair else None,
                    "sc_background": sc.get("background"),
                    "sc_overall_success": (sc.get("overall") or [None, None])[0],
                    "sc_overall_preserve": (sc.get("overall") or [None, None])[1],
                    "pq_naturalness": pq[0] if pq else None,
                    "pq_artifacts": pq[1] if pq else None,
                    "reward": region_reward(sc, rid, pq),
                    "parsed": bool(sc),
                    # Distinguish truncated responses from malformed ones.
                    "finish_reason": getattr(cand, "finish_reason", None),
                    "n_tokens": len(getattr(cand, "token_ids", ()) or ()),
                    "raw": cand.text,
                })
    return pd.DataFrame(recs)


def _gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:       # provenance only; never fail a judged shard over it
        return None


def preflight(rows: pd.DataFrame, bases: Path, variants: Path) -> dict:
    """List every input file build_requests would open that does not exist."""
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
    """One chat message as text: images summarised, prompt shown in full."""
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
    """Build the requests and print the first one, without importing vLLM.

    Lets request construction be checked on a machine with no GPU.
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

    msgs, meta = build_requests(rows, bases, variants, a.presentation,
                                a.bbox_frame)
    per_variant = len(msgs) / max(len(rows), 1)
    print(f"{len(msgs)} requests ({per_variant:.2f} per variant: 1 SC + 1 PQ) "
          f"x n={a.n_samples} = {len(msgs) * a.n_samples} generations")

    # Show what the presentation actually changed.
    if a.presentation != "baseline":
        canon = [r["region_id"] for r in
                 json.loads((bases / rows.iloc[0].base_id / "regions.json").read_text())]
        print(f"\n--- presentation: {a.presentation} ---")
        print(f"  canonical region order (first base): {canon}")
        seen = [(m["variant_id"], m["target_region_id"], m["region_ids"])
                for m in meta if m["kind"] == "sc"]
        for vid, tgt, ids in seen[:5]:
            print(f"  {vid} target={tgt} -> shown {ids}")
        if a.presentation in TEXT_AXES:
            n_ch = sum(1 for _, _, ids in seen if ids != canon)
            print(f"  {n_ch}/{len(seen)} variants differ from the canonical list")
            if n_ch == 0:
                print("  WARNING: this axis changed NOTHING. A nuisance "
                      "condition identical to")
                print("  baseline measures nothing. Fix it before spending "
                      "GPU time.")
        else:
            print("  (this axis changes pixels, not the region list)")
        if a.presentation == "noimg":
            n_img = sum(1 for c in msgs[0][0]["content"] if c["type"] == "image_pil")
            print(f"  image parts in the first request: {n_img} (must be 0)")

    print("\n--- first request ---")
    print(f"meta: {meta[0]}")
    print('messages[0] = [{"role": "user", "content": [')
    print(describe_message(msgs[0]))
    print("]}]")

    print("\n--- sampling params that would be used (not constructed) ---")
    sp = sampling_base(a.n_samples, a.temperature)
    print("  " + " ".join(f"{k}={v}" for k, v in sp.items()))
    print(f"  (max_tokens is {sp['max_tokens']}, not 32: A.4.3 asks for "
          f"per-region reasoning text before the scores.)")
    print("\n--- engine that would be loaded (not loaded) ---")
    print(f"  model={a.model} dtype=bfloat16 "
          f"max_model_len={a.max_model_len} "
          f"gpu_memory_utilization={a.gpu_util}")
    dry_limits, dry_mm_kwargs = _mm_args(a.model, a.max_dynamic_patch)
    print(f"  enforce_eager=True max_num_batched_tokens=2048 "
          f"limit_mm_per_prompt={dry_limits}")
    print(f"  mm_processor_kwargs={dry_mm_kwargs}")
    print(f"  bbox_frame={a.bbox_frame} (the bbox_2d values in the prompt above)")

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
    # Greedy by default. Must match across shards, like --gpu-util.
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="pilot mode: cap variants")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="context reserved per request (default 4096). Lower "
                         "raises max concurrency proportionally; must be the "
                         "same on every VM.")
    ap.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                    help=f"gpu_memory_utilization (default {DEFAULT_GPU_UTIL}); "
                         "all VMs must pass the same value")
    ap.add_argument("--reasoning", default="free", choices=REASONING_MODES,
                    help="how the schema expresses the reasoning field. "
                         "free (default) is 7.5x faster than bounded at the "
                         "same parse rate; none drops the field, which the "
                         "protocol asks for.")
    ap.add_argument("--presentation", default="baseline", choices=PRESENTATIONS,
                    help="re-package the requests for the nuisance and "
                         "exploitability analyses (src/presentation.py). Write "
                         "each condition to its own parquet, not alongside "
                         "out/scores_shard*.parquet: stage 4 does not group by "
                         "presentation.")
    ap.add_argument("--bbox-frame", default="pixel", choices=BBOX_FRAMES,
                    help="coordinate frame of the region boxes in the SC "
                         "prompt: pixel (default), or norm1000, the 0-1000 "
                         "relative frame Qwen3-VL and InternVL3 use.")
    ap.add_argument("--max-dynamic-patch", type=int,
                    default=DEFAULT_MAX_DYNAMIC_PATCH,
                    help="InternVL only: max 448px tiles per image (default "
                         f"{DEFAULT_MAX_DYNAMIC_PATCH}). "
                         "6 needs --max-model-len 8192. Ignored for Qwen.")
    ap.add_argument("--no-structured", action="store_true",
                    help="disable JSON-schema-constrained decoding. Only for "
                         "measuring what the constraint is worth -- "
                         "unconstrained runs drop regions and loop.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the requests and print the first one, without "
                         "importing vLLM — runs on a machine with no GPU")
    a = ap.parse_args()

    if a.n_samples > 1 and a.temperature == 0:
        print(f"WARNING: --n-samples {a.n_samples} at --temperature 0 gives "
              f"{a.n_samples} IDENTICAL samples.\n"
              f"         Greedy decoding is deterministic, so this costs "
              f"{a.n_samples}x and measures no noise floor.\n"
              f"         For a floor run add --temperature 0.7; otherwise drop "
              f"to --n-samples 1.")

    df = shard(pd.read_parquet(a.manifest), a.shard, a.of)
    if a.limit:
        df = df.head(a.limit)
    print(f"judging {len(df)} variants with {a.model}")

    if a.dry_run:
        raise SystemExit(dry_run(df, Path(a.bases), Path(a.variants), a))
    if not a.out:
        ap.error("--out is required unless --dry-run is given")

    pf = preflight(df, Path(a.bases), Path(a.variants))
    n_missing = sum(len(v) for v in pf["missing"].values())
    if n_missing:
        lines = [f"{n_missing} input file(s) missing; nothing was judged."]
        for kind, paths in pf["missing"].items():
            if paths:
                lines.append(f"  {kind}: {len(paths)} missing, e.g. {paths[0]}")
        if pf["missing"]["variant"]:
            lines.append("Variants live in /dev/shm and are wiped on reboot, and "
                         "run_shard.sh deletes them")
            lines.append("when it finishes. Regenerate them (~3s):")
            lines.append(f"  python -m src.stage2_corrupt --manifest {a.manifest} "
                         f"--bases {a.bases} \\")
            lines.append(f"      --out {a.variants} --shard {a.shard} --of {a.of}")
        else:
            lines.append("Run stage 0 and stage 1 first, or point --bases elsewhere.")
        raise SystemExit("\n".join(lines))

    msgs, meta = build_requests(df, Path(a.bases), Path(a.variants), a.presentation,
                                a.bbox_frame)
    print(f"{len(msgs)} requests x n={a.n_samples} (presentation={a.presentation}, "
          f"bbox_frame={a.bbox_frame})")

    llm = load_engine(a.model, max_len=a.max_model_len, util=a.gpu_util,
                      tiles=a.max_dynamic_patch)
    res = run(llm, msgs, meta, a.n_samples, a.temperature,
              structured=not a.no_structured, reasoning=a.reasoning)
    # Provenance: how this parquet was produced.
    res["judge"] = a.model
    res["n_samples"] = a.n_samples
    res["temperature"] = a.temperature
    res["bbox_frame"] = a.bbox_frame
    res["max_model_len"] = a.max_model_len
    res["gpu_util"] = a.gpu_util
    res["max_dynamic_patch"] = (a.max_dynamic_patch if "internvl" in a.model.lower()
                                else None)
    res["gpu_name"] = _gpu_name()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out)

    ok = res.parsed.mean() if len(res) else 0.0
    print(f"parse rate: {ok:.1%}")
    if ok < 0.95:
        print("WARNING: low parse rate — fix the prompt before scaling up")
    print(res.reward.describe())


if __name__ == "__main__":
    main()
