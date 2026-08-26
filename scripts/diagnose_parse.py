"""
Why did judge responses fail to parse? Reads a scores parquet. CPU only.

stage3 stores the raw response text, so a low parse rate can be diagnosed
without touching a GPU or re-running the judge. Run this before changing the
prompt: the two failure modes have opposite fixes.

  TRUNCATED     the response hit max_tokens mid-JSON. A.4.3 demands per-region
                `reasoning` before the scores, so a multi-region answer can be
                hundreds of tokens and the cap is the binding constraint.
                Fix: raise max_tokens, or ask for shorter reasoning.
  MALFORMED     the response terminated on its own but is not the shape
                parse_sc expects -- wrong keys, prose instead of JSON, scores
                out of range. Fix: the prompt or the parser.

Usage:
    python -m scripts.diagnose_parse --scores out/smoke_scores.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.judge_prompt import _extract_json, SCORE_MIN, SCORE_MAX  # noqa: E402


def classify(raw: str, finish_reason=None) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "EMPTY"
    if finish_reason == "length":
        return "TRUNCATED (finish_reason=length)"
    # No finish_reason recorded (older parquet): infer it. An unbalanced brace
    # count is the signature of a response cut off mid-object.
    if raw.count("{") > raw.count("}"):
        return "TRUNCATED (unbalanced braces)"
    d = _extract_json(raw)
    if d is None:
        return "MALFORMED (no parseable JSON object)"
    if not isinstance(d, dict):
        return "MALFORMED (JSON is not an object)"
    missing = [k for k in ("edit_region", "background", "overall_score")
               if k not in d]
    if missing:
        return f"MALFORMED (missing keys: {','.join(missing)})"
    regions = d.get("edit_region")
    if not isinstance(regions, list) or not regions:
        return "MALFORMED (edit_region is not a non-empty list)"
    for r in regions:
        if not isinstance(r, dict):
            return "MALFORMED (edit_region entry is not an object)"
        if "id" not in r:
            return "MALFORMED (region has no id)"
        sc = r.get("score")
        if not isinstance(sc, list) or len(sc) != 2:
            return "MALFORMED (score is not a 2-list)"
        for v in sc:
            if not isinstance(v, (int, float)):
                return f"MALFORMED (score entry {v!r} is not a number)"
            if not (SCORE_MIN <= v <= SCORE_MAX):
                return f"MALFORMED (score {v} outside {SCORE_MIN}-{SCORE_MAX})"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--show", type=int, default=2,
                    help="how many failing responses to print in full")
    a = ap.parse_args()

    df = pd.read_parquet(a.scores)
    # One response produced many rows (one per scored region). Diagnose the
    # response, not the row, or every count is multiplied by the region count.
    key = ["variant_id", "sample_idx"]
    resp = df.drop_duplicates(subset=key)[key + ["raw", "parsed"]].copy()
    if "finish_reason" in df.columns:
        resp = resp.merge(df.drop_duplicates(subset=key)[key + ["finish_reason", "n_tokens"]],
                          on=key, how="left")
    else:
        resp["finish_reason"] = None
        resp["n_tokens"] = None

    resp["why"] = [classify(r, f) for r, f in
                   zip(resp.raw, resp.finish_reason)]

    print(f"{len(resp)} distinct responses from {len(df)} rows")
    print(f"parse rate (as stage 3 counts it): {df.parsed.mean():.1%}\n")

    print("=== failure breakdown ===")
    for why, n in resp.why.value_counts().items():
        print(f"  {n:4d}  {why}")

    if resp.n_tokens.notna().any():
        print("\n=== response length (tokens) ===")
        print(resp.groupby(resp.why.str.split(" ").str[0]).n_tokens
              .describe()[["count", "mean", "max"]].round(0).to_string())

    bad = resp[resp.why != "OK"]
    if bad.empty:
        print("\nAll responses are well formed.")
        return 0

    print(f"\n=== {min(a.show, len(bad))} of {len(bad)} failing responses ===")
    for _, row in bad.head(a.show).iterrows():
        print("\n" + "-" * 70)
        print(f"variant={row.variant_id} sample={row.sample_idx} "
              f"finish_reason={row.finish_reason} n_tokens={row.n_tokens}")
        print(f"why: {row.why}")
        print("-" * 70)
        print(row.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
