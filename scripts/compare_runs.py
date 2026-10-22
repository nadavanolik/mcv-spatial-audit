"""
Two runs of the SAME judge on the same manifest: how different are they?  [CPU]

Used twice on Athena:

  hardware  committed A10 run vs the same protocol on A100. Expected to sit at
            the greedy noise floor (identical requests disagree 6-11% of the
            time on `reward` even on one VM); if it does not, A100 numbers are
            not comparable with the committed ones.
  frame     pixel vs norm1000 boxes on A100. The effect of telling the judge
            where the regions are in the frame it grounds in.

Two parts:

  paired.csv    per (variant, region) pair present in both runs: share
                identical, mean |difference|, correlation -- on clean controls
                and on corrupted variants separately, for `reward` and `phi`.
  headline.csv  the stage-4 headline computed on each run (floor-excluded with
                --min-control): pooled AUROC, tie rates, coherence against the
                permutation null, delta redundancy. With a photo-bootstrap
                interval on the AUROC difference b - a.

Usage:
    python -m scripts.compare_runs \\
        --a 'results/main_2026-09-18/scores_shard*.parquet' --a-name a10 \\
        --b 'results/athena/main_qwen_pixel/scores_shard*.parquet' --b-name a100 \\
        --min-control 0 --out out/analysis/compare_hardware
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage4_analyze import (BG, N_BOOT, SEED, _ci, _photo_boot_idx,  # noqa: E402
                                delta_table, drop_floored, load,
                                redundancy_delta, response_coherence, usable)

READOUTS = ["reward", "phi"]
KEY = ["variant_id", "scored_region_id"]


def paired(a: pd.DataFrame, b: pd.DataFrame, names) -> pd.DataFrame:
    rows = []
    for col in READOUTS:
        x = a.groupby(KEY + ["is_control"])[col].mean().rename("a")
        y = b.groupby(KEY + ["is_control"])[col].mean().rename("b")
        m = pd.concat([x, y], axis=1, join="inner").dropna().reset_index()
        m = m[m.scored_region_id != BG]
        for part, g in (("controls", m[m.is_control]), ("corrupted", m[~m.is_control])):
            if g.empty:
                continue
            diff = g.b - g.a
            rows.append(dict(
                readout=col, part=part, a=names[0], b=names[1], n=len(g),
                identical=float((diff == 0).mean()),
                mean_abs_diff=float(diff.abs().mean()),
                mean_diff=float(diff.mean()),
                pearson=float(np.corrcoef(g.a, g.b)[0, 1]) if len(g) > 2 else np.nan))
    return pd.DataFrame(rows)


def deltas(df: pd.DataFrame, col: str, min_control) -> pd.DataFrame:
    d = delta_table(usable(df, col), col)
    if min_control is not None:
        d = drop_floored(d, min_control)
    return d[d.scored_region_id != BG]


def headline(d: pd.DataFrame, name: str, col: str) -> dict:
    tgt, oth = d[d.is_target], d[~d.is_target]
    coh = response_coherence(d)
    red = redundancy_delta(d)
    out = dict(run=name, readout=col, n_rows=len(d), n_bases=d.base_id.nunique(),
               auroc=roc_auc_score(d.is_target, -d.delta),
               target_unchanged=float((tgt.delta == 0).mean()),
               other_unchanged=float((oth.delta == 0).mean()),
               mean_delta_target=float(tgt.delta.mean()),
               mean_delta_other=float(oth.delta.mean()))
    if len(coh):
        c = coh.iloc[0]
        out.update(frac_mixed=c.frac_mixed, mixed_null=c.mixed_null,
                   gap=c.gap, gap_lo=c.gap_lo, gap_hi=c.gap_hi)
    if len(red):
        r = red.iloc[0]
        out.update(r2_delta=r.r2, r2_delta_lo=r.r2_lo, r2_delta_hi=r.r2_hi)
    return out


def auroc_diff_ci(da: pd.DataFrame, db: pd.DataFrame,
                  n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """AUROC(b) - AUROC(a), resampling photographs shared by both runs."""
    photos = sorted(set(da.base_id) & set(db.base_id))
    ia = {p: np.flatnonzero(da.base_id.to_numpy() == p) for p in photos}
    ib = {p: np.flatnonzero(db.base_id.to_numpy() == p) for p in photos}
    ya, sa = da.is_target.to_numpy(), -da.delta.to_numpy()
    yb, sb = db.is_target.to_numpy(), -db.delta.to_numpy()
    diffs = []
    for draw in _photo_boot_idx(len(photos), n_boot, seed):
        ka = np.concatenate([ia[photos[k]] for k in draw])
        kb = np.concatenate([ib[photos[k]] for k in draw])
        if len(set(ya[ka])) < 2 or len(set(yb[kb])) < 2:
            continue
        diffs.append(roc_auc_score(yb[kb], sb[kb]) - roc_auc_score(ya[ka], sa[ka]))
    point = roc_auc_score(yb, sb) - roc_auc_score(ya, sa)
    lo, hi = _ci(diffs)
    return point, lo, hi, len(photos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--min-control", type=float, default=None)
    ap.add_argument("--out", default="out/analysis/compare")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = load(a.a), load(a.b)
    for name, df in ((a.a_name, A), (a.b_name, B)):
        if df.judge.nunique() != 1:
            raise SystemExit(f"{name}: expected one judge, got {list(df.judge.unique())}")
    if A.judge.iloc[0] != B.judge.iloc[0]:
        print(f"NOTE: different judges ({A.judge.iloc[0]} vs {B.judge.iloc[0]}); "
              f"the paired table compares models, not runs")

    p = paired(A, B, (a.a_name, a.b_name))
    print("\n=== paired scores, same (variant, region) in both runs ===")
    print(p.round(4).to_string(index=False))
    p.to_csv(out / "paired.csv", index=False)

    rows = []
    for col in READOUTS:
        da, db = deltas(A, col, a.min_control), deltas(B, col, a.min_control)
        ha, hb = headline(da, a.a_name, col), headline(db, a.b_name, col)
        d, lo, hi, k = auroc_diff_ci(da, db)
        hb.update(auroc_diff=d, auroc_diff_lo=lo, auroc_diff_hi=hi, n_shared_photos=k)
        rows += [ha, hb]
    h = pd.DataFrame(rows)
    print(f"\n=== stage-4 headline per run (min-control {a.min_control}) ===")
    print(h.round(4).to_string(index=False))
    h.to_csv(out / "headline.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
