"""
Stage 4 — the four analyses. Pure pandas over the merged score shards.

Every effect is reported against the judge's own noise floor, measured from
the repeated samples on the clean controls. An effect smaller than that floor
is not an effect.

Usage:
    python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' --out out/analysis
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no score files match {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(files)} shards, judges={df.judge.unique()}")
    return df


def noise_floor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-judge SD across repeated samples of the SAME (variant, region).
    This is the denominator for everything below."""
    g = (df[df.is_control]
         .groupby(["judge", "variant_id", "scored_region_id"])[col]
         .std().reset_index(name="sd"))
    return g.groupby("judge").sd.agg(["mean", "median", "max"]).reset_index()


def delta_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """∆score = corrupted - matching clean control, per (base, scored region)."""
    agg = (df.groupby(["judge", "base_id", "variant_id", "scored_region_id",
                       "target_region_id", "corruption", "severity",
                       "area_bin", "is_control"])[col]
           .mean().reset_index(name="score"))

    ctrl = (agg[agg.is_control]
            .groupby(["judge", "base_id", "scored_region_id"])["score"]
            .mean().reset_index(name="ctrl_score"))

    out = agg[~agg.is_control].merge(ctrl, on=["judge", "base_id", "scored_region_id"])
    out["delta"] = out.score - out.ctrl_score
    out["is_target"] = out.scored_region_id == out.target_region_id
    return out


def localization(d: pd.DataFrame) -> pd.DataFrame:
    """AUROC: can ∆score identify WHICH region was corrupted?

    Label = "this region is the corrupted one". A judge with real spatial
    credit scores high; 0.5 means the per-region number carries no spatial
    information at all.
    """
    rows = []
    for (judge, sev), g in d.groupby(["judge", "severity"]):
        if g.is_target.nunique() < 2:
            continue
        rows.append(dict(judge=judge, severity=sev,
                         auroc=roc_auc_score(g.is_target, -g.delta),
                         n=len(g),
                         mean_delta_target=g[g.is_target].delta.mean(),
                         mean_delta_other=g[~g.is_target].delta.mean()))
    return pd.DataFrame(rows)


def leakage_matrix(d: pd.DataFrame, judge: str) -> pd.DataFrame:
    """Mean ∆score at region j when region i was corrupted.

    Off-diagonal mass IS the finding: it is the penalty landing somewhere other
    than where the damage is.
    """
    g = d[d.judge == judge]
    return g.pivot_table(index="target_region_id", columns="scored_region_id",
                         values="delta", aggfunc="mean")


def redundancy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """How much per-region variance is just the image-level impression?

    Regress each region's score on the per-image mean; R^2 near 1 means the
    region scores are the global score copied across regions.
    """
    rows = []
    for judge, g in df.groupby("judge"):
        a = g.groupby(["variant_id", "scored_region_id"])[col].mean().reset_index()
        img = a.groupby("variant_id")[col].mean().rename("img_mean")
        a = a.join(img, on="variant_id")
        a = a.dropna(subset=[col, "img_mean"])
        if len(a) > 2 and a.img_mean.std() > 0:
            r = np.corrcoef(a[col], a.img_mean)[0, 1]
            rows.append(dict(judge=judge, r2=r ** 2, n=len(a)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="out/analysis")
    ap.add_argument("--col", default="sc_expected",
                    help="sc_expected (logprob) or sc_sampled (integer)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.scores)

    nf = noise_floor(df, a.col)
    print("\n=== noise floor (SD across repeats, clean controls) ===")
    print(nf.to_string(index=False))
    nf.to_csv(out / "noise_floor.csv", index=False)

    d = delta_table(df, a.col)
    d.to_parquet(out / "deltas.parquet")

    loc = localization(d)
    print("\n=== localization AUROC ===")
    print(loc.to_string(index=False))
    loc.to_csv(out / "localization.csv", index=False)

    for judge in d.judge.unique():
        lm = leakage_matrix(d, judge)
        print(f"\n=== leakage matrix: {judge} ===")
        print(lm.round(3).to_string())
        lm.to_csv(out / f"leakage_{judge.replace('/', '_')}.csv")

    red = redundancy(df, a.col)
    print("\n=== redundancy (R^2 vs image mean) ===")
    print(red.to_string(index=False))
    red.to_csv(out / "redundancy.csv", index=False)

    # The headline sanity check: is the target-region effect even above noise?
    floor = nf.set_index("judge")["median"].to_dict()
    print("\n=== effect vs noise floor ===")
    for judge, g in d[d.is_target].groupby("judge"):
        eff = abs(g.delta.mean())
        f = floor.get(judge, float("nan"))
        verdict = "ABOVE noise" if eff > f else "BELOW NOISE — not a signal"
        print(f"{judge}: |mean ∆|={eff:.3f} vs floor={f:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
