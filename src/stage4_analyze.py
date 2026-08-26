"""
Stage 4 - the four analyses. Pure pandas over the merged score shards.

Reads the A.4.3 columns stage 3 actually emits (`sc_success`, `sc_preserve`,
`sc_background`, `pq_*`, `reward`, `parsed`), not the retired 1-5 placeholder
pair `sc_sampled` / `sc_expected`.

Every effect is reported against the judge's own noise floor, measured from the
repeated samples on the clean controls. An effect smaller than that floor is
not an effect.

FOUR READOUTS, and the difference between them is itself a result:

  reward       Equation (3), sqrt(phi * AES)/C. The paper's own per-region
               quantity, and therefore the headline. Carries the image-level
               AES factor, so part of it is global by construction.
  phi          min(success, preserve) - Equation (3) with the global AES factor
               divided out. Strictly more spatially resolved than `reward` can
               be; if `reward` localises worse than `phi`, AES dilution is why.
  sc_preserve  The axis corruption SHOULD move: the edit still follows the
               instruction, it is just damaged.
  sc_success   The axis corruption should NOT move.

Splitting those last two is the specific diagnostic for the open risk that the
judge is blind to degradation of an edit it has already called compliant. phi =
min() hides which axis moved; --col lets you look at each.

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

# The readouts run side by side in every table, so a degenerate one is visible
# next to a working one rather than being discovered on its own.
READOUTS = ["reward", "phi", "sc_preserve", "sc_success"]

BG = "bg"


def load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no score files match {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # scored_region_id mixes ints with the literal "bg", so parquet hands it
    # back as object. Normalise both sides to str once, here, rather than
    # discovering the mixed-dtype comparison in the middle of a groupby.
    df["scored_region_id"] = df.scored_region_id.astype(str)
    df["target_region_id"] = df.target_region_id.astype(str)

    # phi: Equation (3)'s instruction-following term, before AES multiplies it.
    # Background rows take the background score, exactly as region_reward does.
    fg = df[["sc_success", "sc_preserve"]].min(axis=1)
    df["phi"] = np.where(df.scored_region_id == BG, df.sc_background, fg)

    n = len(df)
    parsed = df.parsed.mean() if "parsed" in df else float("nan")
    print(f"loaded {n} rows from {len(files)} shards, judges={list(df.judge.unique())}")
    print(f"parse rate: {parsed:.1%}")
    if parsed < 0.95:
        print("WARNING: low parse rate - every number below is over a biased "
              "subset (the responses that happened to be well-formed).")
    return df


def usable(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows where this readout actually has a value.

    Dropped separately per readout, not once globally: a response can carry a
    valid SC block but a missing PQ, which kills `reward` while leaving `phi`
    perfectly good. Dropping globally would silently shrink every table to the
    intersection.
    """
    d = df[df[col].notna()]
    if len(d) < len(df):
        print(f"  [{col}] using {len(d)}/{len(df)} rows "
              f"({len(df) - len(d)} missing this readout)")
    return d


def noise_floor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-judge SD across repeated samples of the SAME (variant, region).
    This is the denominator for everything below."""
    g = (df[df.is_control]
         .groupby(["judge", "variant_id", "scored_region_id"])[col]
         .std().reset_index(name="sd"))
    if g.empty:
        return pd.DataFrame(columns=["judge", "mean", "median", "max"])
    return g.groupby("judge").sd.agg(["mean", "median", "max"]).reset_index()


def delta_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """delta = corrupted - matching clean control, per (base, scored region).

    The control baseline is averaged over ALL of a base's controls. The manifest
    emits one control row per region, but they are the same clean edit rendered
    under different variant_ids, so pooling them just buys more samples of the
    same quantity.
    """
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
    """AUROC: can delta identify WHICH region was corrupted?

    Label = "this region is the corrupted one". A judge with real spatial credit
    scores high; 0.5 means the per-region number carries no spatial information
    at all.

    Background rows are excluded: `bg` is never a corruption target, so it is a
    guaranteed negative and including it would inflate AUROC for free whenever
    the judge merely distinguishes foreground from background. It stays in the
    leakage matrix, where "damage in region i moved the background score" is a
    real thing to see.
    """
    d = d[d.scored_region_id != BG]
    rows = []
    for (judge, sev), g in d.groupby(["judge", "severity"]):
        if g.is_target.nunique() < 2:
            continue
        rows.append(dict(judge=judge, severity=sev,
                         auroc=roc_auc_score(g.is_target, -g.delta),
                         n=len(g),
                         n_target=int(g.is_target.sum()),
                         ties=float((g.delta == 0).mean()),
                         mean_delta_target=g[g.is_target].delta.mean(),
                         mean_delta_other=g[~g.is_target].delta.mean()))
    return pd.DataFrame(rows)


def localization_by_corruption(d: pd.DataFrame) -> pd.DataFrame:
    """The same AUROC split by corruption type.

    `remove` and `blur` are structurally different damage from `noise`, and the
    synthetic smoke test only ever tried `noise`. If sensitivity exists at all,
    this is the table it shows up in.
    """
    d = d[d.scored_region_id != BG]
    rows = []
    for (judge, corr), g in d.groupby(["judge", "corruption"]):
        if g.is_target.nunique() < 2:
            continue
        rows.append(dict(judge=judge, corruption=corr,
                         auroc=roc_auc_score(g.is_target, -g.delta),
                         n=len(g),
                         mean_delta_target=g[g.is_target].delta.mean(),
                         mean_delta_other=g[~g.is_target].delta.mean()))
    return pd.DataFrame(rows)


def leakage_matrix(d: pd.DataFrame, judge: str) -> pd.DataFrame:
    """Mean delta at region j when region i was corrupted.

    Off-diagonal mass IS the finding: it is the penalty landing somewhere other
    than where the damage is.
    """
    g = d[d.judge == judge]
    return g.pivot_table(index="target_region_id", columns="scored_region_id",
                         values="delta", aggfunc="mean")


def redundancy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """How much per-region variance is just the image-level impression?

    Regress each region's score on the mean over that image's OTHER regions.
    Not on the plain image mean, which includes the region itself: with the ~4
    regions we score, self-inclusion contributes about a quarter of the
    predictor and manufactures correlation even out of pure noise. Leave-one-out
    makes R^2 near 1 mean what it is supposed to mean - the region scores are
    one global impression copied across regions.

    Background is excluded from both sides; it is a different question.
    """
    df = df[df.scored_region_id != BG]
    rows = []
    for judge, g in df.groupby("judge"):
        a = g.groupby(["variant_id", "scored_region_id"])[col].mean().reset_index()
        grp = a.groupby("variant_id")[col]
        a["n_reg"] = grp.transform("size")
        a["others"] = (grp.transform("sum") - a[col]) / (a["n_reg"] - 1)
        a = a[a.n_reg > 1].dropna(subset=[col, "others"])
        if len(a) > 2 and a.others.std() > 0 and a[col].std() > 0:
            r = np.corrcoef(a[col], a.others)[0, 1]
            rows.append(dict(judge=judge, r2=r ** 2, n=len(a)))
    return pd.DataFrame(rows)


def axis_table(df: pd.DataFrame) -> pd.DataFrame:
    """success vs preserve on the TARGETED region, by severity.

    The specific open question: corruption should drive `preserve` down while
    `success` holds, because the edit still follows the instruction - it is
    just damaged. If `preserve` never moves, that is the finding, and it is one
    this table states directly rather than leaving to be inferred from a
    collapsed AUROC.
    """
    d = df[(df.scored_region_id == df.target_region_id) & (df.scored_region_id != BG)]
    if d.empty:
        return pd.DataFrame()
    return (d.groupby(["judge", "corruption", "severity"])
            [["sc_success", "sc_preserve", "phi", "reward"]]
            .mean().round(3).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="out/analysis")
    ap.add_argument("--col", default="reward", choices=READOUTS,
                    help="headline readout (default: reward, Equation 3)")
    ap.add_argument("--all-readouts", action="store_true",
                    help="run localization for every readout in READOUTS - the "
                         "success/preserve split is the diagnostic for a judge "
                         "that ignores damage to an edit it called compliant")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.scores)

    cols = READOUTS if a.all_readouts else [a.col]

    print("\n=== score distributions (all rows) ===")
    print(df[READOUTS].describe().round(3).to_string())
    for c in READOUTS:
        nunq = df[c].nunique(dropna=True)
        if nunq <= 2:
            print(f"WARNING: {c} takes only {nunq} distinct value(s) - "
                  f"delta is all-or-nothing and AUROC degenerates.")

    head = usable(df, a.col)
    nf = noise_floor(head, a.col)
    print(f"\n=== noise floor, {a.col} (SD across repeats, clean controls) ===")
    print(nf.to_string(index=False) if len(nf) else "  no controls found")
    nf.to_csv(out / "noise_floor.csv", index=False)

    print("\n=== targeted region: which axis moves? ===")
    ax = axis_table(df)
    print(ax.to_string(index=False) if len(ax) else "  nothing targeted")
    ax.to_csv(out / "axis_by_severity.csv", index=False)

    d = delta_table(head, a.col)
    d.to_parquet(out / "deltas.parquet")

    print("\n=== localization AUROC by severity ===")
    for c in cols:
        dc = d if c == a.col else delta_table(usable(df, c), c)
        loc = localization(dc)
        if len(loc):
            loc.insert(0, "readout", c)
            print(loc.round(4).to_string(index=False))
        else:
            print(f"  {c}: n/a")
        loc.to_csv(out / f"localization_{c}.csv", index=False)

    print("\n=== localization AUROC by corruption type ===")
    lc = localization_by_corruption(d)
    print(lc.round(4).to_string(index=False) if len(lc) else "  n/a")
    lc.to_csv(out / f"localization_by_corruption_{a.col}.csv", index=False)

    for judge in d.judge.unique():
        lm = leakage_matrix(d, judge)
        print(f"\n=== leakage matrix ({a.col}): {judge} ===")
        print(lm.round(3).to_string())
        lm.to_csv(out / f"leakage_{judge.replace('/', '_')}.csv")

    print("\n=== redundancy (R^2 vs leave-one-out mean of other regions) ===")
    for c in cols:
        red = redundancy(usable(df, c), c)
        if len(red):
            red.insert(0, "readout", c)
            print(red.round(4).to_string(index=False))
        else:
            print(f"  {c}: n/a")
        red.to_csv(out / f"redundancy_{c}.csv", index=False)

    # The headline sanity check: is the target-region effect even above noise?
    floor = nf.set_index("judge")["median"].to_dict() if len(nf) else {}
    print(f"\n=== effect vs noise floor ({a.col}) ===")
    for judge, g in d[d.is_target].groupby("judge"):
        eff = abs(g.delta.mean())
        f = floor.get(judge, float("nan"))
        verdict = "ABOVE noise" if eff > f else "BELOW NOISE - not a signal"
        print(f"{judge}: |mean delta|={eff:.4f} vs floor={f:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
