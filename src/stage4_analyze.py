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

# Corruptions whose severity ladder is not a ladder, collapsed to one condition
# before any groupby. `remove` inpaints the object away at every setting:
# measured on real photographs (2026-08-26), severity 1 and 3 change the masked
# pixels by 35.01 and 35.52 mean 8-bit levels - 0.5 apart, against blur's
# 7.60 -> 21.87 over the same span. The object is gone either way; radius only
# changes how the fill borrows neighbouring texture.
#
# This matters because a flat response across remove's severities is an ABSENT
# STIMULUS, not judge insensitivity. Left in a by-severity table it reports our
# own design as a finding about the judge.
FLAT_SEVERITY = {"remove"}
BINARY = "binary"


def collapse_flat_severity(d: pd.DataFrame) -> pd.DataFrame:
    """Fold FLAT_SEVERITY corruptions into a single severity label.

    Nothing is dropped and nothing is pooled across corruptions: `remove` keeps
    its own row in every per-corruption table, and appears in the by-severity
    tables as its own `binary` row rather than contaminating 1 and 3. What goes
    away is only the claim that its severity 1 and severity 3 are different
    stimuli.

    Severity becomes a string for every row, so the label sorts and groups
    alongside the numeric levels. Nothing downstream does arithmetic on it.
    """
    if not {"severity", "corruption"} <= set(d.columns):
        return d
    d = d.copy()
    d["severity"] = d.severity.astype(str)
    d.loc[d.corruption.isin(FLAT_SEVERITY), "severity"] = BINARY
    return d


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


def measurement_quality(df: pd.DataFrame, col: str, nf: pd.DataFrame) -> None:
    """Can this data detect an effect at all? Print before interpreting anything.

    Two ways the answer is no, both seen on the first real pilot:

    RAILS. If the judge mostly emits the extremes (0 or 25), the score is closer
    to a coin flip than a measurement, and repeated samples of the SAME input
    land on different rails. That shows up as a huge within-variant SD.

    FLOOR. A region whose CLEAN control already scored 0 cannot go down. Damaging
    it is uninformative by construction, and including it dilutes every average
    toward zero. This is a property of our data, not the judge: FLUX does not
    always execute the instruction, so some regions are legitimately 0 before we
    touch them.
    """
    print("\n=== measurement quality (read this before any result below) ===")
    v = df[col].dropna()
    if v.empty:
        print("  no values")
        return
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    rail = float(((v <= lo + 1e-9) | (v >= hi - 1e-9)).mean())
    print(f"  {col}: range {lo:.3f}-{hi:.3f}, {rail:.1%} of values sit on a rail")

    floor = float(nf["median"].max()) if len(nf) else float("nan")
    if floor != floor:
        # NaN, not zero: pandas .std() over a single sample is undefined. That
        # happens with --n-samples 1, which is the RIGHT way to run this once
        # the sampling noise is understood -- so say what to use instead rather
        # than letting a NaN comparison silently render every verdict "BELOW
        # NOISE", which is exactly what it did on the first greedy run.
        print("  noise floor is undefined (n_samples=1: no within-variant "
              "spread to measure).")
        print("  Judge effects against the between-variant SD below, and "
              "against the tie rate.")
    elif rng > 0:
        print(f"  noise floor {floor:.3f} = {floor / rng:.1%} of that range")
        print(f"  smallest detectable effect ~{2 * floor:.3f} "
              f"({2 * floor / rng:.1%} of range)")
        if floor > 0.15 * rng:
            print("  WARNING: the floor is a large fraction of the range. Repeated")
            print("  samples of the SAME input disagree that much, so nothing")
            print("  smaller can be resolved. Lower the sampling temperature")
            print("  before reading any AUROC below as a property of the judge.")
        elif floor == 0:
            print("  Floor is exactly 0 -- greedy decoding, so within-variant SD is")
            print("  NOT a usable denominator. Judge effects against the spread")
            print("  ACROSS variants instead; see the between-variant SD below.")

    # The alternative denominator: how much do clean controls differ from each
    # other across bases? An effect has to clear this to be interesting.
    ctrl = (df[df.is_control]
            .groupby(["base_id", "scored_region_id"])[col].mean())
    if len(ctrl) > 1:
        print(f"  between-variant SD of clean controls: {ctrl.std():.3f}")


def floor_fraction(d: pd.DataFrame, thresh: float) -> str:
    keep = (d.ctrl_score > thresh).mean() if len(d) else float("nan")
    return f"{keep:.1%}"


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
    # Here, not at the call sites: every severity groupby downstream reads this
    # table, and one that forgot would silently produce the misleading version.
    return collapse_flat_severity(out)


def drop_floored(d: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """Keep only regions whose CLEAN control scored above `thresh`.

    A region already at the bottom on the clean edit cannot drop further when we
    damage it, so it contributes a guaranteed zero delta to both classes and
    pulls AUROC toward 0.5 no matter how well the judge localises. Excluding it
    is not cherry-picking: it is refusing to ask a question the scale cannot
    answer. Say in the report how many regions this removed and at what
    threshold.
    """
    before = len(d)
    d = d[d.ctrl_score > thresh].copy()
    print(f"  --min-control {thresh}: kept {len(d)}/{before} rows "
          f"({len(d) / max(before, 1):.1%})")
    return d


def sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    """How often does the score MOVE AT ALL when we corrupt a region?

    The most direct question the audit can ask, and under greedy decoding it
    needs no statistics: delta == 0 exactly means the judge returned the same
    number for the clean image and the damaged one.

    It matters more than AUROC here. AUROC asks whether the corrupted region
    ranks below the others, which is vacuous when neither moved -- a judge that
    never reacts scores 0.5, and so does one that reacts at random. The tie rate
    tells those apart, and on the first greedy pilot it was 60-81%.
    """
    d = d[d.scored_region_id != BG]
    rows = []
    for (judge, corr, sev), g in d.groupby(["judge", "corruption", "severity"]):
        tgt, oth = g[g.is_target], g[~g.is_target]

        def frac(x, op):
            return float(op(x.delta).mean()) if len(x) else float("nan")

        rows.append(dict(
            judge=judge, corruption=corr, severity=sev, n_target=len(tgt),
            target_unchanged=frac(tgt, lambda v: v == 0),
            target_dropped=frac(tgt, lambda v: v < 0),
            target_rose=frac(tgt, lambda v: v > 0),
            other_unchanged=frac(oth, lambda v: v == 0),
        ))
    return pd.DataFrame(rows)


def response_coherence(d: pd.DataFrame) -> pd.DataFrame:
    """Within ONE variant, do the regions move together or independently?

    The sensitivity table showed target_unchanged and other_unchanged equal to
    three decimals in three of four cells. Similar rates would be unremarkable;
    IDENTICAL rates suggest the judge's answer changes as a whole-image event -
    either every region of a variant differs from its control, or none does.

    That is a much stronger claim than "does not localise", so it gets its own
    test rather than an eyeball. Classify each variant as none / all / mixed by
    how many of its regions moved, and compare the mixed fraction against what
    independent per-region movement would produce at the same overall rate:

        P(mixed) = 1 - p^n - (1-p)^n

    Mixed far below that expectation means regions are not moving independently
    - the per-region scores share one decision. Mixed at or above it is
    consistent with independent per-region behaviour and this reading is wrong.
    """
    d = d[d.scored_region_id != BG]
    rows = []
    for judge, g in d.groupby("judge"):
        per = (g.assign(moved=(g.delta != 0).astype(int))
                 .groupby("variant_id")
                 .agg(n=("moved", "size"), moved=("moved", "sum")))
        per = per[per.n > 1]
        if per.empty:
            continue
        pattern = np.where(per.moved == 0, "none",
                           np.where(per.moved == per.n, "all", "mixed"))
        p = float(per.moved.sum() / per.n.sum())     # overall per-region rate
        n = float(per.n.mode().iloc[0])              # typical regions/variant
        expect_mixed = 1.0 - p ** n - (1.0 - p) ** n
        rows.append(dict(
            judge=judge, n_variants=len(per),
            regions_per_variant=n,
            per_region_move_rate=p,
            frac_none=float((pattern == "none").mean()),
            frac_all=float((pattern == "all").mean()),
            frac_mixed=float((pattern == "mixed").mean()),
            mixed_if_independent=expect_mixed,
        ))
    return pd.DataFrame(rows)


def drift_robustness(d: pd.DataFrame, drift_csv: str, thresh: float) -> pd.DataFrame:
    """Headline numbers on all bases vs bases whose layout survived the edit.

    Stage 0's masks are computed on source.png and applied to edit.png. Where
    FLUX re-composed the scene, the mask no longer covers the object the judge
    is told about, so we corrupt background while claiming to have damaged a
    region - and that looks exactly like a judge that cannot localise. The
    confound points at our own conclusion, so it cannot be waved away.

    It also does not have to be measured precisely. If the headline numbers
    agree across the split, the result does not depend on the bases where the
    geometry is doubtful, and how good the proxy is stops mattering. Only if
    they diverge is a per-region check (an independent detector, never our own
    judge) worth its cost.

    Pooled, not averaged over the per-condition cells of sensitivity(): a mean
    of means would weight a 3-row cell like a 300-row one.
    """
    drift = pd.read_csv(drift_csv, dtype={"base_id": str})
    keep = set(drift.loc[drift.edge_iou >= thresh, "base_id"])
    fg = d[d.scored_region_id != BG]

    rows = []
    for name, g in (("all", fg), (f"edge_iou>={thresh}",
                                  fg[fg.base_id.isin(keep)])):
        tgt = g[g.is_target]
        coh = response_coherence(g)
        auroc = (roc_auc_score(g.is_target, -g.delta)
                 if g.is_target.nunique() > 1 else float("nan"))
        rows.append(dict(
            subset=name, n_bases=g.base_id.nunique(), n_rows=len(g),
            target_unchanged=float((tgt.delta == 0).mean()) if len(tgt) else float("nan"),
            auroc=auroc,
            frac_mixed=float(coh.frac_mixed.iloc[0]) if len(coh) else float("nan"),
            mixed_if_independent=float(coh.mixed_if_independent.iloc[0]) if len(coh) else float("nan"),
        ))
    return pd.DataFrame(rows)


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

    Severity rows 1 and 3 pool every GRADED corruption. FLAT_SEVERITY ones get
    their own `binary` row instead, so reading down the 1 -> 3 column really is
    reading an effect-size ladder.
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
    d = collapse_flat_severity(d)
    return (d.groupby(["judge", "corruption", "severity"])
            [["sc_success", "sc_preserve", "phi", "reward"]]
            .mean().round(3).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="out/analysis")
    ap.add_argument("--col", default="reward", choices=READOUTS,
                    help="headline readout (default: reward, Equation 3)")
    ap.add_argument("--min-control", type=float, default=None,
                    help="drop regions whose CLEAN control scored at or below "
                         "this. A region already at the floor cannot drop when "
                         "damaged, so it forces AUROC toward 0.5 regardless of "
                         "how well the judge localises. Try 0 for reward, or 0 "
                         "for phi/sc_* on the 0-25 scale.")
    ap.add_argument("--drift-csv", default=None,
                    help="out/edit_drift.csv from scripts/verify_edit_drift.py. "
                         "With it, every headline number is reported twice: "
                         "all bases, and only those whose layout survived the "
                         "edit. See --min-edge-iou.")
    ap.add_argument("--min-edge-iou", type=float, default=0.4,
                    help="edge-IoU cut for the --drift-csv robustness split "
                         "(default 0.4)")
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
    measurement_quality(head, a.col, nf)
    print(f"\n=== noise floor, {a.col} (SD across repeats, clean controls) ===")
    print(nf.to_string(index=False) if len(nf) else "  no controls found")
    nf.to_csv(out / "noise_floor.csv", index=False)

    print("\n=== targeted region: which axis moves? ===")
    ax = axis_table(df)
    print(ax.to_string(index=False) if len(ax) else "  nothing targeted")
    ax.to_csv(out / "axis_by_severity.csv", index=False)

    d = delta_table(head, a.col)
    if a.min_control is not None:
        d = drop_floored(d, a.min_control)
    d.to_parquet(out / "deltas.parquet")

    print("\n=== does the score move at all? (delta == 0 exactly) ===")
    sens = sensitivity(d)
    if len(sens):
        print(sens.round(3).to_string(index=False))
        sens.to_csv(out / "sensitivity.csv", index=False)
        worst = sens.target_unchanged.max()
        if worst == worst and worst > 0.5:
            print(f"\n  Up to {worst:.0%} of DAMAGED regions get a score "
                  f"identical to their clean")
            print("  control. Where the score never moves, no ranking metric "
                  "below can say")
            print("  anything: AUROC 0.5 there means 'did not react', not "
                  "'reacted in the")
            print("  wrong place'. Report the tie rate alongside every AUROC.")
    else:
        print("  n/a")

    print("\n=== do a variant's regions move together? ===")
    coh = response_coherence(d)
    if len(coh):
        print(coh.round(3).to_string(index=False))
        coh.to_csv(out / "coherence.csv", index=False)
        for r in coh.itertuples():
            if r.frac_mixed < 0.6 * r.mixed_if_independent:
                print(f"\n  {r.judge}: only {r.frac_mixed:.0%} of variants show "
                      f"SOME regions moving")
                print(f"  while others hold, against {r.mixed_if_independent:.0%} "
                      f"expected if regions moved")
                print("  independently. The judge is revising its whole answer "
                      "for an image, not")
                print("  the score of the region we damaged.")
    else:
        print("  n/a")

    print("\n=== localization AUROC by severity ===")
    print(f"  ({'/'.join(sorted(FLAT_SEVERITY))} has no severity ladder - its "
          f"1 and 3 differ by 0.5 of 35 8-bit")
    print(f"   levels - so it reports as one '{BINARY}' row rather than "
          f"flattening 1 and 3.)")
    for c in cols:
        dc = d if c == a.col else delta_table(usable(df, c), c)
        if a.min_control is not None and c != a.col:
            dc = drop_floored(dc, a.min_control)
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

    if a.drift_csv:
        print(f"\n=== robustness: bases whose layout survived the edit "
              f"(edge IoU >= {a.min_edge_iou}) ===")
        rb = drift_robustness(d, a.drift_csv, a.min_edge_iou)
        print(rb.round(4).to_string(index=False))
        rb.to_csv(out / "drift_robustness.csv", index=False)
        print("  Agreement across the two rows means the conclusion does not "
              "rest on bases")
        print("  where a source-coordinate mask may no longer cover its "
              "object. Divergence")
        print("  means the masks need a per-region check against an "
              "independent detector -")
        print("  never against the judge under audit.")

    # The headline sanity check: is the target-region effect even above noise?
    floor = nf.set_index("judge")["median"].to_dict() if len(nf) else {}
    print(f"\n=== effect vs noise floor ({a.col}) ===")
    for judge, g in d[d.is_target].groupby("judge"):
        eff = abs(g.delta.mean())
        f = floor.get(judge, float("nan"))
        if f != f:
            verdict = ("floor undefined (n_samples=1) - judge this by the tie "
                       "rate and the between-variant SD above")
        elif f == 0:
            # Greedy decoding: repeated samples are identical, so a zero SD says
            # nothing about whether the effect is real. Declaring "ABOVE noise"
            # here would be an artefact of the sampling config, not a result.
            verdict = ("floor is 0 (greedy) - within-variant SD cannot judge "
                       "this; compare against the between-variant SD above")
        elif eff > f:
            verdict = "ABOVE noise"
        else:
            verdict = "BELOW NOISE - not a signal"
        print(f"{judge}: |mean delta|={eff:.4f} vs floor={f:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
