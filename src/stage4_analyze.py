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


# Resampling sizes and seed, fixed so committed CSVs reproduce byte for byte.
N_PERM = 200
N_BOOT = 1000
SEED = 0

# The cells a permutation null shuffles within: one photo under one corruption
# condition. Its variants differ only in which region was damaged, and each
# scores the same regions, so a region's move/no-move flag can be exchanged
# between them without changing anything but which variant it sits in.
COHERENCE_CELL = ["base_id", "corruption", "severity", "area_bin"]


def _photo_boot_idx(n_photos: int, n_boot: int = N_BOOT, seed: int = SEED):
    """(n_boot, n_photos) photo indices drawn with replacement.

    Every interval in this module resamples PHOTOGRAPHS, not rows: a photo's
    regions and variants share one edit, one instruction and one scene, so a
    row-level interval would claim a precision the design does not have.
    """
    return np.random.default_rng(seed).integers(0, n_photos, (n_boot, n_photos))


def _ci(x) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def _mixed(vcode: np.ndarray, moved: np.ndarray, n_var: int) -> np.ndarray:
    """Per variant: did some, but not all, of its regions move?"""
    s = np.bincount(vcode, weights=moved, minlength=n_var)
    n = np.bincount(vcode, minlength=n_var)
    return (s > 0) & (s < n)


def response_coherence(d: pd.DataFrame, n_perm: int = N_PERM,
                       n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Within ONE variant, do the regions move together or independently?

    The sensitivity table showed target_unchanged and other_unchanged equal to
    three decimals in three of four cells. Similar rates would be unremarkable;
    IDENTICAL rates suggest the judge's answer changes as a whole-image event -
    either every region of a variant differs from its control, or none does.

    Classify each variant as none / all / mixed by how many of its regions
    moved, and compare the mixed share against what independent per-region
    movement would produce.

    THE BASELINE IS A PERMUTATION NULL (`mixed_null`), not a formula. Within
    each photo x condition cell, each scored region's moved flag is shuffled
    across that cell's variants. That keeps every region's own move rate in
    that photo and condition and breaks only any tendency of a variant's
    regions to move together. Observed mixed below the null = coupling.

    `mixed_if_independent` is the earlier closed form, 1 - p^n - (1-p)^n with
    ONE pooled rate p. Kept only so older numbers can be traced: move rates
    differ widely across photos and corruptions, and mixing different rates
    always lowers the true mixed share below what one pooled rate predicts, so
    it reads heterogeneity as coupling. On `main` (phi, floor-excluded) it
    gave 51% against a permutation null of 34%. Do not quote it.

    `gap` = frac_mixed - mixed_null, with a 95% bootstrap over photographs
    (the null is averaged over permutations per variant first, so the interval
    carries the photo-to-photo uncertainty of both sides).
    """
    d = d[d.scored_region_id != BG]
    rows = []
    for judge, g in d.groupby("judge"):
        g = g.assign(moved=(g.delta != 0).astype(int))
        g = g[g.groupby("variant_id").moved.transform("size") > 1]
        if g.empty:
            continue
        per = (g.groupby("variant_id")
                .agg(n=("moved", "size"), moved=("moved", "sum"),
                     base_id=("base_id", "first")))
        pattern = np.where(per.moved == 0, "none",
                           np.where(per.moved == per.n, "all", "mixed"))
        p = float(per.moved.sum() / per.n.sum())     # overall per-region rate
        n = float(per.n.mode().iloc[0])              # typical regions/variant
        expect_mixed = 1.0 - p ** n - (1.0 - p) ** n

        # Permutation null. Sorting by (cell+region, random key) gives a random
        # permutation inside every group at once; writing it back over the
        # group-sorted slots moves each flag to another variant of its group.
        vcode = pd.Categorical(g.variant_id, categories=per.index).codes
        gcode = g.groupby(COHERENCE_CELL + ["scored_region_id"]).ngroup().to_numpy()
        moved = g.moved.to_numpy()
        slots = np.argsort(gcode, kind="stable")
        rng = np.random.default_rng(seed)
        null_per_var = np.zeros(len(per))
        null_frac = np.empty(n_perm)
        for i in range(n_perm):
            order = np.lexsort((rng.random(len(moved)), gcode))
            pm = np.empty_like(moved)
            pm[slots] = moved[order]
            mx = _mixed(vcode, pm, len(per))
            null_per_var += mx
            null_frac[i] = mx.mean()
        null_per_var /= n_perm

        obs = (pattern == "mixed").astype(float)
        pcode = pd.Categorical(per.base_id).codes
        k = int(pcode.max()) + 1
        ph_n = np.bincount(pcode, minlength=k).astype(float)
        ph_obs = np.bincount(pcode, weights=obs, minlength=k)
        ph_null = np.bincount(pcode, weights=null_per_var, minlength=k)
        idx = _photo_boot_idx(k, n_boot, seed)
        bn = ph_n[idx].sum(1)
        b_obs = ph_obs[idx].sum(1) / bn
        b_gap = b_obs - ph_null[idx].sum(1) / bn
        m_lo, m_hi = _ci(b_obs)
        nl_lo, nl_hi = _ci(null_frac)
        g_lo, g_hi = _ci(b_gap)
        rows.append(dict(
            judge=judge, n_variants=len(per), n_photos=k,
            regions_per_variant=n,
            per_region_move_rate=p,
            frac_none=float((pattern == "none").mean()),
            frac_all=float((pattern == "all").mean()),
            frac_mixed=float(obs.mean()),
            frac_mixed_lo=m_lo, frac_mixed_hi=m_hi,
            mixed_null=float(null_per_var.mean()),
            mixed_null_lo=nl_lo, mixed_null_hi=nl_hi,
            gap=float(obs.mean() - null_per_var.mean()),
            gap_lo=g_lo, gap_hi=g_hi,
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
    of means would weight a 3-row cell like a 300-row one. One row per judge
    and subset: an earlier version pooled every judge in the file into one row.
    """
    drift = pd.read_csv(drift_csv, dtype={"base_id": str})
    keep = set(drift.loc[drift.edge_iou >= thresh, "base_id"])
    fg = d[d.scored_region_id != BG]
    rows = []
    for name, sub in (("all", fg), (f"edge_iou>={thresh}",
                                    fg[fg.base_id.isin(keep)])):
        rows += _subset_rows(name, sub)
    return pd.DataFrame(rows)


def _by_judge(t: pd.DataFrame) -> pd.DataFrame:
    return t.set_index("judge") if "judge" in t.columns else pd.DataFrame()


def _subset_rows(name: str, sub: pd.DataFrame) -> list:
    """The pooled headline numbers for one subset of the delta table, per judge.
    Shared by the drift and overlap robustness splits so they read alike."""
    coh = _by_judge(response_coherence(sub)) if len(sub) else pd.DataFrame()
    red = _by_judge(redundancy_delta(sub)) if len(sub) else pd.DataFrame()
    rows = []
    for judge, g in sub.groupby("judge"):
        tgt, oth = g[g.is_target], g[~g.is_target]
        c = coh.loc[judge] if judge in coh.index else None
        r = red.loc[judge] if judge in red.index else None
        rows.append(dict(
            subset=name, judge=judge, n_bases=g.base_id.nunique(), n_rows=len(g),
            target_unchanged=float((tgt.delta == 0).mean()) if len(tgt) else float("nan"),
            other_unchanged=float((oth.delta == 0).mean()) if len(oth) else float("nan"),
            auroc=(roc_auc_score(g.is_target, -g.delta)
                   if g.is_target.nunique() > 1 else float("nan")),
            frac_mixed=c.frac_mixed if c is not None else float("nan"),
            mixed_null=c.mixed_null if c is not None else float("nan"),
            gap=c.gap if c is not None else float("nan"),
            gap_lo=c.gap_lo if c is not None else float("nan"),
            gap_hi=c.gap_hi if c is not None else float("nan"),
            r2_delta=r.r2 if r is not None else float("nan"),
            r2_delta_lo=r.r2_lo if r is not None else float("nan"),
            r2_delta_hi=r.r2_hi if r is not None else float("nan"),
            mixed_if_independent=(c.mixed_if_independent if c is not None
                                  else float("nan")),
        ))
    # An empty subset is a result (e.g. no photo is free of overlap), so it
    # still gets a row rather than vanishing from the table.
    return rows or [dict(subset=name, n_bases=0, n_rows=0)]


# --- region overlap (#1): does damage to one region land in another's box? --

OVERLAP_BINS = [-np.inf, 0.0, 0.01, 0.1, 0.5, np.inf]
OVERLAP_LABELS = ["0", "(0,0.01]", "(0.01,0.1]", "(0.1,0.5]", ">0.5"]


def attach_overlap(d: pd.DataFrame, overlap_csv: str) -> pd.DataFrame:
    """Join scripts/region_overlap.py's measurement onto the delta table.

    `other_box_damaged` is the share of a NON-target region's box whose pixels
    the corruption visibly changed. The judge is told about regions as boxes,
    so where it is above zero a moving non-target score is not clearly a leak:
    some of what the judge was told is that region really was damaged. Target
    rows get 0 (not applicable). Non-target rows with no measurement get NaN
    and are kept out of every clean subset.
    """
    ov = pd.read_csv(overlap_csv, dtype={"variant_id": str, "other_region_id": str})
    ov = ov.rename(columns={"other_region_id": "scored_region_id"})
    out = d.merge(ov[["variant_id", "scored_region_id", "other_box_damaged"]],
                  on=["variant_id", "scored_region_id"], how="left")
    out.loc[out.is_target, "other_box_damaged"] = 0.0
    miss = int((~out.is_target & (out.scored_region_id != BG)
                & out.other_box_damaged.isna()).sum())
    if miss:
        print(f"  WARNING: {miss} non-target rows have no overlap measurement; "
              f"they are excluded from the clean subsets")
    return out


def overlap_split(d: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Headline numbers on all pairs vs pairs where the damage stayed out of the
    other region's box.

    Two cuts per threshold t:
      pairs<=t   drop each (variant, non-target region) whose box took more
                 than t of visible damage. Keeps every photo.
      photos<=t  keep only photos where NO pair exceeds t. Stricter: removes
                 any coupling that overlap could induce within a variant.
    """
    fg = d[d.scored_region_id != BG]
    bad = fg[~fg.is_target]
    rows = _subset_rows("all", fg)
    for t in thresholds:
        clean_pair = fg.is_target | (fg.other_box_damaged <= t)
        rows += _subset_rows(f"pairs<={t}", fg[clean_pair])
        dirty = set(bad.loc[~(bad.other_box_damaged <= t), "base_id"])
        rows += _subset_rows(f"photos<={t}", fg[~fg.base_id.isin(dirty)])
    return pd.DataFrame(rows)


def leakage_by_overlap(d: pd.DataFrame) -> pd.DataFrame:
    """Leakage by spatial relation, replacing the slot-indexed matrix.

    Non-target rows binned by how much of their box the damage reached, against
    the target's own response. A judge that localises shows movement in the
    damaged bins and none in the `0` bin; a global judge moves every bin alike.
    """
    fg = d[d.scored_region_id != BG]
    rows = []
    for judge, g in fg.groupby("judge"):
        t = g[g.is_target]
        rows.append(dict(judge=judge, relation="target", n=len(t),
                         n_bases=t.base_id.nunique(),
                         mean_delta=t.delta.mean(),
                         frac_moved=float((t.delta != 0).mean())))
        o = g[~g.is_target].dropna(subset=["other_box_damaged"])
        o = o.assign(bin=pd.cut(o.other_box_damaged, OVERLAP_BINS,
                                labels=OVERLAP_LABELS))
        for b, ob in o.groupby("bin", observed=False):
            rows.append(dict(judge=judge, relation=f"other, box damaged {b}",
                             n=len(ob), n_bases=ob.base_id.nunique(),
                             mean_delta=ob.delta.mean() if len(ob) else float("nan"),
                             frac_moved=(float((ob.delta != 0).mean())
                                         if len(ob) else float("nan"))))
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


def _r2_ci(x, y, photo, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Squared Pearson r of x on y, with a 95% bootstrap over photographs.

    Computed from per-photo sums, so a bootstrap draw is a sum over resampled
    photos rather than a re-concatenation of rows.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    codes = pd.Categorical(photo).codes
    k = int(codes.max()) + 1

    def tot(w):
        return np.bincount(codes, weights=w, minlength=k)

    n = np.bincount(codes, minlength=k).astype(float)
    sx, sy, sxx, syy, sxy = tot(x), tot(y), tot(x * x), tot(y * y), tot(x * y)

    def r2(idx):
        N, X, Y = n[idx].sum(-1), sx[idx].sum(-1), sy[idx].sum(-1)
        cov = sxy[idx].sum(-1) - X * Y / N
        vx = sxx[idx].sum(-1) - X * X / N
        vy = syy[idx].sum(-1) - Y * Y / N
        with np.errstate(divide="ignore", invalid="ignore"):
            return cov * cov / (vx * vy)

    point = float(r2(np.arange(k)[None, :])[0])
    lo, hi = _ci(r2(_photo_boot_idx(k, n_boot, seed)))
    return dict(r2=point, r2_lo=lo, r2_hi=hi, n=len(x), n_photos=k)


def redundancy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """LEVEL redundancy: each region's score against the leave-one-out mean of
    the image's other regions, over every variant.

    Not the redundancy test to quote. Score LEVELS mostly track whether FLUX's
    edits on that photo worked, which is shared across a photo's regions for a
    perfectly spatial judge too, so a high value here does not show a copied
    global impression. Kept, now with a photo interval, for continuity with the
    numbers already reported; quote redundancy_delta() instead.

    Leave-one-out, not the plain image mean: with ~4 regions, self-inclusion
    contributes about a quarter of the predictor and manufactures correlation.
    Background is excluded from both sides.
    """
    df = df[df.scored_region_id != BG]
    rows = []
    for judge, g in df.groupby("judge"):
        a = (g.groupby(["variant_id", "scored_region_id"])
              .agg(**{col: (col, "mean"), "base_id": ("base_id", "first")})
              .reset_index())
        grp = a.groupby("variant_id")[col]
        a["n_reg"] = grp.transform("size")
        a["others"] = (grp.transform("sum") - a[col]) / (a["n_reg"] - 1)
        a = a[a.n_reg > 1].dropna(subset=[col, "others"])
        if len(a) > 2 and a.others.std() > 0 and a[col].std() > 0:
            rows.append(dict(judge=judge, version="level",
                             **_r2_ci(a[col], a.others, a.base_id)))
    return pd.DataFrame(rows)


def redundancy_delta(d: pd.DataFrame) -> pd.DataFrame:
    """DELTA redundancy on untouched regions: the redundancy test to quote.

    Takes the delta table (corrupted minus clean, per region) and keeps only
    regions that were NOT damaged, whose true change is zero by design. Each
    one's delta is regressed on the leave-one-out mean delta of the variant's
    OTHER untouched regions.

    A judge scoring each region on its own content gives untouched regions
    nothing but independent noise: R^2 near 0. A judge whose per-region numbers
    are one image-level impression moves them together when ANY region is
    damaged: R^2 high. Unlike the level version, a perfect judge cannot score
    high here. Needs >= 2 untouched scored regions per variant.
    """
    u = d[(d.scored_region_id != BG) & ~d.is_target]
    rows = []
    for judge, g in u.groupby("judge"):
        grp = g.groupby("variant_id").delta
        n_unt = grp.transform("size")
        g = g.assign(others=(grp.transform("sum") - g.delta) / (n_unt - 1))
        g = g[n_unt > 1].dropna(subset=["delta", "others"])
        if len(g) > 2 and g.others.std() > 0 and g.delta.std() > 0:
            rows.append(dict(judge=judge, version="delta_untouched",
                             **_r2_ci(g.delta, g.others, g.base_id)))
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
    ap.add_argument("--overlap-csv", default=None,
                    help="overlap.csv from scripts/region_overlap.py. With it, "
                         "the headline numbers are reported on all pairs and "
                         "on pairs/photos where the damage stayed out of the "
                         "other region's box (see --max-overlap), plus leakage "
                         "binned by that overlap.")
    ap.add_argument("--max-overlap", type=float, nargs="+", default=[0.0, 0.1],
                    help="other_box_damaged thresholds for the overlap split "
                         "(default 0 and 0.1)")
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
        print("  Baseline is mixed_null (permutation within photo x condition),")
        print("  NOT mixed_if_independent (one pooled rate; kept for tracing).")
        for r in coh.itertuples():
            if r.gap_hi < 0:
                print(f"\n  {r.judge}: {r.frac_mixed:.0%} of variants mixed "
                      f"against {r.mixed_null:.0%} if regions moved")
                print(f"  independently (gap {r.gap:+.3f} [{r.gap_lo:+.3f}, "
                      f"{r.gap_hi:+.3f}], photo bootstrap): regions co-move.")
    else:
        print("  n/a")

    print("\n=== localization AUROC by severity ===")
    print(f"  ({'/'.join(sorted(FLAT_SEVERITY))} has no severity ladder - its "
          f"1 and 3 differ by 0.5 of 35 8-bit")
    print(f"   levels - so it reports as one '{BINARY}' row rather than "
          f"flattening 1 and 3.)")
    deltas = {}
    for c in cols:
        dc = d if c == a.col else delta_table(usable(df, c), c)
        if a.min_control is not None and c != a.col:
            dc = drop_floored(dc, a.min_control)
        deltas[c] = dc
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

    print("\n=== redundancy on DELTAS of untouched regions (quote this) ===")
    for c in cols:
        red = redundancy_delta(deltas[c])
        if len(red):
            red.insert(0, "readout", c)
            print(red.round(4).to_string(index=False))
        else:
            print(f"  {c}: n/a")
        red.to_csv(out / f"redundancy_delta_{c}.csv", index=False)

    print("\n=== redundancy on score LEVELS (legacy; a perfect judge scores "
          "high too) ===")
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

    if a.overlap_csv:
        print(f"\n=== robustness: damage kept out of other regions' boxes "
              f"(thresholds {a.max_overlap}) ===")
        dov = attach_overlap(d, a.overlap_csv)
        os_ = overlap_split(dov, a.max_overlap)
        print(os_.round(4).to_string(index=False))
        os_.to_csv(out / "overlap_split.csv", index=False)
        for t in a.max_overlap:
            fg = dov[dov.scored_region_id != BG]
            clean = fg[fg.is_target | (fg.other_box_damaged <= t)]
            lo = localization(clean)
            lo.insert(0, "subset", f"pairs<={t}")
            lo.to_csv(out / f"localization_clean{t}_{a.col}.csv", index=False)
        lb = leakage_by_overlap(dov)
        print("\n--- leakage by spatial relation ---")
        print(lb.round(4).to_string(index=False))
        lb.to_csv(out / "leakage_by_overlap.csv", index=False)

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
