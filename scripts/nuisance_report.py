"""
Nuisance and exploitability, read off a sweep of presentation conditions. [CPU]

Here the image is fixed and the packaging changes, so scores are paired on
(judge, variant_id, scored_region_id) across presentations; stage 4, which
pairs corrupted variants with clean controls, does not apply. A nuisance change
carries no information about edit quality, so it is compared with what real
damage to a region does.

Usage:
    python -m scripts.nuisance_report --scores 'out/nuisance/scores_*.parquet'
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.presentation import (EXPLOIT_AXES, NUISANCE_AXES,      # noqa: E402
                              TEXT_AXES)
from src.stage4_analyze import (READOUTS, BG, delta_table,      # noqa: E402
                                drop_floored, load, noise_floor,
                                sensitivity, usable)

BASE = "baseline"


def paired(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per (judge, variant, region); one column per presentation
    (mean over samples)."""
    a = (df.groupby(["judge", "variant_id", "scored_region_id", "presentation"])
         [col].mean().reset_index())
    return a.pivot_table(index=["judge", "variant_id", "scored_region_id"],
                         columns="presentation", values=col)


def damage_reference(df: pd.DataFrame, col: str,
                     min_control: float = None) -> tuple:
    """What real damage does to the damaged region, from baseline-condition
    scores that include corrupted variants (the main run, via --damage-scores).

    `min_control` drops regions whose clean score is at or below it, as stage
    4's --min-control does.
    """
    b = df[df.presentation == BASE]
    if b.empty:
        return float("nan"), float("nan"), pd.DataFrame(), pd.DataFrame()
    d = delta_table(usable(b, col), col)
    if min_control is not None:
        d = drop_floored(d, min_control)
    if d.empty:
        return float("nan"), float("nan"), pd.DataFrame(), pd.DataFrame()
    tgt = d[(d.is_target) & (d.scored_region_id != BG)]
    return (float(tgt.delta.abs().mean()) if len(tgt) else float("nan"),
            float((tgt.delta == 0).mean()) if len(tgt) else float("nan"),
            sensitivity(d),
            pd.DataFrame({"abs": tgt.delta.abs().values,
                          "base": tgt.base_id.values}))


def rerun_floor(df: pd.DataFrame, col: str) -> tuple:
    """Noise floor from byte-identical requests.

    A photo's clean controls under `baseline` are the same request, yet greedy
    vLLM sometimes scores them differently. Returns (mean |delta| between
    them, share unchanged, number of pairs).
    """
    b = df[(df.presentation == BASE) & df.is_control]
    d = []
    for _, g in b.groupby(["judge", "base_id", "scored_region_id"]):
        v = g[col].to_numpy()
        d += [abs(x - y) for i, x in enumerate(v) for y in v[i + 1:]]
    if not d:
        return float("nan"), float("nan"), 0
    d = np.asarray(d)
    return float(d.mean()), float((d == 0).mean()), len(d)


def nuisance_table(w: pd.DataFrame, damage: float, floor: float,
                   base_of: pd.Series = None,
                   min_control: float = None,
                   dmg: pd.DataFrame = None) -> pd.DataFrame:
    """|delta| for each nuisance change, against what real damage moved.

    `base_of` (variant_id -> base_id) adds the per-photograph mean and its
    bootstrap CI. `min_control` keeps foreground regions whose baseline score is
    above it. `dmg` (per damaged region: |delta| and base) adds `ratio`, the
    ratio of per-photograph means, with a CI from `ratio_ci`."""
    if min_control is not None:
        w = w[(w.index.get_level_values("scored_region_id") != BG)
              & (w[BASE] > min_control)]
    rows = []
    for mode in [c for c in w.columns if c in NUISANCE_AXES]:
        p = w[[BASE, mode]].dropna()
        if p.empty:
            continue
        d = p[mode] - p[BASE]
        ci = {}
        if base_of is not None:
            base = pd.Series(p.index.get_level_values("variant_id"),
                             index=d.index).map(base_of)
            lo, hi = boot_ci(d.abs(), base)
            ci = dict(photo_mean_abs=cluster_mean(d.abs(), base),
                      photo_lo=lo, photo_hi=hi, n_bases=int(base.nunique()))
            if dmg is not None and len(dmg):
                r, rlo, rhi = ratio_ci(d.abs(), base, dmg["abs"], dmg["base"])
                ci.update(ratio=r, ratio_lo=rlo, ratio_hi=rhi)
        rows.append(dict(
            presentation=mode,
            kind="text" if mode in TEXT_AXES else "image",
            n=len(d),
            unchanged=float((d == 0).mean()),
            mean_abs_delta=float(d.abs().mean()),
            mean_delta=float(d.mean()),
            vs_damage=float(d.abs().mean() / damage) if damage else float("nan"),
            vs_floor=float(d.abs().mean() / floor) if floor else float("nan"),
            **ci,
        ))
    return pd.DataFrame(rows)


def slot_effect(df: pd.DataFrame, col: str, mode: str = "shuffle") -> pd.DataFrame:
    """Under `shuffle`, does the score depend on the region's list position?

    The raw per-slot mean is confounded (slots 3-4 exist only on photos with 4-5
    regions); `within` subtracts each region's own mean over its slots, leaving
    the position effect. Its CI resamples photographs.
    """
    d = df[(df.presentation == mode) & (df.scored_region_id != BG)]
    d = d[d.slot_idx >= 0].copy()
    if d.empty:
        return pd.DataFrame()
    d["within"] = d[col] - d.groupby(["judge", "base_id", "scored_region_id"]
                                     )[col].transform("mean")
    rows = []
    for (judge, slot), g in d.groupby(["judge", "slot_idx"]):
        lo, hi = boot_ci(g.within, g.base_id)
        rows.append(dict(judge=judge, slot_idx=slot, mean=g[col].mean(),
                         std=g[col].std(), size=len(g),
                         within=cluster_mean(g.within, g.base_id),
                         within_lo=lo, within_hi=hi))
    return pd.DataFrame(rows).round(3)


def identity_null(df: pd.DataFrame, col: str, mode: str = "shuffle",
                  min_control: float = None) -> pd.DataFrame:
    """Shuffles that kept the original order, as a null inside the run.

    Their requests are identical to `baseline`, so their |delta| is the
    run-to-run noise in the same pairing the nuisance deltas use. They are
    mostly 3-region photos, so this is a check, not a matched control.
    Background excluded.
    """
    k = ["judge", "variant_id", "scored_region_id"]
    fg = df[df.scored_region_id != BG]
    s = fg[fg.presentation == mode][k + ["base_id", "slot_idx", col]]
    b = fg[fg.presentation == BASE][k + ["slot_idx", col]]
    m = s.merge(b, on=k, suffixes=("", "_b")).dropna(subset=[col, col + "_b"])
    if min_control is not None:
        m = m[m[col + "_b"] > min_control]
    if m.empty:
        return pd.DataFrame()
    same = (m.slot_idx == m.slot_idx_b).groupby([m.judge, m.variant_id]).all()
    m["identity"] = [same[(j, v)] for j, v in zip(m.judge, m.variant_id)]
    m["absd"] = (m[col] - m[col + "_b"]).abs()
    rows = []
    for (judge, ident), g in m.groupby(["judge", "identity"]):
        lo, hi = boot_ci(g.absd, g.base_id)
        rows.append(dict(judge=judge, order="identity" if ident else "reordered",
                         n_variants=g.variant_id.nunique(), n=len(g),
                         n_bases=g.base_id.nunique(),
                         mean_abs_delta=g.absd.mean(),
                         photo_mean_abs=cluster_mean(g.absd, g.base_id),
                         photo_lo=lo, photo_hi=hi,
                         unchanged=float((g.absd == 0).mean())))
    return pd.DataFrame(rows)


def cluster_mean(gain: pd.Series, base_id: pd.Series) -> float:
    """Mean of per-photograph means: the quantity `boot_ci` brackets."""
    g = pd.DataFrame({"gain": gain.values, "base": base_id.values}).dropna()
    if g.empty:
        return float("nan")
    return float(g.groupby("base").gain.mean().mean())


def boot_ci(gain: pd.Series, base_id: pd.Series, n_boot: int = 2000,
            seed: int = 0) -> tuple:
    """95% CI for `cluster_mean`, resampling photographs (regions within a
    photo are not independent). Fixed seed; not corrected for multiple
    comparisons."""
    g = pd.DataFrame({"gain": gain.values, "base": base_id.values}).dropna()
    if g.empty:
        return float("nan"), float("nan")
    per_base = g.groupby("base").gain.mean()
    if len(per_base) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_base), size=(n_boot, len(per_base)))
    means = per_base.values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ratio_ci(num: pd.Series, num_base: pd.Series, den: pd.Series,
             den_base: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Ratio of two per-photograph means, with a 95% CI that resamples the
    photographs both share, so both sides' uncertainty is included."""
    a = pd.DataFrame({"v": num.values, "b": num_base.values}).dropna()
    c = pd.DataFrame({"v": den.values, "b": den_base.values}).dropna()
    a, c = a.groupby("b").v.mean(), c.groupby("b").v.mean()
    shared = a.index.intersection(c.index)
    if len(shared) < 2:
        return float("nan"), float("nan"), float("nan")
    a, c = a[shared].values, c[shared].values
    idx = np.random.default_rng(seed).integers(0, len(shared),
                                               size=(n_boot, len(shared)))
    r = a[idx].mean(axis=1) / c[idx].mean(axis=1)
    return (float(a.mean() / c.mean()), float(np.percentile(r, 2.5)),
            float(np.percentile(r, 97.5)))


def reversion(df: pd.DataFrame, w_by_col: dict) -> pd.DataFrame:
    """Does the condition pull every score toward one value?

    Selecting failed edits on the judge's own baseline and measuring change from
    that baseline favours gains on low scores even without an exploit. `slope`
    regresses the gain on the baseline score (-1: the condition replaces the
    score with a constant). `resid_failed` / `resid_ok` repeat the split after
    removing that linear term.
    """
    succ = paired(usable(df, "sc_success"), "sc_success")
    if BASE not in succ.columns:
        return pd.DataFrame()
    base_success = succ[BASE].rename("base_success")
    v = df.drop_duplicates("variant_id").set_index("variant_id")
    rows = []
    for mode in EXPLOIT_AXES:
        for col, w in w_by_col.items():
            if mode not in w.columns:
                continue
            p = w[[BASE, mode]].dropna().join(base_success).dropna().reset_index()
            p = p[(p.scored_region_id != BG)
                  & p.variant_id.map(v.is_control).astype(bool)]
            if len(p) < 3:
                continue
            gain = (p[mode] - p[BASE]).values
            x = p[BASE].values
            slope, intercept = np.polyfit(x, gain, 1)
            resid = gain - (slope * x + intercept)
            failed = (p.base_success < 20).values
            rows.append(dict(
                presentation=mode, readout=col, n=len(p),
                slope=float(slope), corr=float(np.corrcoef(x, gain)[0, 1]),
                sd_baseline=float(p[BASE].std()), sd_condition=float(p[mode].std()),
                raw_failed=float(gain[failed].mean()) if failed.any() else float("nan"),
                raw_ok=float(gain[~failed].mean()) if (~failed).any() else float("nan"),
                resid_failed=float(resid[failed].mean()) if failed.any() else float("nan"),
                resid_ok=float(resid[~failed].mean()) if (~failed).any() else float("nan"),
            ))
    return pd.DataFrame(rows)


def local_contrast(df: pd.DataFrame, w_by_col: dict) -> pd.DataFrame:
    """Within each variant, does the target region gain more than the others?

    `enhance` lifts the whole frame, so it cannot favour one region; its
    contrast is the baseline that `enhance_target` must exceed.
    """
    v = df.drop_duplicates("variant_id").set_index("variant_id")
    rows = []
    for mode in EXPLOIT_AXES:
        for col, w in w_by_col.items():
            if mode not in w.columns:
                continue
            p = w[[BASE, mode]].dropna().reset_index()
            p = p[p.scored_region_id != BG]
            if p.empty:
                continue
            tgt = p.variant_id.map(v.target_region_id).astype(str)
            p = p.assign(gain=p[mode] - p[BASE],
                         base_id=p.variant_id.map(v.base_id),
                         where=np.where(p.scored_region_id.astype(str) == tgt,
                                        "target", "other"))
            piv = (p.pivot_table(index=["variant_id", "base_id"], columns="where",
                                 values="gain").dropna().reset_index())
            if len(piv) < 3 or "target" not in piv or "other" not in piv:
                continue
            diff = piv["target"] - piv["other"]
            lo, hi = boot_ci(diff, piv.base_id)
            rows.append(dict(presentation=mode, readout=col,
                             n_variants=len(piv), n_bases=int(piv.base_id.nunique()),
                             target_minus_other=float(diff.mean()),
                             ci_lo=lo, ci_hi=hi,
                             uniform_control=mode == "enhance"))
    return pd.DataFrame(rows)


def pq_by_presentation(df: pd.DataFrame) -> pd.DataFrame:
    """Change in the two image-level quality scores under each presentation.

    AES = min(PQ) multiplies every region's reward, so this is how a cosmetic
    change would raise it.
    """
    d = df[df.parsed].copy()
    pq = (d.groupby(["judge", "variant_id", "presentation"])
          [["pq_naturalness", "pq_artifacts"]].first().reset_index())
    rows = []
    for col in ("pq_naturalness", "pq_artifacts"):
        w = pq.pivot_table(index=["judge", "variant_id"],
                           columns="presentation", values=col)
        if BASE not in w.columns:
            continue
        for mode in [c for c in w.columns if c != BASE]:
            p = w[[BASE, mode]].dropna()
            if p.empty:
                continue
            g = p[mode] - p[BASE]
            rows.append(dict(readout=col, presentation=mode, n=len(p),
                             baseline_mean=float(p[BASE].mean()),
                             condition_mean=float(p[mode].mean()),
                             mean_gain=float(g.mean()),
                             frac_rose=float((g > 0).mean()),
                             frac_fell=float((g < 0).mean())))
    return pd.DataFrame(rows)


def exploit_table(df: pd.DataFrame, w_by_col: dict) -> pd.DataFrame:
    """Mean gain over baseline under each exploit condition, for `reward` and
    `phi` (which can move in opposite directions)."""
    v = df.drop_duplicates("variant_id").set_index("variant_id")
    rows = []
    for mode in EXPLOIT_AXES:
        for col, w in w_by_col.items():
            if mode not in w.columns:
                continue
            p = w[[BASE, mode]].dropna().reset_index()
            if p.empty:
                continue
            gain = p[mode] - p[BASE]
            b = p.variant_id.map(v.base_id)
            lo, hi = boot_ci(gain, b)
            rows.append(dict(presentation=mode, readout=col, n=len(p),
                             n_bases=int(b.nunique()),
                             baseline_mean=float(p[BASE].mean()),
                             condition_mean=float(p[mode].mean()),
                             mean_gain=cluster_mean(gain, b),
                             ci_lo=lo, ci_hi=hi,
                             frac_rose=float((gain > 0).mean())))
    return pd.DataFrame(rows)


def exploit_split(df: pd.DataFrame, w_by_col: dict,
                  failed_below: float) -> pd.DataFrame:
    """Exploit gain on clean edits, split by target vs other region and by
    whether the judge called the edit failed (baseline sc_success below
    `failed_below`)."""
    succ = paired(usable(df, "sc_success"), "sc_success")
    if BASE not in succ.columns:
        return pd.DataFrame()
    base_success = succ[BASE].rename("base_success")
    v = df.drop_duplicates("variant_id").set_index("variant_id")
    rows = []
    for mode in EXPLOIT_AXES:
        for col, w in w_by_col.items():
            if mode not in w.columns:
                continue
            p = w[[BASE, mode]].dropna().join(base_success).dropna().reset_index()
            p = p[(p.scored_region_id != BG)
                  & p.variant_id.map(v.is_control).astype(bool)]
            if p.empty:
                continue
            tgt = p.variant_id.map(v.target_region_id).astype(str)
            p["where"] = np.where(p.scored_region_id.astype(str) == tgt,
                                  "target", "other")
            p["edit"] = np.where(p.base_success < failed_below, "failed", "ok")
            p["gain"] = p[mode] - p[BASE]
            p["base_id"] = p.variant_id.map(v.base_id)
            for (where, edit), g in p.groupby(["where", "edit"]):
                lo, hi = boot_ci(g.gain, g.base_id)
                rows.append(dict(presentation=mode, readout=col, where=where,
                                 edit=edit, n=len(g),
                                 n_bases=int(g.base_id.nunique()),
                                 mean_gain=cluster_mean(g.gain, g.base_id),
                                 ci_lo=lo, ci_hi=hi,
                                 frac_rose=float((g.gain > 0).mean())))
    return pd.DataFrame(rows)


def exploit_figure(split: pd.DataFrame, path: Path, col: str = "reward") -> bool:
    """Bar chart of the target-region gain per presentation, failed vs ok
    edits, with bootstrap CIs."""
    import matplotlib
    matplotlib.use("Agg")                      # no display on any of our boxes
    import matplotlib.pyplot as plt

    # split["where"]: `.where` is a DataFrame method.
    d = split[(split.readout == col) & (split["where"] == "target")]
    if d.empty:
        return False
    modes = list(dict.fromkeys(d.presentation))
    fig, ax = plt.subplots(figsize=(7, 4))
    width, colours = 0.38, {"failed": "#c44e52", "ok": "#4c72b0"}
    for k, edit in enumerate(("failed", "ok")):
        sub = d[d.edit == edit].set_index("presentation").reindex(modes)
        x = np.arange(len(modes)) + (k - 0.5) * width
        err = np.vstack([(sub.mean_gain - sub.ci_lo).values,
                         (sub.ci_hi - sub.mean_gain).values])
        ax.bar(x, sub.mean_gain.values, width, label=f"{edit} edits",
               color=colours[edit], yerr=np.abs(err), capsize=4, ecolor="0.3")
    ax.axhline(0, color="0.2", lw=1)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel(f"mean gain in {col} over baseline")
    ax.set_title("Can the score be raised without improving the edit?\n"
                 "target region, 95% CI over photographs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Parse rate and share of scored regions, per condition."""
    fg = df[df.scored_region_id != BG]
    return (fg.groupby("presentation")
            .agg(rows=("parsed", "size"), parse_rate=("parsed", "mean"),
                 scored=("sc_success", lambda s: float(s.notna().mean())),
                 reward=("reward", lambda s: float(s.notna().mean())))
            .round(3).reset_index())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/nuisance/scores_*.parquet",
                    help="the greedy presentation conditions, one file each")
    # A separate glob: a sampled baseline run is indistinguishable from the
    # greedy baseline by its columns.
    ap.add_argument("--sampled", default="out/nuisance/floor_*.parquet",
                    help="the n>1 baseline run that supplies the noise floor. "
                         "Without it the floor comes from identical greedy "
                         "baseline requests instead.")
    ap.add_argument("--damage-scores", default=None,
                    help="greedy baseline parquets that hold corrupted variants, "
                         "for the damage reference. Needed when --scores is a "
                         "controls-only sweep (nuisance150); point it at the "
                         "main run, which judged the same controls.")
    ap.add_argument("--min-control", type=float, default=None,
                    help="drop regions whose clean score is at or below this, "
                         "in the damage reference and the nuisance table alike "
                         "(background excluded too). 0 matches the main run's "
                         "floor-excluded headline.")
    ap.add_argument("--out", default="out/analysis/nuisance")
    ap.add_argument("--col", default="reward", choices=READOUTS,
                    help="headline readout for the nuisance table "
                         "(default: reward, Equation 3)")
    ap.add_argument("--failed-below", type=float, default=20.0,
                    help="baseline sc_success (0-25) below which a region's edit "
                         "counts as failed in the exploit split. A knob, not a "
                         "measured cut; the table prints n and n_bases per cell")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    greedy = load(a.scores)
    for c in ("presentation", "slot_idx"):
        if c not in greedy.columns:
            raise SystemExit(
                f"no `{c}` column: these parquets predate --presentation. "
                f"Re-run stage 3 with --presentation, or point --scores at "
                f"out/nuisance/.")
    if greedy.sample_idx.max() > 0:
        print("WARNING: --scores contains multi-sample rows. The floor run "
              "belongs in --sampled;\n         averaged in here it inflates "
              "every condition's mean toward the baseline.")
    sampled = pd.DataFrame()
    if glob.glob(a.sampled):
        sampled = load(a.sampled)

    if BASE not in set(greedy.presentation):
        raise SystemExit(
            "no `baseline` condition in the greedy runs. Every nuisance delta "
            "is measured against it, so it is not optional -- re-run stage 3 "
            "with --presentation baseline --temperature 0 --n-samples 1.")

    print("\n=== conditions ===")
    cov = coverage(greedy)
    print(cov.to_string(index=False))
    cov.to_csv(out / "coverage.csv", index=False)
    if len(sampled):
        print(f"  + {len(sampled)} rows from the sampled floor run "
              f"({int(sampled.sample_idx.max()) + 1} samples/request)")

    # Noise floor: from a sampled run if given, else from identical requests.
    floor = float("nan")
    if len(sampled):
        nf = noise_floor(usable(sampled, a.col), a.col)
        if len(nf):
            floor = float(nf["median"].max())
            print(f"\n=== noise floor, {a.col} (sampled baseline run) ===")
            print(nf.to_string(index=False))
            nf.to_csv(out / "noise_floor.csv", index=False)
    if floor != floor:
        floor, rerun_ties, n_pairs = rerun_floor(usable(greedy, a.col), a.col)
        print(f"\n=== noise floor, {a.col} (identical baseline requests) ===")
        print(f"  {n_pairs} pairs of byte-identical requests in one greedy run")
        print(f"  mean |delta| {floor:.4f}, unchanged {rerun_ties:.1%}")
        print("  vs_floor below is against this mean |delta|.")

    damage_src = load(a.damage_scores) if a.damage_scores else greedy
    damage, damage_ties, sens, dmg = damage_reference(damage_src, a.col,
                                                      a.min_control)
    print(f"\n=== reference: what REAL damage does ({a.col}) ===")
    if len(sens):
        print(sens.round(3).to_string(index=False))
    print(f"  mean |delta| on the DAMAGED region: {damage:.4f}")
    print(f"  share of damaged regions that did not move at all: {damage_ties:.1%}")
    sens.to_csv(out / "damage_reference.csv", index=False)

    w_by_col = {c: paired(usable(greedy, c), c) for c in ("reward", "phi")}
    w = w_by_col[a.col] if a.col in w_by_col else paired(usable(greedy, a.col), a.col)

    print(f"\n=== NUISANCE: does a null change move the score ({a.col})? ===")
    print("  vs_damage: 1.0 means a nuisance change moves the score as much as")
    print("  real damage to the region does. text axes change NO pixels.")
    nt = nuisance_table(w, damage, floor,
                        greedy.drop_duplicates("variant_id")
                        .set_index("variant_id").base_id, a.min_control, dmg)
    if len(nt):
        print(nt.round(4).to_string(index=False))
        nt.to_csv(out / "nuisance.csv", index=False)
        bad = nt[(nt.kind == "text") & (nt.vs_damage >= 1.0)]
        for r in bad.itertuples():
            print(f"\n  {r.presentation}: a change that touches no pixel moves "
                  f"{a.col} by {r.mean_abs_delta:.4f},")
            print(f"  {r.vs_damage:.2f}x what damaging the region does. The "
                  f"score is responding to")
            print("  the packaging, not the region.")
    else:
        print("  n/a -- only the baseline condition is present")

    inull = identity_null(greedy, a.col, min_control=a.min_control)
    if len(inull):
        print(f"\n=== identity shuffles: the null inside the shuffle run ({a.col}) ===")
        print("  identity = canonical order kept, request byte-identical to baseline.")
        print(inull.round(4).to_string(index=False))
        inull.to_csv(out / "identity_null.csv", index=False)

    se = slot_effect(greedy, a.col)
    if len(se):
        print(f"\n=== does the score follow LIST POSITION? (shuffle, {a.col}) ===")
        print("  Region ids are shuffled per variant, so slot and region id are")
        print("  decorrelated. A trend across slots is position, not content.")
        print(se.to_string(index=False))
        se.to_csv(out / "slot_effect.csv", index=False)

    print("\n=== EXPLOITABILITY: can the score be pushed UP for free? ===")
    print("  enhance is a global cosmetic lift; no edit is improved by it.")
    print("  enhance_target is the same lift inside the target region's box only.")
    print("  reward carries AES = min(PQ), an image-level factor; phi does not.")
    ex = exploit_table(greedy, w_by_col)
    if len(ex):
        print(ex.round(4).to_string(index=False))
        ex.to_csv(out / "exploitability.csv", index=False)
        for r in ex[ex.readout == "reward"].itertuples():
            real = not (r.ci_lo <= 0 <= r.ci_hi)
            if real and r.mean_gain > 0:
                print(f"\n  {r.presentation} raised mean reward by "
                      f"{r.mean_gain:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
                      f"({r.frac_rose:.0%} of regions rose)")
                print("  with no edit improved. An editor trained on this "
                      "reward learns the trick.")
            elif not real:
                print(f"\n  {r.presentation}: {r.mean_gain:+.4f} "
                      f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] -- CI covers zero over "
                      f"{r.n_bases} photographs.")
                print("  Report as no effect, with the interval; a bare mean "
                      "cannot tell that from too few samples.")
    else:
        print("  n/a -- no exploitability condition in this glob")

    print("\n=== EXPLOIT on clean edits: failed vs ok, target vs other ===")
    print(f"  failed = baseline sc_success below {a.failed_below:g}. n_bases is the")
    print("  effective sample; regions within one photo are not independent.")
    split = exploit_split(greedy, w_by_col, a.failed_below)
    if len(split):
        print(split.round(4).to_string(index=False))
        split.to_csv(out / "exploit_split.csv", index=False)
        s = split.set_index(["presentation", "readout", "where", "edit"]).mean_gain
        k = ("enhance_target", "phi")
        if k + ("target", "failed") in s.index and k + ("other", "failed") in s.index:
            t, o = s[k + ("target", "failed")], s[k + ("other", "failed")]
            print(f"\n  enhance_target on FAILED edits: phi {t:+.4f} on the lifted "
                  f"region, {o:+.4f} elsewhere.")
            print("  A gap means the judge is flattered locally; no gap means the "
                  "lift acts on the whole image.")
        if exploit_figure(split, out / "exploit_gain.png"):
            print(f"\n  figure: {out / 'exploit_gain.png'}")
    else:
        print("  n/a -- needs a baseline with sc_success and an exploit condition")

    print("\n=== is the split just reversion to the mean? ===")
    print("  Splitting on the judge's OWN baseline and measuring change from")
    print("  that same baseline is circular. slope -1 = the condition replaces")
    print("  the score with a constant. If resid_* collapse to ~0, the split")
    print("  carried nothing the slope did not.")
    rev = reversion(greedy, w_by_col)
    if len(rev):
        print(rev.round(3).to_string(index=False))
        rev.to_csv(out / "reversion.csv", index=False)
        for r in rev[rev.readout == a.col].itertuples():
            if abs(r.resid_failed) < 0.25 * abs(r.raw_failed or 1):
                print(f"\n  {r.presentation}: slope {r.slope:+.3f}; the "
                      f"failed/ok gap ({r.raw_failed:+.3f} vs {r.raw_ok:+.3f}) "
                      f"falls to")
                print(f"  {r.resid_failed:+.3f} vs {r.resid_ok:+.3f} once one "
                      f"linear term in the baseline is removed. Report this as "
                      f"reversion,")
                print("  not as an exploit, unless a NON-circular split "
                      "(scripts/judge_agreement.py) agrees.")

    print("\n=== local flattery: target minus other, paired within variant ===")
    print("  enhance is the NEGATIVE CONTROL -- it lifts the whole frame, so it")
    print("  cannot flatter one region. Its contrast is the floor; anything")
    print("  enhance_target does not clear is not local flattery.")
    lc = local_contrast(greedy, w_by_col)
    if len(lc):
        print(lc.round(3).to_string(index=False))
        lc.to_csv(out / "local_contrast.csv", index=False)
        ctl = lc[(lc.presentation == "enhance") & (lc.readout == "phi")]
        if len(ctl) and not (ctl.ci_lo.iloc[0] <= 0 <= ctl.ci_hi.iloc[0]):
            print(f"\n  WARNING: the uniform control itself gives "
                  f"{ctl.target_minus_other.iloc[0]:+.3f} "
                  f"[{ctl.ci_lo.iloc[0]:+.3f}, {ctl.ci_hi.iloc[0]:+.3f}],")
            print("  excluding zero. Target regions differ from their "
                  "neighbours under a lift that")
            print("  cannot flatter locally, so this contrast cannot support "
                  "ANY local-flattery claim.")

    n_ci = 2 * (len(ex) + len(split) + len(lc))
    print(f"\n=== multiplicity: ~{n_ci} intervals in this run, uncorrected ===")
    print("  At 95% and this family size, roughly one null cell is expected to")
    print("  exclude zero by chance. Treat a single exclusion as exploratory;")
    print("  the claims worth making are the ones that survive a non-circular")
    print("  split and are large against the reversion slope.")

    print("\n=== the AES channel: what each condition does to image-level PQ ===")
    print("  AES = min(PQ), so the LOWER of the two rows sets the reward.")
    print("  A lift that raises one and lowers the other has not helped.")
    pq = pq_by_presentation(greedy)
    if len(pq):
        print(pq.round(3).to_string(index=False))
        pq.to_csv(out / "pq_by_presentation.csv", index=False)
    else:
        print("  n/a -- no parsed PQ rows in this glob")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
