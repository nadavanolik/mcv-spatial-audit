"""
Nuisance and exploitability, read off a sweep of presentation conditions. [CPU]

Lives in scripts/ rather than stage 4 for one reason: the delta is a different
delta. stage4_analyze compares a CORRUPTED variant against its CLEAN control.
Here the image is fixed and the PACKAGING changes, so the pairing key is
(judge, variant_id, scored_region_id) across presentations. Feeding these
parquets to stage 4 would not just answer the wrong question -- delta_table
does not group by `presentation`, so it would pool every condition's clean
controls into one baseline and say nothing about it.

Everything else is stage 4's vocabulary, imported rather than re-derived, so
the tables read next to its output: the same tie rate, the same noise floor,
the same |delta|.

THE HEADLINE. A nuisance condition changes nothing that carries information
about edit quality. If the score moves as much for a shuffled region list as
it does for a genuinely damaged region, the per-region number is not measuring
the region.

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
                                load, noise_floor, sensitivity, usable)

BASE = "baseline"


def paired(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per (judge, variant, region); one column per presentation.

    Aggregating over sample_idx first is defensive -- the greedy runs have a
    single sample each, but a mean over one value is a no-op and a mean over
    five is the right thing if someone runs a condition sampled.
    """
    a = (df.groupby(["judge", "variant_id", "scored_region_id", "presentation"])
         [col].mean().reset_index())
    return a.pivot_table(index=["judge", "variant_id", "scored_region_id"],
                         columns="presentation", values=col)


def damage_reference(df: pd.DataFrame, col: str) -> tuple:
    """What REAL damage does, from the baseline condition alone.

    The pilot design is [none, blur, remove], so the baseline parquet already
    holds clean controls and damaged variants -- the sweep is self-contained
    and needs nothing from the main run.
    """
    b = df[df.presentation == BASE]
    if b.empty:
        return float("nan"), float("nan"), pd.DataFrame()
    d = delta_table(usable(b, col), col)
    if d.empty:
        return float("nan"), float("nan"), pd.DataFrame()
    tgt = d[(d.is_target) & (d.scored_region_id != BG)]
    return (float(tgt.delta.abs().mean()) if len(tgt) else float("nan"),
            float((tgt.delta == 0).mean()) if len(tgt) else float("nan"),
            sensitivity(d))


def nuisance_table(w: pd.DataFrame, damage: float, floor: float) -> pd.DataFrame:
    """|delta| for each null change, against what real damage moved."""
    rows = []
    for mode in [c for c in w.columns if c in NUISANCE_AXES]:
        p = w[[BASE, mode]].dropna()
        if p.empty:
            continue
        d = p[mode] - p[BASE]
        rows.append(dict(
            presentation=mode,
            kind="text" if mode in TEXT_AXES else "image",
            n=len(d),
            unchanged=float((d == 0).mean()),
            mean_abs_delta=float(d.abs().mean()),
            mean_delta=float(d.mean()),
            vs_damage=float(d.abs().mean() / damage) if damage else float("nan"),
            vs_floor=float(d.abs().mean() / floor) if floor else float("nan"),
        ))
    return pd.DataFrame(rows)


def slot_effect(df: pd.DataFrame, col: str, mode: str = "shuffle") -> pd.DataFrame:
    """Under `shuffle`, does the score follow the region's POSITION in the list?

    This is the mechanism behind a nuisance effect, not just its size. A score
    that tracks slot 0 rather than the region named in slot 0 is a per-region
    reward in name only. Meaningful only for shuffle, where position varies
    independently of region id; under every other condition the two are the
    same column.
    """
    d = df[(df.presentation == mode) & (df.scored_region_id != BG)]
    d = d[d.slot_idx >= 0]
    if d.empty:
        return pd.DataFrame()
    return (d.groupby(["judge", "slot_idx"])[col]
            .agg(["mean", "std", "size"]).round(3).reset_index())


def cluster_mean(gain: pd.Series, base_id: pd.Series) -> float:
    """The mean of per-photograph means -- the estimand `boot_ci` brackets.

    A row-weighted mean is a DIFFERENT estimator, and pairing one with the
    other's interval produced point estimates sitting on their own confidence
    bound (worst case seen: -1.345 quoted against a CI centred on -2.258, a 68%
    discrepancy). Point and interval must estimate the same thing, so both come
    from here.
    """
    g = pd.DataFrame({"gain": gain.values, "base": base_id.values}).dropna()
    if g.empty:
        return float("nan")
    return float(g.groupby("base").gain.mean().mean())


def boot_ci(gain: pd.Series, base_id: pd.Series, n_boot: int = 2000,
            seed: int = 0) -> tuple:
    """95% CI for `cluster_mean`, resampling PHOTOGRAPHS, not regions.

    Regions inside one photograph share an edit, an instruction and a scene, so
    they are not independent draws; a CI over rows would be far too tight. The
    honest sample size is the number of bases, which is why `exploit_split`
    already reports `n_bases`.

    Seeded and fixed at 0 so a committed CSV reproduces exactly. This is an
    analysis seed and has nothing to do with the corruption seeds in schema.py.

    NOT multiplicity-corrected. A full run emits 30 of these per judge, so
    "this one interval excludes zero" is weak evidence on its own -- see the
    family-size warning in main().
    """
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


def reversion(df: pd.DataFrame, w_by_col: dict) -> pd.DataFrame:
    """Does the condition shrink every score toward the judge's grand mean?

    This is the control the failed/ok split needs and did not have. Splitting on
    the judge's OWN baseline and then measuring change from that same baseline
    is circular: any condition that reverts toward a central value produces a
    positive gain on the low tail and a negative one on the high tail, with no
    exploit involved.

    `slope` regresses the gain on the baseline score. A slope near -1 means the
    condition simply replaces the score with a constant (that is `noimg`); a
    slope near 0 means the condition moves scores independently of where they
    started, which is what a genuine exploit would look like. `resid_failed` and
    `resid_ok` re-run the split on the residual after removing one linear term:
    if those collapse to zero, the split carried nothing the slope did not.
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
    """Within each variant, does the TARGET region gain more than its neighbours?

    Paired inside the variant, because that is how the design is built --
    comparing two marginal confidence intervals is not a test of their
    difference and can hide or invent one.

    Read `enhance` as the built-in negative control: it lifts the whole frame
    uniformly, so it CANNOT flatter one region, and whatever contrast it
    produces is the floor for this statistic. If `enhance_target` does not beat
    that floor, there is no local flattery to report -- and if `enhance` itself
    clears zero, the contrast is measuring something about target regions rather
    than about the lift.
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
    """What each presentation does to the two IMAGE-level quality scores.

    Eq. (3) multiplies every region's phi by AES = min(PQ), so this is the
    channel a cosmetic lift would have to exploit. PQ is one value per image,
    not per region, hence `first()` per (variant, presentation).

    Read `pq_artifacts` and `pq_naturalness` separately: AES is their MINIMUM,
    so whichever is lower is the one that actually sets the reward, and a lift
    that raises one while lowering the other has not helped.
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
    """Can the score be pushed UP without the image getting better?

    `reward` and `phi` are reported side by side because Equation (3) is
    sqrt(phi * AES)/C with AES = min(PQ) a single image-level term. A global
    cosmetic lift can raise AES -- and therefore every region's reward -- with
    no edit improved. It may at the same time LOWER phi, because sharpening the
    edit makes it differ more from the source and the prompt asks about
    preservation. Those two moving in opposite directions is a result, not a
    contradiction; a single collapsed number would hide it.
    """
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
    """The exploit gain, split by WHERE the region is and whether its edit FAILED.

    Clean controls only: the scenario is an edit that already went wrong (the
    car came out blue, not red) with no corruption added, and the question is
    whether a change that fixes nothing raises that region's score. "Failed" is
    the judge's own baseline `sc_success` below `failed_below` -- not circular
    here, because a policy trained on this reward climbs the judge's number,
    not the truth.

    `where` separates the two exploit mechanisms. Under `enhance` (global) the
    target is just a designated region and target == other is expected. Under
    `enhance_target`, a target gain above other's on `phi` means the judge can
    be flattered locally; equal gains mean the lift acts through the whole
    image. `n_bases` is the effective sample -- regions within one photograph
    are not independent.
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
    """One bar per (presentation, failed/ok), with the bootstrap CI as the bar.

    The failed-vs-ok split is the whole exploitability story, so it is the whole
    figure. A CI that straddles zero is the visual form of "this axis does
    nothing", which is the finding for both cosmetic axes and needs to be as
    legible as the one axis that works.
    """
    import matplotlib
    matplotlib.use("Agg")                      # no display on any of our boxes
    import matplotlib.pyplot as plt

    # split["where"], never split.where -- `.where` is a DataFrame METHOD, so
    # attribute access returns the method and the comparison is silently False.
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
    """Did the judge answer at all, per condition.

    Carries the noimg exploit's headline directly: a full, schema-valid,
    every-region-scored response to a request containing no image.
    """
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
    # A separate argument, not a filter over --scores. The floor run also
    # carries presentation == "baseline", so it is indistinguishable from the
    # greedy baseline by any column: same variants, same regions, same label.
    # Globbed together they would average silently. Two globs, two files, no
    # detection logic to get wrong.
    ap.add_argument("--sampled", default="out/nuisance/floor_*.parquet",
                    help="the n>1 baseline run that supplies the noise floor. "
                         "Greedy decoding has no within-variant spread, so "
                         "without this there is no denominator.")
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

    # The denominator. Greedy runs have no within-variant spread by
    # construction, so the floor has to come from the sampled run or not at all.
    floor = float("nan")
    if len(sampled):
        nf = noise_floor(usable(sampled, a.col), a.col)
        if len(nf):
            floor = float(nf["median"].max())
            print(f"\n=== noise floor, {a.col} (sampled baseline run) ===")
            print(nf.to_string(index=False))
            nf.to_csv(out / "noise_floor.csv", index=False)
    if floor != floor:
        print("\n=== noise floor: NOT MEASURED ===")
        print("  No sampled run in this glob. Every ratio below is against")
        print("  real damage only, and a nuisance effect cannot be told from")
        print("  judge instability. Run the baseline condition once more at")
        print("  --temperature 0.7 --n-samples 5 over the SAME variants.")

    damage, damage_ties, sens = damage_reference(greedy, a.col)
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
    nt = nuisance_table(w, damage, floor)
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
        # A CI straddling zero is the difference between "no effect" and
        # "underpowered", and the two look identical in a mean alone.
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
