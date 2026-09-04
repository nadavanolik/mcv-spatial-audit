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
    rows = []
    for mode in EXPLOIT_AXES:
        for col, w in w_by_col.items():
            if mode not in w.columns:
                continue
            p = w[[BASE, mode]].dropna()
            if p.empty:
                continue
            gain = p[mode] - p[BASE]
            rows.append(dict(presentation=mode, readout=col, n=len(p),
                             baseline_mean=float(p[BASE].mean()),
                             condition_mean=float(p[mode].mean()),
                             mean_gain=float(gain.mean()),
                             frac_rose=float((gain > 0).mean())))
    return pd.DataFrame(rows)


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
    print("  reward carries AES = min(PQ), an image-level factor; phi does not.")
    ex = exploit_table(greedy, w_by_col)
    if len(ex):
        print(ex.round(4).to_string(index=False))
        ex.to_csv(out / "exploitability.csv", index=False)
        for r in ex[(ex.readout == "reward") & (ex.mean_gain > 0)].itertuples():
            print(f"\n  {r.presentation} raised mean reward by {r.mean_gain:+.4f} "
                  f"({r.frac_rose:.0%} of regions rose)")
            print("  with no edit improved. An editor trained on this reward "
                  "learns the trick.")
    else:
        print("  n/a -- no exploitability condition in this glob")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
