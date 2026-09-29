"""
Do two judges agree about the same picture? [CPU]

The only analysis in this repo that deliberately spans judges.
`nuisance_report.py` must be run once per judge -- its tables carry no judge
column and would average two families into one meaningless row. This script
exists because the opposite question is also worth asking: given the SAME image,
the same regions and the same prompt, how much do two judges agree?

That matters beyond curiosity. The four papers this audit examines train an
editor against one judge's numbers. If two judges cannot agree which edits
failed, then "the reward" is not a property of the edit at all, and a policy
trained on one of them is fitting that judge rather than image quality.

Pairing key is (variant_id, scored_region_id, presentation). Anything the two
runs do not share is dropped, so an unparsed row on either side removes the
pair rather than biasing one column.

Usage:
    python -m scripts.judge_agreement --scores 'out/nuisance/scores_*.parquet'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage4_analyze import BG, load, usable                 # noqa: E402

KEY = ["variant_id", "scored_region_id", "presentation"]


def paired_judges(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per (variant, region, presentation); one column per judge."""
    d = usable(df, col)
    d = d[d.scored_region_id != BG]
    a = d.groupby(KEY + ["judge"])[col].mean().reset_index()
    return a.pivot_table(index=KEY, columns="judge", values=col).dropna()


def agreement(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Correlation between the two judges, per presentation.

    Spearman as well as Pearson because these scores sit on a coarse grid with
    heavy mass on the rails -- a linear correlation alone would understate rank
    agreement, and a rank correlation alone would hide a scale mismatch.
    """
    w = paired_judges(df, col)
    if w.shape[1] != 2:
        return pd.DataFrame()
    x, y = w.columns
    rows = []
    for pres, g in w.groupby(level="presentation"):
        rows.append(dict(presentation=pres, n=len(g),
                         mean_a=float(g[x].mean()), mean_b=float(g[y].mean()),
                         pearson=float(g[x].corr(g[y])),
                         spearman=float(g[x].corr(g[y], method="spearman")),
                         mean_abs_diff=float((g[x] - g[y]).abs().mean())))
    return pd.DataFrame(rows).assign(judge_a=x, judge_b=y)


def failed_edit_overlap(df: pd.DataFrame, failed_below: float,
                        presentation: str = "baseline") -> pd.DataFrame:
    """Do the two judges call the SAME edits failed?

    This is the question with teeth. `exploit_split` splits on each judge's own
    `sc_success`, so "failed" means something slightly different in each judge's
    table; if the two sets barely overlap, the two exploit results are about
    different photographs and only agree by luck.

    Jaccard over the region sets, plus the raw 2x2, so a low overlap can be read
    as disagreement rather than one judge simply being harsher.
    """
    w = paired_judges(df[df.presentation == presentation], "sc_success")
    if w.shape[1] != 2:
        return pd.DataFrame()
    x, y = w.columns
    fa, fb = w[x] < failed_below, w[y] < failed_below
    both, only_a, only_b = int((fa & fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())
    neither = int((~fa & ~fb).sum())
    union = both + only_a + only_b
    return pd.DataFrame([dict(
        presentation=presentation, n=len(w), failed_below=failed_below,
        judge_a=x, judge_b=y,
        failed_a=int(fa.sum()), failed_b=int(fb.sum()),
        both_failed=both, only_a=only_a, only_b=only_b, neither=neither,
        jaccard=float(both / union) if union else float("nan"),
        agree=float(((fa == fb).mean())),
    )])


def crossjudge_exploit(df: pd.DataFrame, failed_below: float,
                       col: str = "reward") -> pd.DataFrame:
    """The exploit split with the circularity removed.

    `nuisance_report.exploit_split` decides "this edit failed" from a judge's
    own baseline and then measures that same judge's change from that same
    baseline. Any condition that pulls scores toward a central value scores
    positive on the low tail for purely arithmetic reasons.

    Here judge B's baseline picks the failed edits and judge A's gain is
    measured on them. The two judges' errors are not the same, so a gain that
    survives is about the condition rather than about the selection. Both
    directions are reported because neither judge is the reference.
    """
    # `paired` pivots presentations for ONE judge; `paired_judges` pivots the
    # two judges. This needs both, which is why it lives here and not there.
    from scripts.nuisance_report import boot_ci, cluster_mean, paired
    from src.stage4_analyze import usable as _usable

    succ = paired_judges(df[df.presentation == "baseline"], "sc_success")
    if succ.shape[1] != 2:
        return pd.DataFrame()
    v = df.drop_duplicates("variant_id").set_index("variant_id")
    rows = []
    for gain_judge in succ.columns:
        split_judge = [c for c in succ.columns if c != gain_judge][0]
        one = df[df.judge == gain_judge]
        w = paired(_usable(one, col), col).droplevel("judge")
        sel = succ[split_judge].droplevel("presentation")
        for mode in [c for c in w.columns if c != "baseline"]:
            p = w[["baseline", mode]].dropna()
            p = p.join(sel.rename("sel"), how="inner").dropna().reset_index()
            p = p[p.variant_id.map(v.is_control).astype(bool)]
            if p.empty:
                continue
            p = p.assign(gain=p[mode] - p["baseline"],
                         base_id=p.variant_id.map(v.base_id))
            for label, g in (("failed", p[p.sel < failed_below]),
                             ("ok", p[p.sel >= failed_below])):
                if len(g) < 3:
                    continue
                lo, hi = boot_ci(g.gain, g.base_id)
                rows.append(dict(
                    presentation=mode, readout=col, gain_judge=gain_judge,
                    split_by=split_judge, edit=label, n=len(g),
                    n_bases=int(g.base_id.nunique()),
                    mean_gain=cluster_mean(g.gain, g.base_id),
                    ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/nuisance/scores_*.parquet",
                    help="BOTH judges' parquets -- unlike nuisance_report, "
                         "this script needs them in one glob")
    ap.add_argument("--out", default="out/analysis/judge_agreement")
    ap.add_argument("--col", default="reward")
    ap.add_argument("--failed-below", type=float, default=20.0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.scores)
    judges = sorted(df.judge.unique())
    if len(judges) != 2:
        raise SystemExit(
            f"need exactly 2 judges in the glob, found {len(judges)}: {judges}. "
            f"This is the one script that wants both families at once.")
    print(f"{len(df)} rows, judges: {judges[0]} vs {judges[1]}")

    print(f"\n=== agreement on {a.col}, per presentation ===")
    print("  mean_abs_diff is on the readout's own scale; pearson/spearman say")
    print("  whether they rank the same regions the same way.")
    ag = agreement(df, a.col)
    if ag.empty:
        raise SystemExit("no paired rows -- do both judges cover the same variants?")
    print(ag.round(3).to_string(index=False))
    ag.to_csv(out / f"agreement_{a.col}.csv", index=False)

    print(f"\n=== do they call the SAME edits failed? (sc_success < {a.failed_below:g}) ===")
    print("  jaccard 1.0 = identical sets. Low overlap means each judge's")
    print("  exploit split is about a different set of photographs.")
    ov = failed_edit_overlap(df, a.failed_below)
    if len(ov):
        print(ov.round(3).to_string(index=False))
        ov.to_csv(out / "failed_edit_overlap.csv", index=False)

    print("\n=== the exploit split, WITHOUT the circularity ===")
    print("  One judge picks the failed edits, the other's gain is measured on")
    print("  them. A cell whose CI covers zero here is reversion to the mean,")
    print("  not an exploit, however tight its own-judge interval looked.")
    xj = crossjudge_exploit(df, a.failed_below, a.col)
    if len(xj):
        print(xj.round(3).to_string(index=False))
        xj.to_csv(out / "crossjudge_exploit.csv", index=False)
        dead = xj[(xj.edit == "failed") & (xj.ci_lo <= 0) & (xj.ci_hi >= 0)]
        for r in dead.itertuples():
            print(f"\n  {r.presentation} on {r.gain_judge.split('/')[-1]}: "
                  f"{r.mean_gain:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] when "
                  f"'failed' comes from the other judge.")
            print("  Do not report this axis as an exploit.")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
