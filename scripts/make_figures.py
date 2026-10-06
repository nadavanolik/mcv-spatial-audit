"""
Report figures and tables, rebuilt from the committed result CSVs.  [CPU]

Nothing here re-judges or re-analyses: every input is a CSV already under
results/, written by stage 4 or a helper script (see each folder's
REPRODUCE.md). This script only draws them, so a figure in the report can always
be traced back to a file in the repo.

    python -m scripts.make_figures            # -> results/figures/

Outputs:
  fig_localization.png   floor-excluded reward AUROC by severity, both judges
                         (two_judge_2026-09-26/localization_reward.csv)
  fig_tie_rate.png       Qwen: share of damaged vs untouched regions whose
                         score did not move, per corruption
                         (main_2026-09-18/floor_excluded/sensitivity.csv)
  tab_tie_rate.tex       the same numbers as a LaTeX tabular
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Two categorical slots, validated for CVD separation on a light surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

JUDGE_LABEL = {
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "OpenGVLab/InternVL3-2B": "InternVL3-2B",
}
SEVERITY_LABEL = {"1": "Severity 1", "3": "Severity 3", "binary": "Removal"}
CORRUPTION_ORDER = ["blur", "jpeg", "noise", "saturate", "remove"]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_localization(out: Path, plt) -> None:
    df = pd.read_csv(RESULTS / "two_judge_2026-09-26" / "localization_reward.csv",
                     dtype={"severity": str})
    sevs = ["1", "3", "binary"]
    judges = list(JUDGE_LABEL)
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for j, (judge, color) in enumerate(zip(judges, (BLUE, ORANGE))):
        rows = df[df["judge"] == judge].set_index("severity").loc[sevs]
        xs = [i + (j - 0.5) * width for i in range(len(sevs))]
        bars = ax.bar(xs, rows["auroc"], width=width - 0.04, color=color,
                      label=JUDGE_LABEL[judge])
        for bar, v in zip(bars, rows["auroc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(range(len(sevs)), [SEVERITY_LABEL[s] for s in sevs])
    ax.set_ylim(0.48, 0.545)
    ax.set_ylabel("ROC-AUC (reward)", color=INK)
    ax.set_xlabel("Corruption condition", color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_localization.png", dpi=200)
    plt.close(fig)


def _tie_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "main_2026-09-18" / "floor_excluded" / "sensitivity.csv",
                     dtype={"severity": str})
    df["order"] = df["corruption"].map(CORRUPTION_ORDER.index)
    df = df.sort_values(["order", "severity"]).reset_index(drop=True)
    df["label"] = [c if s == "binary" else f"{c} s{s}"
                   for c, s in zip(df["corruption"], df["severity"])]
    return df


def fig_tie_rate(out: Path, plt) -> None:
    df = _tie_table()
    n = len(df)
    height = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ys = range(n)
    ax.barh([y - height / 2 for y in ys], df["target_unchanged"] * 100,
            height=height - 0.04, color=BLUE, label="Damaged region")
    ax.barh([y + height / 2 for y in ys], df["other_unchanged"] * 100,
            height=height - 0.04, color=ORANGE, label="Untouched regions")
    ax.set_yticks(list(ys), df["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Regions whose score did not change (%)", color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_tie_rate.png", dpi=200)
    plt.close(fig)


def tab_tie_rate(out: Path) -> None:
    df = _tie_table()
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Corruption & $n$ & Damaged unchanged & Untouched unchanged \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['label']} & {r['n_target']} & "
                     f"{100 * r['target_unchanged']:.1f}\\% & "
                     f"{100 * r['other_unchanged']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "tab_tie_rate.tex").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=RESULTS / "figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")                      # no display on any of our boxes
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    fig_localization(args.out, plt)
    fig_tie_rate(args.out, plt)
    tab_tie_rate(args.out)
    for p in sorted(args.out.iterdir()):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
