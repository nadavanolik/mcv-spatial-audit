"""
Report figures, rebuilt from one run's stage-4 outputs.  [CPU]

Nothing here re-judges: every input is a file stage 4 or pq_response.py already
wrote under results/<run>/<tag>/analysis/. The one computation is a photo
bootstrap of per-corruption AUROC on stage 4's own deltas.parquet, with stage
4's definition (background excluded, label = is_target, score = -delta).

    python -m scripts.make_figures                       # -> results/figures/
    python -m scripts.make_figures --run main8b_2026-10-22

Outputs:
  fig_perception_localization.png
      (a) image-level AES drop per corruption at the strong setting, in units
          of the between-photo SD of clean AES (pq_response)
      (b) floor-excluded phi AUROC per corruption at the same setting
      both judges, 95% photo-bootstrap intervals
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Two categorical slots, validated for CVD separation on a light surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

# Run subdirectory -> legend label, in plotting order.
JUDGES = {"main_qwen_norm": "Qwen3-VL-8B", "main_ivl8": "InternVL3-8B"}
CORRUPTIONS = [("blur", "3", "Blur"), ("jpeg", "3", "JPEG"),
               ("noise", "3", "Noise"), ("saturate", "3", "Saturate"),
               ("remove", "binary", "Remove")]
N_BOOT = 1000


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def perception(run: Path, tag: str) -> pd.DataFrame:
    """dAES per corruption at the strong setting, in between-photo SD units."""
    a = run / tag / "analysis"
    txt = (a / "pq_response.txt").read_text()
    sd = float(re.search(r"between-base SD ([\d.]+)", txt).group(1))
    df = pd.read_csv(a / "pq_response.csv", dtype={"severity": str})
    rows = []
    for corr, sev, label in CORRUPTIONS:
        r = df[(df.corruption == corr) & (df.severity == sev)].iloc[0]
        # Plotted as a drop, so positive = the judge lowered AES.
        rows.append(dict(label=label, value=-r.d_AES / sd,
                         lo=-r.d_AES_hi / sd, hi=-r.d_AES_lo / sd))
    return pd.DataFrame(rows)


def localization(run: Path, tag: str, seed: int = 0) -> pd.DataFrame:
    """phi AUROC per corruption at the strong setting, photo bootstrap."""
    d = pd.read_parquet(run / tag / "analysis" / "phi" / "deltas.parquet")
    d = d[d.scored_region_id.astype(str) != "bg"]
    rng = np.random.default_rng(seed)
    rows = []
    for corr, sev, label in CORRUPTIONS:
        g = d[(d.corruption == corr) & (d.severity.astype(str) == sev)]
        auc = roc_auc_score(g.is_target, -g.delta)
        by_photo = {b: x for b, x in g.groupby("base_id")}
        photos = np.array(list(by_photo))
        boot = []
        for _ in range(N_BOOT):
            s = pd.concat([by_photo[b] for b in rng.choice(photos, len(photos))])
            if s.is_target.nunique() == 2:
                boot.append(roc_auc_score(s.is_target, -s.delta))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(label=label, value=auc, lo=lo, hi=hi,
                         n_photos=len(photos)))
    return pd.DataFrame(rows)


def _grouped(ax, tables: dict[str, pd.DataFrame], colors, bars: bool) -> None:
    """One slot per corruption, one offset mark per judge, 95% intervals.

    Bars only where zero is the natural baseline (a drop). AUROC is drawn as
    points: its floor is 0.5, and bars cut at an arbitrary axis minimum would
    exaggerate the differences.
    """
    width = 0.38
    for j, ((name, t), color) in enumerate(zip(tables.items(), colors)):
        xs = np.arange(len(t)) + (j - 0.5) * width
        err = [t.value - t.lo, t.hi - t.value]
        if bars:
            ax.bar(xs, t.value, width=width - 0.04, color=color, label=name)
            ax.errorbar(xs, t.value, yerr=err, fmt="none", ecolor=INK,
                        elinewidth=0.8, capsize=2)
        else:
            ax.errorbar(xs, t.value, yerr=err, fmt="o", color=color,
                        markersize=4, elinewidth=1.2, capsize=2, label=name)
    labels = next(iter(tables.values())).label
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def fig_perception_localization(run: Path, out: Path, plt) -> pd.DataFrame:
    per = {name: perception(run, tag) for tag, name in JUDGES.items()}
    loc = {name: localization(run, tag) for tag, name in JUDGES.items()}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    _grouped(a, per, (BLUE, ORANGE), bars=True)
    a.set_ylabel("AES drop (SD of clean AES)", color=INK)
    a.set_title("(a) Perception: image-level AES falls", color=INK, loc="left")
    a.set_ylim(0, 1.0)
    _style(a)
    a.legend(frameon=False, loc="upper left")

    _grouped(b, loc, (BLUE, ORANGE), bars=False)
    b.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    b.text(-0.55, 0.503, "chance", color=MUTED, fontsize=7, va="bottom")
    b.set_ylim(0.45, 0.65)
    b.set_ylabel(r"AUROC on $\phi$ (perfect = 1)", color=INK)
    b.set_title("(b) Localization: damaged region vs. the others", color=INK,
                loc="left")
    _style(b)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig_perception_localization.png", dpi=300)
    fig.savefig(out / "fig_perception_localization.pdf")
    plt.close(fig)

    rows = []
    for kind, tabs in (("perception_sd", per), ("auroc_phi", loc)):
        for name, t in tabs.items():
            rows.append(t.assign(judge=name, quantity=kind))
    return pd.concat(rows)[["quantity", "judge", "label", "value", "lo", "hi"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="main8b_2026-10-22",
                    help="results/<run>/ holding main_qwen_norm/ and main_ivl8/")
    ap.add_argument("--out", type=Path, default=RESULTS / "figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")                      # no display on any of our boxes
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    nums = fig_perception_localization(RESULTS / args.run, args.out, plt)
    nums.to_csv(args.out / "fig_perception_localization.csv", index=False)
    print(nums.round(3).to_string(index=False))
    for p in sorted(args.out.glob("fig_perception_localization.*")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
