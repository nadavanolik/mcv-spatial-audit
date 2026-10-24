"""
Appendix figures drawn on one photograph: the pipeline, and where the score
moved.  [CPU, needs data/bases]

Variants are re-rendered here from the base edit with stage 2's own function
and seed, so the pixels shown are byte-identical to what the judge saw. Scores
come from stage 4's deltas.parquet for the run; nothing is re-judged.

    python -m scripts.make_examples                          # -> results/figures/
    python -m scripts.make_examples --base 000000381607 --corruption noise

Outputs:
  fig_example_pipeline.png   source | edit (region boxes) | two corrupted variants
  fig_example_heatmap.png    one panel per damaged region: every region tinted
                             by its delta-phi against the clean control
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.corruptions import apply_corruption
from src.schema import seed_for, variant_id
from src.stage2_corrupt import load_rgb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INK, MUTED = "#1f1f1e", "#6b6a64"
REGION_COLORS = ["#2a78d6", "#eb6834", "#2f9e6a", "#9a5bd1", "#c9a227"]


def render(bases: Path, base: str, rid: int, corruption: str,
           severity: int) -> np.ndarray:
    """The exact variant stage 2 renders for this manifest row."""
    edit = load_rgb(bases / base / "edit.png")
    mask = cv2.imread(str(bases / base / "masks" / f"r{rid}.png"),
                      cv2.IMREAD_GRAYSCALE)
    vid = variant_id(base, rid, corruption, severity, "full")
    return apply_corruption(edit, mask, corruption, severity, "full",
                            seed_for(vid))


def _mask(bases: Path, base: str, rid: int) -> np.ndarray:
    return cv2.imread(str(bases / base / "masks" / f"r{rid}.png"),
                      cv2.IMREAD_GRAYSCALE) > 127


def _contour(ax, mask: np.ndarray, color: str, lw: float, ls="-") -> None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if len(c) < 3:
            continue
        xy = np.vstack([c[:, 0, :], c[:1, 0, :]])
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, linestyle=ls)


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_pipeline(bases: Path, base: str, regions: list[dict], shown: list,
                 out: Path, plt) -> None:
    src = load_rgb(bases / base / "source.png")
    edit = load_rgb(bases / base / "edit.png")
    panels = [(src, "COCO source"), (edit, "FLUX.1-Kontext edit")]
    names = {"noise": "noise (strong)", "blur": "blur (strong)",
             "jpeg": "JPEG (strong)", "saturate": "saturation (strong)",
             "remove": "removal"}
    for rid, corr, sev in shown:
        panels.append((render(bases, base, rid, corr, sev),
                       f"{names[corr]}: {regions[rid]['label']}"))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 7.0 / len(panels) + 0.3))
    for i, (ax, (img, title)) in enumerate(zip(axes, panels)):
        ax.imshow(img)
        ax.set_title(title, fontsize=7.5, color=INK)
        _clean(ax)
        if i == 1:
            for r in regions:
                x, y, w, h = r["bbox"]
                c = REGION_COLORS[r["region_id"] % len(REGION_COLORS)]
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                           edgecolor=c, linewidth=1.2))
                ax.text(x + 4, y + 4, r["label"], color="white", fontsize=6,
                        va="top", bbox=dict(facecolor=c, edgecolor="none",
                                            pad=1.0))
        if i >= 2:
            rid = shown[i - 2][0]
            _contour(ax, _mask(bases, base, rid), "white", 0.9, "--")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(out / "fig_example_pipeline.png", dpi=300)
    plt.close(fig)


def fig_heatmap(bases: Path, base: str, regions: list[dict], d: pd.DataFrame,
                corruption: str, severity: str, out: Path, plt) -> pd.DataFrame:
    from matplotlib.colors import TwoSlopeNorm

    edit_sev = 3 if severity == "binary" else int(severity)
    g = d[(d.base_id == base) & (d.corruption == corruption)
          & (d.severity.astype(str) == severity)
          & (d.scored_region_id.astype(str) != "bg")].copy()
    g["scored_region_id"] = g.scored_region_id.astype(int)
    g["target_region_id"] = g.target_region_id.astype(int)
    if severity == "binary":                   # one variant per target
        g = g[g.variant_id.isin({variant_id(base, r["region_id"], corruption,
                                            edit_sev, "full") for r in regions})]

    cmap = plt.get_cmap("RdBu")
    norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)
    masks = {r["region_id"]: _mask(bases, base, r["region_id"]) for r in regions}

    fig, axes = plt.subplots(1, len(regions), figsize=(7.0, 7.0 / len(regions) + 0.35))
    for ax, tr in zip(axes, regions):
        t = tr["region_id"]
        img = render(bases, base, t, corruption, edit_sev).astype(np.float32) / 255
        rows = g[g.target_region_id == t].set_index("scored_region_id")
        for rid, m in masks.items():
            if rid not in rows.index:
                continue
            # Opacity scales with |delta|: an unmoved region stays untinted, so
            # the damage itself stays visible under it.
            delta = rows.loc[rid, "delta"]
            col = np.array(cmap(norm(delta))[:3])
            alpha = 0.6 * min(abs(delta) / 25, 1)
            img[m] = (1 - alpha) * img[m] + alpha * col
        ax.imshow(np.clip(img, 0, 1))
        _clean(ax)
        for rid, m in masks.items():
            _contour(ax, m, "black" if rid == t else "white",
                     2.0 if rid == t else 0.6, "-" if rid == t else ":")
            if rid in rows.index:
                r = rows.loc[rid]
                ys, xs = np.nonzero(m)
                ax.text(xs.mean(), ys.mean(),
                        f"{regions[rid]['label']}\n{r.ctrl_score:.0f}"
                        rf"$\to${r.score:.0f}",
                        ha="center", va="center", fontsize=6.5, color=INK,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="none", pad=1.2))
        ax.set_title(f"damage on the {tr['label']}", fontsize=8, color=INK)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.025, pad=0.01)
    cb.set_label(r"$\Delta\phi$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out / "fig_example_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return g[["target_region_id", "scored_region_id", "ctrl_score", "score",
              "delta"]].sort_values(["target_region_id", "scored_region_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="main8b_2026-10-22")
    ap.add_argument("--judge-tag", default="main_qwen_norm")
    ap.add_argument("--bases", type=Path, default=ROOT / "data" / "bases")
    ap.add_argument("--base", default="000000381607")
    ap.add_argument("--corruption", default="noise")
    ap.add_argument("--severity", default="3", help="1, 3 or binary")
    ap.add_argument("--out", type=Path, default=RESULTS / "figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = json.loads((args.bases / args.base / "regions.json").read_text())
    regions = sorted(regions, key=lambda r: r["region_id"])
    d = pd.read_parquet(RESULTS / args.run / args.judge_tag / "analysis" / "phi"
                        / "deltas.parquet")
    args.out.mkdir(parents=True, exist_ok=True)

    # The heatmap's corruption on the first region, and removal of the second.
    shown = [(0, args.corruption, 3), (1, "remove", 3)]
    fig_pipeline(args.bases, args.base, regions, shown, args.out, plt)
    tab = fig_heatmap(args.bases, args.base, regions, d, args.corruption,
                      args.severity, args.out, plt)
    print((args.bases / args.base / "instruction.txt").read_text().strip())
    print(tab.to_string(index=False))
    for p in ("fig_example_pipeline.png", "fig_example_heatmap.png"):
        print(f"wrote {(args.out / p).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
