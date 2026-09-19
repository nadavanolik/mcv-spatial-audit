"""
PQ response -- does the IMAGE-LEVEL quality term react to per-region corruption?

The reward is sqrt(phi_r * AES)/C (Equation 3), where phi_r = min(success,
preserve) is per-region but AES = min(PQ) is a SINGLE image-level term shared by
every region of an image. stage 4 asks whether the per-region part localises the
damage. This asks the complementary question about the global part: when we
corrupt one region, does the image-level PQ move at all, and does it move more
for stronger damage?

It is the check that decides how to read a flat localization result:

  PQ does NOT move  -> "the score ignored the damage" is ambiguous with "the
                       judge could not SEE it" (region ~4.5% of the image, and
                       max_pixels is capped, so fine texture may be lost).
  PQ DOES move,     -> the judge perceives the corruption but books it globally;
  monotone in sev      the damage information exists and simply never reaches a
                       per-region score. That is the sharper finding, and it is
                       what the main run shows.

So this table is read BESIDE stage 4's localization tables, never instead of
them: stage 4 says the per-region scores do not localise; this says that is not
because the judge is blind.

PQ is image-level, hence constant across a variant's region rows, so we collapse
to one row per variant before differencing each corrupted variant against the
mean PQ of its base's clean (corruption == none) controls. AES = min(PQ) matches
the reward's own aggregation.

CPU only; reads the same scores parquet stage 4 reads.

Usage:
    python -m scripts.pq_response \
        --scores 'results/main_2026-09-18/scores_shard*.parquet' \
        --out results/main_2026-09-18/pq_response.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage4_analyze import collapse_flat_severity  # noqa: E402

PQ_COLS = ["pq_naturalness", "pq_artifacts"]


def pq_response(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per-corruption mean change in image-level PQ vs the base's clean control.

    Returns the by-(corruption, severity) table and the control PQ reference
    (its between-base SD is the yardstick for whether a delta is meaningful).
    """
    # One row per variant: PQ is identical across a variant's region rows.
    v = (df.drop_duplicates("variant_id")
         [["variant_id", "base_id", "corruption", "severity", "is_control"]
          + PQ_COLS].copy())
    v["aes"] = v[PQ_COLS].min(axis=1)          # AES = min(PQ), as in Equation 3

    ctrl = (v[v.is_control].groupby("base_id")[PQ_COLS + ["aes"]].mean()
            .rename(columns={c: "c_" + c for c in PQ_COLS + ["aes"]}))

    m = v[~v.is_control].merge(ctrl, on="base_id")
    for c in PQ_COLS + ["aes"]:
        m["d_" + c] = m[c] - m["c_" + c]

    # remove has no severity ladder (inpaint deletes the object at every radius),
    # so it reports as one 'binary' condition -- identical handling to stage 4.
    m = collapse_flat_severity(m)

    tbl = (m.groupby(["corruption", "severity"])
           .agg(n=("variant_id", "size"),
                d_naturalness=("d_pq_naturalness", "mean"),
                d_artifacts=("d_pq_artifacts", "mean"),
                d_AES=("d_aes", "mean"),
                AES_unchanged=("d_aes", lambda x: float((x == 0).mean())),
                AES_dropped=("d_aes", lambda x: float((x < 0).mean())))
           .reset_index())
    return tbl, ctrl["c_aes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default="out/analysis/pq_response.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(a.scores))
    if not files:
        raise SystemExit(f"no score files match {a.scores}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    tbl, c_aes = pq_response(df)

    print("=== image-level PQ delta: corrupted variant vs its base clean control ===")
    print("(PQ on 0-25; AES = min(naturalness, artifacts), the reward's own term)")
    print(tbl.round(3).to_string(index=False))
    print()
    print(f"control AES: mean {c_aes.mean():.2f}, between-base SD {c_aes.std():.3f}")
    print("Read beside stage 4: a monotone AES drop means the judge SEES the")
    print("damage, so a flat per-region localization is about attribution, not")
    print("perception.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tbl.round(4).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
