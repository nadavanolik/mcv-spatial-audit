"""
Stage 2 — regenerate corrupted variants locally.

The corrupted variants are NEVER transferred between VMs and never written to
the 90G root disk. Each VM regenerates its own shard into /dev/shm (RAM-backed,
mode 1777, the only large writable surface we have) from:
    base edits (~300MB, transferred once) + manifest (a few MB)

Usage:
    python -m src.stage2_corrupt --manifest out/manifest.parquet \
        --bases data/bases --shard 0 --of 5 --out /dev/shm/mcv/variants
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .corruptions import apply_corruption
from .schema import shard


def load_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    # PNG level 1: we are writing to RAM, so favour speed over size.
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def render_shard(manifest: pd.DataFrame, bases_dir: Path, out_dir: Path,
                 k: int, of: int) -> list[Path]:
    rows = shard(manifest, k, of)
    print(f"shard {k}/{of}: {len(rows)} of {len(manifest)} variants")

    written = []
    # Group by base so each base edit is decoded once, not once per variant.
    for base_id, grp in tqdm(rows.groupby("base_id"), desc="bases"):
        base_img = load_rgb(bases_dir / base_id / "edit.png")
        mask_cache: dict[int, np.ndarray] = {}

        for row in grp.itertuples():
            rid = row.target_region_id
            if rid not in mask_cache:
                mp = bases_dir / base_id / "masks" / f"r{rid}.png"
                mask_cache[rid] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            mask = mask_cache[rid]
            if mask is None:
                raise FileNotFoundError(f"missing mask for {base_id} r{rid}")

            out = apply_corruption(base_img, mask, row.corruption, row.severity,
                                   row.area_bin, int(row.seed))
            p = out_dir / f"{row.variant_id}.png"
            save_rgb(p, out)
            written.append(p)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--out", default="/dev/shm/mcv/variants")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()

    df = pd.read_parquet(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # /dev/shm competes with the vLLM KV cache for RAM. 440GB total means this
    # is not tight, but check anyway so a full tmpfs fails loudly and early.
    # shutil, not os.statvfs: the latter is POSIX-only and this module is also
    # run on the Windows laptop for CPU-side checks.
    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"{out}: {free_gb:.0f}GB free")
    if free_gb < 10:
        raise SystemExit("under 10GB free in scratch — refusing to start")

    w = render_shard(df, Path(a.bases), out, a.shard, a.of)
    print(f"wrote {len(w)} variants to {out}")


if __name__ == "__main__":
    main()
