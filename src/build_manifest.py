"""Expand data/bases/bases.json into the full manifest. Run once, commit the
resulting parquet hash to the repo so all five VMs provably share one design."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import yaml

from .schema import BaseSpec, Region, build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="data/bases")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--profile", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--out", default="out/manifest.parquet")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())[a.profile]
    specs = json.loads((Path(a.bases) / "bases.json").read_text())[: cfg["n_bases"]]

    bases = [BaseSpec(base_id=s["base_id"], source_file=s["source_file"],
                      instruction=s["instruction"],
                      regions=tuple(Region(r["region_id"], r["label"], tuple(r["bbox"]),
                                           r["mask_file"], r["area_frac"])
                                    for r in s["regions"]))
             for s in specs]

    df = build_manifest(bases, cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)

    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]
    print(f"{len(df)} variants ({df.is_control.sum()} controls) -> {a.out}")
    print(f"manifest sha256[:16] = {h}   <- all VMs must report this value")


if __name__ == "__main__":
    main()
