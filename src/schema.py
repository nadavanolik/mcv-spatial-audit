"""
Manifest schema + deterministic identity.

This module is the contract between all five stages. Freeze it early; every
analysis in the report is a groupby over these columns.

CRITICAL: variant_id and seed are pure functions of the design-matrix fields.
That is what lets any VM regenerate any variant byte-identically without ever
transferring the corrupted images.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd

# --- design matrix -----------------------------------------------------------

CORRUPTIONS = ["none", "blur", "saturate", "noise", "jpeg", "remove"]
SEVERITIES = [0, 1, 2, 3]          # 0 is only valid with corruption="none"
AREA_BINS = ["full", "half", "quarter"]

MANIFEST_COLUMNS = [
    "variant_id",       # str, sha1 of the identity tuple
    "base_id",          # str, which stage-1 edited image this derives from
    "source_file",      # str, original COCO file name
    "instruction",      # str, the edit instruction given to the editor
    "n_regions",        # int
    "target_region_id", # int, region that was corrupted; -1 for clean control
    "corruption",       # str, one of CORRUPTIONS
    "severity",         # int
    "area_bin",         # str
    "seed",             # int64, derived from variant_id
    "is_control",       # bool, corruption == "none"
]


@dataclass(frozen=True)
class Region:
    """One annotated region of a base image. Masks are stored as PNG next to
    the base edit, never inline in the manifest."""
    region_id: int
    label: str                    # COCO category name, used to build instructions
    bbox: tuple                   # (x, y, w, h) in pixels
    mask_file: str                # relative path, e.g. "masks/000123_r0.png"
    area_frac: float              # mask area / image area


@dataclass(frozen=True)
class BaseSpec:
    """A stage-1 unit: one source image + instruction + its regions."""
    base_id: str
    source_file: str
    instruction: str
    regions: tuple = field(default=())

    def to_json(self) -> str:
        d = asdict(self)
        d["regions"] = [asdict(r) for r in self.regions]
        return json.dumps(d, sort_keys=True)


def variant_id(base_id: str, target_region_id: int, corruption: str,
               severity: int, area_bin: str) -> str:
    """Stable 16-hex-char id. Changing ANY field yields a new variant."""
    key = f"{base_id}|{target_region_id}|{corruption}|{severity}|{area_bin}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def seed_for(vid: str) -> int:
    """Derive the RNG seed from the row itself, never from a global counter.
    A counter would make shard k produce different noise than shard j."""
    return int(hashlib.sha256(vid.encode()).hexdigest()[:8], 16)


def build_manifest(bases: list[BaseSpec], cfg: dict) -> pd.DataFrame:
    """Expand base specs into the full design matrix.

    Emits, per base:
      - one clean control per region (corruption="none")
      - corruption x severity x area_bin for each region, subject to cfg caps
    """
    corruptions = cfg["corruptions"]
    severities = cfg["severities"]
    area_bins = cfg["area_bins"]

    rows = []
    for b in bases:
        for r in b.regions:
            # clean control: needed as the ∆score baseline for this region
            vid = variant_id(b.base_id, r.region_id, "none", 0, "full")
            rows.append(dict(
                variant_id=vid, base_id=b.base_id, source_file=b.source_file,
                instruction=b.instruction, n_regions=len(b.regions),
                target_region_id=r.region_id, corruption="none", severity=0,
                area_bin="full", seed=seed_for(vid), is_control=True,
            ))
            for c in corruptions:
                if c == "none":
                    continue
                for s in severities:
                    if s == 0:
                        continue
                    for a in area_bins:
                        vid = variant_id(b.base_id, r.region_id, c, s, a)
                        rows.append(dict(
                            variant_id=vid, base_id=b.base_id,
                            source_file=b.source_file, instruction=b.instruction,
                            n_regions=len(b.regions), target_region_id=r.region_id,
                            corruption=c, severity=s, area_bin=a,
                            seed=seed_for(vid), is_control=False,
                        ))

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    assert df.variant_id.is_unique, "variant_id collision — check identity tuple"
    return df


def shard(df: pd.DataFrame, k: int, of: int) -> pd.DataFrame:
    """Deterministic sharding by variant_id hash, NOT by row order.

    Hash-based sharding survives manifest regeneration and row reordering; a
    positional `df.iloc[k::of]` does not.
    """
    assert 0 <= k < of
    h = df.variant_id.map(lambda v: int(v[:8], 16) % of)
    return df[h == k].copy()
