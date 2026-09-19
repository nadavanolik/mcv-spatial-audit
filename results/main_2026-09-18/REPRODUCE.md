# Reproducing the main-run numbers

Everything in the report comes from the five raw score parquets in this folder
(`scores_shard0..4.parquet`, ~1.8MB total). No GPU and no re-judging are needed —
stage 4 and the two helper scripts are pure CPU/pandas. Run these from the repo
root.

The `main` run itself (for the record, needs a GPU VM):
`mcvgpu2025s-0050`, 2026-09-18, `--profile main`, greedy, sharded five ways on
one machine. Manifest `sha256[:16] = 1d862ad6ce725e26`.

## The numbers as reported (floor-excluded, primary)

37% of region-deltas come from regions whose clean control already scored 0
(FLUX failed the edit); such a region cannot drop when damaged, so it is a
guaranteed tie. The headline numbers exclude them:

```bash
python -m src.stage4_analyze \
    --scores 'results/main_2026-09-18/scores_shard*.parquet' \
    --out results/main_2026-09-18/floor_excluded \
    --all-readouts --min-control 0
```

Writes `floor_excluded/`: `sensitivity.csv` (tie rate), `coherence.csv`,
`localization_{reward,phi,sc_preserve,sc_success}.csv`,
`localization_by_corruption_reward.csv`, `redundancy_*.csv`, `axis_by_severity.csv`,
`leakage_*.csv`.

Coherence is read on `phi`, not `reward` (reward carries the shared image-level
AES factor, which mechanically pulls regions together). The phi-based coherence
and sensitivity tables:

```bash
python -m src.stage4_analyze \
    --scores 'results/main_2026-09-18/scores_shard*.parquet' \
    --out out/_phi_tmp --col phi --all-readouts --min-control 0
# floor_excluded/coherence_phi.csv, sensitivity_phi.csv are copies of
# out/_phi_tmp/{coherence,sensitivity}.csv
```

## The perception check (does image-level PQ react?)

```bash
python -m scripts.pq_response \
    --scores 'results/main_2026-09-18/scores_shard*.parquet' \
    --out results/main_2026-09-18/pq_response.csv
```

## The all-region view (for comparison, includes floored regions)

The CSVs directly in this folder (not under `floor_excluded/`) are the
all-region analysis, drift-robustness split included:

```bash
python -m src.stage4_analyze \
    --scores 'results/main_2026-09-18/scores_shard*.parquet' \
    --out out/analysis --all-readouts --drift-csv data/bases/edit_drift.csv
```

`edit_drift.csv` ships inside `bases.tar.gz` (see the repo README); it is only
needed for the `--drift-csv` robustness split, not for anything else above.

## What each file is

- `scores_shard*.parquet` — raw judge outputs, one row per (variant, region).
  The reproducibility anchor; everything else derives from these.
- `report.txt` — full stdout of the all-region stage-4 run.
- `pq_response.csv` — image-level PQ delta vs clean control, by corruption.
- `floor_excluded/` — the numbers as reported (see above).
- `*.csv` (top level) — the all-region stage-4 outputs.
