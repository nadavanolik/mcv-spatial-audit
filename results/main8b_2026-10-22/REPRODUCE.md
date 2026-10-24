# Reproducing the report numbers (8B judges, 0-1000 boxes)

This is the run the report is written from. Two judges, Qwen3-VL-8B-Instruct and
InternVL3-8B, judged the `main` design (150 photos, 476 regions, 5,236 variants)
and the `nuisance150` clean controls under all seven presentations. Boxes were
given in the 0-1000 frame both models are trained on (`--bbox-frame norm1000`).
One Qwen run repeats the committed pixel-box protocol, to separate a re-run from
the change of box frame.

Everything below the judging step is CPU-only and runs from the raw score
parquets in this folder. Run from the repo root.

## Provenance

`gate/` holds what was checked before any judging: the code commit and model
snapshots (`provenance.txt`), the determinism fixture hash
(`determinism.txt`, `776feeddd281fa726195bf504c7b19c8`, same as the A10 VMs),
the test suites (`tests.txt`), the manifest hashes (`manifest_*.txt`: `main`
`1d862ad6ce725e26`, `nuisance150` `8f5c7b61541fda69`) and `pip_freeze.txt`.

## Judging (GPU, for the record)

One cell per (run, presentation, shard). Stage 2 renders the shard and stage 3
judges it in the same session (see `.claude/CLAUDE.md` on `/dev/shm`).

| Folder | Model | `--bbox-frame` | Extra flags | Manifest | Presentations | Shards |
|---|---|---|---|---|---|---|
| `main_qwen_norm/` | `Qwen/Qwen3-VL-8B-Instruct` | `norm1000` | `--max-model-len 4096` | `main` | `baseline` | 5 |
| `main_ivl8/` | `OpenGVLab/InternVL3-8B` | `norm1000` | `--max-model-len 8192 --max-dynamic-patch 6` | `main` | `baseline` | 5 |
| `main_qwen_pixel/` | `Qwen/Qwen3-VL-8B-Instruct` | `pixel` | `--max-model-len 4096` | `main` | `baseline` | 5 |
| `nuis_qwen/` | `Qwen/Qwen3-VL-8B-Instruct` | `norm1000` | `--max-model-len 4096` | `nuisance150` | all seven | 1 |
| `nuis_ivl8/` | `OpenGVLab/InternVL3-8B` | `norm1000` | `--max-model-len 8192 --max-dynamic-patch 6` | `nuisance150` | all seven | 1 |

The seven presentations are `baseline`, `noimg`, `enhance`, `enhance_target`,
`shuffle`, `subset`, `box`. Per cell:

```bash
python -m src.build_manifest --profile main --out out/manifest_main.parquet
python -m src.stage2_corrupt --manifest out/manifest_main.parquet --bases data/bases \
    --out /dev/shm/mcv/variants --shard $SHARD --of 5
python -m src.stage3_judge --manifest out/manifest_main.parquet --bases data/bases \
    --variants /dev/shm/mcv/variants --model $MODEL --shard $SHARD --of 5 \
    --presentation baseline --bbox-frame norm1000 --max-model-len 4096 \
    --gpu-util 0.89 --temperature 0 --n-samples 1 \
    --out results/main8b_2026-10-22/main_qwen_norm/scores_shard$SHARD.parquet
```

Nuisance cells use `--profile nuisance150`, `--shard 0 --of 1`, and write
`nuis_*/scores_<presentation>.parquet`.

## Checks on the stimuli (CPU, needs `data/bases`)

Render every `main` variant once (`--shard 0 --of 1`), then:

```bash
python -m scripts.verify_corruption --manifest out/manifest_main.parquet \
    --bases data/bases --variants /dev/shm/mcv/variants   # checks/verify_corruption.txt
python -m scripts.region_overlap --manifest out/manifest_main.parquet \
    --bases data/bases --variants /dev/shm/mcv/variants \
    --out results/main8b_2026-10-22/checks/overlap.csv
python -m scripts.bbox_frame_check --scores 'results/main_2026-09-18/scores_shard*.parquet' \
    --bases data/bases --out results/main8b_2026-10-22/checks/bbox_frame_committed.csv
```

## Analysis (CPU)

Per main run (`main_qwen_norm`, `main_ivl8`, `main_qwen_pixel`), with
`R=results/main8b_2026-10-22` and `A=$R/$TAG/analysis`:

```bash
for COL in reward phi; do
  python -m src.stage4_analyze --scores "$R/$TAG/scores_shard*.parquet" \
      --out $A/$COL --col $COL --all-readouts --min-control 0 \
      --drift-csv data/bases/edit_drift.csv --overlap-csv $R/checks/overlap.csv
done
python -m scripts.pq_response --scores "$R/$TAG/scores_shard*.parquet" \
    --out $A/pq_response.csv --gate
python -m scripts.bbox_frame_check --scores "$R/$TAG/scores_shard*.parquet" \
    --bases data/bases --out $A/bbox_frame.csv
```

Run against run:

```bash
python -m scripts.compare_runs \
    --a 'results/main_2026-09-18/scores_shard*.parquet' --a-name committed \
    --b "$R/main_qwen_pixel/scores_shard*.parquet" --b-name rerun \
    --min-control 0 --out $R/compare/rerun
python -m scripts.compare_runs \
    --a "$R/main_qwen_pixel/scores_shard*.parquet" --a-name pixel \
    --b "$R/main_qwen_norm/scores_shard*.parquet" --b-name norm1000 \
    --min-control 0 --out $R/compare/frame
```

Nuisance and exploitability, per judge (`nuis_qwen` with damage from
`main_qwen_norm`, `nuis_ivl8` with damage from `main_ivl8`), for `--col reward`
and `phi`, with and without `--min-control 0` (`floor_excluded/` and `all/`):

```bash
python -m scripts.nuisance_report --scores "$R/nuis_qwen/scores_*.parquet" \
    --damage-scores "$R/main_qwen_norm/scores_shard*.parquet" \
    --col phi --min-control 0 --out $R/nuis_qwen/analysis/floor_excluded/phi
```

Cross-judge (the failed-edit selection by the other judge):

```bash
python -m scripts.judge_agreement --scores "$R/nuis_*/scores_*.parquet" \
    --out $R/agreement
```

`committed/` re-runs the current stage 4 on the two committed A10 runs
(`main_2026-09-18`, `internvl3_2b_2026-09-26`) with the same flags, for
comparison; the committed CSVs in those folders are left untouched.

Figures: `python -m scripts.make_figures` and `python -m scripts.make_examples`
(the second needs `data/bases`) write to `results/figures/`.
