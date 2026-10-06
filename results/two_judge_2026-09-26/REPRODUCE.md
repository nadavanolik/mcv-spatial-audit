# Reproducing the two-judge numbers

The `main` manifest (150 bases, 5,236 variants) judged by a second model,
InternVL3-2B, and analysed together with the Qwen3-VL-8B run. The raw InternVL
parquets are in [`../internvl3_2b_2026-09-26/`](../internvl3_2b_2026-09-26/);
the Qwen ones are in [`../main_2026-09-18/`](../main_2026-09-18/). No GPU is
needed for anything below. Run from the repo root.

`results/*/scores_shard*.parquet` matches exactly those ten files, five per
judge, so one glob loads both judges.

## Reward readout (this folder)

```bash
python -m src.stage4_analyze \
    --scores 'results/*/scores_shard*.parquet' \
    --out results/two_judge_2026-09-26 \
    --all-readouts --min-control 0
```

Floor-excluded, like the `main` headline. `localization_reward.csv` is the
source of the report's localization figure. `drift_robustness.csv` additionally
needs `--drift-csv data/bases/edit_drift.csv`, which ships in `bases.tar.gz`
(see the top-level README), not in this repo.

## phi readout (`phi/`)

```bash
python -m src.stage4_analyze \
    --scores 'results/*/scores_shard*.parquet' \
    --out results/two_judge_2026-09-26/phi \
    --col phi --all-readouts --min-control 0
```

Coherence is read here, on `phi`, for the reason given in
[`../main_2026-09-18/REPRODUCE.md`](../main_2026-09-18/REPRODUCE.md).

## Image-level PQ response, InternVL

```bash
python -m scripts.pq_response \
    --scores 'results/internvl3_2b_2026-09-26/scores_shard*.parquet' \
    --out results/two_judge_2026-09-26/pq_response_internvl.csv
```

The Qwen equivalent is `../main_2026-09-18/pq_response.csv`.

All of the above were re-run from the committed parquets on 2026-10-06 and
matched the committed CSVs.
