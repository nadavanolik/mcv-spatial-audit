# Nuisance axes at 150 bases, 2026-10-14

Qwen3-VL-8B on `mcvgpu2025s-0004`, 2026-10-14 19:02-20:42, then InternVL3-2B
21:44-22:29 (`--gpu-util 0.60`; a first attempt at 21:34 died when the VM went
down, and the condition restarted from scratch), repo at `b0dba7e`
(`src/` byte-identical to `5431dd4`, which produced the exploit run). Controls
only: the `nuisance150` manifest (476 clean controls, `sha256[:16] =
8f5c7b61541fda69`), greedy, one VM, three presentations: `shuffle`, `subset`,
`box`.

This supersedes `results/archive/nuisance_pilot_2026-09-15/` for the nuisance axes
(5 photographs there, 150 here).

Two inputs come from earlier runs, deliberately:

- **`scores_{qwen,internvl}_baseline.parquet` are byte copies** of the same
  files in `results/archive/exploit_150_2026-09-29/` (same VM, same manifest, same
  `src/`). Every nuisance delta pairs against its judge's copy.
- **The damage reference comes from `results/archive/main_2026-09-18/`** (Qwen) and
  `results/archive/internvl3_2b_2026-09-26/` (InternVL), which judged the same 150
  photographs, same controls, same model. A nuisance delta needs
  only clean controls; only its denominator needs damaged images, and `main`
  already measured that. Run-to-run noise does not distort the pairing: identical
  requests disagree as often within one run as across runs and VMs.

## The run (needs a GPU VM)

```bash
python -m src.build_manifest --profile nuisance150 --out out/manifest_nuisance150.parquet
python -m src.stage2_corrupt --manifest out/manifest_nuisance150.parquet \
    --bases data/bases --out /dev/shm/mcv/nuisance --shard 0 --of 1
for P in shuffle subset box; do
  python -m src.stage3_judge --manifest out/manifest_nuisance150.parquet \
      --bases data/bases --variants /dev/shm/mcv/nuisance \
      --model Qwen/Qwen3-VL-8B-Instruct --shard 0 --of 1 --presentation "$P" \
      --temperature 0 --n-samples 1 --out "out/nuisance/scores_qwen_$P.parquet"
done
```

Stage 2 and stage 3 must share one login session (`RemoveIPC=yes` empties
`/dev/shm` when the last session ends). Use tmux. 25-43 min per condition
including engine start (`shuffle` slowest).

## The numbers (CPU, no GPU, reproduces the committed CSVs)

```bash
python -m scripts.nuisance_report \
    --scores 'results/archive/nuisance_150_2026-10-14/scores_qwen_*.parquet' \
    --damage-scores 'results/archive/main_2026-09-18/scores_shard*.parquet' \
    --out results/archive/nuisance_150_2026-10-14

# the same on phi, the per-region term without the image-level AES factor
python -m scripts.nuisance_report \
    --scores 'results/archive/nuisance_150_2026-10-14/scores_qwen_*.parquet' \
    --damage-scores 'results/archive/main_2026-09-18/scores_shard*.parquet' \
    --col phi --out results/archive/nuisance_150_2026-10-14/phi

# the HEADLINE: floor-excluded, as the main run's numbers are (add --col phi
# and /phi for the phi version)
python -m scripts.nuisance_report \
    --scores 'results/archive/nuisance_150_2026-10-14/scores_qwen_*.parquet' \
    --damage-scores 'results/archive/main_2026-09-18/scores_shard*.parquet' \
    --min-control 0 --out results/archive/nuisance_150_2026-10-14/floor_excluded
```

InternVL, the same four commands with `scores_internvl_*`,
`--damage-scores 'results/archive/internvl3_2b_2026-09-26/scores_shard*.parquet'` and
`--out results/archive/nuisance_150_2026-10-14/internvl[/phi|/floor_excluded[/phi]]`.
Run each judge separately: the output tables carry no judge column. On the VM
the InternVL run is the same loop with `--model OpenGVLab/InternVL3-2B
--gpu-util 0.60` and `scores_internvl_$P.parquet`.

Quote `floor_excluded/`. A region whose clean score is already 0 can rise under
a packaging change but cannot fall under damage, so the all-region cut tilts the
ratio toward packaging: `box` is 1.56x damage there and 1.12x [0.82, 1.49] once
floored regions are dropped.

## What each file is

- `scores_qwen_*.parquet` - raw judge outputs, one row per (variant, region).
- `report.txt` - the full console report (`reward`); `phi/report.txt` the same
  on `phi`.
- `nuisance.csv` - per presentation against baseline: pooled mean |delta|
  with its ratios to real damage (`vs_damage`) and to the noise floor
  (`vs_floor`); the per-photograph mean with a 95% bootstrap CI over
  photographs (`photo_*`); and `ratio` [`ratio_lo`, `ratio_hi`], per-photo
  nuisance mean over per-photo damage mean with a CI that resamples the shared
  photographs, so the damage side's uncertainty is included. Quote `ratio`.
- `damage_reference.csv` - what real damage does to the damaged region, from
  `main`.
- `identity_null.csv` - `shuffle` split into variants that kept the canonical
  order (request byte-identical to baseline, judged in the shuffle run) and
  variants actually reordered. The first is the noise in exactly the nuisance
  pairing; for InternVL it is the comparison to use, since its noise is larger
  between runs than within one.
- `internvl/` - the same files for InternVL3-2B. Read them against its own
  noise only: its `ratio` to damage is not meaningful (its damage effect is ~2x
  its noise floor; it fails the perception precondition).
- `slot_effect.csv` - under `shuffle`, score by list position; `within` is
  position with each region compared against itself, which removes the
  confound that slots 3-4 exist only on 4-5 region photographs.
- `coverage.csv` - parse rate per condition (100% everywhere).
- `pq_by_presentation.csv` - image-level PQ per condition.

The noise floor is the mean |delta| between byte-identical baseline requests in
the same greedy run: a photo's 3-5 controls are the same request, since a clean
control's prompt does not depend on its target. See `rerun_floor()` in
`scripts/nuisance_report.py`.
