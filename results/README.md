# Results

Every number in the report comes from a CSV in this folder, and every CSV
regenerates on a laptop with no GPU from the raw score parquets committed next
to it. Each run folder has a `REPRODUCE.md` with the exact commands.

## Runs

| Folder | What it is | Judge(s) | Size |
|---|---|---|---|
| [`main8b_2026-10-22/`](main8b_2026-10-22/) | **The run the report is written from.** The main audit and all seven presentations on the clean edits, boxes in the 0-1000 frame; plus a pixel-box Qwen re-run and the stimulus checks | Qwen3-VL-8B, InternVL3-8B | 22,176 scored regions per main run |
| [`main_2026-09-18/`](main_2026-09-18/) | The main audit: 150 COCO photos, 476 regions, 5,236 variants (5 corruptions), greedy | Qwen3-VL-8B | 22,176 scored regions |
| [`internvl3_2b_2026-09-26/`](internvl3_2b_2026-09-26/) | Raw scores only: the same 5,236 variants judged by a second model | InternVL3-2B | 22,176 scored regions |
| [`two_judge_2026-09-26/`](two_judge_2026-09-26/) | Analysis of both judges side by side, built from the two folders above | both | - |
| [`exploit_150_2026-09-29/`](exploit_150_2026-09-29/) | Exploitability: the 476 clean edits re-judged under 4 presentations (`baseline`, `noimg`, `enhance`, `enhance_target`), plus cross-judge agreement | both | 8,064 rows per judge |
| [`nuisance_150_2026-10-14/`](nuisance_150_2026-10-14/) | Nuisance: the 476 clean edits re-judged under `shuffle`, `subset`, `box`, against the exploit run's baseline; damage reference from `main` | both (InternVL under `internvl/`) | 476 clean controls per condition |
| [`nuisance_pilot_2026-09-15/`](nuisance_pilot_2026-09-15/) | Early pilot of the presentation sweep on 5 photos. Superseded by `exploit_150` (exploit axes) and `nuisance_150` (nuisance axes) | Qwen3-VL-8B | 5 photos |
| [`figures/`](figures/) | Report figures and tables, drawn from the CSVs above by `scripts/make_figures.py` | - | - |

## Where each report number comes from

All from [`main8b_2026-10-22/`](main8b_2026-10-22/). `<judge>` is
`main_qwen_norm` (Qwen3-VL-8B) or `main_ivl8` (InternVL3-8B); `<nuis>` is
`nuis_qwen` or `nuis_ivl8`.

| Report | Quantity | File |
|---|---|---|
| Fig. 1a, Table 2 | image-level AES change by corruption and setting, with photo-bootstrap intervals | `<judge>/analysis/pq_response.csv` |
| Fig. 1b, Table 1 | AUROC on `phi`, floor-excluded, per corruption | `<judge>/analysis/phi/localization_by_corruption_phi.csv`; per-region deltas in `phi/deltas.parquet` |
| Table 1 | share of corrupted vs untouched regions whose `phi` did not move | `<judge>/analysis/phi/sensitivity.csv` |
| Results, robustness | untouched regions the damage reached; layout drift | `<judge>/analysis/phi/{overlap_split,drift_robustness}.csv` |
| Results, robustness | pixel vs 0-1000 boxes, Qwen | `compare/frame/headline.csv` |
| Results, coupling | mixed variants vs the permutation null | `<judge>/analysis/phi/coherence.csv` |
| Results, coupling | R^2 of an untouched region's delta on the others' | `<judge>/analysis/phi/redundancy_delta_phi.csv` |
| Results, noise floor | disagreement between identical clean-control requests | `<judge>/analysis/phi/noise_floor.csv` |
| Table 3 | nuisance ratio (`shuffle`, `subset`, `box`) against damage, floor-excluded | `<nuis>/analysis/floor_excluded/phi/nuisance.csv` |
| Results, exploitability | `noimg`, `enhance`, `enhance_target` gains on failed edits | `<nuis>/analysis/floor_excluded/phi/exploitability.csv` |
| Results, exploitability | the same with failed edits picked by the other judge | `agreement/crossjudge_exploit.csv` |
| Fig. 2-3 | one photograph, pipeline and per-region delta | `figures/fig_example_*.png` (`scripts/make_examples.py`) |

The earlier folders below hold the A10 runs (pixel boxes, InternVL3-2B as the
second judge) that preceded the report. Their numbers point the same way but
are not the ones printed in it.

`scripts/make_figures.py` writes `figures/fig_perception_localization.{png,pdf,csv}`
(Fig. 1) and `scripts/make_examples.py` the two example figures (needs
`data/bases`). `fig_localization.png`, `fig_tie_rate.png` and `tab_tie_rate.tex`
are from the earlier A10 runs.

```bash
python -m scripts.make_figures
python -m scripts.make_examples
```

## Conventions

- **Floor-excluded** means regions whose clean control already scored 0 are
  dropped (`--min-control 0`). Such a region cannot drop when damaged, so it is
  a guaranteed tie. The headline numbers are floor-excluded; the CSVs directly
  in `main_2026-09-18/` (not under `floor_excluded/`) include every region.
  In `main8b_2026-10-22/`, every main-run analysis is floor-excluded and the
  nuisance reports carry both cuts (`all/`, `floor_excluded/`).
- **`reward`** is the paper's Equation (3), `sqrt(phi * AES) / C`. **`phi`** is
  its per-region part, `min(success, preserve)` on a 0-25 scale, without the
  image-level AES factor that every region of an image shares.
- Intervals are 95% bootstrap over photographs, not regions.
- Full interpretation and caveats: [`../docs/FINDINGS.md`](../docs/FINDINGS.md).
