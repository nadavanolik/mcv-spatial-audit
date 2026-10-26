# Results

Every number in the report comes from a CSV in
[`main8b_2026-10-22/`](main8b_2026-10-22/), and every CSV regenerates on a
laptop with no GPU from the raw score parquets committed next to it. The exact
commands are in [`main8b_2026-10-22/REPRODUCE.md`](main8b_2026-10-22/REPRODUCE.md).

| Folder | Contents |
|---|---|
| [`main8b_2026-10-22/`](main8b_2026-10-22/) | The run the report is written from: Qwen3-VL-8B and InternVL3-8B on the main design (150 photos, 476 regions, 5,236 variants) and on the 476 clean edits under all seven presentations, boxes in the 0-1000 frame. Also a pixel-box Qwen re-run and the stimulus checks. |
| [`figures/`](figures/) | Figure 1 and the two appendix figures, written by `scripts/make_figures.py` and `scripts/make_examples.py`. `paper/figures/` holds copies. |
| [`archive/`](archive/) | Earlier runs on the same photos (pixel boxes, InternVL3-2B as the second judge). They point the same way but are not the numbers in the report. Each keeps its own `REPRODUCE.md`. |

## Where each report number comes from

Paths are under `main8b_2026-10-22/`. `<judge>` is `main_qwen_norm`
(Qwen3-VL-8B) or `main_ivl8` (InternVL3-8B); `<nuis>` is `nuis_qwen` or
`nuis_ivl8`.

| Report | Quantity | File |
|---|---|---|
| Fig. 1a, Table 2 | image-level AES change by corruption and setting | `<judge>/analysis/pq_response.csv` |
| Fig. 1b, Table 1 | AUROC on `phi`, floor-excluded, per corruption | `<judge>/analysis/phi/localization_by_corruption_phi.csv`; per-region deltas in `phi/deltas.parquet` |
| Table 1 | share of corrupted vs untouched regions whose `phi` did not move | `<judge>/analysis/phi/sensitivity.csv` |
| Results, robustness | untouched regions the damage reached; layout drift | `<judge>/analysis/phi/{overlap_split,drift_robustness}.csv` |
| Results, robustness | pixel vs 0-1000 boxes, Qwen | `compare/frame/headline.csv` |
| Results, coupling | mixed variants vs the permutation null | `<judge>/analysis/phi/coherence.csv` |
| Results, coupling | R^2 of an untouched region's delta on the others' | `<judge>/analysis/phi/redundancy_delta_phi.csv` |
| Results, noise floor | disagreement between identical clean-control requests | `<nuis>/analysis/report_floor_excluded_phi.txt` ("noise floor") |
| Table 3 | nuisance ratio (`shuffle`, `subset`, `box`) against damage | `<nuis>/analysis/floor_excluded/phi/nuisance.csv` |
| Results, nuisance | identity shuffles vs reordered lists | `<nuis>/analysis/floor_excluded/phi/identity_null.csv` |
| Results, exploitability | blind default and `noimg`, `enhance`, `enhance_target` gains (overall; on failed edits) | `<nuis>/analysis/floor_excluded/phi/{exploitability,exploit_split}.csv` |
| Results, exploitability | the same with failed edits picked by the other judge | `agreement/crossjudge_exploit.csv` |
| Fig. 2-3 | one photograph, pipeline and per-region delta | `../figures/fig_example_*.png` |

## Conventions

- **Floor-excluded** means regions whose clean control already scored 0 are
  dropped (`--min-control 0`). Such a region cannot drop when damaged, so it is
  a guaranteed tie. Every main-run analysis is floor-excluded; the nuisance
  reports carry both cuts (`all/`, `floor_excluded/`).
- **`reward`** is the paper's Equation (3), `sqrt(phi * AES) / C`. **`phi`** is
  its per-region part, `min(success, preserve)` on a 0-25 scale, without the
  image-level AES factor that every region of an image shares.
- Intervals are 95% bootstrap over photographs, not regions, and are not
  corrected for multiple comparisons.
- Numbers computed on Windows can differ from the committed CSVs in the last
  floating-point digit (around 1e-15); they are otherwise identical.
