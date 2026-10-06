# Results

Every number in the report comes from a CSV in this folder, and every CSV
regenerates on a laptop with no GPU from the raw score parquets committed next
to it. Each run folder has a `REPRODUCE.md` with the exact commands.

## Runs

| Folder | What it is | Judge(s) | Size |
|---|---|---|---|
| [`main_2026-09-18/`](main_2026-09-18/) | The main audit: 150 COCO photos, 476 regions, 5,236 variants (5 corruptions), greedy | Qwen3-VL-8B | 22,176 scored regions |
| [`internvl3_2b_2026-09-26/`](internvl3_2b_2026-09-26/) | Raw scores only: the same 5,236 variants judged by a second model | InternVL3-2B | 22,176 scored regions |
| [`two_judge_2026-09-26/`](two_judge_2026-09-26/) | Analysis of both judges side by side, built from the two folders above | both | - |
| [`exploit_150_2026-09-29/`](exploit_150_2026-09-29/) | Exploitability: the 476 clean edits re-judged under 4 presentations (`baseline`, `noimg`, `enhance`, `enhance_target`), plus cross-judge agreement | both | 8,064 rows per judge |
| [`nuisance_pilot_2026-09-15/`](nuisance_pilot_2026-09-15/) | Early pilot of the presentation sweep on 5 photos. Superseded by `exploit_150` for the exploit axes; still the only run of `shuffle`/`subset`/`box` | Qwen3-VL-8B | 5 photos |
| [`figures/`](figures/) | Report figures and tables, drawn from the CSVs above by `scripts/make_figures.py` | - | - |

## Where each report number comes from

| Report section | Quantity | File |
|---|---|---|
| 3.1 Sensitivity | image-level PQ / AES change by corruption and severity | `main_2026-09-18/pq_response.csv` |
| 3.2 Localization | ROC-AUC, `reward`, floor-excluded | `two_judge_2026-09-26/localization_reward.csv` |
| 3.2 Localization | ROC-AUC, `phi` | `two_judge_2026-09-26/phi/localization_phi.csv` |
| 3.2 Localization | share of damaged vs untouched regions whose score did not move | `main_2026-09-18/floor_excluded/sensitivity.csv` |
| 3.3 Global coupling | redundancy R^2 on `phi` | `two_judge_2026-09-26/redundancy_phi.csv` |
| 3.3 Global coupling | coherence (mixed vs expected-if-independent), `phi` | `two_judge_2026-09-26/phi/coherence.csv` |
| 3.3 Robustness | layout-drift split | `main_2026-09-18/drift_robustness.csv` |
| Exploitability | blind (`noimg`) and cosmetic gains, with bootstrap intervals | `exploit_150_2026-09-29/{qwen,internvl}/exploitability.csv` |
| Judge agreement | Cohen's kappa, Jaccard, correlation between the two judges | `exploit_150_2026-09-29/agreement/` |

`scripts/make_figures.py` writes `figures/fig_localization.png`,
`figures/fig_tie_rate.png` and `figures/tab_tie_rate.tex`:

```bash
python -m scripts.make_figures
```

## Conventions

- **Floor-excluded** means regions whose clean control already scored 0 are
  dropped (`--min-control 0`). Such a region cannot drop when damaged, so it is
  a guaranteed tie. The headline numbers are floor-excluded; the CSVs directly
  in `main_2026-09-18/` (not under `floor_excluded/`) include every region.
- **`reward`** is the paper's Equation (3), `sqrt(phi * AES) / C`. **`phi`** is
  its per-region part, `min(success, preserve)` on a 0-25 scale, without the
  image-level AES factor that every region of an image shares.
- Intervals in `exploit_150_2026-09-29/` are 95% bootstrap over photographs,
  not regions.
- Full interpretation and caveats: [`../docs/FINDINGS.md`](../docs/FINDINGS.md).
