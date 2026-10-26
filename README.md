# Does spatial credit survive the reward model?

Auditing region-level feedback in instruction-based image editing.
MCV final project: Akkerman, Anolik, Barhoum, Kravets, Livneh.
Report: [`paper/final_report.pdf`](paper/final_report.pdf).

Recent RL methods for image editing (SpatialFlow-GRPO, Edit-GRPO,
RC-GRPO-Editing, SpatialReward) reward each region of an edit separately, with
scores from a VLM judge. We test whether the score attached to a region reflects
that region: corrupt exactly one region of an edited image, ask the judge to
score every region, and check where the score reacts. Inference only, no
training.

On 150 COCO photographs (5,236 variants), Qwen3-VL-8B and InternVL3-8B both
register the damage at the image level, but neither attributes it to the
damaged region (AUROC 0.52-0.54), and reordering the region list in the prompt
moves a region's score more than corrupting it does.

## Reproduce the report's numbers (CPU only)

The judging needs GPUs, but everything after it runs on a laptop from the
committed score parquets in a few minutes:

```bash
git clone https://github.com/nadavanolik/mcv-spatial-audit.git && cd mcv-spatial-audit
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt

R=results/main8b_2026-10-22

# localization, coupling and robustness on phi (Table 1); one run per judge
for TAG in main_qwen_norm main_ivl8; do
  python -m src.stage4_analyze --scores "$R/$TAG/scores_shard*.parquet" \
      --out out/$TAG/phi --col phi --all-readouts --min-control 0 \
      --overlap-csv $R/checks/overlap.csv
done

# nuisance and exploitability (Table 3), Qwen; use nuis_ivl8 + main_ivl8 for InternVL
python -m scripts.nuisance_report --scores "$R/nuis_qwen/scores_*.parquet" \
    --damage-scores "$R/main_qwen_norm/scores_shard*.parquet" \
    --col phi --min-control 0 --out out/nuis_qwen/phi

# Figure 1, written to results/figures/
python -m scripts.make_figures
```

[`results/README.md`](results/README.md) maps every number in the report to the
CSV it comes from, and
[`results/main8b_2026-10-22/REPRODUCE.md`](results/main8b_2026-10-22/REPRODUCE.md)
has the full command list, including the GPU runs.

## Repository layout

```
src/schema.py             manifest schema, variant ids, seeds, hash sharding
src/corruptions.py        the 5 seeded, feathered corruptions
src/judge_prompt.py       judge prompts, JSON schemas, Eq. (3) reward
src/presentation.py       nuisance / exploitability variants of a request
src/stage0_coco.py        COCO filter -> photos, regions, instructions
src/stage1_edit.py        FLUX.1 Kontext editing                     [GPU]
src/build_manifest.py     expand the photos into the design matrix
src/stage2_corrupt.py     render one shard's corrupted variants
src/stage3_judge.py       vLLM judging, schema-constrained           [GPU]
src/stage4_analyze.py     localization, ties, coupling, redundancy, noise floor

scripts/                  setup, sharded runner, analysis and figure scripts,
                          stimulus checks (see each file's docstring)
tests/                    5 CPU-only suites
config.yaml               design profiles: pilot / main / nuisance150 / full_cross
results/                  committed raw scores and every analysis table
paper/                    report source (LaTeX) and PDF
```

## Pipeline

```
stage 0  COCO filter        CPU   once     -> data/bases/{regions,masks,source}
stage 1  editing (FLUX)     GPU   once     -> data/bases/*/edit.png
         published to the HF Hub as bases.tar.gz (146MB)
build_manifest              CPU   once     -> out/manifest.parquet
stage 2  corruption         CPU   per VM   -> /dev/shm (regenerated, never copied)
stage 3  judging (vLLM)     GPU   per VM   -> out/scores_shard{k}.parquet
stage 4  analysis           CPU   once     -> out/analysis/
```

Stage 2 is a deterministic function of the edit, the mask and the manifest row,
so each machine renders its own shard instead of copying gigabytes of
variants. Stage 1 is not reproducible from a seed (diffusion output drifts
across library versions), so its output was generated once and published.

## Installation

The runs used five Azure VMs, each with an NVIDIA A10 24GB (vGPU), no sudo, and
no shared filesystem. Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh <role>      # core | judge | editor | coco
```

| File | For | Adds |
|---|---|---|
| `requirements.txt` | every machine | numpy, opencv-python-headless, pillow, pandas, scikit-learn (pinned; the corruption bytes depend on them) |
| `requirements-judge.txt` | stage 3 | vllm |
| `requirements-editor.txt` | stage 1 | diffusers, transformers, accelerate |
| `requirements-coco.txt` | stage 0 | pycocotools |

The editor needs its own venv, since diffusers and vLLM pin different torch
builds, and a CUDA 12.8 torch build:

```bash
python -m venv .venv-editor && source .venv-editor/bin/activate
pip install -r requirements-editor.txt
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
hf auth login     # FLUX.1-Kontext-dev is gated
```

## Data

The edited photographs are public on the Hub, so only the machine that runs
stage 0 needs COCO:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mcv-spatial-audit/mcv-spatial-audit', 'bases.tar.gz', repo_type='dataset'))"
tar xzf <the path it prints> -C data       # creates data/bases
```

To rebuild them from COCO instead, `bash scripts/stage0.sh` downloads the
train2017 annotations, fetches only the ~150 images that pass the filter, and
writes `data/bases`. Do not download a whole image split; the filter keeps about
one image in a hundred.

## Running the audit (GPU)

```bash
bash scripts/stage0.sh                                  # or download bases.tar.gz
python -m src.stage1_edit                               # editor VM, ~9h for 150 photos
python -m src.build_manifest --profile main
SHARD=<0-4> OF=5 bash scripts/run_shard.sh              # stages 2+3, one per VM
python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' --all-readouts \
    --min-control 0 --drift-csv data/bases/edit_drift.csv
```

The report's runs also passed `--bbox-frame norm1000` to stage 3, and for
InternVL3-8B `--max-model-len 8192 --max-dynamic-patch 6`; see
[`results/main8b_2026-10-22/REPRODUCE.md`](results/main8b_2026-10-22/REPRODUCE.md).

Notes:

- Stages 2 and 3 must run in one login session (use `tmux`). The VMs clear
  `/dev/shm` when a user's last session ends, so rendering in one `ssh` call
  and judging in the next leaves stage 3 with no inputs.
- Decoding is greedy. vLLM is still not bit-reproducible: identical requests
  disagree a few percent of the time, and that disagreement is the noise floor
  the analysis compares against.
- `--gpu-util` defaults to 0.89, which fits an 8B judge on the A10. A 2B judge
  needs about 0.60.
- The nuisance and exploitability conditions use `--profile nuisance150` (476
  clean edits) and `--presentation <name>`, one output file per presentation,
  analysed with `scripts/nuisance_report.py` rather than stage 4.
- Cheap checks before a long run: `stage1_edit --preflight`,
  `stage3_judge --dry-run` (builds every request without loading vLLM), and
  `scripts/verify_corruption.py` (was the damage real, and only inside the mask).

## Testing

```bash
python tests/test_determinism.py   # corruption determinism and locality
python tests/test_stage0.py        # photo selection, via a stub COCO
python tests/test_stage4.py        # synthetic judges with known behaviour
python tests/test_nuisance.py      # presentation variants and nuisance analysis
python tests/test_syntax.py        # every file parses; GPU modules import without torch
```

`bash scripts/verify_determinism.sh` prints a hash of all corruption outputs.
It must be identical on every machine that renders shards; on Linux it is
`776feeddd281fa726195bf504c7b19c8`. Windows prints a different value because of
a different libjpeg build.

## Deviations from the audited protocol

- We apply SpatialFlow-GRPO's region-scoring prompt (its Appendix A.4.3,
  verbatim in `src/judge_prompt.py`) to the **base** Qwen3-VL-8B and
  InternVL3-8B, not to its fine-tuned reward model.
- The paper does not print its image-quality prompt or how regions are inserted
  into the prompt; both are ours.
- Judge output is constrained to a JSON schema. This fixes the format only;
  without it the judge silently skips regions.
