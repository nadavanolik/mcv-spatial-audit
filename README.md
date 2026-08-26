# Does spatial credit survive the reward model?

Auditing region-level feedback in instruction-based image editing.
MCV final project — Akkerman, Anolik, Barhoum, Kravets, Livneh.

## Hardware this is built for

Five VMs, each: **NVIDIA A10 24GB** (vGPU `A10-24Q`, full framebuffer),
~440GB RAM, 36 vCPU, **90G writable root disk, no sudo**.

Constraints baked into the code:

| Constraint | Consequence |
|---|---|
| No shared filesystem (`/datashare` is read-only, uid=0) | Artefacts move via HF Hub; corrupted variants are **regenerated**, never transferred |
| `/mnt` root-owned | `HF_HOME` lives in `$HOME`, not `/mnt` |
| No sudo | `opencv-python-headless` (GUI build needs `libGL.so.1` → apt); no system packages anywhere |
| A10 = SM 8.6 | bf16, **not** fp8 — fp8 kernels need SM 8.9+ |
| A10-24Q leaves 21.37 of 23.72GiB free | `--gpu-util` has a narrow two-sided window ~(0.861, 0.901); default **0.89**, same on every VM |
| FLUX transformer is 23.8GB in bf16 | Stage 1 needs **sequential** CPU offload; model-level offload cannot fit and OOMs at step 0 |
| Driver caps at CUDA 12.8 | `torch` must be a **cu128** build; plain `pip install torch` fetches a cu13 wheel that reports "no CUDA device" |
| 90G disk | VMs are **role-specialised**: the editor VM never holds judges, judge VMs never hold the editor |
| `/dev/shm` = 217G, mode 1777 | Scratch for regenerated variants |

## Setup (every VM)

```bash
cd ~ && git clone <repo> mcv_project && cd mcv_project
python -m venv .venv && source .venv/bin/activate

echo 'export HF_HOME=$HOME/hf_cache'        >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1'   >> ~/.bashrc
source ~/.bashrc

bash scripts/setup.sh <role>    # judge | editor | coco | core
```

One command. It installs the right dependency set for your role, prints the
pinned versions the corruption bytes depend on, warns if more than one OpenCV
is installed, and finishes by running the determinism check and printing the
hash to compare against a teammate's.

Dependencies are split by role because the 90G disk means **no VM holds both
the editor and a judge**:

| File | Who | Adds |
|---|---|---|
| `requirements.txt` | everyone, incl. the no-GPU laptop | numpy/opencv/pillow/pandas/sklearn — pinned, determinism-critical |
| `requirements-judge.txt` | stage 3 VMs | `vllm` (which pins its own torch) |
| `requirements-editor.txt` | the one stage 1 VM | `diffusers`, `transformers`, `accelerate` |
| `requirements-coco.txt` | whoever runs stage 0 | `pycocotools` — ships manylinux wheels; no compiler needed |

Each role file pulls in `requirements.txt`, so you install one thing, not two.

**The fixture hash must match across all five VMs.** If it does not, your
shards are not comparable and every downstream number is meaningless. On this
container the reference value is `776feeddd281fa726195bf504c7b19c8`; yours may
differ if a pinned version drifts, but all five of yours must agree.

## Pipeline

```
stage 0  COCO filter        CPU     once     -> data/bases/{regions,masks,source}
stage 1  editing (FLUX)     GPU     once     -> data/bases/*/edit.png   [EDITOR VM ONLY]
         ---- tar + upload ~300MB to HF Hub. The only large transfer. ----
build_manifest              CPU     once     -> out/manifest.parquet
stage 2  corruption         CPU     per-VM   -> /dev/shm  (regenerated, never shipped)
stage 3  judging (vLLM)     GPU     per-VM   -> out/scores_shard{k}.parquet
stage 4  analysis           CPU     once     -> out/analysis/
```

Measured on one A10 (2026-08-26): stage 1 **189s/image**, stage 3
**7.9s/request** — `main` is ~3.1h/VM sharded five ways.

Stage 2 is a pure function of `(base edit, mask, manifest row)`, which is why
the multi-gigabyte corrupted set never crosses the network. Stage 1 is **not**
reproducible from a seed — diffusion drifts across library versions — so its
output is an immutable artefact generated exactly once.

## Check before you spend

Every expensive stage has a cheap dry run that fetches no weights and needs no
GPU. Use them — each was added after the expensive path failed for a reason the
cheap one could have caught.

```bash
python -m src.stage0_coco --coco .../instances_val2017.json --survey   # yield, no images needed
python -m src.stage1_edit --preflight        # API, auth, gating, disk, VRAM fit
python -m src.stage3_judge ... --dry-run     # builds every request, never imports vLLM
python -m scripts.diagnose_parse --scores out/scores_shard0.parquet    # why responses failed
```

## Editor VM: a second venv

diffusers and vLLM pin different torch builds, so the editor cannot share the
judge's environment:

```bash
python -m venv .venv-editor && source .venv-editor/bin/activate
pip install -r requirements-editor.txt
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
hf auth login        # FLUX.1-Kontext-dev is gated; accept the licence on its model page
python -m src.stage1_edit --preflight
python -m scripts.smoke_edit                 # one real edit on a synthetic image
```

## Run the pilot

```bash
python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
    --images data/coco/val2017 --out data/bases --n 100
python -m src.stage1_edit --limit 5          # ~190s per image
python -m src.build_manifest --profile pilot
SHARD=0 OF=1 bash scripts/run_shard.sh
python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' --all-readouts
```

**Look at the axis table before anything else.** Corruption should push
`sc_preserve` down while `sc_success` holds — the edit still follows the
instruction, it is just damaged. If neither moves, the audit has no signal to
measure and you need to know in week 1, not week 4.

## Role split

| VM | Owner | Role |
|---|---|---|
| 0 | | **Editor VM**: stages 0+1, owns `bases.tar.gz`. Then joins judging. |
| 1 | | Judge harness + prompt fidelity (`src/judge_prompt.py`) |
| 2 | | Corruption engine + manifest |
| 3 | | Analysis + figures |
| 4 | | Second judge family + nuisance/exploitability tests |

All five run `scripts/run_shard.sh` with their own `SHARD` once stage 3 starts.

## Before you report anything

`src/judge_prompt.py` now carries A.4.3 **verbatim** from arXiv:2606.26872.
Three caveats belong in the report rather than buried in a docstring:

- **The PQ prompt is reconstructed, not verbatim.** The paper shows SFReward's
  PQ *output* (A.4.4) but never its PQ *prompt*.
- **How the instruction and region list are appended is ours.** A.4.3 says only
  "you will be provided with pre-identified editing regions" and never shows
  the injection format.
- **SFReward is a fine-tuned model**; A.4.3 is the prompt that labelled its
  training data with a Gemini-3-Pro teacher. We apply it to *base* Qwen3-VL-8B,
  so we audit the prompt-based protocol, not the released reward model.

One finding falls straight out of Equation (3): the region reward is
`sqrt(phi(IF) * AES) / C`, where `AES = min(PQ)` is a single **image-level**
term multiplying every region of that image. Part of each "region" reward is
global by construction, before any judge behaviour is measured. Stage 4 reports
`reward` and `phi` side by side precisely so this dilution is visible.

## Timeline to 30.9

- **Week 1** — setup, COCO filter, pilot, judge histogram. Go/no-go.
- **Week 2** — 100 bases edited + shipped; manifest frozen; localization + redundancy on `main`.
- **Week 3** — second judge family; nuisance + exploitability tests.
- **Week 4** — figures, LaTeX, repo cleanup.
- **Week 5** — buffer + the 5-minute talk. Do not plan work here.

The `full_cross` profile and the "Optimal Reward ∆" stretch goal are explicitly
out of scope unless week 3 finishes early.
