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
| `requirements-coco.txt` | whoever runs stage 0 | `pycocotools` — builds from source, needs gcc |

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

Stage 2 is a pure function of `(base edit, mask, manifest row)`, which is why
the multi-gigabyte corrupted set never crosses the network. Stage 1 is **not**
reproducible from a seed — diffusion drifts across library versions — so its
output is an immutable artefact generated exactly once.

## Run the pilot first

```bash
python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
    --images data/coco/val2017 --out data/bases --n 5
python -m src.stage1_edit --bases data/bases --limit 5
python -m src.build_manifest --profile pilot
SHARD=0 OF=1 bash scripts/run_shard.sh
```

~240 requests, minutes to run. **Look at the printed score histogram before
doing anything else.** If the judge returns 4/5 for every region regardless of
damage, the audit has no signal to measure and you need to redesign in week 1,
not week 4. That is the single highest-value thing you can learn right now.

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
