# Does spatial credit survive the reward model?

Auditing region-level feedback in instruction-based image editing.
MCV final project — Akkerman, Anolik, Barhoum, Kravets, Livneh.

2026 RL post-training methods for image editing (SpatialFlow-GRPO, Edit-GRPO,
RC-GRPO-Editing, SpatialReward) replaced whole-image rewards with per-region
scores from a VLM judge. This repository asks whether the number attached to a
region actually reflects that region's content.

The method: take an edited image, corrupt exactly one region, ask the judge to
score every region. We hold per-region ground truth the judge does not, so any
mismatch between where we corrupted and where the score reacts is directly
measurable. Inference only — no training.

**Status:** pipeline verified end to end; the go/no-go pilot returned GO. See
[`TEAM_BRIEF.md`](TEAM_BRIEF.md) for current state, findings and assignments.

## Repository layout

```
src/schema.py             manifest schema, variant_id, seed derivation, hash sharding
src/corruptions.py        5 seeded feathered degradations (determinism-critical)
src/judge_prompt.py       A.4.3 prompt verbatim, JSON schemas, Eq. (3) reward
src/stage0_coco.py        COCO instance-seg filter -> multi-region base specs
src/stage1_edit.py        FLUX Kontext editing            [editor VM]
src/build_manifest.py     expand base specs into the design matrix
src/stage2_corrupt.py     regenerate this VM's shard into /dev/shm
src/stage3_judge.py       sharded vLLM judging            [judge VM]
src/stage4_analyze.py     measurement quality, tie rate, coherence, AUROC,
                          leakage matrix, redundancy, noise floor

scripts/setup.sh          one-command bootstrap for a role, then the hash check
scripts/run_shard.sh      one VM's share of stages 2+3
scripts/smoke_edit.py     one real FLUX edit on a synthetic image   [editor VM]
scripts/smoke_judge.py    one real judge call on synthetic images   [judge VM]
scripts/diagnose_parse.py why judge responses failed, from a parquet     [CPU]
scripts/verify_corruption.py  did the corruption damage the image, and
                          only inside the mask?                         [CPU]
scripts/verify_determinism.sh  cross-VM hash check

tests/                    4 suites, all CPU, all run in seconds
config.yaml               pilot / main / full_cross profiles
```

## Requirements

Five VMs, each: **NVIDIA A10 24GB** (vGPU `A10-24Q`), ~440GB RAM, 36 vCPU,
90G writable root disk, **no sudo**. Python 3.12.

Constraints baked into the code. Deviating from them breaks things in ways that
are hard to diagnose:

| Constraint | Consequence |
|---|---|
| No shared filesystem between VMs | Artefacts move via HF Hub; corrupted variants are **regenerated**, never transferred |
| No sudo | `opencv-python-headless` only — the GUI build needs `libGL.so.1` via apt |
| A10 = SM 8.6 | bf16, **not** fp8 — fp8 kernels need SM 8.9+ |
| A10-24Q leaves 21.37 of 23.72GiB free | `--gpu-util` has a narrow two-sided window ~(0.861, 0.901); default **0.89**, identical on every VM |
| FLUX transformer is 23.8GB in bf16 | Stage 1 needs **sequential** CPU offload; model-level cannot fit |
| Driver caps at CUDA 12.8 | `torch` must be a **cu128** build |
| 90G root shared by weights and data | Leave `HF_HOME` unset. VMs are role-specialised: no VM holds both the editor and a judge |
| `/dev/shm` = 217G, RAM-backed | Scratch for regenerated variants; nothing large goes on `/` |

## Installation

```bash
git clone <repo> mcv-spatial-audit && cd mcv-spatial-audit
python -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh <role>      # judge | editor | coco | core
```

`setup.sh` installs the dependency set for your role, prints the pinned
versions the corruption bytes depend on, warns if more than one OpenCV is
present, and finishes by running the determinism check.

| File | Who | Adds |
|---|---|---|
| `requirements.txt` | everyone, incl. the no-GPU laptop | numpy / opencv / pillow / pandas / sklearn — pinned, determinism-critical |
| `requirements-judge.txt` | stage 3 VMs | `vllm` |
| `requirements-editor.txt` | the one stage 1 VM | `diffusers`, `transformers`, `accelerate` |
| `requirements-coco.txt` | whoever runs stage 0 | `pycocotools` |

Each role file pulls in `requirements.txt`, so installation is one command.

**The editor VM needs a second venv**, because diffusers and vLLM pin different
torch builds:

```bash
python -m venv .venv-editor && source .venv-editor/bin/activate
pip install -r requirements-editor.txt
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
hf auth login     # FLUX.1-Kontext-dev is gated; accept the licence on its model page
```

## Pipeline

```
stage 0  COCO filter        CPU     once     -> data/bases/{regions,masks,source}
stage 1  editing (FLUX)     GPU     once     -> data/bases/*/edit.png   [editor VM]
         ---- tar + upload ~300MB to HF Hub. The only large transfer. ----
build_manifest              CPU     once     -> out/manifest.parquet
stage 2  corruption         CPU     per-VM   -> /dev/shm  (regenerated, never shipped)
stage 3  judging (vLLM)     GPU     per-VM   -> out/scores_shard{k}.parquet
stage 4  analysis           CPU     once     -> out/analysis/
```

Stage 2 is a pure function of `(base edit, mask, manifest row, seed)`, which is
why the multi-gigabyte corrupted set never crosses the network. Stage 1 is
**not** reproducible from a seed — diffusion drifts across library versions — so
its output is an immutable artefact generated exactly once and described by
`data/bases/stage1_provenance.json`.

Measured on one A10: stage 1 **189s/image**, stage 3 **2.84s/request** at greedy
decoding. The `main` profile is **~1.1h/VM** sharded five ways.

## Usage

```bash
# stages 0 + 1 — editor VM only, once
python -m src.stage0_coco --coco data/coco/annotations/instances_val2017.json \
    --images data/coco/val2017 --out data/bases --n 100
python -m src.stage1_edit --limit 100

# everyone, once bases.tar.gz is distributed
python -m src.build_manifest --profile pilot          # or main
SHARD=<0-4> OF=5 bash scripts/run_shard.sh
python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' --all-readouts
```

`run_shard.sh` runs stages 2 and 3, and deletes its `/dev/shm` scratch when it
finishes. Set `KEEP_SCRATCH=1` while iterating on stage 3.

In stage 4's output, read `=== does the score move at all? ===` first. It gives
the share of damaged regions whose score is byte-identical to their clean
control, split by target and non-target. AUROC is 0.5 both for a judge that
never reacts and one that reacts at random; only the tie rate separates those.

### Check before you spend

Every expensive stage has a cheap dry run that fetches no weights and needs no
GPU. Each was added after the expensive path failed for a reason the cheap one
would have caught.

```bash
python -m src.stage0_coco --coco … --survey       # base yield, no images needed
python -m src.stage1_edit --preflight             # API, auth, gating, disk, VRAM fit
python -m src.stage3_judge … --dry-run            # builds every request, never imports vLLM
python -m scripts.diagnose_parse --scores …       # why judge responses failed
python -m scripts.verify_corruption --manifest …  # was the damage real, and only in the mask?
```

`verify_corruption` is the one to run before claiming the judge ignored
something — it separates "the judge did not react" from "there was nothing to
react to".

## Testing

```bash
python tests/test_determinism.py   # 5 determinism properties
python tests/test_stage0.py        # selection logic, via a stub COCO
python tests/test_stage4.py        # 3 synthetic judges with known behaviour
python tests/test_syntax.py        # everything parses; GPU modules import without torch
```

All four are CPU-only and finish in seconds. Keep them passing.

**The cross-VM fixture hash must match on all five VMs.** If it does not, the
shards are not comparable and every downstream number is meaningless. The Linux
reference is `776feeddd281fa726195bf504c7b19c8`. A Windows laptop prints a
different value — expected, different libjpeg, and not a valid reference to
compare a VM against.

## Reproducibility notes

`src/judge_prompt.py` carries Appendix A.4.3 of arXiv:2606.26872 **verbatim**.
Three deviations belong in any write-up:

- **The PQ prompt is reconstructed.** The paper shows SFReward's PQ *output*
  (A.4.4) but never its PQ *prompt*.
- **The instruction and region-list injection format is ours.** A.4.3 says only
  "you will be provided with pre-identified editing regions".
- **SFReward is a fine-tuned model** (Qwen3-VL-8B + SFReward-14K); A.4.3 is the
  prompt that labelled its training data. We apply it to *base* Qwen3-VL-8B, so
  this audits the prompt-based protocol, not the released reward model.

Judge output is constrained to a JSON schema. That fixes format only — every
score in 0–25 stays reachable — but it is a deviation from free generation and
should be stated. Without it the judge silently drops regions and covers only
~43% of what it was asked to score.

One property falls straight out of Equation (3): the region reward is
`sqrt(phi(IF) * AES) / C`, where `AES = min(PQ)` is a single **image-level**
term multiplying every region of that image. Part of each "region" reward is
global by construction, before any judge behaviour is measured. Stage 4 reports
`reward` and `phi` side by side so this is visible.
