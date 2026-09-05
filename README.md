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
src/presentation.py       nuisance / exploitability packaging axes
src/stage0_coco.py        COCO instance-seg filter -> multi-region base specs
src/stage1_edit.py        FLUX Kontext editing            [editor VM]
src/build_manifest.py     expand base specs into the design matrix
src/stage2_corrupt.py     regenerate this VM's shard into /dev/shm
src/stage3_judge.py       sharded vLLM judging            [judge VM]
src/stage4_analyze.py     measurement quality, tie rate, coherence, AUROC,
                          leakage matrix, redundancy, noise floor

scripts/setup.sh          one-command bootstrap for a role, then the hash check
scripts/stage0.sh         data/bases from nothing: annotations, images, select
scripts/run_shard.sh      one VM's share of stages 2+3
scripts/smoke_edit.py     one real FLUX edit on a synthetic image   [editor VM]
scripts/smoke_judge.py    one real judge call on synthetic images   [judge VM]
scripts/diagnose_parse.py why judge responses failed, from a parquet     [CPU]
scripts/nuisance_report.py  nuisance + exploitability, across presentations [CPU]
scripts/verify_corruption.py  did the corruption damage the image, and
                          only inside the mask?                         [CPU]
scripts/verify_determinism.sh  cross-VM hash check
scripts/verify_edit_drift.py  did stage 1 keep the layout the masks describe?

tests/                    5 suites, all CPU, all run in seconds
config.yaml               pilot / main / full_cross profiles

docs/DECISIONS.md         why the harness is what it is: measurements behind
                          every constraint, settled questions, bug history
docs/FINDINGS.md          what we measured about the judge, and its caveats
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

## Data

Only the machine running stage 0 needs COCO. Everyone else works from
`bases.tar.gz`, published on the Hub and public, so no token is needed:

```bash
cd <repo>
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mcv-spatial-audit/mcv-spatial-audit', 'bases.tar.gz', repo_type='dataset'))"
tar xzf <the path it prints> -C data       # creates data/bases
```

146MB: 150 bases, 476 regions, both `source.png` and `edit.png`, the masks,
`bases.json`, `stage1_provenance.json` and `edit_drift.csv`. That single
download replaces stages 0 and 1 for every machine that is not the editor VM.

**Never download an image split.** The selection filter keeps roughly 1 image
in 100, so fetching train2017's 18GB to keep ~150 files is 700x more transfer
than the job needs, and does not fit beside FLUX on a 90G root. Annotations
first, then only the images that actually qualify.

One command does all of it — annotations, only the qualifying images, and the
selection itself:

```bash
bash scripts/stage0.sh          # 150 bases from train2017; N=5 for a smoke run
```

It refuses to overwrite an existing `data/bases` (stage 0 output is regenerated,
never merged in place) and re-fetches only images that are missing, so an
interrupted run is safe to repeat. **Base specs are regenerated per machine, not
transferred**: stage 0 is a pure function of the annotations file, `config.yaml`
and `--seed`, so every VM that runs this gets the same 150 photographs, regions
and instructions. `build_manifest` prints a hash that proves it.

The same thing by hand, if you want to see the steps:

```bash
mkdir -p data/coco && cd data/coco

# Annotations for both splits, 241MB.
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip \
    annotations/instances_train2017.json annotations/instances_val2017.json
rm annotations_trainval2017.zip
cd ../..

# What would the filter yield? No images, no writes, ~1 minute.
python -m src.stage0_coco --coco data/coco/annotations/instances_train2017.json --survey

# Fetch ONLY the qualifying images. Ask for more than you need, so one failed
# download does not shift which images get used.
python -m src.stage0_coco --coco data/coco/annotations/instances_train2017.json \
    --n 200 --list-urls > /tmp/urls.txt
wget -q -P data/coco/train2017 -i /tmp/urls.txt      # ~200 files, ~30MB
```

val2017 (5,000 images) yields only ~46 bases under the category-uniqueness
rule, so **use train2017** — 118,287 images at the same rate is ~1,090. The
train/val distinction carries no meaning here: this audit trains nothing, and
both splits are equally public to the models involved.

## Pipeline

```
stage 0  COCO filter        CPU     once     -> data/bases/{regions,masks,source}
stage 1  editing (FLUX)     GPU     once     -> data/bases/*/edit.png   [editor VM]
         ---- tar + upload 146MB to HF Hub. The only large transfer. ----
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

Measured over the full 150-base run: stage 1 **213.1s/image** (8h53m total);
stage 3 **2.84s/request** at greedy decoding. Stage 1 is PCIe-bound under
sequential offload, so it varies by host - 191.4s on one A10, 213.1s on another.
The `main` profile is 150 bases: **~1.7h/VM** sharded five ways.

## Usage

```bash
# stage 0 — anywhere, CPU only, ~5 minutes. Run it on the editor VM before
# stage 1; run it anywhere else instead of copying data/bases around.
bash scripts/stage0.sh

# stage 1 — editor VM only, once, ~9h for 150 bases. Detach it: an ssh drop
# would otherwise kill a run that survives everything else.
nohup python -m src.stage1_edit > ~/stage1.log 2>&1 < /dev/null &

# everyone, once bases.tar.gz is distributed
python -m src.build_manifest --profile pilot          # or main
SHARD=<0-4> OF=5 bash scripts/run_shard.sh
python -m src.stage4_analyze --scores 'out/scores_shard*.parquet' --all-readouts \
    --drift-csv data/bases/edit_drift.csv --min-edge-iou 0.4
```

`run_shard.sh` runs stages 2 and 3, and deletes its `/dev/shm` scratch when it
finishes. Set `KEEP_SCRATCH=1` while iterating on stage 3.

Judging is **greedy** (`--temperature 0 --n-samples 1`), passed explicitly by
`run_shard.sh` and matching `stage3_judge`'s own defaults. Like `--gpu-util` it
must be identical on every VM, so there is no environment-variable override.
The one sampled run — which is what supplies the noise floor, since greedy has
no within-variant spread — is a separate command with its own `--out`.

### Nuisance and exploitability

The same images, judged again with different packaging. `--presentation` is a
stage-3 flag rather than a manifest field, so no variant is re-rendered and no
`variant_id` changes; the pixel-side axes are applied in memory at request
build. Runs on **one** VM, ~1h for the whole sweep.

```bash
python -m src.build_manifest --profile pilot
python -m src.stage2_corrupt --manifest out/manifest.parquet \
    --bases data/bases --out /dev/shm/mcv/nuisance --shard 0 --of 1
mkdir -p out/nuisance

for P in baseline shuffle subset box noimg enhance; do
  python -m src.stage3_judge --manifest out/manifest.parquet --bases data/bases \
      --variants /dev/shm/mcv/nuisance --shard 0 --of 1 --presentation "$P" \
      --temperature 0 --n-samples 1 --out "out/nuisance/scores_$P.parquet"
done

# the noise floor: same variants, baseline packaging, sampled
python -m src.stage3_judge --manifest out/manifest.parquet --bases data/bases \
    --variants /dev/shm/mcv/nuisance --shard 0 --of 1 --presentation baseline \
    --temperature 0.7 --n-samples 5 --out out/nuisance/floor_baseline.parquet

python -m scripts.nuisance_report
```

Three things about that recipe are load-bearing:

- **`--profile pilot`, not `--limit`.** A row limit can take a base's controls
  and drop its corrupted rows, leaving nothing to compare against.
- **`scores_*` and `floor_*` are different prefixes.** The floor run also
  carries `presentation == "baseline"` and is indistinguishable from the greedy
  baseline by any column, so the two globs are what keeps them apart.
- **`out/nuisance/` must not match `out/scores_shard*.parquet`.** Stage 4 does
  not group by `presentation`; pointed at these files it would pool every
  condition's clean controls into one baseline without saying so. Read them
  with `scripts/nuisance_report.py` instead.

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
python -m src.stage0_coco --coco … --list-urls    # fetch only qualifying images
python -m src.stage1_edit --preflight             # API, auth, gating, disk, VRAM fit
python -m src.stage3_judge … --dry-run            # builds every request, never imports vLLM
python -m scripts.diagnose_parse --scores …       # why judge responses failed
python -m scripts.verify_corruption --manifest …  # was the damage real, and only in the mask?
python -m scripts.verify_edit_drift               # did the edit move what the masks describe?
```

`verify_corruption` is the one to run before claiming the judge ignored
something — it separates "the judge did not react" from "there was nothing to
react to".

## Testing

```bash
python tests/test_determinism.py   # 5 determinism properties
python tests/test_stage0.py        # selection logic, via a stub COCO
python tests/test_stage4.py        # 3 synthetic judges with known behaviour
python tests/test_nuisance.py      # presentation axes + 3 judges with known
                                   # nuisance behaviour
python tests/test_syntax.py        # everything parses; GPU modules import without torch
```

All five are CPU-only and finish in seconds. Keep them passing.

`test_nuisance.py` also builds the synthetic base fixture that makes
`stage3_judge --dry-run` runnable on a machine with no stage-1 output.

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
`reward` and `phi` side by side so this is visible, and the `enhance`
presentation tests it directly: a global cosmetic lift that improves no edit
should, if that reading is right, raise every region's reward through `AES`
alone.

Decoding is greedy and every scores parquet records `n_samples` and
`temperature` alongside `judge`, so a merged set of shards can always say how it
was produced. The `shuffle` presentation permutes the JSON schema's region slots
along with the prompt, since the grammar pins slot *k* to the *k*-th listed
region; the constraint is on format only either way.
