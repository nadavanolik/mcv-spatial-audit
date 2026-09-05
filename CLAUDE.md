# CLAUDE.md

Read this before changing anything. It carries the rules and the current state.
The evidence behind them lives in two companion files — read the relevant one
before questioning a default or re-opening a settled question:

- **[`docs/DECISIONS.md`](docs/DECISIONS.md)** — why the harness is what it is:
  hardware derivations, settled questions, bug history, per-stage verification.
- **[`docs/FINDINGS.md`](docs/FINDINGS.md)** — what we measured about the judge:
  pilot verdict, retired hypotheses, numbers for the report.

## The project

MCV final project, 5 students, due **30 September 2026**. An inference-only
audit asking whether per-region VLM reward scores are actually spatially
resolved.

RL post-training methods for image editing (SpatialFlow-GRPO, Edit-GRPO,
RC-GRPO-Editing, SpatialReward — all 2026) replaced whole-image rewards with
per-region scores from a VLM judge. Nobody checked whether the number attached
to a region reflects that region's content, rather than a global impression or a
prompt artifact.

The test: take an edited image, corrupt exactly one region, ask the judge to
score all regions. We hold per-region ground truth the judge does not, so any
mismatch between where we corrupted and where the score reacts is directly
measurable. Four analyses: localization (AUROC), redundancy (R² vs image-level
score), nuisance (do irrelevant presentation changes move scores as much as real
damage?), exploitability (can scores be pushed up without visual improvement?).

No training. Inference only.

## Where code runs — laptop vs VMs

Development happens on a **Windows laptop with no GPU**. The five A10 VMs are
separate machines reached over SSH, and **code reaches them only via git
push/pull** — no shared filesystem, no ad-hoc file copying.

| Runs on the laptop | GPU VM only |
|---|---|
| `src/schema.py`, `src/corruptions.py`, `src/build_manifest.py` | `src/stage1_edit.py` — FLUX Kontext, editor VM |
| `tests/`, `scripts/verify_determinism.sh` | `src/stage3_judge.py` — vLLM |
| `src/stage2_corrupt.py`, `src/stage4_analyze.py` — CPU-only, given their inputs | `scripts/run_shard.sh` — wraps both of the above |
| | `src/stage0_coco.py` — needs the COCO download |

Only `stage1_edit.py` and `stage3_judge.py` import torch/diffusers/vllm.

**Consequence for Claude Code:** it runs on the laptop and *cannot execute the
GPU stages at all*. When one needs testing, produce the exact command to run over
SSH and wait for the user to paste the output back. Never report a GPU stage as
verified on the strength of a local run, and never add code whose only validation
path is running it locally.

`stage3_judge.py --dry-run` exists for this split: it runs `preflight()` over the
shard, builds every chat message, prints the first one and the engine/sampling
config it *would* use, and returns without importing vLLM or torch. Use it to
settle manifest plumbing and message construction on the laptop, so an SSH
session only ever debugs the vLLM API surface. It needs stage-1 bases and stage-2
variants on disk; without them it prints an inventory and exits 1.

Anything that runs on both sides must stay OS-portable — no `os.statvfs`, no
POSIX-only paths, and printed strings stay ASCII (the Windows console is not
UTF-8; an em-dash in a `print` mojibakes).

Local env: `.venv/`, Python **3.12** — the pinned `numpy==1.26.4` and
`opencv-python-headless==4.11.0.86` have no 3.13 wheels. Invoke it explicitly
(`./.venv/Scripts/python.exe tests/test_determinism.py`).

## Hardware — hard constraints, not preferences

Five Azure NV36ads_A10_v5, one per student: **A10 24GB** (`A10-24Q` vGPU, full
framebuffer), ~440GB RAM, 36 vCPU, 90G free on `/`. **No sudo, no apt, ever.**
`/mnt` is root-owned and unusable; `/datashare` is read-only CIFS; `/dev/shm` is
217G, writable, RAM-backed, wiped on reboot. **No shared filesystem between the
five VMs.**

| Constraint | Consequence |
|---|---|
| A10 = SM 8.6 (Ampere) | **bf16 only, never fp8** — those kernels need SM 8.9+. The official Qwen FP8 checkpoint is unusable. |
| `A10-24Q` leaves 21.37 of 23.72GiB free | `gpu_memory_utilization` has a narrow **two-sided** window, ~(0.861, 0.901). Both ends fail. `DEFAULT_GPU_UTIL = 0.89`. |
| Qwen3-VL accepts video | `limit_mm_per_prompt` **must** carry `"video": 0`, or vLLM sizes the encoder cache for a max-length video and OOMs. |
| 16.8GiB of weights on a 20.16GiB budget | `load_engine` runs **eager**, `max_num_batched_tokens=2048`, `max_model_len=4096`. |
| KV cache is 0.70GiB = 5,072 tokens | The judge is effectively serial. **This does not matter, and a 4B judge does not fix it.** |
| No shared FS | Corrupted variants are **regenerated per-VM**, never transferred. Only ~450MB of base edits moves, once, via HF Hub. |
| No sudo | `opencv-python-headless` — the normal build needs `libGL.so.1` via apt. Never swap it. |
| 90G disk | VMs are **role-specialised**: the editor VM holds the diffusion model, judge VMs hold judges. Never both. **Do not set `HF_HOME`.** |
| `/dev/shm` | Scratch for regenerated variants. Nothing large goes on `/`. |

Every number above was measured; the derivations are in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Invariants — do not undo these

1. **Regenerate, don't transfer.** Stage 2 corruption is a pure deterministic
   function of `(base edit, mask, manifest row, seed)`. Each VM renders only its
   own shard into `/dev/shm`. This is the load-bearing decision that makes five
   disconnected machines workable. Never ship variants between VMs or write them
   to `/`.

2. **Determinism is a correctness requirement.** If two VMs produce different
   bytes for the same `variant_id`, scores from different shards are not
   comparable and the audit is invalid. Seeds derive from `sha256(variant_id)`,
   never from a counter or wall clock. `numpy`, `opencv-python-headless` and
   `Pillow` are version-pinned for this reason — the OpenCV pin is exactly
   `==4.11.0.86` and **must never be relaxed to a range**. To bump one, re-run
   `scripts/verify_determinism.sh` on two machines and compare hashes first.
   `tests/test_determinism.py` guards repeatability, seed-sensitivity, order
   independence, spatial locality and monotone area bins. Keep it passing.

3. **Sharding is by `variant_id` hash, not row position.** `df.iloc[k::5]`
   breaks the moment the manifest is regenerated or reordered. See
   `schema.shard()`.

4. **Stage 1 (editing) is an immutable artefact.** Diffusion sampling drifts
   across library versions and attention backends even at fixed seed, so base
   edits are generated once, on one VM, tarred and uploaded. Never regenerate
   them per-VM. This is the opposite of the stage 2 rule and the asymmetry is
   deliberate.

5. **Grammar cost is the judge's bottleneck** — not batching, not prefill.
   Schema-constrained decoding is mandatory (unconstrained covers ~43% of
   regions while reporting a healthy parse rate), `maxLength` in a schema costs
   7.5x and buys nothing, and `--reasoning free` is the default for that reason.
   `max_pixels` stays capped: Qwen3-VL tokenizes by area and vLLM sizes the
   encoder cache from the cap regardless of what you send.

6. **No expected-score readout.** `expected_score_from_logprobs` raises
   `NotImplementedError` and **must keep raising**. The score ties are not a
   measurement artefact to be smoothed away — they ARE the finding. A continuous
   logprob readout would turn "the judge did not react" into a small non-zero
   number. `sensitivity()` in stage 4 reports the tie rate directly, split by
   target vs non-target; report it alongside every AUROC.

7. **Presentation is a stage-3 flag, never a manifest column.** A sixth
   `variant_id` field would change every id, hence every seed, hence every
   rendered byte — voiding the fixture hash three VMs have confirmed.
   `--presentation` re-packages the same images in memory at request-build time.

## Repo layout

```
src/schema.py           manifest schema, variant_id, seed derivation, hash sharding
src/corruptions.py      5 seeded feathered degradations (determinism-critical)
src/judge_prompt.py     A.4.3 prompt verbatim, JSON schemas, Eq. (3) reward
src/presentation.py     6 nuisance/exploitability packaging axes, applied in RAM
src/stage0_coco.py      COCO instance-seg filter -> multi-region base specs
src/stage1_edit.py      FLUX Kontext editing, sequential offload  [EDITOR VM ONLY]
src/build_manifest.py   expand base specs into the design matrix
src/stage2_corrupt.py   regenerate this VM's shard into /dev/shm
src/stage3_judge.py     sharded vLLM judging, schema-constrained
src/stage4_analyze.py   measurement quality, tie rate, coherence, AUROC,
                        leakage matrix, redundancy, noise floor

scripts/setup.sh        one-command bootstrap for a role, then the hash check
scripts/run_shard.sh    one VM's share of stages 2+3
scripts/verify_determinism.sh   cross-VM hash check
scripts/smoke_judge.py  one real judge call on synthetic images  [JUDGE VM ONLY]
scripts/smoke_edit.py   one real FLUX edit on a synthetic image  [EDITOR VM ONLY]
scripts/diagnose_parse.py     why judge responses failed, from a parquet  [CPU]
scripts/verify_corruption.py  did the corruption damage the image, and only
                        inside the mask? [CPU] -- run before any insensitivity claim
scripts/verify_edit_drift.py  did stage 1 keep the layout the masks describe?
                        [CPU] -- feeds stage4's --drift-csv robustness split
scripts/nuisance_report.py    paired-delta analysis across presentations  [CPU]

tests/test_determinism.py   5 determinism properties
tests/test_stage0.py        selection logic via a stub COCO (no pycocotools)
tests/test_stage4.py        3 synthetic judges with known behaviour + floor filter
tests/test_nuisance.py      presentation axes + 3 judges; also builds the fixture
                            that --dry-run needs
tests/test_syntax.py        every file parses; GPU modules import without torch

config.yaml             pilot / main / full_cross profiles
requirements.txt        core, every machine (determinism-critical pins)
requirements-{judge,editor,coco}.txt   role add-ons, each -r requirements.txt
```

## Current state

**The full pipeline runs end to end on real data** (2026-08-26,
`mcvgpu2025s-0050`): stage 0 -> 1 -> manifest -> 2 -> 3 -> 4, 100 base specs,
5 edited, 75 pilot variants rendered, judged and analysed. Parse rate 100%,
region coverage 100%. **The go/no-go pilot returned GO** — see
[`docs/FINDINGS.md`](docs/FINDINGS.md).

All five test suites pass on the laptop. Cross-VM determinism is confirmed on
**three of five VMs** (`0050`, `0043`, `0053`), all printing
`776feeddd281fa726195bf504c7b19c8`.

**Stage 1 is done** (2026-09-05, `mcvgpu2025s-0004`): 150 bases edited in
8h53m at 213.1s each, every `edit.png` at source resolution, every instruction
hash fresh. The tarball is published and downstream is unblocked:
`mcv-spatial-audit/mcv-spatial-audit` on the Hub, `bases.tar.gz`, 146MB,
public, carrying `bases.json`, `stage1_provenance.json` and `edit_drift.csv`.

**Outstanding:**

- Determinism hash from the last two VMs. This is the only unreported
  verification.
- `annToMask` in stage 0 has never been exercised.
- The nuisance/exploitability sweep has never run on a GPU. The one thing that
  could still invalidate the `shuffle` axis is whether vLLM/xgrammar accepts a
  permuted `prefixItems` schema at all.
- Whether the judge reads `score_preserve` as preservation or as overediting —
  the open question in [`docs/DECISIONS.md`](docs/DECISIONS.md). Settle it by
  measurement on the pilot parquet, not by re-reading the paper.
- `jpeg`, `noise` and `saturate` have never been judged on real data; the pilot
  ran only `[none, blur, remove]`.
- **Stage 1 does not always preserve layout.** Edge IoU between source and edit
  is below 0.40 on 54 of 150 bases, mostly indoor furniture. Masks are computed
  on `source.png` and applied to `edit.png`, so a re-composed scene means we
  corrupt background while claiming a region — a confound that MIMICS the
  finding. Settled by reporting every headline twice (`stage4_analyze
  --drift-csv out/edit_drift.csv`), not by filtering. See
  [`docs/DECISIONS.md`](docs/DECISIONS.md).

**Do next, in order:**

1. ~~Editor VM: stage 0, then edit 150 bases, then upload the tarball.~~
   **Done 2026-09-05.** Every VM now pulls `bases.tar.gz` from
   `mcv-spatial-audit/mcv-spatial-audit` instead. Base ids are COCO
   train2017 image ids; anything from the pilot is dead data.
2. Cross-VM determinism hash from the two VMs that have not reported it.
3. `main`: 150 bases, **~1.7h/VM** at greedy, sharded five ways.
4. The nuisance/exploitability sweep: ~1h on **one** VM that has `data/bases` and
   a judge checkpoint. Independent of `main` and of which photographs it uses, so
   it can run the moment any VM has bases. Run `--dry-run` on real bases first —
   30 seconds, no GPU — since the only untested assumption left is the shape of a
   real `regions.json`.
5. Pick and wire the second judge family (cross-family agreement is a finding).
   A 4B Qwen is a cross-scale comparison, not a second family.
6. Figures for the report.

## Do not re-litigate

Each was measured, not argued. The evidence is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

- **`--gpu-util 0.89`.** The window is (0.861, 0.901) and both ends fail.
- **Stage 1 `--offload sequential`.** Model-level offload cannot fit, ever — the
  FLUX transformer alone is 23.8GB against 21.37GiB free. It is a VRAM limit;
  freeing disk changes nothing.
- **`--reasoning free`.** `bounded` costs 7.5x for identical quality.
- **Schema-constrained decoding is mandatory.** Unconstrained covers ~43% of
  regions.
- **A 4B judge does not fix throughput.** 26x concurrency bought 2%.
- **Greedy in both `stage3_judge` and `run_shard.sh`**, passed explicitly, with
  no env-var override on purpose. The noise floor is a separate
  `--temperature 0.7 --n-samples 5` run over the identical variants.
- **`remove` is reported as one binary condition**, not a severity ladder — its
  s1 and s3 differ by 0.5 of 35 levels because inpainting deletes the object at
  every radius. `FLAT_SEVERITY` handles it so no caller can forget.
- **Presentation axes are a stage-3 flag, not a manifest column.**
- **The nuisance analysis lives in `scripts/nuisance_report.py`, not stage 4.**
  It pairs on `(variant_id, scored_region_id)` across presentations;
  `delta_table` would silently pool the conditions' controls.
- **`main` is 150 bases.** Not pool-limited. The 50 over 100 narrow intervals by
  ~18% and cost 2.7h on the editor VM. Do not push to 200 without re-timing
  stage 1.
- **"Semantic yes, photometric no" is retired.** It did not replicate on real
  photographs.
- **Do not download a COCO image split.** The filter discards 99 of every 100
  images; `--list-urls` fetches only what qualifies.

## Guardrails

- Don't run the `full_cross` profile casually: 24,800 variants -> 99,200
  requests. `main` at 150 bases x 3.19 regions is ~5,300 variants / ~10,500
  requests, ~1.7h/VM at greedy, sized to the proposal's stated budget.
- Don't add dependencies needing apt/sudo.
- Don't write variants, weights or datasets to `/` beyond the README's budget.
- Don't commit HF tokens, `data/` or `out/` (see `.gitignore`).
- The "Optimal Reward ∆" stretch goal in the proposal is out of scope unless
  everything else finishes early.

## Deliverables

- Report, 2-3 pages, LaTeX: abstract, intro + related work, method, results,
  discussion. Figures matter.
- **Public** GitHub repo with reproducible code.
- 5-minute talk by one representative.

## The five docs have different jobs — keep them separate

They drifted into near-duplicates once and had to be pulled apart. Before adding
anything, decide which one it belongs in.

| | Audience | Contains | Does NOT contain |
|---|---|---|---|
| `README.md` | anyone who opens the repo | what the project is, layout, requirements, installation, pipeline, usage, testing, reproducibility caveats | findings, assignments, timeline, status |
| `TEAM_BRIEF.md` | the four other students | plain-language explanation, current status, what the pilot found, **their missions**, decisions to make together, VM gotchas, timeline | setup commands (link to README), implementation detail, measurement tables |
| `CLAUDE.md` | future Claude Code sessions | the rules, the constraints, current state, what not to re-litigate | evidence, derivations, history, anything a human needs to copy-paste |
| `docs/DECISIONS.md` | future Claude Code sessions, and anyone questioning a default | every measurement behind a constraint, every settled question and why, bug history, per-stage verification record | current state, task lists |
| `docs/FINDINGS.md` | whoever writes the report | pilot verdict, retired hypotheses, judge behaviour, the numbers and their caveats | harness decisions, setup |

`TEAM_BRIEF.md` is deliberately the *simplest*: short paragraphs, no jargon,
missions up front. It is the teammate-facing status page — **when project state
changes, update it in the same session.** Triggers: a pilot or `main` result, an
open decision being settled, a VM reporting its determinism hash, or a finding
being retired the way "semantic yes, photometric no" was. A number reaches the
brief only in the form a teammate would repeat out loud ("53-80% of damaged
regions score identically", not "ties 0.811 / 0.722 on phi").
