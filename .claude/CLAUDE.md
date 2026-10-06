# CLAUDE.md

Read this before changing anything. It carries the rules and the current state.
The evidence behind them lives in two companion files — read the relevant one
before questioning a default or re-opening a settled question:

- **[`docs/DECISIONS.md`](../docs/DECISIONS.md)** — why the harness is what it is:
  hardware derivations, settled questions, bug history, per-stage verification.
- **[`docs/FINDINGS.md`](../docs/FINDINGS.md)** — what we measured about the judge:
  pilot verdict, retired hypotheses, numbers for the report.

## The project

MCV final project, 5 students. The 30 September 2026 deadline was **extended**;
new date TBC. An inference-only
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

**Stage 2 and stage 3 must share one login session.** systemd's `RemoveIPC=yes`
empties the user's `/dev/shm` when their last session ends, and every
`ssh host 'cmd'` is its own session — so rendering variants in one SSH call and
judging them in the next leaves stage 3 with nothing, reporting
`MISSING INPUTS`. Use `tmux`, or have the runner re-render when the directory is
missing. Regeneration is deterministic and costs ~16s for 476 controls, so the
guard is free. This never shows up in an interactive session; it only bites
automation.

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
| `A10-24Q` leaves 21.37 of 23.72GiB free | `gpu_memory_utilization` has a narrow **two-sided** window, ~(0.861, 0.901). Both ends fail. `DEFAULT_GPU_UTIL = 0.89`. **That window is for a ~16GiB checkpoint.** A *small* judge needs a LOWER value, not the same one: at 0.89 InternVL3-2B sized a 16.76GiB KV cache, then OOMed in sampler warmup needing 38MiB. `--gpu-util 0.60` works. |
| Qwen3-VL accepts video | `limit_mm_per_prompt` **must** carry `"video": 0`, or vLLM sizes the encoder cache for a max-length video and OOMs. |
| 16.8GiB of weights on a 20.16GiB budget | `load_engine` runs **eager**, `max_num_batched_tokens=2048`, `max_model_len=4096`. |
| KV cache is 0.70GiB = 5,072 tokens | The judge is effectively serial. **This does not matter, and a 4B judge does not fix it.** |
| No shared FS | Corrupted variants are **regenerated per-VM**, never transferred. Only 146MB of base edits moves, once, via HF Hub. |
| No sudo | `opencv-python-headless` — the normal build needs `libGL.so.1` via apt. Never swap it. |
| 90G disk | VMs are **role-specialised**: the editor VM holds the diffusion model, judge VMs hold judges. Never both. **Do not set `HF_HOME`.** |
| `/dev/shm` | Scratch for regenerated variants. Nothing large goes on `/`. |

Every number above was measured; the derivations are in
[`docs/DECISIONS.md`](../docs/DECISIONS.md).

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
src/presentation.py     7 nuisance/exploitability packaging axes, applied in RAM
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
scripts/pq_response.py        does image-level PQ react to per-region damage? the
                        perception control for a flat localization result  [CPU]
scripts/verify_corruption.py  did the corruption damage the image, and only
                        inside the mask? [CPU] -- run before any insensitivity claim
scripts/verify_edit_drift.py  did stage 1 keep the layout the masks describe?
                        [CPU] -- feeds stage4's --drift-csv robustness split
scripts/nuisance_report.py    paired-delta analysis across presentations, with
                        bootstrap intervals over photographs  [CPU]
scripts/judge_agreement.py    do two judges agree on the same image? the only
                        analysis that spans judges  [CPU]
scripts/make_figures.py       report figures + tie-rate table, drawn from the
                        committed results/ CSVs only  [CPU]

tests/test_determinism.py   5 determinism properties
tests/test_stage0.py        selection logic via a stub COCO (no pycocotools)
tests/test_stage4.py        3 synthetic judges with known behaviour + floor filter
tests/test_nuisance.py      presentation axes + 3 judges; also builds the fixture
                            that --dry-run needs
tests/test_syntax.py        every file parses; GPU modules import without torch

config.yaml             pilot / main / nuisance150 / full_cross profiles
requirements.txt        core, every machine (determinism-critical pins)
requirements-{judge,editor,coco}.txt   role add-ons, each -r requirements.txt

results/README.md       index: every report number -> the CSV it comes from
results/*/REPRODUCE.md  exact commands per run (all re-run 2026-10-06, match)
results/figures/        output of make_figures.py
.claude/CLAUDE.md       this file (moved out of the root 2026-10-06 so the
                        public repo opens on README.md)
docs/internal/TEAM_BRIEF.md   teammate status page
```

## Current state

**Stage 1 is done and the tarball is published** (2026-09-05,
`mcvgpu2025s-0004`): 150 train2017 bases, 476 regions, edited in 8h53m at
213.1s each, every `edit.png` at source resolution, every instruction hash
fresh. `mcv-spatial-audit/mcv-spatial-audit` on the Hub, `bases.tar.gz`, 146MB,
public, carrying `bases.json`, `stage1_provenance.json` and `edit_drift.csv`.
**Nothing downstream is blocked.**

**The `main` run is DONE** (2026-09-18, `mcvgpu2025s-0050`): 150 bases, 476
regions, 5,236 variants, all five corruptions, greedy, judged and analysed end
to end on one VM (all five shards serial in tmux — the disconnected teammates
never materialised, ~8.5h). Parse 100%, 22,176 rows. **Finding: the judge
perceives the corruption but the per-region reward does not localise it.**
Image-level PQ/AES falls monotonically with severity (noise s3 −2.7, remove −2.5,
blur/jpeg s3 ~−1.7 on a between-base SD of 3.36), so a flat per-region result is
about attribution, not perception. Per region: floor-excluded AUROC ~0.52-0.54,
the damaged region no more likely to move than an untouched one (target ties ≈
other ties in every cell), redundancy R^2 0.52 (`phi`), and whole-image
co-movement on `phi` is 28% mixed vs 51% under independence. The layout-drift
split (edge IoU >= 0.4, 96 bases) agrees, so the confound does not drive it.
`jpeg`/`noise`/`saturate` judged on real data for the first time; behave like
`blur`/`remove`.

Two honesty corrections landed after a self-critique pass (2026-09-18) and are
baked into those numbers: headline figures are **floor-excluded**
(`--min-control 0` — 37% of deltas came from regions already at 0, guaranteed
ties that inflate the tie rate and drag AUROC to 0.5), and **coherence is read on
`phi`, not `reward`** (reward's shared AES factor mechanically co-moves regions;
the first-pass 22%-vs-68% was AES-inflated). Scope limit to keep loud: base
Qwen3-VL-8B + the A.4.3 prompt, **not** the fine-tuned SFReward model or its
Gemini teacher. Numbers, raw parquets and `REPRODUCE.md` are in
`results/main_2026-09-18/`; `scripts/pq_response.py` is the new perception check;
full account in [`docs/FINDINGS.md`](../docs/FINDINGS.md). What remains is the
write-up, not the harness.

**Exploitability at 150 bases is DONE** (2026-09-29, `mcvgpu2025s-0004`): the
`nuisance150` profile — 476 clean controls, no corrupted variants — judged by
Qwen3-VL-8B under `baseline`, `noimg`, `enhance`, `enhance_target`, 8,064 rows,
~100 photographs per exploit cell against the pilot's 4. **`noimg` is the one
real exploit and it replicates**: mean `reward` +0.219 overall, **+0.591 on edits
the judge had already called failed, 94% of those regions rising**, against a
0.114 noise floor. Ok edits drift the other way (-0.107): with no image the judge
falls to a high default near `phi` 19. Describe `noimg` as a **constant**, not a
gain: the slope of gain on baseline is -0.922 and `sd` falls 0.417 -> 0.173, so
"+0.55 on failed edits" is arithmetic, not a second finding. Quote the default
(blind `reward` 0.71 vs sighted 0.46). **The cosmetic axes are reversion to the
mean, not exploits** — an earlier "small but real" claim was **retracted
2026-09-29** after an audit: the failed/ok split selects on the judge's own
baseline and measures change from that same baseline, one linear term absorbs the
entire gap (+0.058/-0.056 -> -0.009/+0.009), and letting the *other* judge pick
the failed edits puts all four cells over zero. **No local flattery, and the
contrast is unusable**: the uniform `enhance` control, which cannot flatter
locally, gives +0.30 [+0.11, +0.51] target-minus-other against
`enhance_target`'s +0.33 [-0.03, +0.69]. AES moves the wrong way too — sharpening
*lowers* PQ for both judges. All intervals are 95% bootstrap over photographs,
~72 per run and **uncorrected**, so no single exclusion of zero carries a claim.
Qwen's `noimg` parses at 91.2% (every other condition 100%) because with no
image the model loops in `reasoning` until the token cap — itself a result, but
it biases the parsed subset toward simpler scenes, so quote it.

**InternVL3-2B ran the same four conditions** (2026-09-29 20:31, same VM,
`--gpu-util 0.60`): blind default `phi` 20.4, cosmetic axes at zero. It parses
`noimg` at 100%, so the looping is Qwen's, not a property of image-free prompts.

**The cross-family localization claim is RETRACTED (2026-09-29).** InternVL3-2B
**fails the perception precondition**: `scripts/pq_response.py` gives it dAES
**-0.109** on noise s3 against Qwen's **-2.68**, a control AES of 7.17/25 with a
between-base SD of 0.67, and 63% of its per-region `phi` at exactly 0. A judge
that does not register the stimulus cannot inform on attribution, so its AUROC of
~0.50 measures the instrument, not the protocol. **Every localization claim is
Qwen3-VL-8B only.** The blind-default and judge-disagreement results still stand
across families, because neither needs sensitivity to damage. Evidence in
`results/two_judge_2026-09-26/pq_response_internvl.csv`. A genuine cross-family
claim needs a second judge that passes the control first — a GPU run.
**The two judges barely agree with each other** (`scripts/judge_agreement.py`,
the only analysis here that spans judges): on identical controls **Cohen's kappa
on the failed/not call is 0.203** and their failed sets overlap at Jaccard 0.479.
Lead with kappa, not Pearson — the scale is near-binary, and the rails *inflate*
r (0.212 overall, but **0.094** among regions where both judges are off the
rails). The per-region number does not track the region, and does not track
anything stable across judges either. This one IS cross-family despite the
retraction above: it needs no sensitivity to damage, only two opinions of the
same clean edits.

**The scale is effectively binary and this conditions everything.** `phi` takes
**7 distinct values, 85.6% of them 0 or 25**; `reward` is 42.1% exactly 0 with
four values covering 82.1%. Report the reward as 2-level in practice. This also
vindicates invariant 6 — a continuous logprob readout on a two-valued scale would
have manufactured a result.

Numbers and raw parquets in `results/exploit_150_2026-09-29/` (`qwen/`,
`internvl/`, `agreement/`, plus `exploit_gain.png` per judge); full account in
[`docs/FINDINGS.md`](../docs/FINDINGS.md).

Downloaded, unpacked and built into a pilot manifest on `mcvgpu2025s-0043`
(2026-09-15): `bases.json` holds 150 bases, pilot takes 5 with 16 regions -> 80
variants, all 80 rendered by stage 2 and built into requests by `--dry-run`.

**The full pipeline ran end to end once before this** (2026-08-26,
`mcvgpu2025s-0050`): stage 0 -> 1 -> manifest -> 2 -> 3 -> 4 on 100 base specs,
5 edited, 75 pilot variants judged and analysed. Parse rate 100%, region
coverage 100%. **The go/no-go pilot returned GO** — see
[`docs/FINDINGS.md`](../docs/FINDINGS.md). Those base ids are dead: the split
changed and every id changed with it.

All five test suites pass on the laptop. Cross-VM determinism is confirmed on
**four of five VMs** (`0050`, `0043`, `0053`, `0004`), all printing
`776feeddd281fa726195bf504c7b19c8`.

**Outstanding (none blocks the write-up):**

- The `T=0.7 n=5` noise-floor run over the identical `main` variants. Greedy gave
  no within-run floor; the tie rate and between-variant SD carry the finding, but
  the floor is worth having for the report.
- Determinism hash from the last VM. Still the only unreported verification.
- **The nuisance axes are still at pilot n.** `shuffle`, `subset` and `box` ran
  only on the 2026-09-15 pilot's 5 photos, where `shuffle` moved `reward` 1.55x
  what real damage does. The 2026-09-29 rerun was controls-only, so it could not
  carry them — a nuisance delta needs corrupted variants for its denominator.
  Raising them to 150 bases means a manifest with corruptions, roughly 4x the
  cost of the exploit run.
- The `T=0.7 n=5` noise floor is the only denominator this run lacks: the
  0.114 it is compared against comes from the 2026-09-15 pilot, over different
  photographs. A better within-run floor already exists and is unused --
  `enhance_target` minus `enhance` (two near-identical stimuli, same images)
  moves `reward` by 0.09.
- **A second judge family that passes the perception control.** InternVL3-2B
  does not (see above), so the cross-family localization claim is retracted and
  this is back to being the top strengthening job. Run `scripts/pq_response.py`
  on any new judge BEFORE writing up its localization numbers.
- The `score_preserve` overediting question is now partly answered — the main run
  shows the axis moves but does not localise (see
  [`docs/FINDINGS.md`](../docs/FINDINGS.md)). The direct removal-vs-recolour
  `sc_preserve` comparison on the parquet is still unrun.

The layout-drift confound is **resolved, not outstanding**: the `main`
drift-robustness split agrees across all-150 and layout-survived subsets, so the
source-coordinate masks do not manufacture the null. See the current-state
paragraph above and [`docs/FINDINGS.md`](../docs/FINDINGS.md).

**Do next, in order:**

1. ~~Editor VM: stage 0, then edit 150 bases, then upload the tarball.~~
   **Done 2026-09-05.** Every VM pulls `bases.tar.gz` from
   `mcv-spatial-audit/mcv-spatial-audit`. Base ids are COCO train2017 image ids.
2. ~~`main`: 150 bases at greedy, sharded.~~ **Done 2026-09-18** on `0050`, all
   five shards serial on one VM, ~8.5h. Results in `results/main_2026-09-18/`.
3. ~~The nuisance/exploitability sweep on one VM.~~ **Done 2026-09-15** on `0043`,
   8 conditions including `enhance_target`.
4. ~~Exploitability at reportable n.~~ **Done 2026-09-29** on `0004`, 150 bases,
   Qwen3-VL-8B. Results in `results/exploit_150_2026-09-29/`.
5. **Figures and the report.** The tie-rate and coherence tables are the
   headline, not AUROC. This is the critical path now.
6. Cross-VM determinism hash from the one VM that has not reported it.
7. Optional: the `T=0.7 n=5` noise-floor run, a large-region check
   (`area_bin: half`, since `main` used only `full` at mean 4.5% area), and the
   nuisance axes at 150 bases. Full rationale for all of these is in
   `docs/internal/TEAM_BRIEF.md` "Recommended strengthening".

## Do not re-litigate

Each was measured, not argued. The evidence is in
[`docs/DECISIONS.md`](../docs/DECISIONS.md).

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
| `README.md` | anyone who opens the repo | what the project is, a CPU-only reproduce quickstart, layout, requirements, installation, pipeline, usage, testing, reproducibility caveats | findings (point to `results/README.md` instead), assignments, timeline, status |
| `docs/internal/TEAM_BRIEF.md` | the four other students | plain-language explanation, current status, what the pilot found, **their missions**, decisions to make together, VM gotchas, timeline | setup commands (link to README), implementation detail, measurement tables |
| `.claude/CLAUDE.md` | future Claude Code sessions | the rules, the constraints, current state, what not to re-litigate | evidence, derivations, history, anything a human needs to copy-paste |
| `docs/DECISIONS.md` | future Claude Code sessions, and anyone questioning a default | every measurement behind a constraint, every settled question and why, bug history, per-stage verification record | current state, task lists |
| `docs/FINDINGS.md` | whoever writes the report | pilot verdict, retired hypotheses, judge behaviour, the numbers and their caveats | harness decisions, setup |

`TEAM_BRIEF.md` is deliberately the *simplest*: short paragraphs, no jargon,
missions up front. It is the teammate-facing status page — **when project state
changes, update it in the same session.** Triggers: a pilot or `main` result, an
open decision being settled, a VM reporting its determinism hash, or a finding
being retired the way "semantic yes, photometric no" was. A number reaches the
brief only in the form a teammate would repeat out loud ("53-80% of damaged
regions score identically", not "ties 0.811 / 0.722 on phi").
