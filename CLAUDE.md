# CLAUDE.md

Context for Claude Code working on this repo. Read this before changing anything.

## The project

MCV final project, 5 students, due **30 September 2026**. An inference-only audit
asking whether per-region VLM reward scores are actually spatially resolved.

RL post-training methods for image editing (SpatialFlow-GRPO, Edit-GRPO,
RC-GRPO-Editing, SpatialReward — all 2026) replaced whole-image rewards with
per-region scores from a VLM judge. Nobody checked whether the number attached
to a region reflects that region's content, rather than a global impression or
a prompt artifact.

The test: take an edited image, deliberately corrupt exactly one region, ask the
judge to score all regions. We hold per-region ground truth the judge does not,
so any mismatch between where we corrupted and where the score reacts is
directly measurable. Four analyses: localization (AUROC), redundancy (R² vs
image-level score), nuisance (do irrelevant presentation changes move scores as
much as real damage?), exploitability (can scores be pushed up without visual
improvement?).

No training. Inference only.

## Hardware — these are hard constraints, not preferences

Five VMs, one per student. Each is an Azure NV36ads_A10_v5:

- **GPU**: NVIDIA A10 24GB (`A10-24Q` vGPU, full framebuffer, not time-sliced)
- **RAM**: ~440GB. **CPU**: 36 vCPU
- **Disk**: 90G free on `/`. This is the only large persistent writable space.
- **No sudo.** No apt. No system package installation, ever.
- **`/mnt`**: root-owned, not writable by us. Do not use.
- **`/datashare`**: CIFS mount, `uid=0,gid=0,file_mode=0744`. Read-only in
  practice. **There is no shared filesystem between the five VMs.**
- **`/dev/shm`**: 217G, mode 1777, writable. RAM-backed, wiped on reboot.

Consequences, all already reflected in the code:

| Constraint | Consequence |
|---|---|
| A10 = SM 8.6 (Ampere) | **bf16 only. Never fp8** — those kernels need SM 8.9+. The official Qwen FP8 checkpoint is not usable here. |
| No shared FS | Corrupted variants are **regenerated per-VM**, never transferred. Only ~300MB of base edits moves, once, via HF Hub. |
| No sudo | `opencv-python-headless` — the normal build needs `libGL.so.1` via apt. Never swap this. |
| 90G disk | VMs are **role-specialised**: the editor VM holds the diffusion model, judge VMs hold judges. Never both. |
| `/dev/shm` | Scratch for regenerated variants. Nothing large goes on `/`. |

## Architectural decisions — do not undo these

1. **Regenerate, don't transfer.** Stage 2 corruption is a pure deterministic
   function of `(base edit, mask, manifest row, seed)`. Each VM renders only its
   own shard into `/dev/shm`. This is the load-bearing decision that makes five
   disconnected machines workable. Do not add code that ships variants between
   VMs or writes them to `/`.

2. **Determinism is a correctness requirement, not a nicety.** If two VMs
   produce different bytes for the same `variant_id`, scores from different
   shards are not comparable and the whole audit is invalid.
   - Seeds derive from `sha256(variant_id)`, never from a counter or wall clock.
   - `numpy`, `opencv-python-headless`, `Pillow` are **version-pinned** in
     `requirements.txt` for this reason. If you must bump one, re-run
     `scripts/verify_determinism.sh` on two machines and compare hashes first.
   - `tests/test_determinism.py` guards repeatability, seed-sensitivity, order
     independence, spatial locality, and monotone area bins. Keep it passing.

3. **Sharding is by `variant_id` hash, not row position.** `df.iloc[k::5]`
   breaks the moment the manifest is regenerated or reordered. See
   `schema.shard()`.

4. **Stage 1 (editing) is an immutable artefact.** Diffusion sampling drifts
   across library versions and attention backends even at fixed seed, so base
   edits are generated **once**, on one VM, tarred, and uploaded. Never
   regenerate them per-VM. This is the opposite of the stage 2 rule and the
   asymmetry is deliberate.

5. **Judge efficiency.** vLLM `SamplingParams(n=k)` shares the prefill across
   samples. With image inputs, prefill dominates (outputs are ~15 tokens), so
   this is a near-free 4-5x versus k separate requests. Also cap `max_pixels` —
   Qwen3-VL tokenizes by area and an uncapped 2000px image explodes latency.

6. **Two score readouts.** `sc_sampled` (the integer the judge emitted) and
   `sc_expected` (Σ p(k)·k over digit-token logprobs). The integer scale makes
   ∆score granularity 1, which produces massive ties and a degenerate AUROC.
   The logprob expectation is continuous and lower-variance. Report both;
   the comparison between them is itself a result.

## Repo layout

```
src/schema.py           manifest schema, variant_id, seed derivation, hash sharding
src/corruptions.py      5 seeded feathered degradations (determinism-critical)
src/stage0_coco.py      COCO instance-seg filter -> multi-region base specs
src/stage1_edit.py      FLUX Kontext editing w/ CPU offload  [EDITOR VM ONLY]
src/build_manifest.py   expand base specs into the design matrix
src/stage2_corrupt.py   regenerate this VM's shard into /dev/shm
src/stage3_judge.py     sharded vLLM judging
src/stage4_analyze.py   AUROC, leakage matrix, redundancy, noise floor
scripts/run_shard.sh    one VM's share of stages 2+3
scripts/verify_determinism.sh   cross-VM hash check
tests/test_determinism.py
config.yaml             pilot / main / full_cross profiles
```

## Status — what is verified vs what has never run

**Verified by actual execution:**
- `tests/test_determinism.py` — all 5 properties pass.
- Cross-VM fixture hash on a reference container: `776feeddd281fa726195bf504c7b19c8`
  (the team's five VMs must agree with *each other*; a pin drift may shift the
  absolute value).
- Manifest expansion and hash sharding: 200 bases → 24,800 variants, shards
  `[5025, 4895, 4930, 4954, 4996]`, lossless and unique.
- All modules parse.

**Never executed — expect real bugs here:**
- `stage0_coco.py` — needs COCO downloaded; `pycocotools` API calls unverified.
- `stage1_edit.py` — FLUX Kontext is gated on HF; pipeline class name and
  offload behaviour unverified on this hardware.
- `stage3_judge.py` — **the highest-risk file.** vLLM's `llm.chat()` multimodal
  message format for Qwen3-VL, the `image_pil` content type, `mm_processor_kwargs`
  key names, and the logprob structure in `expected_score_from_logprobs` all need
  checking against the installed vLLM version. Expect to fix this file.
- `stage4_analyze.py` — logic is straightforward but has never seen real data.

## Open TODOs

1. **`src/judge_prompt.py` contains a PLACEHOLDER prompt.** Right shape,
   invented wording. It must be replaced verbatim from SpatialFlow-GRPO's
   appendix (arXiv:2606.26872) before any reportable run. The project's claim is
   that it audits *the published protocol*; an invented prompt voids the
   comparison. This is isolated to one file on purpose.
2. Pick and wire the second judge family (cross-family agreement is a finding).
3. Nuisance and exploitability tests are designed but not implemented — they
   reuse the stage 3 harness with varied presentation (box/mask/crop, prompt
   order, region count).
4. Figures for the report.

## Guardrails

- Don't run the `full_cross` profile casually: 24,800 variants → 99,200 requests
  → ~6.6h per VM. `main` (4,400 variants, ~1.2h/VM) is sized to the proposal's
  stated budget.
- Don't add dependencies needing apt/sudo.
- Don't write variants, weights, or datasets to `/` beyond the budget in README.
- Don't commit HF tokens, `data/`, or `out/` (see `.gitignore`).
- The "Optimal Reward ∆" stretch goal in the proposal is explicitly out of scope
  unless everything else finishes early.

## Deliverables

- Report, 2-3 pages, LaTeX: abstract, intro + related work, method, results,
  discussion. Figures matter.
- **Public** GitHub repo with reproducible code.
- 5-minute talk by one representative.
