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

## Where code runs — laptop vs VMs

Development happens on a **Windows laptop with no GPU**. The five A10 VMs are
separate machines reached over SSH, and **code reaches them only via git
push/pull** — there is no shared filesystem with the laptop, and no ad-hoc file
copying.

| Runs on the laptop | GPU VM only |
|---|---|
| `src/schema.py`, `src/corruptions.py`, `src/build_manifest.py` | `src/stage1_edit.py` — FLUX Kontext, editor VM |
| `tests/test_determinism.py`, `scripts/verify_determinism.sh` | `src/stage3_judge.py` — vLLM |
| `src/stage2_corrupt.py`, `src/stage4_analyze.py` — CPU-only, given their inputs | `scripts/run_shard.sh` — wraps both of the above |
| | `src/stage0_coco.py` — needs the COCO download |

Only `stage1_edit.py` and `stage3_judge.py` import torch/diffusers/vllm; the
rest is pure CPU.

**Consequence for Claude Code:** it runs on the laptop and *cannot execute the
GPU stages at all*. When one needs testing, it must produce the exact command to
run over SSH and wait for the user to paste the output back. Never report a GPU
stage as verified on the strength of a local run, and never add code whose only
validation path is running it locally.

`stage3_judge.py --dry-run` exists for this split: it runs `preflight()` over the
shard, builds every chat message, prints the first one and the engine/sampling
config it *would* use, and returns without importing vLLM or torch. Use it to
settle manifest plumbing and message construction on the laptop, so an SSH
session is only ever debugging the vLLM API surface itself. It needs stage-1
bases and stage-2 variants on disk; with them missing it prints an inventory of
what it could not find and exits 1.

Anything that runs on both sides must stay OS-portable — no `os.statvfs`, no
POSIX-only paths, and printed strings stay ASCII (the Windows console is not
UTF-8; an em-dash in a `print` mojibakes).

Local env: `.venv/`, Python **3.12** — the pinned `numpy==1.26.4` and
`opencv-python-headless==4.10.0.84` have no 3.13 wheels. Invoke it explicitly
(`./.venv/Scripts/python.exe tests/test_determinism.py`).

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
| `A10-24Q` reserves ~2.4GiB | Only **21.34 of 23.72GiB is free** at startup. vLLM budgets `gpu_memory_utilization` against *total* but demands that much *free*, so 0.90 misses by 0.01GiB and the engine dies before loading a weight. `DEFAULT_GPU_UTIL = 0.85`. |
| Qwen3-VL accepts video | `limit_mm_per_prompt` **must** carry `"video": 0`. Left unset, vLLM sizes the encoder cache for a max-length video (151250 tokens) and OOMs in `profile_run` trying to allocate 4.62GiB on top of 16.8GiB of weights. |
| 16.8GiB of weights on a 20.16GiB budget | Everything else has to fit in ~3.4GiB, so `load_engine` runs **eager** (no CUDA graphs), caps `max_num_batched_tokens=2048`, and `max_model_len=4096`. At 8192 + graphs the KV cache came out at **-0.40GiB**. |
| **Qwen3-VL-8B bf16 barely fits** | At `--gpu-util 0.89` it starts, but the KV cache is **0.70GiB = 5,072 tokens, max concurrency 1.24x** — effectively serial. The `main` profile's ~1.2h/VM budget assumed real batching. **A 4B judge is the structural fix**; see the throughput note below. |
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
   - The OpenCV pin is **coupled to vLLM**: vllm 0.11.0 requires
     `opencv-python-headless>=4.11.0`, so the judge VMs cannot install an older
     one. The pin is `==4.11.0.86` — the floor, exactly pinned. Never relax it
     to a range to end a resolver fight: a range lets two VMs land on different
     builds, which is precisely the failure determinism exists to prevent.
     If a future vLLM raises the floor again, bump to the new floor exactly and
     re-run the equivalence check below before accepting it.
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

6. **Two score readouts — HALF BROKEN, needs a design pass.** The plan was
   `sc_sampled` (the emitted integer) plus `sc_expected` (Σ p(k)·k over
   digit-token logprobs), because a 1-5 integer gives ∆score granularity 1 and
   a tie-ridden, degenerate AUROC.
   The real A.4.3 protocol changes the premise: scores are **two-digit numbers
   on 0-25**, nested in a list, inside a per-region object, after a
   variable-length free-text `reasoning` field. The old readout located a score
   by regex on a running prefix and summed over single digit tokens; none of
   that survives. `expected_score_from_logprobs` now **raises
   NotImplementedError** rather than silently returning wrong numbers.
   Mitigating factor: 0-25 is a 26-point scale, so the tie problem that
   motivated the continuous readout is far less severe than it was on 1-5.
   Decide whether it is still needed before rebuilding it.

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
scripts/setup.sh        one-command bootstrap: deps for a role, then the hash check
scripts/smoke_judge.py  one real judge call on synthetic images  [JUDGE VM ONLY]
scripts/run_shard.sh    one VM's share of stages 2+3
scripts/verify_determinism.sh   cross-VM hash check
tests/test_determinism.py
config.yaml             pilot / main / full_cross profiles

requirements.txt          core, every machine (determinism-critical pins)
requirements-judge.txt    + vllm            [judge VMs]
requirements-editor.txt   + diffusers etc.  [editor VM]
requirements-coco.txt     + pycocotools     [stage 0 only]
```

Dependencies are split by role, not commented out in one file. The role files
each `-r requirements.txt`, so setup is always exactly one install command.
Don't pin `torch` in the role files — vLLM and diffusers pin the build they were
compiled against, and a second pin from us produces either a resolver conflict
or a silently mismatched CUDA build. Don't move `pycocotools` back into core: it
builds from source, needs a compiler we cannot install, and only stage 0 imports
it — a failure there must not block the other four VMs.

## Status — what is verified vs what has never run

**Verified by actual execution:**
- `tests/test_determinism.py` — all 5 properties pass, on the reference container
  and (2026-08-25) on the Windows laptop under `.venv` / Python 3.12.
- Cross-VM fixture hash on a reference container: `776feeddd281fa726195bf504c7b19c8`.
  The same script on the Windows laptop prints `5073799d8511586afc2dc504b330abea`.
  **This is expected and is not a failure:** the invariant that matters is that
  the five *Linux* VMs agree with each other. The laptop is a different platform
  (different OpenCV/libjpeg build), so its hash is **not** a valid reference —
  never compare a VM's hash against it. Per-corruption laptop hashes, if anyone
  wants to localise a future cross-VM mismatch:
  `blur b9d191e7be9487bc`, `saturate 55ab816f6c3f49f4`, `noise 1692c77f84ed2c3a`,
  `jpeg 4a0f806a2ce9d42e`, `remove 5fc3f8333affcbc2`.
- Manifest expansion and hash sharding: 200 bases → 24,800 variants, shards
  `[5025, 4895, 4930, 4954, 4996]`, lossless and unique.
- All modules parse.
- Repo layout was flattened in the scaffold commit and restored to the
  documented `src/` / `tests/` / `scripts/` tree on 2026-08-25 — the relative
  imports (`from .schema import ...`) and both shell scripts require it.
- **OpenCV 4.10.0.84 -> 4.11.0.86 is byte-equivalent** for our corruptions.
  Checked twice: on the laptop by rendering the fixture under both versions
  with `numpy`/`Pillow` held identical (all five per-corruption hashes and the
  total matched), and then on VM `mcvgpu2025s-0050` (2026-08-25), which printed
  `776feeddd281fa726195bf504c7b19c8` — the pre-bump container reference —
  while running 4.11.0.86. The bump forced by vLLM re-baselines nothing.
- **Cross-VM agreement is confirmed, not assumed** (2026-08-25). Two
  independent VMs — `mcvgpu2025s-0050` and `mcvgpu2025s-0043` — both print
  `776feeddd281fa726195bf504c7b19c8` on `numpy 1.26.4 / cv2 4.11.0 /
  pillow 10.4.0`, matching the reference container. This is the invariant the
  whole five-way shard split rests on and it had never been tested before now.
  The remaining three VMs must reproduce it as they come online.
- **The laptop's hash differs for platform reasons only, as suspected.** Linux
  gives `776feedd…`; Windows gives `5073799d…` on identical pins. Both are
  internally stable. Compare VM hashes to `776feedd…`; never to the laptop's.
- `stage2_corrupt.py` — ran end-to-end on the laptop (2026-08-25) against a
  synthetic 3-base fixture: 81/81 variants rendered. Its free-space guard used
  `os.statvfs`, which does not exist on Windows; now `shutil.disk_usage`.
- `stage3_judge.py` **up to but not including the engine** — `--dry-run` built
  all 243 requests from that fixture with `vllm` and `torch` confirmed absent
  from `sys.modules`. This exercises manifest → shard → regions.json → prompt →
  message assembly. It says nothing about whether vLLM accepts those messages.

**Never executed — expect real bugs here:**
- `stage0_coco.py` — needs COCO downloaded; `pycocotools` API calls unverified.
- `stage1_edit.py` — FLUX Kontext is gated on HF; pipeline class name and
  offload behaviour unverified on this hardware.
- `stage3_judge.py` — was the highest-risk file; most of that risk is now
  retired. Verified against the installed vllm 0.11.0 on a judge VM
  (2026-08-25), all without a GPU:
  - `LLM.chat` accepts `list[list[message]]`, so batching message-lists is right.
  - `MM_PARSER_MAP` in `vllm/entrypoints/chat_utils.py` contains an `"image_pil"`
    entry, so `{"type": "image_pil", ...}` is the correct discriminator.
  - `LLM.__init__` ends in `**kwargs: Any`, forwarded to `EngineArgs`. That is
    how `max_model_len` and `limit_mm_per_prompt` reach the engine even though
    neither is an explicit parameter — **don't "fix" that by deleting them.**
    `mm_processor_kwargs`, `dtype` and `gpu_memory_utilization` are explicit.

  - A **raw** PIL image is what the parser wants, confirmed at runtime:
    `MM_PARSER_MAP["image_pil"]({"type": ..., "image_pil": img})` returns the
    identical object. Ignore `CustomChatCompletionContentPILImageParam`'s
    annotation of `Optional[PILImage]` (a pydantic wrapper) — its own docstring
    example and `parse_image_pil`'s `Optional[Image.Image]` both say raw, and
    TypedDict annotations are not runtime-enforced. Do not add a wrapper.
  - `EngineArgs` really does carry `max_model_len`, `limit_mm_per_prompt`,
    `mm_processor_kwargs`, `dtype`, `gpu_memory_utilization`.

  **`build_requests` and `load_engine` therefore need no changes.**

  **A real judge call now runs end to end** (2026-08-25, `smoke_judge.py`,
  `--gpu-util 0.89`), which closes the last two:
  - `chat_template_content_format="auto"` resolves to `'openai'` for Qwen3-VL,
    which preserves our custom content parts. `'string'` would have silently
    dropped the images.
  - The logprob structure is `list[dict[int, vllm.logprobs.Logprob]]` with
    `.decoded_token` and `.logprob` — exactly what
    `expected_score_from_logprobs` assumes. It returned
    `{'SC': 4.9993, 'PQ': 5.0000}`. **No rewrite needed.**

  `stage3_judge.py` is verified as far as synthetic data can take it.
- `stage4_analyze.py` — logic is straightforward but has never seen real data.

## BLOCKER (cause found, fix landed, awaiting re-test)

**Status 2026-08-25:** the vision path is confirmed working and the placeholder
prompt was the cause. The real A.4.3 prompt is now in. Re-run `smoke_judge` to
confirm the gap appears; until it does, do not generate data.

The diagnostic that settled it — same images, plain question instead of the
scoring prompt, temperature 0:

```
second image is the RED edit  -> 'red'    (correct)
second image is the BLUE copy -> 'blue'   (correct)
```

So the model reads the images, distinguishes them, and answers correctly.
Anything degenerate from here is prompt or judge behaviour, **not plumbing** —
don't go looking at the harness again. The evidence below is what the
*placeholder* produced and is kept only as the before-picture.

---

### The failure, with the placeholder prompt

`smoke_judge.py` on 2026-08-25, Qwen3-VL-8B, **placeholder prompt**:

| request | images | SC (expected) |
|---|---|---|
| A followed | source + genuinely edited | 4.9991 |
| B ignored | source + unchanged copy | 4.9437 |
| **C text-only** | **none at all** | **5.0000** |

A − B = **+0.055**. The images are definitely in the prompt (+396 tokens vs C).
But a request with **no images scores 5.0**, so the number is a function of the
prompt, not the pixels.

**Do not generate data until this gap is real.** Every analysis in the report —
AUROC, the leakage matrix, redundancy — is computed on ∆score. If ∆score is
noise, all four analyses return noise, and 17,600 requests per profile buys
nothing. This is cheap to keep re-testing and ruinous to discover afterwards.

What this is *not* yet: a finding. The prompt is the invented placeholder, so
this says nothing about the published protocol. A degenerate judge is one of the
outcomes the audit exists to detect, but claiming it requires the real prompt.

Order to work through it:
1. **Get the real SFReward prompt in** (TODO 1 below). Now on the critical path,
   not a nicety — until it is in, we cannot tell a degenerate judge from a
   degenerate prompt of our own making.
2. Run `smoke_judge` again. Section 2b probes the vision path with a plain
   question ("what colour is the square?"). If it answers correctly, the model
   sees fine and the scoring prompt is the problem; if not, look at the harness.
3. Only when A − B is clearly positive, run the `pilot` profile end to end and
   look at the score histogram before scaling to `main`.

## Open TODOs

1. ~~Placeholder prompt~~ **DONE (2026-08-25).** `src/judge_prompt.py` now
   carries A.4.3 verbatim from arXiv:2606.26872. Three caveats to state in the
   report rather than bury:
   - **The PQ prompt is reconstructed, not verbatim.** The paper shows
     SFReward's PQ *output* (A.4.4) but never its PQ *prompt*; A.5.2's PQ prompt
     is the MultiEditBench/VIEScore one on 0-10 for GPT-4.1, a different
     purpose. Ours matches A.4.4's output shape. Marked in the file.
   - **How the instruction and region list are appended is ours.** A.4.3 says
     only "You will be provided with pre-identified editing regions" and never
     shows the injection format.
   - **SFReward is a fine-tuned model** (Qwen3-VL-8B + SFReward-14K); A.4.3 is
     the prompt that labelled that data with a Gemini-3-Pro teacher. We apply it
     to *base* Qwen3-VL-8B, so we audit the prompt-based protocol, not the
     released reward model.

   Consequences already implemented: scale is **0-25, not 1-5**; requests are
   **2 per variant** (one SC scoring every region at once, one image-level PQ)
   rather than one per region — which is the protocol's own shape and cheaper
   than what we had; `max_tokens` 32 -> 1024, since A.4.3 demands per-region
   `reasoning` before the scores; COCO `(x,y,w,h)` is converted to A.4.3's
   `bbox_2d [x1,y1,x2,y2]`.

   **A finding that falls straight out of reading Equation (3):** the region
   reward is √(φ(IF_{i,r}) · AES_i)/C where AES_i = min(PQ) is a *single
   image-level* term multiplying every region of that image. Part of each
   "region" reward is global by construction, before any judge behaviour is
   measured. Within one image it cancels from region-to-region comparisons;
   across variants it does not. Worth a paragraph in the report.

1b. **`stage4_analyze.py` still expects the old columns.** stage 3 now emits
   `sc_success`, `sc_preserve`, `sc_background`, `sc_overall_*`, `pq_*`,
   `reward`, `parsed` — not `sc_sampled`/`sc_expected`. Migrate before running
   any analysis.
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
