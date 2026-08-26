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
| `A10-24Q` reserves ~2.4GiB | Only **21.37 of 23.72GiB is free** at startup, and `gpu_memory_utilization` has a **narrow two-sided window, ~(0.861, 0.901)**. Too high: vLLM budgets against *total* but demands that much *free*, so >0.901 dies before loading a weight. Too low: weights (16.64GiB) + encoder cache + activation peak need ~20.41GiB, so <0.861 sizes the KV cache **negative**. Measured: 0.85 -> **-0.25GiB, dies**; 0.87 -> +0.23; 0.89 -> +0.70. `DEFAULT_GPU_UTIL = 0.89`. |
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
only stage 0 imports it, and a failure there must not block the other four VMs.
(It ships manylinux wheels as of 2.0.11 and needs no compiler; the old "builds
from source" warning is retired.)

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
- **A latent crash at the end of every stage-3 run, found and fixed
  (2026-08-26).** `scored_region_id` mixed region ints with the literal `"bg"`,
  so pyarrow inferred `int64` from the ints and raised `ArrowInvalid` on the
  first `"bg"` — in `res.to_parquet(a.out)`, i.e. *after* a whole shard had
  been judged. An hour of A10 time per VM, discarded at the write. Both
  `scored_region_id` and `target_region_id` are now stringified in
  `build_requests`/`run`. Reproduced on pandas 2.2.2 / pyarrow 17.0.0.
- `stage3_judge.py` **up to but not including the engine** — `--dry-run` built
  all 243 requests from that fixture with `vllm` and `torch` confirmed absent
  from `sys.modules`. This exercises manifest → shard → regions.json → prompt →
  message assembly. It says nothing about whether vLLM accepts those messages.

**Verified on VM `mcvgpu2025s-0050`, 2026-08-26 — stage 1 runs:**
- **`enable_model_cpu_offload` CANNOT work on an A10 and never could.** It
  makes one whole *component* resident, and FLUX Kontext's transformer is
  **23.8GB** in bf16 (download: 9.95 + 9.98 + 3.87) against the **21.37GiB**
  an A10-24Q leaves free. It OOMed at step 0 of 28 after a clean 34GB fetch.
  This is VRAM, not disk — unloading a judge or clearing the cache changes
  nothing. `enable_sequential_cpu_offload` streams submodules and is now the
  default. Peak VRAM **2.39GiB of ~21GiB**.
- **191.4s/image at 28 steps.** pilot (5) = 16 min, `main` (100) = 5.3h,
  `full_cross` (200) = 10.6h. `main` fits an overnight run, so no quantization
  is needed. If it ever is, the lever is an NF4 transformer (~6GB, resident,
  removes the PCIe round trip that dominates), **not** fewer steps — the card
  is 90% idle, not compute-bound.
- **Kontext returns 1024x1024 for a 448x448 input.** The resize-back-to-source
  in `stage1_edit` is therefore load-bearing, not defensive: without it every
  stage-0 mask would index a wrong-sized `edit.png` and stage 2 would corrupt
  the wrong pixels in every variant.
- **The editor follows instructions.** Target region `[30,30,200]` ->
  `[244,4,6]` (redness -85 -> +239); the untouched control moved 15.6 against
  the target's 144.7. Some global drift exists and will show up in the noise
  floor.
- **`pycocotools` ships manylinux wheels** (2.0.11) and needs no compiler. The
  "builds from source, needs gcc" note in README/requirements-coco is stale.
- **Stage 0 yield: 187 usable bases from val2017's 5000 images** (3.7%).
  Histogram of instructable regions per image: `0:2206  1:1883  2:724  3:150
  4:35  5:2`. 187 covers `main`'s 100 with room to spare; it does not cover
  `full_cross`'s 200, which is out of scope anyway. Mean 3.21 regions/base.
  If more are ever needed, train2017 at the same rate gives ~4,400 — but it is
  19GB and will not fit next to FLUX on the 90G disk.

**The full pipeline runs on real data (2026-08-26, `mcvgpu2025s-0050`).**
stage 0 -> 1 -> manifest -> 2 -> 3 -> 4 end to end, 100 base specs, 5 edited,
75 pilot variants rendered, judged and analysed. Four harness bugs were found
and fixed by doing it, none of which any synthetic test could have surfaced:

1. **`DEFAULT_GPU_UTIL = 0.85` could never have worked.** It sizes the KV
   cache at **-0.25GiB**. Every smoke test had passed `--gpu-util 0.89`
   explicitly, so the default was first exercised on the first real run. The
   window is two-sided and narrow, ~(0.861, 0.901). Now 0.89.
2. **A repetition loop ate 30% of responses.** The judge repeated one sentence
   ~110 times to the token cap, mid-JSON. `repetition_penalty` did not hold it.
3. **Silent coverage loss.** `parse_sc` needs only `id` and `score`, so
   responses that dropped a region (2 of 3 scored) or omitted
   `background`/`overall_score` still counted as PARSED. Coverage was 60%.
4. **Grammar whitespace runaway.** With a JSON schema applied, responses ran to
   the full 1536-token cap holding ~100 tokens of JSON; the rest was
   indentation, which is always grammar-valid. `disable_any_whitespace` needs
   `backend="xgrammar"` named explicitly or vLLM rejects it.

Fixed by schema-constrained decoding (`prefixItems` pinning each slot to a
region id, `maxLength` on `reasoning`, `required` on the top-level keys) plus
compact output. Result: **parse rate 100%, region coverage 30/30, background /
overall / PQ all 100%**, responses down to a 240-token mean.
`scripts/diagnose_parse.py` reads a scores parquet on CPU and classifies every
response; run it after any judge change.

**RESOLVED: throughput.** The grammar was the bottleneck, and inside the
grammar it was one keyword. Four configurations, 20 requests each:

| reasoning mode | s/req | vs bounded | `main` h/VM | parse | coverage |
|---|---|---|---|---|---|
| `bounded` (maxLength) | 58.90 | 1.0x | 22.97 | 100% | 100% |
| **`free` (plain string)** | **7.85** | **7.5x** | **3.06** | **100%** | **100%** |
| `none` (field dropped) | 5.15 | 11.4x | 2.01 | 100% | 100% |
| no schema at all | 2.85 | 20.7x | 1.11 | 100% | **~43%** |

**`maxLength` cost 7.5x for nothing.** A length bound makes xgrammar track a
character counter, which multiplies FSM states; removing it left parse rate and
region coverage both at 100% (50 responses, mean 333 tokens against a 1536
cap). `free` is now the default.

Two things this corrects:
- **Batching was never the constraint.** A 4B judge with **26.22x** the
  concurrency finished **2% faster** than the 8B at 1.24x. This file called a
  4B judge "the structural fix" long before anything was measured; it is not.
  It remains worth running as a cross-scale comparison (TODO 2), just not for
  speed.
- **Dropping the schema is not an option.** Unconstrained parses fine but
  covers only ~43% of regions, because the judge silently omits regions and
  `background`/`overall_score`. Speed there is bought with missing data.

`main` now projects to **~3.1h/VM** against the proposal's 1.2h estimate. That
is a one-time overnight run across five VMs in parallel, so it is a schedule
note rather than a blocker; `n_samples` 5 -> 3 would bring it to ~1.8h if it
ever needs to fit, at the cost of a noisier noise floor.

**Never executed — expect real bugs here:**
- `stage0_coco.py` — **selection logic is now laptop-tested** by
  `tests/test_stage0.py` (2026-08-26), which drives `select`/`write_base` with
  a stub COCO. That was made possible by deleting an `assert isinstance(coco,
  COCO)` whose only effect was to force a `pycocotools` import — a library that
  only stage 0 needs — into the one function
  carrying the real risk. Still unverified: pycocotools' own API
  (`annToMask`, `getAnnIds(iscrowd=...)`) and, more importantly, **whether
  val2017 even contains 200 qualifying images**. Run `--survey` first; it needs
  only the annotations JSON, no images and no writes.
  Three bugs fixed while testing: `config.yaml`'s `selection:` block was
  ignored in favour of hardcoded values; `instruction_for` was drawn twice per
  region (once to test for `None`, once for real) so the instruction written
  was a different draw from the one that passed the filter; and `"trash can"`
  is not a COCO category, so it matched nothing, silently, forever.
- `stage1_edit.py` — **`--preflight` now settles most of it without a
  download.** It checks the `FluxKontextPipeline` class name, its `__call__`
  signature, the offload API, HF auth, gated-repo access, disk, and the stage-0
  inputs. `torch` is imported lazily so the input checks run on the laptop.
  `scripts/smoke_edit.py` does one real edit on a synthetic image and reports
  s/image, peak VRAM and the extrapolated budget. Still unverified until both
  are run on the editor VM: whether the weights load under offload on a 24GB
  A10, and whether Kontext actually follows the instructions.
  **A latent misalignment fixed:** `out.resize(src.size)` ran *after*
  `src.thumbnail()` had mutated `src` in place, so any source larger than
  `--max-side` would have produced an `edit.png` at the thumbnailed size while
  stage 0's masks stayed at source resolution — stage 2 would then corrupt the
  wrong pixels. COCO images are <=640px so it never fired, but `--max-side`
  is a knob and the bug was one config change away. Now resizes to the size
  captured before thumbnailing.
  Stage 1 also writes `stage1_provenance.json` (model revision, diffusers /
  transformers / torch versions, steps, guidance). Diffusion output is not
  seed-reproducible across versions, so for an immutable artefact that file
  *is* the reproducibility claim.
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
    `.decoded_token` and `.logprob`. Note this was verified against the
    *placeholder's* flat 1-5 output; the vLLM-side structure is confirmed, but
    `expected_score_from_logprobs` itself no longer applies under A.4.3 — see
    architectural decision 6.

  `stage3_judge.py` is verified as far as synthetic data can take it.
- `stage4_analyze.py` — **no longer unexercised.** `tests/test_stage4.py`
  (2026-08-26, laptop) fabricates three judges in stage 3's exact output schema
  and asserts stage 4 recovers each: `perfect` (only the corrupted region
  drops) -> AUROC 1.00; `global` (damage moves every region equally) -> AUROC
  0.50 and R^2 0.998; `blind` (a constant) -> noise floor exactly 0 and
  correctly reported as no signal. All 30 checks pass. `global` is the one that
  matters: it is precisely the failure this audit exists to detect, and a
  stage 4 that gave it a healthy AUROC would be worse than useless.
  Still never seen *real* data — but it is no longer true that its logic is
  unverified.

## Judge behaviour — what is settled and what is the top risk

Settled 2026-08-25 (synthetic squares, real A.4.3 prompt, Qwen3-VL-8B):

- **The harness is correct.** Prompt parses, `background` / `overall_score`
  present, Equation (3) runs, images demonstrably reach the model (+396 tokens),
  vision path confirmed by a plain-question probe answering `red`/`blue`
  correctly. **Never re-debug the request path.**
- **The judge discriminates instruction-following.** Obeyed 25.0 vs ignored 0.0
  on the targeted region — the full width of the scale. The old placeholder
  prompt was the entire cause of the earlier all-5s result.

### TOP RISK, NARROWED (2026-08-26): semantic yes, photometric no

Re-ran on `mcvgpu2025s-0050` across all four corruptions. The earlier "the
judge ignores corruption entirely" was drawn from `noise` alone and was too
broad. What actually holds:

| corruption | region 0 succ/pres | region 1 (untouched) |
|---|---|---|
| clean | 25/25 | 25/25 |
| **remove** s1 | **20/15** | 25/25 |
| **remove** s2 | **20/15** | **20/15**  <- leakage |
| **remove** s3 | **20/15** | 25/25 |
| blur s1-s3 | 25/25 | 25/25 |
| jpeg s1-s3 | 25/25 | 25/25 |
| noise s1-s3 | 25/25 | 25/25 |

**The judge tracks semantic change, not degradation.** `remove` moves it a full
10 points and moves `success` too (25 -> 20), which is coherent: delete the
square and the instruction is no longer satisfied. Blur, JPEG and noise return
a flat 25 at every severity.

Two caveats: the `remove` severity ladder is flat (15/15/15), so the reaction
is not graded; and the region-1 drop at s2 but not s1 or s3 is non-monotone,
which reads as instability rather than a clean spatial effect. Still synthetic
squares — the pilot on real COCO edits is what decides this.

### Original framing (kept for context)

The observation that started it — every cell 25.0 under `noise` only:

Same instruction, correctly applied, then region 0 corrupted with `noise` at
severities 1/2/3:

| variant | region 0 phi | region 1 phi | bg |
|---|---|---|---|
| clean | 25.0 | 25.0 | 25.0 |
| noise s1 | 25.0 | 25.0 | 25.0 |
| noise s2 | 25.0 | 25.0 | 25.0 |
| noise s3 | 25.0 | 25.0 | 25.0 |
| text-only (no images) | 25.0 | 25.0 | 25.0 |

Every cell 25.0. **If this holds on real data, ∆score is 0 everywhere and AUROC
is 0.5 by construction — the whole audit measures nothing.**

It is *not* that the judge is stuck at 25: it emitted 0.0 when the instruction
was ignored. It appears blind specifically to **degradation of an edit it has
already judged compliant** — which is precisely what stage 2 perturbs.

Do not conclude this yet. Confounds, in the order worth eliminating:

1. **Flat synthetic squares are far out of distribution.** Real COCO edits are
   textured; "is this region preserved" may be a very different question there.
   **Re-run this on real edits as the first thing in the pilot.**
2. **Only `noise` was tried.** `remove` and `blur` are structurally different
   damage. `smoke_judge --corruption remove|blur|jpeg`.
3. **`phi = min(success, preserve)` hides which axis moved.** Corruption should
   hit `preserve` while `success` stays high — the edit still follows the
   instruction, it is just damaged. `smoke_judge` now prints both separately.
   If `preserve` never moves, that is the specific finding.

If it survives all three, it is a genuine and reportable result — a per-region
reward that is insensitive to per-region damage is exactly the failure this
project set out to look for. But it needs real data behind it.

### Free exploitability data point

The judge scores an edit **it was shown no images of** — text-only returns all
25s. Worth a line in the report.

## PILOT RESULT, 2026-08-26 — inconclusive, and the reason is measurable

75 variants, 5 bases, `[none, blur, remove]`, Qwen3-VL-8B, n=5 @ temperature
0.7. Parse 99.2%, coverage 99.2%, `run_shard.sh` verified end to end.

**AUROC 0.46-0.52 on every readout. Do NOT report this as "the judge has no
spatial signal" — the design could not have detected one.**

| | |
|---|---|
| noise floor (SD over 5 samples of the SAME input) | **0.363** |
| reward range | 0.959 |
| floor as share of range | **37.9%** |
| smallest detectable effect (2 SD) | 0.727 = **75.8% of range** |
| largest effect observed | 0.013 = 1.4% |
| values sitting on a rail (0 or max) | **45.6%** |

Two impossibilities confirm it is noise, not signal: `blur` **raised** the
targeted region's score (+1.16 phi vs clean), and `remove` severity 3 scored
**higher** than severity 1 (12.49 vs 10.50).

**Cause: sampling.** `n_samples=5 @ temperature=0.7` was OUR choice to estimate
a noise floor, not the paper's protocol. The judge's output is bimodal at 0 and
25, so at 0.7 it flips rails between samples of an identical input and the floor
swamps any effect. **Next: `--temperature 0 --n-samples 1`** (also 5x cheaper,
~5 min for the pilot).

**A hypothesis that was WRONG, recorded so nobody re-runs it:** floor effects
were suspected — 25% of scores are exactly 0, and a region already at 0 cannot
drop. But `--min-control 0` kept **100%** of rows: the control baseline averages
5 samples over several control variants, so essentially no control mean lands at
0. The per-sample zeros do not become baseline zeros. `--min-control` stays in
stage 4 as a diagnostic, but it is not the problem here.

**What survives the noise:** leave-one-out redundancy **R^2 = 0.57-0.73**
(sc_preserve highest at 0.734). Region scores co-move strongly within an image.
Noise *attenuates* correlation, so the true figure is higher than measured —
this is the one pilot number that points at the "global impression" hypothesis,
and it is the most interesting result so far.

## Session close-out, 2026-08-26

The pipeline runs end to end on real data. What remains is the experiment, not
the harness.

**Do next, in order:**
1. Editor VM: edit the remaining 95 bases (~5h at 189s each), then
   `tar czf bases.tar.gz -C data bases` and upload. Everything downstream is
   blocked on that tarball.
2. Run the pilot and read the axis table (`sc_preserve` vs `sc_success` on the
   targeted region). That is the go/no-go on the top risk.
3. Cross-VM determinism hash from the three VMs that have not reported it.
4. `main`: ~3.1h/VM, sharded five ways.

**Do not re-litigate** (each was measured, not argued):
- `--gpu-util` 0.89; the window is (0.861, 0.901) and both ends fail.
- Stage 1 `--offload sequential`; model-level cannot fit, ever.
- `--reasoning free`; `bounded` costs 7.5x for identical quality.
- Schema-constrained decoding is mandatory — unconstrained covers ~43% of
  regions while reporting a healthy parse rate.
- A 4B judge does NOT fix throughput. 26x concurrency bought 2%.

**Disk is the live constraint on `mcvgpu2025s-0050`:** FLUX (~34GB) +
Qwen3-VL-8B (~16GB) + Qwen3-VL-4B (~9GB) + COCO val2017 (~1GB) + a second venv
with CUDA wheels, against a 90G root. `run_shard.sh` used to force
`HF_HOME=$HOME/hf_cache` while every manual run used the default
`~/.cache/huggingface`, so it re-downloaded a cached 16GB checkpoint into an
empty second cache and ran the disk out mid-transfer. Fixed: the script now
inherits `HF_HOME`, reports which cache it will use, and refuses to start if the
model is uncached with under 25G free. **Do not set `HF_HOME`.**

**Still unverified:** `run_shard.sh` end to end; determinism on VMs 3-5; stage 4
against real multi-base data (it has only seen 2- and 10-variant smoke runs).

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

1b. ~~`stage4_analyze.py` still expects the old columns.~~ **DONE
   (2026-08-26).** Migrated to the A.4.3 columns and verified end to end by
   `tests/test_stage4.py`. Four readouts now run side by side — `reward`
   (Equation 3, the headline), `phi` (Eq. 3 with the global AES factor divided
   out), `sc_preserve` and `sc_success`. The last two exist to answer the top
   risk directly: corruption should drive `preserve` down while `success`
   holds, and `phi = min()` hides which axis moved. `--all-readouts` runs the
   lot. Two further changes worth knowing:
   - **Redundancy now regresses on the leave-one-out mean of the image's other
     regions**, not the plain image mean. The plain mean includes the region
     itself, which at ~4 regions is a quarter of the predictor and manufactures
     correlation out of pure noise. R^2 near 1 now means what it claims to.
   - **Background rows are excluded from AUROC** (`bg` is never a corruption
     target, so it is a guaranteed negative that would inflate AUROC for free)
     but kept in the leakage matrix, where "damage in region i moved the
     background score" is a real thing to see.
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
