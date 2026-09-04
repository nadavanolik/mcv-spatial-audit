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
`opencv-python-headless==4.11.0.86` have no 3.13 wheels. Invoke it explicitly
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
| **Qwen3-VL-8B bf16 barely fits** | At `--gpu-util 0.89` it starts, but the KV cache is **0.70GiB = 5,072 tokens, max concurrency 1.24x** — effectively serial. **This does NOT matter and a 4B judge does NOT fix it**: measured 2026-08-26, a 4B at 26.22x the concurrency finished 2% faster. Grammar decoding, not batching, set the pace. See "RESOLVED: throughput". |
| No shared FS | Corrupted variants are **regenerated per-VM**, never transferred. Only ~450MB of base edits moves, once, via HF Hub. |
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

5. **Judge efficiency — REWRITTEN 2026-08-26 after measurement.** The original
   claim was that `SamplingParams(n=k)` shares the prefill so k samples are
   nearly free, because "outputs are ~15 tokens". Both halves were wrong under
   A.4.3: outputs are ~330 tokens (per-region `reasoning`), so decode is not
   negligible, and the grammar dominates everything anyway. What actually holds:
   - **Grammar cost is the bottleneck**, not batching and not prefill. Measured:
     constrained 13.5 out tok/s vs unconstrained 265.9.
   - **`maxLength` in a JSON schema costs 7.5x** and buys nothing. Never use it.
   - **`--reasoning free`** is the default for that reason.
   - `max_pixels` is still capped — Qwen3-VL tokenizes by area and vLLM sizes
     the encoder cache from the cap regardless of what you send.
   - Sampling: the pilot showed `n=5 @ T=0.7` is unusable (see PILOT VERDICT).
     Greedy `n=1` is both correct and 5x cheaper.

6. **Two score readouts — QUESTION ANSWERED 2026-08-26. Do not rebuild it.**
   The plan was `sc_sampled` (the emitted integer) plus `sc_expected`
   (Σ p(k)·k over digit-token logprobs), because a 1-5 integer gives ∆score
   granularity 1 and a tie-ridden AUROC. A.4.3 broke the implementation (scores
   are two-digit, nested, after variable-length reasoning), and
   `expected_score_from_logprobs` raises `NotImplementedError`.

   **Leave it raising.** The pilot showed the ties are not a measurement
   artefact to be smoothed away — they ARE the finding. 53-80% of damaged
   regions receive a byte-identical score to their clean control under greedy
   decoding. A continuous logprob readout would paper over exactly the
   observation the audit exists to make, by turning "the judge did not react"
   into a small non-zero number.

   What replaced it: `sensitivity()` in stage 4 reports the tie rate directly,
   split by target vs non-target. Report the tie rate alongside every AUROC —
   AUROC is 0.5 both for a judge that never reacts and one that reacts at
   random, and only the tie rate separates those.

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
scripts/setup.sh        one-command bootstrap: deps for a role, then the hash check
scripts/smoke_judge.py  one real judge call on synthetic images  [JUDGE VM ONLY]
scripts/smoke_edit.py   one real FLUX edit on a synthetic image  [EDITOR VM ONLY]
scripts/diagnose_parse.py   why judge responses failed, from a scores parquet [CPU]
scripts/verify_corruption.py  did the corruption damage the image, and only
                        inside the mask? [CPU] -- run before any insensitivity claim
scripts/nuisance_report.py  paired-delta analysis across presentations [CPU]
scripts/run_shard.sh    one VM's share of stages 2+3
scripts/verify_determinism.sh   cross-VM hash check
tests/test_determinism.py   5 determinism properties
tests/test_stage0.py        selection logic via a stub COCO (no pycocotools)
tests/test_stage4.py        3 synthetic judges with known behaviour + floor filter
tests/test_nuisance.py      presentation axes + 3 judges with known nuisance
                        behaviour; also builds the fixture --dry-run needs
tests/test_syntax.py        every file parses; GPU modules import without torch
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
- **Cross-VM agreement is confirmed, not assumed.** Three independent VMs —
  `mcvgpu2025s-0050` and `mcvgpu2025s-0043` (2026-08-25), `mcvgpu2025s-0053`
  (2026-09-03) — all print `776feeddd281fa726195bf504c7b19c8` on
  `numpy 1.26.4 / cv2 4.11.0 / pillow 10.4.0`, matching the reference
  container. This is the invariant the whole five-way shard split rests on and
  it had never been tested before 2026-08-25. The remaining two VMs must
  reproduce it as they come online.
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
- **191.4s/image at 28 steps.** pilot (5) = 16 min, `main` (150) = 8.0h,
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
- **Stage 0 yield, and why the dataset changed to train2017 (2026-09-04).**
  Measured on `mcvgpu2025s-0053`, val2017's 5000 images:

  | | usable bases | histogram of usable regions per image |
  |---|---|---|
  | before category uniqueness (2026-08-26) | **187** (3.7%) | `0:2206 1:1883 2:724 3:150 4:35 5:2` |
  | after (2026-09-03) | **46** (0.92%) | `0:3202 1:1414 2:338 3:38 4:6 5:2` |

  The rule cost 75% of the pool, as suspected — `person` and `car` are COCO's
  two commonest categories and rarely appear alone. 46 does not cover `main`'s
  100, let alone 150.

  **Resolved by switching to train2017, keeping the strict rule.** 118,287
  images at the same 0.92% is ~1,090 bases. The old note here said train2017
  "is 19GB and will not fit next to FLUX on the 90G disk" — true of the split,
  irrelevant to us: **we never need the split.** `--survey` reads annotations
  only, and `--list-urls` prints the download URL of each qualifying image
  (from the record's own `coco_url`), so `wget -i` fetches ~200 files / ~30MB.
  Do not download an image split; the filter discards 99 of every 100.

  The train/val distinction carries no meaning for this audit: nothing is
  trained, and both splits are equally public to FLUX and to Qwen. Worth one
  sentence in the report to pre-empt the question, not a caveat.

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
region id, `required` on the top-level keys) plus compact output. **`maxLength`
was tried and removed** — it cost 7.5x for no quality gain; see below. Result: **parse rate 100%, region coverage 30/30, background /
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

**SUPERSEDED by greedy decoding.** Those figures assume `n=5 @ T=0.7`. The
pilot showed that config is unusable anyway (see PILOT VERDICT), and greedy
(`--temperature 0 --n-samples 1`) measures **2.84 s/request** — `main` at the
then-current 100 bases = 7,018 requests = **1.1h/VM across five VMs**, inside
the proposal's 1.2h estimate. Greedy fixed the measurement and the budget at
once. (`main` is 150 bases as of 2026-09-04: ~10,500 requests, ~1.7h/VM. The
s/request measurement is unaffected.)

**Per-stage notes (all four stages have now executed on real data; this
section is history plus the residual risk in each):**
- `stage0_coco.py` — **selection logic is now laptop-tested** by
  `tests/test_stage0.py` (2026-08-26), which drives `select`/`write_base` with
  a stub COCO. That was made possible by deleting an `assert isinstance(coco,
  COCO)` whose only effect was to force a `pycocotools` import — a library that
  only stage 0 needs — into the one function
  carrying the real risk. Still unverified: `annToMask`, and more importantly
  **whether val2017 contains enough qualifying images**. Run `--survey` first;
  it needs only the annotations JSON, no images and no writes.
  Three bugs fixed while testing: `config.yaml`'s `selection:` block was
  ignored in favour of hardcoded values; `instruction_for` was drawn twice per
  region (once to test for `None`, once for real) so the instruction written
  was a different draw from the one that passed the filter; and `"trash can"`
  is not a COCO category, so it matched nothing, silently, forever.

  **Category uniqueness, added 2026-09-03.** A region's category must appear
  exactly ONCE in the image — not merely once among the regions we kept. The
  old rule (`label in seen`) ran *after* the area band, so a 30%-area car was
  dropped for size and a 10%-area car was then happily kept: region list clean,
  photograph containing two cars. `"make the car red"` is then ambiguous to
  FLUX *and* to the judge, which puts noise straight into the `success` axis
  for a reason we created ourselves. `duplicate_categories()` now counts every
  annotation in the frame, **crowd annotations included** (a "crowd of cars"
  blob is exactly as ambiguous as a second individual car), and disqualifies
  any category appearing twice at or above `selection.duplicate_area_frac`
  (0.01, half of `min_area_frac`: visible, but too small to be a region). A
  40-pixel background car confuses nobody and discarding the image for it is
  pure yield lost — that floor is the whole reason the rule is not strict.

  Two structural consequences worth not undoing:
  - `select` and `survey` now share one filter, `candidates()`. They used to
    carry two copies (`instruction_for(...) is None` vs `label not in
    INSTRUCTABLE`) that were equivalent by accident. A survey that reports a
    yield the selector cannot deliver is worse than no survey.
  - `getAnnIds` is called with **no** `iscrowd` argument and crowds are dropped
    in Python. We need the crowd annotations for the duplicate count, and this
    also removes our dependence on pycocotools' `iscrowd=` comparison, which
    was on the unverified list above.

  **Instruction-family distribution, measured 2026-09-04** on the ~120 val2017
  bases then on disk (360 regions, from `cat data/bases/*/instruction.txt`):

  | family | regions | share |
  |---|---|---|
  | recolour | 204 | **56.7%** |
  | remove / erase | 57 | 15.8% |
  | add sunglasses | 50 | 13.9% |
  | make older | 49 | 13.6% |

  Removal targets by category: cup 13, potted plant 11, bowl 11, book 6,
  bottle 5, traffic light 3, parking meter 3, clock 2, fire hydrant 2,
  stop sign 1. `person` is **27.5% of all regions** — COCO's commonest
  category — and half of those draw "make the person look older", the vaguest
  instruction in the set and the hardest to score on the success axis.

  Two things this corrects:
  - **Removal is ~16% of regions, not the ~1/3** implied by "10 of 29
    categories". Categories are not equally frequent. Whatever the
    `score_preserve` question resolves to, it costs 16% of target regions.
  - **The colour monoculture already exists.** 57% of regions are "change this
    object's hue"; dropping removal would take it to ~67%. Removal is not what
    protects instruction diversity, so adding a material/texture family is
    worth doing on its own merits and independent of the removal decision.

  Caveat on both: those bases predate the uniqueness rule, which will hit
  `person` hardest of all (people almost never appear alone). Every proportion
  above will shift. Treat it as the shape of the old design, not a prediction.

  **Instruction design changed 2026-09-04, in three ways.** All three shift
  which string a region gets; none shifts which regions survive selection.

  - **A MATERIAL family.** `MATERIAL_TEMPLATES` x `MATERIALS = [wood, metal,
    glass, marble, leather]`, answering the colour monoculture above directly.
    `MATERIALIZABLE` is deliberately a **subset of `RECOLORABLE | REMOVABLE`**,
    so `INSTRUCTABLE` — and therefore every survey number ever measured
    against it, including the 46/1,090 yields — is **unchanged**. Categories
    move between instruction pools; none joins or leaves selection. Colour vs
    material is drawn **per region, not per category**, so one photograph gets
    a mix. `used_materials` mirrors `used_colors`: no two regions of an image
    get the same material, for the same reason.
  - **At most one removal per base** (`MAX_REMOVALS_PER_BASE = 1`). Some
    val2017 bases were nothing but removals ("remove the bottle, erase the cup,
    erase the book"): that base's edited image is mostly inpainted background,
    its `background` score largely scores our own inpainting, and corrupting an
    already-emptied region is a weak stimulus. Over the cap a removable
    category **falls back to the material family** rather than being dropped —
    dropping would change the region count `candidates()` already reported to
    `survey`, so `select` and `survey` would silently disagree again. This is
    why `REMOVABLE` must stay a subset of `MATERIALIZABLE`; `tests/test_stage0.py`
    asserts both containments, because either one breaking is silent.
  - **`ATTR_TEMPLATES` gained beard and moustache**, so `person` is no longer
    half "make the person look older" — the vaguest instruction in the set.

  Note this is independent of the open `score_preserve` question below: the cap
  reduces removal's blast radius without deciding whether removal stays.

  `select` also now rejects on the region-count window *before* drawing
  instructions, so an unusable image no longer advances the RNG stream.
  **Any change to this filter restales every `edit.png`** — `stage1_edit`
  fingerprints edits against a hash of their instruction, and shifting which
  regions survive shifts every instruction. Cheap now (5 edits); 8h after the
  150-base run.
- `stage1_edit.py` — **`--preflight` now settles most of it without a
  download.** It checks the `FluxKontextPipeline` class name, its `__call__`
  signature, the offload API, HF auth, gated-repo access, disk, and the stage-0
  inputs. `torch` is imported lazily so the input checks run on the laptop.
  `scripts/smoke_edit.py` does one real edit on a synthetic image and reports
  s/image, peak VRAM and the extrapolated budget. **Both have since run on the
  editor VM** — see the stage-1 block above. Nothing here is unverified now.
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
  **It has since run on real pilot data** (75 variants, 5 bases) and produced
  every table without incident. Two analyses were added afterwards because the
  real data demanded them: `sensitivity` (the tie rate) and `response_coherence`
  (do a variant's regions move together?). Both are covered by fixtures.

## Judge behaviour — settled, and the risks that are now closed

Settled 2026-08-25 (synthetic squares, real A.4.3 prompt, Qwen3-VL-8B):

- **The harness is correct.** Prompt parses, `background` / `overall_score`
  present, Equation (3) runs, images demonstrably reach the model (+396 tokens),
  vision path confirmed by a plain-question probe answering `red`/`blue`
  correctly. **Never re-debug the request path.**
- **The judge discriminates instruction-following.** Obeyed 25.0 vs ignored 0.0
  on the targeted region — the full width of the scale. The old placeholder
  prompt was the entire cause of the earlier all-5s result.

### RETIRED (2026-08-26): "semantic yes, photometric no" did NOT replicate

**This was the top risk for a week. The pilot on real COCO edits closed it, and
the answer was no.** Kept because the synthetic table below is a good example of
how far out of distribution flat squares are, and because the report should say
we checked.

Share of damaged regions whose score dropped, greedy, real photographs:

| | blur | remove |
|---|---|---|
| severity 1 | 20.0% | **20.0%** |
| severity 3 | 26.7% | **26.7%** |
| AUROC | 0.461 | 0.469 |

Identical at matched severities. There is no semantic/photometric split on real
data — the judge is equally insensitive to both, which is subsumed by the
stronger pilot finding that it does not respond per region at all.

**Still untested on real data:** `jpeg`, `noise`, `saturate`. The pilot ran only
`[none, blur, remove]`; `main` includes all five and gives the split a properly
powered second look. Do not treat it as permanently closed.

Everything below this line is the SYNTHETIC-SQUARES result that motivated the
hypothesis. It is superseded.

Re-ran on `mcvgpu2025s-0050` across all four corruptions. The earlier "the
judge ignores corruption entirely" was drawn from `noise` alone and was too
broad. What actually held **on synthetic squares**:

| corruption | region 0 succ/pres | region 1 (untouched) |
|---|---|---|
| clean | 25/25 | 25/25 |
| **remove** s1 | **20/15** | 25/25 |
| **remove** s2 | **20/15** | **20/15**  <- leakage |
| **remove** s3 | **20/15** | 25/25 |
| blur s1-s3 | 25/25 | 25/25 |
| jpeg s1-s3 | 25/25 | 25/25 |
| noise s1-s3 | 25/25 | 25/25 |

On synthetic squares the judge appeared to track semantic change and not
degradation: `remove` moved it a full 10 points, and moved `success` too
(25 -> 20), while blur, JPEG and noise returned a flat 25 at every severity.

**None of that survived contact with real photographs.** See the retirement note
above. The caveats we flagged at the time were the right ones and they were what
broke it: flat textureless squares are far out of distribution, the `remove`
severity ladder was already flat (15/15/15), and the region-1 drop at s2 but not
s1 or s3 was non-monotone — instability, not a spatial effect.

Worth keeping as a methodological point for the report: a synthetic sanity check
produced a clean, plausible, entirely wrong hypothesis, and only real data
caught it.

### Original framing — SUPERSEDED, kept only as history

Every confound listed here was subsequently eliminated: (1) the pilot ran on
real COCO edits; (2) blur and remove were both tried and behave identically;
(3) `success` and `preserve` are reported separately and neither localises.
Do not re-run any of it. The observation that started it, every cell 25.0
under `noise` only:

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

## PILOT VERDICT, 2026-08-26 - GO. Run `main`.

5 bases, 75 variants, `[none, blur, remove]`, Qwen3-VL-8B, **greedy
(--temperature 0 --n-samples 1)**. Parse 100%, coverage 100%.

### The stimulus is real and perfectly localised (verified, not assumed)

`scripts/verify_corruption.py`, mean 8-bit levels, region = 4.5% of image:

| corruption | inside mask | outside mask | masked pixels changed |
|---|---|---|---|
| blur s1 | 7.60 | 0.042 | 39% |
| blur s3 | 21.87 | 0.110 | 79% |
| remove s1 | 35.01 | 0.001 | 78% |
| remove s3 | 35.52 | 0.001 | 83% |

Contrast inside:outside is 179x to 25,000x. **"The judge did not react" is
therefore about the judge**, and the leakage analysis' core assumption -- that
untouched regions really are untouched -- holds on real edits, not just on the
synthetic fixture.

*Design note:* `remove` s1 and s3 differ by 0.5 levels. The severity ladder is
effectively binary for `remove`; only `blur` is graded. Do not read a flat
remove-severity response as insensitivity to severity.

### The finding, three ways, all pointing the same direction

1. **The score usually does not move.** 53-80% of DAMAGED regions receive a
   score identical to their clean control. Deterministic decoding, so this is
   not sampling noise.
2. **When it moves, it is not the damaged region.** target_unchanged equals
   other_unchanged to three decimals in three of four cells (0.667/0.667,
   0.667/0.667, 0.533/0.533). The region we damaged is no more likely to change
   than one we did not touch.
3. **The judge revises the whole image at once.** Only **27%** of variants show
   some regions moving while others hold, against **67%** expected if regions
   moved independently at the same overall rate. Corroborated by leave-one-out
   redundancy **R^2 = 0.52-0.56**.

Together these say the per-region score behaves like **one whole-image
judgement replicated across region slots**, which is precisely the failure this
audit was built to detect. AUROC 0.45-0.47 is a *consequence*, and on its own it
would have been unreadable -- AUROC is 0.5 both for a judge that never reacts
and one that reacts at random.

### Caveats, to state in the report

- **n = 5 photographs.** 90 rows per severity come from five images; the
  effective independent sample is 5. Suggestive, not reportable.
- Only `blur` and `remove` tested; one judge, one family.
- Regions average 4.5% of image area -- the low end of the 2-25% band.

### Greedy also fixed the budget

2.84 s/request measured. `main` at 150 bases = ~10,500 requests,
**~1.7h/VM across five VMs**, inside the proposal's budget (n=5 would be ~4.7h).
`main` also lifts n_target per cell from 15 to ~479, a **32x** power increase.

**Worth reporting separately:** at temperature 0.7 the same judge's score varied
across samples of an IDENTICAL input by 38% of the scale (SD 0.363 on a 0.959
range). A reward model that unstable is a problem for RL training regardless of
whether it localises. Consider a small n=5 @ T=0.7 run on ~10 bases purely to
characterise it.

## Nuisance and exploitability — implemented 2026-09-04, not yet run on a GPU

The last two of the four promised analyses. Both ask what the score does when
something changes that carries **no information about edit quality**.

### Presentation is a stage-3 flag, never a manifest column

`variant_id` is `sha1(base_id|target_region_id|corruption|severity|area_bin)`
(`schema.py:66-70`). A sixth field changes every id, hence every `seed_for()`,
hence every rendered byte — voiding `776feeddd281fa726195bf504c7b19c8`, which
three VMs have now confirmed. So `--presentation` re-packages the SAME images at
request-build time. Any pixel change happens in memory in `build_requests` and
is never written to disk. No manifest change, no re-render, less code.

For the same reason nothing was added to `corruptions.py`:
`verify_determinism.sh` hashes exactly the five ops named in its own list, so
*adding* a function there is safe but *editing* one is not. `presentation.py` is
a separate module and cannot touch that hash at all.

### The six axes

`baseline` is the identity — byte-identical messages to what stage 3 built
before this existed.

| axis | changes | role |
|---|---|---|
| `shuffle` | region list order in the prompt **and** the grammar's slots | nuisance, pure text |
| `subset` | drops exactly one non-target region | nuisance, pure text |
| `box` | bboxes drawn on **both** images | nuisance, visual |
| `noimg` | every image stripped | exploitability |
| `enhance` | global unsharp + contrast + saturation on the edit only | exploitability |

Two orthogonal groupings, and the distinction matters: TEXT/IMAGE is about the
**stimulus** (does a pixel change?), NUISANCE/EXPLOIT is about the **claim** (a
nuisance axis is one where movement is the finding; an exploit axis is one where
movement is the hypothesis). Reporting `enhance` under "does a null change move
the score" is a category error, and the first version of the report did it.

Design notes that are not obvious:
- **`shuffle` must permute the schema too.** `_build_sc_schema` fills
  `prefixItems` positionally from the id list (`judge_prompt.py:233`), so the
  grammar pins slot k to `ids[k]`. Permuting the prompt but not the schema
  would force the model back into canonical order — an axis that runs, costs
  GPU time and measures nothing. `test_schema_slots_follow_the_presented_order`
  exists solely to catch that.
- **Order and drop-choice come from `sha256(variant_id|mode|region_id)` sorts,
  not `random.Random`.** No RNG means no dependence on the interpreter's random
  implementation. ~1/n! of `shuffle` variants draw the identity permutation;
  that is the honest null, not a bug.
- **`box` draws on both images.** On the edit alone the box would itself be a
  difference from the source and could read as an editing artifact.
- **`enhance` touches the edit only.** The source is the real photograph; the
  exploit scenario is an RL editor learning to emit sharpened output.
- **Expect `enhance` to raise `reward` while possibly LOWERING `phi`.**
  Sharpening the edit makes it differ more from the source and the SC prompt
  asks about preservation. Eq. (3) is `sqrt(phi * AES)/C` with `AES = min(PQ)` a
  single image-level factor, so a cosmetic lift can raise every region's reward
  with no edit improved. Those two moving in opposite directions is the result,
  not a contradiction — read `reward` and `phi` side by side, never collapsed.
- **`judge_prompt.py` needed no changes.** `build_sc_prompt` and
  `sc_json_schema` already accepted arbitrary region lists in arbitrary order.

### The analysis is in `scripts/`, and that is not just about merge conflicts

**The delta is a different delta.** `stage4_analyze.delta_table` compares a
CORRUPTED variant against its CLEAN control. Here the image is fixed and the
packaging changes, so the pairing key is `(judge, variant_id,
scored_region_id)` across presentations.

Worse than answering the wrong question: `delta_table` does not group by
`presentation`, so a glob catching several conditions **pools every condition's
clean controls into one baseline, silently**. Same hole in `redundancy` and
`leakage_matrix`. Hence `out/nuisance/`, whose filenames cannot match
`out/scores_shard*.parquet`. If stage 4 ever grows a `presentation` groupby,
this separation can relax; until then do not point stage 4 at these files.

`nuisance_report.py` imports `delta_table`, `sensitivity`, `noise_floor`,
`usable` and `load` from stage 4 rather than re-deriving them, so the tables read
in the same vocabulary — same tie rate, same floor, same |delta|.

**`scores_*.parquet` vs `floor_*.parquet` is load-bearing naming.** The floor
run also carries `presentation == "baseline"`: same variants, same regions, same
label. It is indistinguishable from the greedy baseline by any column, so
detecting it programmatically does not work (both share a groupby key). Two
globs, two filename prefixes, no detection logic to get wrong.

### The sampling tension, and how it resolves

Nuisance needs **greedy** to be attributable — at T=0.7 you cannot tell "the
score moved because I shuffled the list" from "the judge is stochastic". But
greedy destroys the **denominator**: `noise_floor` is the SD across repeated
samples, and at n=1 it is NaN.

Resolution: run every condition greedy, and take the floor from one sampled
baseline run **over the identical variants** — not a separately chosen slice, or
the floor and the deltas are over different images. That sampled run is the
degenerate nuisance axis, the case where nothing changed at all, so it is the
natural baseline for every presentation delta. **It is not an optional report
extra; it is the denominator this analysis needs.**

### Sizing, and the one-VM constraint

Run on `--profile pilot` (5 bases), **not `--limit`**: `stage3_judge` does
`df.head(limit)` after sharding and the manifest is ordered base -> region ->
control -> corruptions, so a row limit can take a base's controls and drop its
corrupted rows. The profile route also makes the sweep a strict subset of
`main`: `build_manifest.py:22` slices `bases.json[:n_bases]`, and pilot's
corruption/severity/area sets are subsets of main's, so **every pilot
`variant_id` is also a main `variant_id`**.

~75 variants -> 150 requests per condition. Seven runs (six greedy + one
sampled) is ~1h plus ~20 min of repeated engine startup. On `main` the same
sweep would be ~20x that — don't.

**Run the whole sweep on ONE VM.** Cross-VM byte-equality of `enhance` and `box`
is unverified (PIL, not covered by the fixture hash) and staying on one machine
makes the question moot. Do not shard these.

The sweep is **self-contained**: pilot is `[none, blur, remove]`, so
`scores_baseline.parquet` holds clean controls AND damaged variants. Real-damage
deltas and nuisance deltas both come out of it; nothing from the main run is
needed, and it need not use the same photographs as an earlier pilot.

### Verified vs unproven

**Verified on the laptop, 2026-09-04** (`tests/test_nuisance.py`, 16 checks):
- The grammar follows the shuffled prompt order.
- All six axes build every request end to end on a synthetic 3-base fixture,
  with region ids parsed back **out of the prompt text** and checked against
  meta. `shuffle` 12/18 non-canonical, `subset` 18/18 with the target retained.
- `noimg` emits zero image parts in both the SC and the PQ request.
- `enhance` reaches both requests (PQ is where AES comes from), leaves the
  source byte-identical, is deterministic, size-preserving and global.
- Eq. (3): doubling PQ at fixed phi multiplies every region's reward by exactly
  root 2.
- `nuisance_report` on three fabricated judges: a nuisance-immune one reports
  exactly 0, a nuisance-sensitive one reports `vs_damage > 1`, and an AES
  exploit raises `reward` while `phi` stays flat.

**Unproven — only a judge VM can settle these:**
- Whether vLLM/xgrammar accepts a permuted `prefixItems` schema at all. This is
  the one thing that could still invalidate the `shuffle` axis.
- Grammar compile cost when `shuffle` turns one schema per base into up to n!.
  `_sc_schema_cached` is `lru_cache(maxsize=64)` (`judge_prompt.py:197`); fine
  at 5 bases, a thrash hazard at `main` scale.
- Cross-VM equality of `enhance`/`box`. Deliberately untested; see above.
- Every actual score.

### Incidental: `dry_run` no longer restates `run()`

`dry_run` printed `repetition_penalty=1.05` (the code has used 1.1 since
760707a) and "max_tokens is 1024" beside a printed 1536 (`32 -> 1024` in
38f2072, `-> 1536` in 4756cc5). Root cause was not carelessness: it hand-copied
`run()`'s sampling dict as a **string literal**, so every change needed a second
edit elsewhere and never got one. Both now read `sampling_base()`. Its own
comment claimed it "mirrors run() rather than restating them from memory" — that
is now true.

## Session close-out, 2026-08-26

The pipeline runs end to end on real data. What remains is the experiment, not
the harness.

**DECIDE FIRST (deferred 2026-08-26, not handled):**

- ~~**Make greedy the default in `run_shard.sh`?**~~ **DECIDED 2026-09-04:
  greedy everywhere, in both places, plus one deliberate sampled run.**

  `n=5 @ T=0.7` was not an arbitrary default. `git log -L` on that line shows
  it bought two things: the **noise floor** (`stage4.noise_floor` is the SD
  across repeated samples of one input, and there is no other route to it), and
  the **expected-score readout**, which needed `logprobs=20` over several
  samples. Reason two is dead — logprobs are gone and
  `expected_score_from_logprobs` raises. Reason one is still real, and that is
  the point: **n=5 is a measuring instrument, not a production setting**, and
  it had been left switched on for the main run.

  What changed:
  - `stage3_judge.py` defaults are now `--n-samples 1 --temperature 0`.
  - `run_shard.sh` passes both **explicitly**. It previously passed neither, so
    whatever the module happened to default to silently became the main run's
    configuration. Stating it in the script means changing the module default
    can never again change the main run without someone noticing.
  - **Deliberately not overridable by an environment variable.** Sampling must
    match across all five VMs for the shards to be comparable, exactly like
    `--gpu-util`; a per-VM override is a way to break that quietly. `TEMP` as a
    variable name is also a collision with the conventional temp-directory
    variable. And `run_shard.sh` hardcodes `--out out/scores_shard${SHARD}`, so
    it is the wrong vehicle for a floor run regardless — that run has its own
    `--out`.
  - The greedy default creates a NEW footgun, so `main()` guards it: asking for
    `--n-samples 5` without also raising the temperature returns 5 **identical**
    samples (greedy is deterministic at a fixed seed), costs 5x, and measures no
    floor. Stage 4 reports a zero floor honestly, but by then the GPU time is
    gone, so the warning fires before the engine loads.
  - Scores parquets now carry `n_samples` and `temperature` columns. `judge`
    was already recorded; sampling moves the numbers further than the choice
    between two Qwen sizes does, and five merged shards otherwise cannot say
    how any of them was decoded.

  Safe to switch now only because nobody has run `main` yet. Mid-run this would
  have made shards incomparable.
- ~~**Whether to keep `remove` on the severity axis.**~~ **DECIDED
  2026-09-04: dropped from severity comparisons, reported as binary.** Its
  severity 1 and 3 differ by 0.5 8-bit levels out of 35 (against blur's
  7.60 -> 21.87), because inpainting removes the object at every radius. A flat
  remove-severity response is an ABSENT STIMULUS, not judge insensitivity, and
  pooled into a by-severity table it reports our own design as a finding.
  `stage4_analyze.FLAT_SEVERITY = {"remove"}` and `collapse_flat_severity()`
  relabel it to a single `binary` condition inside `delta_table` and
  `axis_table`, so no caller can forget. Nothing is dropped or pooled across
  corruptions: `remove` keeps its own row in every per-corruption table and
  appears as its own `binary` row in the by-severity ones, so reading down the
  1 -> 3 column really is reading an effect-size ladder.
  The rejected alternative was redefining severity for `remove` in
  `corruptions.py` (e.g. alpha-blending the inpaint). Better science, but it
  changes corruption bytes, so all five VMs re-run the determinism check and
  any judged data is void -- too much for a ladder on the one corruption we
  already know is the strongest stimulus.
- **`REMOVE_TEMPLATES` in stage 0 — STILL OPEN, but the framing was wrong.**
  It generates "remove the X" instructions, which collide with `remove` being
  a corruption. The stronger objection was that "how well is this region
  preserved" is ill-posed for a region the instruction deleted — a correct
  removal would score ~0 on preserve, hence phi = min(success, preserve) = 0,
  the floor, and `drop_floored` would then discard those regions entirely.

  **That claim is NOT established and was overstated (corrected 2026-09-04).**
  A.4.3's OUTPUT FORMAT names the axis `score_preserve` and defines it as
  "0=completely different, 25=minimal effective edit" — but its RULES section
  calls the same quantity the "overall overediting score", and the BACKGROUND
  rule is framed identically ("Penalize unexpected edits, layout changes,
  artifacts outside editing regions"). Under the overediting reading, a clean
  removal that did not disturb its surroundings is a minimal effective edit and
  should score HIGH. The prompt is genuinely ambiguous and the paper's own
  gloss argues against the pessimistic reading.

  **Settle it by measurement, not by reading.** The question is not what the
  rubric means but what base Qwen3-VL-8B does with it. Compare `sc_preserve` on
  removal-target regions against recolour-target regions, clean controls only,
  same images:

      python - <<'EOF'
      import pandas as pd, glob
      df = pd.concat(pd.read_parquet(f) for f in glob.glob("out/scores_shard*.parquet"))
      c = df[df.is_control & (df.scored_region_id == df.target_region_id)]
      print(c[["base_id","instruction","scored_region_id","sc_success","sc_preserve"]].to_string())
      EOF

  The pilot scores parquet lives on `mcvgpu2025s-0050`, not 0053.
  Whatever the answer: if base Qwen reads that axis differently from how the
  paper means it, that gap is itself reportable. We audit the prompt-based
  protocol, not the released fine-tuned reward model, and an axis ambiguous
  enough to be read two ways is exactly what that distinction exists to catch.

  Two things hold regardless of the answer:
  - **At most one removal per base.** Some existing bases are entirely
    removals ("remove the bottle, erase the cup, erase the book"). That base's
    edited image is mostly inpainted background, its `background` score is
    largely scoring our own inpainting, and corrupting an emptied region is a
    weak stimulus.
  - **The colour monoculture is the real diversity problem, not removal.**
    See the instruction-family distribution in the stage-0 notes above.

**FOR THE WRITE-UP (do not lose):**

- **Every pilot number rests on 5 photographs.** 90 rows per severity come from
  five images; the effective independent sample is 5, not 90. Say so explicitly
  wherever pilot numbers appear.
- **The temperature-0.7 instability is its own finding.** SD 0.363 on a 0.959
  range, across samples of an IDENTICAL input. A reward model that unstable is
  a problem for RL training regardless of whether it localises.

**Do next, in order:**
1. ~~Re-run `--survey` under the category-uniqueness rule~~ **DONE 2026-09-03:
   val2017 yields 46. Switched to train2017 (~1,090 expected); confirm with
   `--survey` on `instances_train2017.json`, then `--list-urls | wget`.**
2. Editor VM: re-run stage 0 against train2017, then edit the chosen number of
   bases (~189s each), then `tar czf bases.tar.gz -C data bases` and upload.
   Everything downstream is blocked on that tarball. Note the base ids change
   completely — they are COCO image ids, and the split changed.
3. ~~Run the pilot~~ **DONE — verdict GO, see PILOT VERDICT above.** The axis
   table turned out to be the wrong readout: the informative one is the tie
   rate (`sensitivity`), because 53-80% of damaged regions do not move at all
   and an axis mean cannot show that.
4. Cross-VM determinism hash from the two VMs that have not reported it.
5. `main`: 150 bases, **~1.7h/VM** at greedy, sharded five ways.
6. The nuisance/exploitability sweep: ~1h on ONE VM that has `data/bases` and a
   judge checkpoint. Independent of `main` and of which photographs it uses, so
   it can run the moment any VM has bases. Run the `--dry-run` on real bases
   first — 30 seconds, no GPU — since the only untested assumption left is the
   shape of a real `regions.json`.

**Base count DECIDED 2026-09-04: `main` is 150 bases.** `config.yaml` says so.
Not pool-limited — train2017 supplies ~1,090. The 50 over 100 cost 2.7 extra
hours on the editor VM (5.3h -> 8.0h, serial, blocking) and nothing anywhere
else: stage 3 goes 1.1 -> 1.7h/VM, disk is a rounding error.

State the gain honestly in the report rather than overselling it: regions
within an image are NOT independent (that is the finding — coherence 27%,
redundancy R^2 0.52-0.56), so effective n is the photograph count, and
100 -> 150 narrows intervals by 1/sqrt(1.5), ~18%. It crosses no threshold —
the tie rate was already comfortable at 100 and AUROC is not resolvable at
either. It is cheap insurance, not a new claim.

Do not push to 200 without re-timing: stage 1 stops fitting one night, and it
is the serial stage everything else waits on. If more power is ever wanted,
more PHOTOGRAPHS is the lever, never more regions per photograph.

**Do not re-litigate** (each was measured, not argued):
- `--gpu-util` 0.89; the window is (0.861, 0.901) and both ends fail.
- Stage 1 `--offload sequential`; model-level cannot fit, ever.
- `--reasoning free`; `bounded` costs 7.5x for identical quality.
- Schema-constrained decoding is mandatory — unconstrained covers ~43% of
  regions while reporting a healthy parse rate.
- A 4B judge does NOT fix throughput. 26x concurrency bought 2%.
- Greedy in both `stage3_judge` and `run_shard.sh`; the floor is a separate
  `--temperature 0.7 --n-samples 5` run, and no env-var override exists on
  purpose. Decided 2026-09-04, see the DECIDE FIRST entry.
- Presentation axes are a stage-3 flag, not a manifest column. A sixth
  `variant_id` field re-renders everything and voids the fixture hash.
- The nuisance analysis lives in `scripts/nuisance_report.py`, not stage 4:
  it pairs on `(variant_id, scored_region_id)` across presentations, and
  `delta_table` would pool the conditions' controls without saying so.

**Disk is the live constraint on `mcvgpu2025s-0050`:** FLUX (~34GB) +
Qwen3-VL-8B (~16GB) + Qwen3-VL-4B (~9GB) + COCO val2017 (~1GB) + a second venv
with CUDA wheels, against a 90G root. `run_shard.sh` used to force
`HF_HOME=$HOME/hf_cache` while every manual run used the default
`~/.cache/huggingface`, so it re-downloaded a cached 16GB checkpoint into an
empty second cache and ran the disk out mid-transfer. Fixed: the script now
inherits `HF_HOME`, reports which cache it will use, and refuses to start if the
model is uncached with under 25G free. **Do not set `HF_HOME`.**

**Still unverified:** determinism on the last two VMs. `mcvgpu2025s-0053`
reported `776feeddd281fa726195bf504c7b19c8` on 2026-09-03, making three of five.
That is the only outstanding verification. (`run_shard.sh` ran end to end for
the pilot; stage 4 has seen real 5-base data.)

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
   than what we had; `max_tokens` 32 -> 1536, since A.4.3 demands per-region
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
3. ~~Nuisance and exploitability tests are designed but not implemented.~~
   **IMPLEMENTED 2026-09-04, never run on a GPU.** See "Nuisance and
   exploitability" below. Laptop-side is verified; the sweep is ~1h on one VM
   and blocked only on that VM having `data/bases`.
4. Figures for the report.

## Guardrails

- Don't run the `full_cross` profile casually: 24,800 variants → 99,200 requests.
  `main` at 150 bases x 3.19 regions is ~5,300 variants / ~10,500 requests,
  **~1.7h/VM at greedy**, sized to the proposal's stated budget.
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

## The team status page (KEEP THIS UPDATED)

There is a published Artifact serving as the shareable status report for the
other four students:

**https://claude.ai/code/artifact/be35225a-248e-4e38-8b62-c18a695a2407**

It is the teammate-facing summary of everything in this file: pipeline status,
the assumptions the hardware overturned, the pilot verdict, the judge
selection, the open decisions, and the step-by-step determinism-hash
instructions.

**When you change project state, update it in the same session.** To do so:

1. Read the current page with the Artifact tool: `action: "read"`, passing that
   URL, which returns the raw HTML.
2. Edit the HTML, then publish with `url` set to that same URL so it updates in
   place rather than creating a second artifact. Publishing without `url` from
   a later conversation creates a duplicate and the team keeps reading the
   stale one.
3. Keep the favicon as the existing emoji and the `<title>` stable — people
   find it by both.

Things that should trigger an update: a pilot or `main` result, a decision from
the "DECIDE FIRST" list being made, a VM reporting its determinism hash, or any
finding being retired the way "semantic yes, photometric no" was.

## The three docs have different jobs — keep them separate

They drifted into near-duplicates once and had to be pulled apart. Before adding
anything, decide which one it belongs in:

| | Audience | Contains | Does NOT contain |
|---|---|---|---|
| `README.md` | anyone who opens the repo | what the project is, layout, requirements, installation, pipeline, usage, testing, reproducibility caveats | findings, assignments, timeline, status |
| `TEAM_BRIEF.md` | the four other students | plain-language explanation, current status, what the pilot found, **their missions**, decisions to make together, VM gotchas, timeline | setup commands (link to README), implementation detail, measurement tables |
| `CLAUDE.md` | future Claude Code sessions | everything measured, every retired hypothesis, why each decision was made, what not to re-litigate | anything a human needs to copy-paste |

`TEAM_BRIEF.md` is deliberately the *simplest* of the three: short paragraphs,
no jargon, missions up front. Do not push measurement tables or implementation
detail into it — those belong here. A number reaches the brief only in the form
a teammate would repeat out loud ("53-80% of damaged regions score identically",
not "ties 0.811 / 0.722 on phi").
