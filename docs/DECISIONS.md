# Decision log

Why the harness is the way it is. Every entry here was measured, not argued.
`CLAUDE.md` carries the conclusions; this file carries the evidence, so a future
session can check a default instead of re-deriving it.

Judge findings live in [`FINDINGS.md`](FINDINGS.md).

---

## Hardware

### The `--gpu-util` window is two-sided, ~(0.861, 0.901)

`A10-24Q` reserves ~2.4GiB, leaving **21.37 of 23.72GiB** free at startup.

- Too high: vLLM budgets against *total* VRAM but demands that much *free*, so
  >0.901 dies before loading a weight.
- Too low: weights (16.64GiB) + encoder cache + activation peak need ~20.41GiB,
  so <0.861 sizes the KV cache **negative**.

Measured: 0.85 -> **-0.25GiB, dies**; 0.87 -> +0.23; 0.89 -> +0.70.
`DEFAULT_GPU_UTIL = 0.89`.

`DEFAULT_GPU_UTIL` was 0.85 until 2026-08-26 and could never have worked. Every
smoke test passed `--gpu-util 0.89` explicitly, so the default was first
exercised on the first real run.

### `limit_mm_per_prompt` must carry `"video": 0`

Qwen3-VL accepts video. Left unset, vLLM sizes the encoder cache for a
max-length video (151,250 tokens) and OOMs in `profile_run` trying to allocate
4.62GiB on top of 16.8GiB of weights.

### Eager mode, small batches

16.8GiB of weights on a 20.16GiB budget leaves ~3.4GiB for everything else, so
`load_engine` runs **eager** (no CUDA graphs), `max_num_batched_tokens=2048`,
`max_model_len=4096`. At 8192 + graphs the KV cache came out at **-0.40GiB**.

### The serial judge does not matter, and a 4B judge does not fix it

At `--gpu-util 0.89` the KV cache is 0.70GiB = 5,072 tokens, max concurrency
**1.24x** — effectively serial. Measured 2026-08-26: a 4B judge at **26.22x**
the concurrency finished **2% faster**. Grammar decoding, not batching, set the
pace. See "Grammar cost" below.

A 4B remains worth running as a cross-scale comparison, just not for speed.

### Disk is the live constraint on `mcvgpu2025s-0050`

FLUX (~34GB) + Qwen3-VL-8B (~16GB) + Qwen3-VL-4B (~9GB) + COCO val2017 (~1GB) +
a second venv with CUDA wheels, against a 90G root.

`run_shard.sh` used to force `HF_HOME=$HOME/hf_cache` while every manual run
used the default `~/.cache/huggingface`, so it re-downloaded a cached 16GB
checkpoint into an empty second cache and ran the disk out mid-transfer. The
script now inherits `HF_HOME`, reports which cache it will use, and refuses to
start if the model is uncached with under 25G free. **Do not set `HF_HOME`.**

---

## Determinism

### Version pins are load-bearing

If two VMs produce different bytes for the same `variant_id`, scores from
different shards are not comparable and the audit is invalid. `numpy`,
`opencv-python-headless` and `Pillow` are pinned for that reason. To bump one,
re-run `scripts/verify_determinism.sh` on two machines and compare hashes first.

### The OpenCV pin is coupled to vLLM

vllm 0.11.0 requires `opencv-python-headless>=4.11.0`, so judge VMs cannot
install an older one. The pin is `==4.11.0.86` — the floor, exactly pinned.

**Never relax it to a range to end a resolver fight.** A range lets two VMs land
on different builds, which is precisely the failure determinism exists to
prevent. If a future vLLM raises the floor, bump to the new floor exactly and
re-run the equivalence check below.

### 4.10.0.84 -> 4.11.0.86 is byte-equivalent for our corruptions

Checked twice. On the laptop by rendering the fixture under both versions with
`numpy`/`Pillow` held identical (all five per-corruption hashes and the total
matched), then on VM `mcvgpu2025s-0050` (2026-08-25), which printed
`776feeddd281fa726195bf504c7b19c8` — the pre-bump container reference — while
running 4.11.0.86. The bump forced by vLLM re-baselines nothing.

### Cross-VM agreement is confirmed, not assumed

Three independent VMs — `mcvgpu2025s-0050` and `mcvgpu2025s-0043` (2026-08-25),
`mcvgpu2025s-0053` (2026-09-03) — all print `776feeddd281fa726195bf504c7b19c8`
on `numpy 1.26.4 / cv2 4.11.0 / pillow 10.4.0`, matching the reference
container. This is the invariant the five-way shard split rests on and it had
never been tested before 2026-08-25. The remaining two VMs must reproduce it.

### The laptop's hash differs for platform reasons only

Linux gives `776feedd…`; Windows gives `5073799d…` on identical pins. Both are
internally stable. **Compare VM hashes to `776feedd…`, never to the laptop's.**

Per-corruption laptop hashes, to localise a future cross-VM mismatch:
`blur b9d191e7be9487bc`, `saturate 55ab816f6c3f49f4`, `noise 1692c77f84ed2c3a`,
`jpeg 4a0f806a2ce9d42e`, `remove 5fc3f8333affcbc2`.

### `verify_determinism.sh` hashes a fixed op list

It hashes exactly the five ops named in its own list, so *adding* a function to
`corruptions.py` is safe but *editing* one is not. `presentation.py` is a
separate module and cannot touch that hash at all.

---

## Stage 0 — selection

### Category uniqueness (added 2026-09-03)

A region's category must appear exactly ONCE in the image — not merely once
among the regions we kept. The old rule (`label in seen`) ran *after* the area
band, so a 30%-area car was dropped for size and a 10%-area car was then kept:
region list clean, photograph containing two cars. `"make the car red"` is then
ambiguous to FLUX *and* to the judge, putting noise into the `success` axis for
a reason we created ourselves.

`duplicate_categories()` counts every annotation in the frame, **crowd
annotations included** (a "crowd of cars" blob is as ambiguous as a second
individual car), and disqualifies any category appearing twice at or above
`selection.duplicate_area_frac` (0.01, half of `min_area_frac`: visible, but too
small to be a region). A 40-pixel background car confuses nobody and discarding
the image for it is pure yield lost — that floor is why the rule is not strict.

Two structural consequences worth not undoing:

- `select` and `survey` share one filter, `candidates()`. They used to carry two
  copies (`instruction_for(...) is None` vs `label not in INSTRUCTABLE`) that
  were equivalent by accident. A survey reporting a yield the selector cannot
  deliver is worse than no survey.
- `getAnnIds` is called with **no** `iscrowd` argument and crowds are dropped in
  Python. We need crowd annotations for the duplicate count, and this removes
  our dependence on pycocotools' `iscrowd=` comparison.

### The dataset switched to train2017 (2026-09-04)

Measured on `mcvgpu2025s-0053`, val2017's 5000 images:

| | usable bases | histogram of usable regions per image |
|---|---|---|
| before category uniqueness (2026-08-26) | **187** (3.7%) | `0:2206 1:1883 2:724 3:150 4:35 5:2` |
| after (2026-09-03) | **46** (0.92%) | `0:3202 1:1414 2:338 3:38 4:6 5:2` |

The rule cost 75% of the pool — `person` and `car` are COCO's two commonest
categories and rarely appear alone. 46 does not cover `main`'s 150.

Resolved by switching to train2017 and keeping the strict rule: 118,287 images
at the same 0.92% is ~1,090 bases. An earlier note claimed train2017 "is 19GB
and will not fit next to FLUX" — true of the split, irrelevant to us, because
**we never need the split**. `--survey` reads annotations only, and
`--list-urls` prints each qualifying image's own `coco_url`, so `wget -i`
fetches ~200 files / ~30MB. **Do not download an image split**; the filter
discards 99 of every 100.

The train/val distinction carries no meaning here: nothing is trained, and both
splits are equally public to FLUX and to Qwen. Worth one sentence in the report
to pre-empt the question, not a caveat.

### Three bugs found by `tests/test_stage0.py` (2026-08-26)

- `config.yaml`'s `selection:` block was ignored in favour of hardcoded values.
- `instruction_for` was drawn twice per region (once to test for `None`, once
  for real), so the instruction written was a different draw from the one that
  passed the filter.
- `"trash can"` is not a COCO category, so it matched nothing, silently.

Testing this was made possible by deleting an `assert isinstance(coco, COCO)`
whose only effect was to force a `pycocotools` import into the one function
carrying the real risk.

`annToMask` was the last unverified path and is now exercised: `scripts/stage0.sh`
on train2017 (2026-09-05) decoded 476 real COCO polygons into masks across 150
bases, and every one of them survived the round trip through stage 1's
resolution check.

### Instruction design changed 2026-09-04, in three ways

All three shift which string a region gets; none shifts which regions survive
selection.

- **A MATERIAL family.** `MATERIAL_TEMPLATES` x `MATERIALS = [wood, metal,
  glass, marble, leather]`, answering the colour monoculture (57% of regions
  were hue changes — see [`FINDINGS.md`](FINDINGS.md)). `MATERIALIZABLE` is
  deliberately a **subset of `RECOLORABLE | REMOVABLE`**, so `INSTRUCTABLE` —
  and every survey number ever measured against it, including the 46/1,090
  yields — is **unchanged**. Categories move between instruction pools; none
  joins or leaves selection. Colour vs material is drawn **per region, not per
  category**, so one photograph gets a mix. `used_materials` mirrors
  `used_colors`: no two regions of an image get the same material.
- **At most one removal per base** (`MAX_REMOVALS_PER_BASE = 1`). Some val2017
  bases were nothing but removals ("remove the bottle, erase the cup, erase the
  book"): that base's edited image is mostly inpainted background, its
  `background` score largely scores our own inpainting, and corrupting an
  already-emptied region is a weak stimulus. Over the cap, a removable category
  **falls back to the material family** rather than being dropped — dropping
  would change the region count `candidates()` already reported to `survey`, so
  `select` and `survey` would silently disagree again. This is why `REMOVABLE`
  must stay a subset of `MATERIALIZABLE`; `tests/test_stage0.py` asserts both
  containments, because either one breaking is silent.
- **`ATTR_TEMPLATES` gained beard and moustache**, so `person` is no longer half
  "make the person look older" — the vaguest instruction in the set.

`select` also now rejects on the region-count window *before* drawing
instructions, so an unusable image no longer advances the RNG stream.

**Any change to this filter restales every `edit.png`.** `stage1_edit`
fingerprints edits against a hash of their instruction, and shifting which
regions survive shifts every instruction. Cheap at 5 edits; 8h after the
150-base run.

---

### Cross-VM determinism: four of five

`776feeddd281fa726195bf504c7b19c8` on `0050`, `0043`, `0053` and `0004`
(2026-09-05), each with numpy 1.26.4 / cv2 4.11.0 / pillow 10.4.0 and all five
properties passing.

## Stage 1 — editing

Verified on `mcvgpu2025s-0050`, 2026-08-26.

### `enable_model_cpu_offload` CANNOT work on an A10 and never could

It makes one whole *component* resident, and FLUX Kontext's transformer is
**23.8GB** in bf16 (download: 9.95 + 9.98 + 3.87) against the **21.37GiB** an
A10-24Q leaves free. It OOMed at step 0 of 28 after a clean 34GB fetch. This is
VRAM, not disk — unloading a judge or clearing the cache changes nothing.

`enable_sequential_cpu_offload` streams submodules and is the default. Peak VRAM
**2.39GiB of ~21GiB**.

### 191.4s/image at 28 steps on `0050`; 215.4s on `0004`

Re-measured on `mcvgpu2025s-0004`, 2026-09-04: **215.4s/image**, same 28 steps,
same sequential offload, same peak VRAM (2.39GiB). 13% slower than `0050` — the
stage is PCIe-bound under sequential offload, so per-VM host bandwidth moves it.
Time the VM you will actually run on; do not inherit a sibling's figure.

At 215.4s: pilot (5) = 18 min, `main` (150) = **9.0h**, `full_cross` (200) =
12.0h, plus ~6 min to load the pipeline. `main` still fits one night, so no
quantization is needed. If it ever is, the lever is an NF4 transformer (~6GB,
resident, removing the PCIe round trip that dominates), **not** fewer steps —
the card is 90% idle, not compute-bound.

### The resize-back is load-bearing

Kontext returns 1024x1024 for a 448x448 input. Without the resize-back in
`stage1_edit`, every stage-0 mask would index a wrong-sized `edit.png` and stage
2 would corrupt the wrong pixels in every variant.

A latent misalignment was fixed here: `out.resize(src.size)` ran *after*
`src.thumbnail()` had mutated `src` in place, so any source larger than
`--max-side` would have produced an `edit.png` at the thumbnailed size while
stage 0's masks stayed at source resolution. COCO images are <=640px so it never
fired, but `--max-side` is a knob and the bug was one config change away. It now
resizes to the size captured before thumbnailing.

### The editor follows instructions

Target region `[30,30,200]` -> `[244,4,6]` (redness -85 -> +239); the untouched
control moved 15.6 against the target's 144.7. Some global drift exists and
shows up in the noise floor.

### The edit does not always preserve layout, and that confound points our way

Measured on all 150 main bases, 2026-09-05, `scripts/verify_edit_drift.py`:

| edge IoU (Canny, dilated, colour-invariant) | |
|---|---|
| p5 / p25 / p50 / p75 / p90 | 0.24 / 0.36 / 0.46 / 0.58 / 0.69 |
| min / max | 0.16 / 0.93 |
| below 0.40 | 54 of 150 |
| below 0.30 | 19 of 150 |

Photometric drift outside every mask is high too (median 44.9 of 255, 10 sources
are black-and-white photographs that come back colorized), but that number is
mostly harmless and was nearly over-read: stage 2's baseline is `edit.png`, not
`source.png`, so a global recolour cancels out of every delta. The colour-blind
edge measure is the one that matters.

Why it matters: stage 0 computes masks on `source.png`, stage 2 applies them to
`edit.png`, and stage 3 shows the judge boxes in SOURCE coordinates. Where FLUX
re-poses a person or re-composes an interior, the mask named "person" covers
whatever moved into that spot — so we corrupt background while telling the judge
we damaged a region. **That looks exactly like a judge that cannot localise,
which is this project's finding.** The confound mimics the result, so it cannot
be waved away.

The low tail is not random: it is overwhelmingly indoor furniture (bed, couch,
chair, vase). Street scenes hold their geometry; rooms get re-composed. The
verified pairs bear this out — `000000268556` (motorcycle, layout visibly
intact) scores 0.86, and the visibly re-posed `000000559665` sits far below.

**Decision: measure and report, do not filter the data.** Every headline number
is reported twice — all 150, and the subset above `--min-edge-iou` (0.4 by
default) — via `stage4_analyze --drift-csv`. If the two agree, the result does
not rest on the doubtful bases and the proxy's weakness stops mattering. Only if
they diverge is a per-region check worth its cost, and that check must use an
INDEPENDENT detector (OWLv2); the judge under audit cannot certify our ground
truth.

Two alternatives were considered and rejected. Dropping the low-IoU bases and
re-editing replacements is cheapest before the manifest is frozen, but it trades
a measurable confound for an invisible selection bias — the discarded scenes are
precisely the dense, cluttered, multi-object interiors where per-region scoring
should earn its keep — and it destroys the evidence that the result is robust.
Running the detector pass up front buys per-region truth for ~2h, but it is
insurance against a risk not yet observed: if the split agrees, no amount of
detector precision improves on that.

`edge_iou` is an UPPER BOUND on displacement. "Change the chair to glass" is
supposed to destroy that object's edges, so a low score means "cannot vouch for
this base", never "this base moved".

### `stage1_provenance.json` is the reproducibility claim

Model revision, diffusers / transformers / torch versions, steps, guidance.
Diffusion output is not seed-reproducible across versions, so for an immutable
artefact that file *is* the claim.

### `--preflight` settles most of stage 1 without a download

It checks the `FluxKontextPipeline` class name, its `__call__` signature, the
offload API, HF auth, gated-repo access, disk, and the stage-0 inputs. `torch`
is imported lazily so the input checks run on the laptop.

### `pycocotools` ships manylinux wheels

2.0.11, no compiler needed. The old "builds from source, needs gcc" note is
retired. Do not move it back into core `requirements.txt`: only stage 0 imports
it, and a failure there must not block the other four VMs.

---

## Stage 3 — judging

### The vLLM API surface, verified against installed vllm 0.11.0 (2026-08-25, no GPU)

- `LLM.chat` accepts `list[list[message]]`, so batching message-lists is right.
- `MM_PARSER_MAP` in `vllm/entrypoints/chat_utils.py` contains an `"image_pil"`
  entry, so `{"type": "image_pil", ...}` is the correct discriminator.
- `LLM.__init__` ends in `**kwargs: Any`, forwarded to `EngineArgs`. That is how
  `max_model_len` and `limit_mm_per_prompt` reach the engine even though neither
  is an explicit parameter — **don't "fix" that by deleting them.**
  `mm_processor_kwargs`, `dtype` and `gpu_memory_utilization` are explicit.
- `EngineArgs` really does carry all five.
- A **raw** PIL image is what the parser wants, confirmed at runtime:
  `MM_PARSER_MAP["image_pil"]({"type": ..., "image_pil": img})` returns the
  identical object. Ignore `CustomChatCompletionContentPILImageParam`'s
  annotation of `Optional[PILImage]` (a pydantic wrapper) — its own docstring
  example and `parse_image_pil`'s `Optional[Image.Image]` both say raw, and
  TypedDict annotations are not runtime-enforced. **Do not add a wrapper.**

Closed by a real judge call (2026-08-25, `smoke_judge.py`, `--gpu-util 0.89`):

- `chat_template_content_format="auto"` resolves to `'openai'` for Qwen3-VL,
  which preserves our custom content parts. `'string'` would have silently
  dropped the images.
- The logprob structure is `list[dict[int, vllm.logprobs.Logprob]]` with
  `.decoded_token` and `.logprob`. Verified against the *placeholder's* flat 1-5
  output; the vLLM-side structure is confirmed, but
  `expected_score_from_logprobs` no longer applies under A.4.3 — see Stage 4.

**`build_requests` and `load_engine` therefore need no changes.** Never re-debug
the request path.

### Grammar cost is the bottleneck, and inside the grammar it is one keyword

Four configurations, 20 requests each:

| reasoning mode | s/req | vs bounded | `main` h/VM | parse | coverage |
|---|---|---|---|---|---|
| `bounded` (maxLength) | 58.90 | 1.0x | 22.97 | 100% | 100% |
| **`free` (plain string)** | **7.85** | **7.5x** | **3.06** | **100%** | **100%** |
| `none` (field dropped) | 5.15 | 11.4x | 2.01 | 100% | 100% |
| no schema at all | 2.85 | 20.7x | 1.11 | 100% | **~43%** |

Constrained decode ran 13.5 out tok/s against 265.9 unconstrained.

**`maxLength` cost 7.5x for nothing.** A length bound makes xgrammar track a
character counter, which multiplies FSM states; removing it left parse rate and
region coverage both at 100% (50 responses, mean 333 tokens against a 1536 cap).
`--reasoning free` is the default. **Never use `maxLength`.**

**Dropping the schema is not an option.** Unconstrained parses fine but covers
only ~43% of regions, because the judge silently omits regions and
`background`/`overall_score`. Speed there is bought with missing data.

The original efficiency claim — that `SamplingParams(n=k)` shares the prefill so
k samples are nearly free, because "outputs are ~15 tokens" — was wrong in both
halves under A.4.3: outputs are ~330 tokens (per-region `reasoning`), so decode
is not negligible, and the grammar dominates anyway.

`max_pixels` is still capped: Qwen3-VL tokenizes by area and vLLM sizes the
encoder cache from the cap regardless of what you send.

Those s/req figures assume `n=5 @ T=0.7`. Greedy measures **2.84 s/request**.

### Four harness bugs, found only by running on real data (2026-08-26)

None of these could have been surfaced by a synthetic test.

1. **`DEFAULT_GPU_UTIL = 0.85` sized the KV cache at -0.25GiB.** See Hardware.
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
region id, `required` on the top-level keys) plus compact output. Result: parse
rate 100%, region coverage 30/30, background / overall / PQ all 100%, responses
down to a 240-token mean. Run `scripts/diagnose_parse.py` after any judge
change; it reads a scores parquet on CPU and classifies every response.

### A latent crash at the end of every stage-3 run (found 2026-08-26)

`scored_region_id` mixed region ints with the literal `"bg"`, so pyarrow
inferred `int64` from the ints and raised `ArrowInvalid` on the first `"bg"` — in
`res.to_parquet(a.out)`, i.e. *after* a whole shard had been judged. An hour of
A10 time per VM, discarded at the write. Both `scored_region_id` and
`target_region_id` are now stringified in `build_requests`/`run`. Reproduced on
pandas 2.2.2 / pyarrow 17.0.0.

### Greedy everywhere, plus one deliberate sampled run (decided 2026-09-04)

`n=5 @ T=0.7` was not an arbitrary default. `git log -L` shows it bought two
things: the **noise floor** (`stage4.noise_floor` is the SD across repeated
samples of one input, and there is no other route to it) and the
**expected-score readout**, which needed `logprobs=20` over several samples.
Reason two is dead. Reason one is still real — and that is the point: **n=5 is a
measuring instrument, not a production setting**, and it had been left switched
on for the main run.

- `stage3_judge.py` defaults are now `--n-samples 1 --temperature 0`.
- `run_shard.sh` passes both **explicitly**. It previously passed neither, so
  whatever the module happened to default to silently became the main run's
  configuration. Stating it in the script means changing the module default can
  never again change the main run without someone noticing.
- **Deliberately not overridable by an environment variable.** Sampling must
  match across all five VMs for the shards to be comparable, exactly like
  `--gpu-util`; a per-VM override is a way to break that quietly. `TEMP` as a
  variable name also collides with the conventional temp-directory variable. And
  `run_shard.sh` hardcodes `--out out/scores_shard${SHARD}`, so it is the wrong
  vehicle for a floor run regardless — that run has its own `--out`.
- The greedy default creates a NEW footgun, so `main()` guards it: asking for
  `--n-samples 5` without raising the temperature returns 5 **identical**
  samples, costs 5x, and measures no floor. The warning fires before the engine
  loads.
- Scores parquets carry `n_samples` and `temperature` columns. `judge` was
  already recorded; sampling moves the numbers further than the choice between
  two Qwen sizes does, and five merged shards otherwise cannot say how any of
  them was decoded.

Safe to switch only because nobody had run `main` yet. Mid-run this would have
made shards incomparable.

### `dry_run` no longer restates `run()`

`dry_run` printed `repetition_penalty=1.05` (the code has used 1.1 since
760707a) and "max_tokens is 1024" beside a printed 1536 (`32 -> 1024` in
38f2072, `-> 1536` in 4756cc5). Root cause was not carelessness: it hand-copied
`run()`'s sampling dict as a **string literal**, so every change needed a second
edit elsewhere and never got one. Both now read `sampling_base()`.

---

## Stage 4 — analysis

### Four readouts, side by side

`reward` (Equation 3, the headline), `phi` (Eq. 3 with the global AES factor
divided out), `sc_preserve` and `sc_success`. The last two exist because
corruption should drive `preserve` down while `success` holds, and
`phi = min(success, preserve)` hides which axis moved. `--all-readouts` runs the
lot.

### Redundancy regresses on the leave-one-out mean

Not the plain image mean. The plain mean includes the region itself, which at ~4
regions is a quarter of the predictor and manufactures correlation out of pure
noise. R^2 near 1 now means what it claims to.

### Background rows are excluded from AUROC, kept in the leakage matrix

`bg` is never a corruption target, so it is a guaranteed negative that would
inflate AUROC for free. In the leakage matrix, "damage in region i moved the
background score" is a real thing to see.

### `remove` is reported as one condition, not a severity ladder (decided 2026-09-04)

Its severity 1 and 3 differ by 0.5 8-bit levels out of 35 (against blur's
7.60 -> 21.87), because inpainting removes the object at every radius. A flat
remove-severity response is an **absent stimulus**, not judge insensitivity, and
pooled into a by-severity table it reports our own design as a finding.

`FLAT_SEVERITY = {"remove"}` and `collapse_flat_severity()` relabel it to a
single `binary` condition inside `delta_table` and `axis_table`, so no caller
can forget. Nothing is dropped or pooled across corruptions: `remove` keeps its
own row in every per-corruption table and appears as its own `binary` row in the
by-severity ones, so reading down the 1 -> 3 column really is reading an
effect-size ladder.

The rejected alternative was redefining severity for `remove` in
`corruptions.py` (e.g. alpha-blending the inpaint). Better science, but it
changes corruption bytes, so all five VMs re-run the determinism check and any
judged data is void — too much for a ladder on the one corruption already known
to be the strongest stimulus.

### `tests/test_stage4.py` fabricates three judges with known behaviour

In stage 3's exact output schema: `perfect` (only the corrupted region drops) ->
AUROC 1.00; `global` (damage moves every region equally) -> AUROC 0.50 and R^2
0.998; `blind` (a constant) -> noise floor exactly 0, correctly reported as no
signal. All 30 checks pass. `global` is the one that matters: it is precisely
the failure this audit exists to detect, and a stage 4 that gave it a healthy
AUROC would be worse than useless.

`sensitivity` (the tie rate) and `response_coherence` (do a variant's regions
move together?) were added after the real pilot demanded them. Both are covered
by fixtures.

### No expected-score readout — question answered 2026-08-26, do not rebuild

The plan was `sc_sampled` (the emitted integer) plus `sc_expected`
(Σ p(k)·k over digit-token logprobs), because a 1-5 integer gives ∆score
granularity 1 and a tie-ridden AUROC. A.4.3 broke the implementation (scores are
two-digit, nested, after variable-length reasoning), and
`expected_score_from_logprobs` raises `NotImplementedError`.

**Leave it raising.** The pilot showed the ties are not a measurement artefact to
be smoothed away — they ARE the finding. 53-80% of damaged regions receive a
byte-identical score to their clean control under greedy decoding. A continuous
logprob readout would paper over exactly the observation the audit exists to
make, by turning "the judge did not react" into a small non-zero number.

`sensitivity()` reports the tie rate directly, split by target vs non-target.
Report it alongside every AUROC — AUROC is 0.5 both for a judge that never reacts
and one that reacts at random, and only the tie rate separates those.

---

## Nuisance and exploitability

Implemented 2026-09-04. Laptop-verified, never run on a GPU.

Both analyses ask what the score does when something changes that carries **no
information about edit quality**.

### Presentation is a stage-3 flag, never a manifest column

`variant_id` is `sha1(base_id|target_region_id|corruption|severity|area_bin)`
(`schema.py:66-70`). A sixth field changes every id, hence every `seed_for()`,
hence every rendered byte — voiding `776feeddd281fa726195bf504c7b19c8`, which
three VMs have confirmed. So `--presentation` re-packages the SAME images at
request-build time. Any pixel change happens in memory in `build_requests` and
is never written to disk. No manifest change, no re-render, less code.

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
  grammar pins slot k to `ids[k]`. Permuting the prompt but not the schema would
  force the model back into canonical order — an axis that runs, costs GPU time
  and measures nothing. `test_schema_slots_follow_the_presented_order` exists
  solely to catch that.
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

### The analysis lives in `scripts/`, and that is not just about merge conflicts

**The delta is a different delta.** `stage4_analyze.delta_table` compares a
CORRUPTED variant against its CLEAN control. Here the image is fixed and the
packaging changes, so the pairing key is `(judge, variant_id, scored_region_id)`
across presentations.

Worse than answering the wrong question: `delta_table` does not group by
`presentation`, so a glob catching several conditions **pools every condition's
clean controls into one baseline, silently**. Same hole in `redundancy` and
`leakage_matrix`. Hence `out/nuisance/`, whose filenames cannot match
`out/scores_shard*.parquet`. If stage 4 ever grows a `presentation` groupby this
separation can relax; until then do not point stage 4 at these files.

`nuisance_report.py` imports `delta_table`, `sensitivity`, `noise_floor`,
`usable` and `load` from stage 4 rather than re-deriving them, so the tables read
in the same vocabulary — same tie rate, same floor, same |delta|.

**`scores_*.parquet` vs `floor_*.parquet` is load-bearing naming.** The floor run
also carries `presentation == "baseline"`: same variants, same regions, same
label. It is indistinguishable from the greedy baseline by any column, so
detecting it programmatically does not work. Two globs, two filename prefixes,
no detection logic to get wrong.

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

Verified on the laptop, 2026-09-04 (`tests/test_nuisance.py`, 16 checks):

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

Only a judge VM can settle these:

- Whether vLLM/xgrammar accepts a permuted `prefixItems` schema at all. This is
  the one thing that could still invalidate the `shuffle` axis.
- Grammar compile cost when `shuffle` turns one schema per base into up to n!.
  `_sc_schema_cached` is `lru_cache(maxsize=64)` (`judge_prompt.py:197`); fine at
  5 bases, a thrash hazard at `main` scale.
- Cross-VM equality of `enhance`/`box`. Deliberately untested; see above.
- Every actual score.

---

## Base count: `main` is 150 bases (decided 2026-09-04)

`config.yaml` says so. Not pool-limited — train2017 supplies 1,372, measured by
`--survey` on 2026-09-04. The 50 over 100 cost ~3.0 extra hours on the editor VM
(6.0h -> 9.0h at 215.4s/image, serial, blocking) and
nothing anywhere else: stage 3 goes 1.1 -> 1.7h/VM, disk is a rounding error.

State the gain honestly rather than overselling it: regions within an image are
NOT independent (that is the finding — coherence 27%, redundancy R^2 0.52-0.56),
so effective n is the photograph count, and 100 -> 150 narrows intervals by
1/sqrt(1.5), ~18%. It crosses no threshold — the tie rate was already comfortable
at 100 and AUROC is not resolvable at either. Cheap insurance, not a new claim.

Do not push to 200 without re-timing: stage 1 stops fitting one night, and it is
the serial stage everything else waits on. If more power is ever wanted, more
PHOTOGRAPHS is the lever, never more regions per photograph.

---

## Open question: `REMOVE_TEMPLATES` in stage 0

`REMOVE_TEMPLATES` generates "remove the X" instructions, which collide with
`remove` being a corruption. The stronger objection was that "how well is this
region preserved" is ill-posed for a region the instruction deleted — a correct
removal would score ~0 on preserve, hence `phi = min(success, preserve) = 0`, the
floor, and `drop_floored` would discard those regions entirely.

**That claim is NOT established and was overstated (corrected 2026-09-04).**
A.4.3's OUTPUT FORMAT names the axis `score_preserve` and defines it as
"0=completely different, 25=minimal effective edit" — but its RULES section calls
the same quantity the "overall overediting score", and the BACKGROUND rule is
framed identically ("Penalize unexpected edits, layout changes, artifacts outside
editing regions"). Under the overediting reading, a clean removal that did not
disturb its surroundings is a minimal effective edit and should score HIGH. The
prompt is genuinely ambiguous and the paper's own gloss argues against the
pessimistic reading.

**Settle it by measurement, not by reading.** The question is not what the rubric
means but what base Qwen3-VL-8B does with it. Compare `sc_preserve` on
removal-target regions against recolour-target regions, clean controls only, same
images:

```bash
python - <<'EOF'
import pandas as pd, glob
df = pd.concat(pd.read_parquet(f) for f in glob.glob("out/scores_shard*.parquet"))
c = df[df.is_control & (df.scored_region_id == df.target_region_id)]
print(c[["base_id","instruction","scored_region_id","sc_success","sc_preserve"]].to_string())
EOF
```

The pilot scores parquet lives on `mcvgpu2025s-0050`, not 0053.

Whatever the answer: if base Qwen reads that axis differently from how the paper
means it, that gap is itself reportable. We audit the prompt-based protocol, not
the released fine-tuned reward model, and an axis ambiguous enough to be read two
ways is exactly what that distinction exists to catch.

Two things hold regardless of the answer:

- **At most one removal per base** — already implemented, see Stage 0.
- **The colour monoculture is the real diversity problem, not removal.** See
  [`FINDINGS.md`](FINDINGS.md); removal is ~16% of regions, so dropping it would
  take colour from 57% to ~67%.

---

## Repo history

- The repo layout was flattened in the scaffold commit and restored to the
  documented `src/` / `tests/` / `scripts/` tree on 2026-08-25 — the relative
  imports (`from .schema import ...`) and both shell scripts require it.
- `stage2_corrupt.py`'s free-space guard used `os.statvfs`, which does not exist
  on Windows; now `shutil.disk_usage`.
- Manifest expansion and hash sharding: 200 bases -> 24,800 variants, shards
  `[5025, 4895, 4930, 4954, 4996]`, lossless and unique.
- Dependencies are split by role, not commented out in one file. The role files
  each `-r requirements.txt`, so setup is always exactly one install command.
  **Don't pin `torch` in the role files** — vLLM and diffusers pin the build they
  were compiled against, and a second pin produces either a resolver conflict or
  a silently mismatched CUDA build.
