# Findings

What we have measured about the judge, and what belongs in the report. Harness
decisions live in [`DECISIONS.md`](DECISIONS.md); the plain-language version for
teammates is [`../TEAM_BRIEF.md`](../TEAM_BRIEF.md).

---

## MAIN RUN, 2026-09-18 — the finding at full power

`mcvgpu2025s-0050`, `main` profile: 150 bases, 476 regions, 5,236 variants, all
five corruptions `[none, blur, saturate, noise, jpeg, remove]`, Qwen3-VL-8B,
greedy (`--temperature 0 --n-samples 1`), sharded five ways on one VM. Parse rate
**100%**, 22,176 scored rows. Tables committed in
[`../results/main_2026-09-18/`](../results/main_2026-09-18/), including the raw
score parquets — every number below reproduces on a laptop with
`python -m src.stage4_analyze --scores 'results/main_2026-09-18/scores_shard*.parquet' --all-readouts`.

**Headline: the judge perceives the corruption, but the per-region reward does not
localise it.** Image-level quality reacts to the damage; the per-region scores do
not attribute it to the corrupted region. Everything the pilot saw on 5 photos
now stands on 150, and `jpeg`, `noise`, `saturate` — judged on real data for the
first time — behave like `blur` and `remove`.

Two honesty corrections are baked into the numbers below, both from self-critique
after the first pass (an earlier version of this section overstated on both and is
superseded):

- **Floored regions are excluded** (`--min-control 0`). 37% of region-deltas came
  from regions whose clean control already scored 0 (FLUX failed the edit); such
  a region cannot drop when damaged, so it is a guaranteed tie that inflates the
  tie rate and drags AUROC toward 0.5. Primary numbers are floor-excluded; the
  all-region figure is noted alongside.
- **Coherence is read on `phi`, not `reward`.** `reward` carries the shared
  image-level AES factor, which mechanically pulls every region together; `phi`
  (AES divided out) is the honest per-region view.

### 0. The judge perceives the damage (`scripts/pq_response.py`)

Image-level quality (AES = min(PQ), the reward's own term) drops when we corrupt
one region, **monotonically with severity**:

| corruption | ΔAES (severe) | share of variants where AES dropped |
|---|---|---|
| noise s3 | −2.68 | 62% |
| remove | −2.50 | 57% |
| jpeg s3 | −1.77 | 45% |
| blur s3 | −1.61 | 43% |
| saturate s3 | −0.62 | 22% |

against a between-base control-AES SD of 3.36 (control mean AES 19.2 / 25). Mild
severities barely move; severe ones move a real fraction of the base-to-base
spread, in a sensible ordering. So a flat per-region result is about
**attribution, not perception** — the judge is not blind to the damage, and not
defeated by the `max_pixels` cap.

### 1. The per-region score is no more likely to move on the damaged region

Tie rate (`delta == 0` exactly, greedy so not sampling noise), floor-excluded,
`reward`:

| corruption / severity | target unchanged | any-other unchanged |
|---|---|---|
| blur 1 / 3 | 0.539 / 0.327 | 0.576 / 0.388 |
| jpeg 1 / 3 | 0.641 / 0.332 | 0.643 / 0.364 |
| noise 1 / 3 | 0.609 / 0.190 | 0.610 / 0.211 |
| saturate 1 / 3 | 0.648 / 0.560 | 0.654 / 0.565 |
| remove (binary) | 0.264 | 0.288 |

`target_unchanged` tracks `other_unchanged` in every cell — **the region we
damaged is no more likely to move than one we did not touch.** (All-region tie
rates run higher, 48-77%, because of the floored regions; the equality of target
and other holds either way, and the score does react more under strong damage —
noise s3 drops the target 76% of the time — it just does not react *selectively*.)

### 2. AUROC ~0.52-0.54: weak spatial information, not zero

Localization AUROC, floor-excluded (0.5 = none):

| readout | sev 1 | sev 3 | remove |
|---|---|---|---|
| reward | 0.518 | 0.530 | 0.533 |
| phi | 0.513 | 0.521 | 0.538 |

By corruption (all-region) 0.508-0.518. It is **weak signal, not zero**: the
targeted region does drop somewhat more than the others on average (floor-excl
`phi`, `remove`: mean delta −3.76 vs −2.38, ~1.6×), but tie rates of 67-84% and
whole-image co-movement swamp it, so the reward cannot reliably identify the
damaged region. Report the tie rate beside every AUROC: with ties this high, 0.5
means "reacts globally", not "reacts at random". "AUROC exactly 0.5, no signal"
would be an overstatement.

### 3. When scores move, they move largely as a whole-image event

Coherence — fraction of multi-region variants where some regions move while others
hold, vs. what independent per-region movement predicts at the same overall rate:

| readout | frac_mixed | expected if independent |
|---|---|---|
| reward (AES-inflated) | 0.15 | 0.74 |
| **phi (honest)** | **0.278** | **0.511** |

On `phi`, mixed movement is ~half what independence predicts (0.278 vs 0.511) —
still non-independent, but far more modest than the AES-driven `reward` figure
(the shared image-level factor mechanically co-moves regions, which is why the
first pass reported the dramatic 22% vs 68%). Corroborated by leave-one-out
redundancy R^2 = **0.52** (`phi`) / 0.56 (`reward`), unchanged by floor exclusion.

### The layout-drift confound is ruled out

The top defensive worry (masks cut on `source.png`, applied to a possibly
re-composed `edit.png`) does **not** drive the null. Reporting every headline
twice, all 150 bases vs. the 96 whose layout survived (edge IoU >= 0.4):

| subset | bases | target unchanged | AUROC | frac mixed |
|---|---|---|---|---|
| all | 150 | 0.633 | 0.514 | 0.216 |
| edge IoU >= 0.4 | 96 | 0.650 | 0.515 | 0.183 |

The two rows agree, so the finding does not rest on the geometrically doubtful
bases. No independent-detector pass is needed.

### Which axis moves, and the effect size

On the targeted region, `sc_success` and `sc_preserve` fall together and
slightly from the `none` baseline (13.57 / 13.39 clean; ~12.2 / ~11.8 under
`remove`); `sc_preserve` drops marginally more than `sc_success` (e.g. `remove`
target mean delta -1.55 vs -1.26) but with 76-88% ties it does not localise
either. The mean target effect on `reward` is |delta| = 0.028, against a
between-variant SD of clean controls of **0.408** — the effect is a fraction of
the base-to-base spread. 39% of `reward` values sit on a rail (0 or ~0.98): many
edits genuinely failed or maxed out.

### Caveats and scope for the report (state all of these)

- **Base Qwen3-VL-8B + the A.4.3 prompt, NOT the deployed reward model.** SFReward
  is Qwen3-VL-8B *fine-tuned* on 14K examples that a Gemini-3-Pro teacher labelled.
  We audit the prompt-based protocol on the base model. We cannot claim the
  fine-tuned SFReward, or the Gemini teacher, behaves this way — the fine-tuning
  exists precisely to shape this behaviour. This bounds the claim and must be
  loud.
- **One judge, one family** (Qwen3-VL-8B). Claiming this is about the *protocol*
  and not this backbone needs a second, independent family. The main open piece
  of strengthening work.
- **Natural region sizes only** (`area_bin: full`, mean 4.5% of image, the low end
  of the paper's 2-25% band). Whether large-region damage localises better is
  untested.
- **Greedy decoding, so no within-run noise floor.** The finding rests on the tie
  rate and the between-variant SD, both the right instruments here. The
  `T=0.7 n=5` floor run over the identical variants is still to do.
- **`n = 150` photographs.** Regions within an image are not independent (that is
  the finding), so effective n is the photo count, not the 5,236 variants.
- The `score_preserve` overediting question is now partly informed — the axis
  does move, it just does not localise — but the direct removal-vs-recolour
  `sc_preserve` comparison has not been run.

All numbers above regenerate from the committed parquets on a CPU laptop; the
commands are in [`../results/main_2026-09-18/REPRODUCE.md`](../results/main_2026-09-18/REPRODUCE.md).

---

## PILOT VERDICT, 2026-08-26 — GO

5 bases, 75 variants, `[none, blur, remove]`, Qwen3-VL-8B, greedy
(`--temperature 0 --n-samples 1`). Parse 100%, coverage 100%.

### The stimulus is real and perfectly localised (verified, not assumed)

`scripts/verify_corruption.py`, mean 8-bit levels, region = 4.5% of image:

| corruption | inside mask | outside mask | masked pixels changed |
|---|---|---|---|
| blur s1 | 7.60 | 0.042 | 39% |
| blur s3 | 21.87 | 0.110 | 79% |
| remove s1 | 35.01 | 0.001 | 78% |
| remove s3 | 35.52 | 0.001 | 83% |

Contrast inside:outside is 179x to 25,000x. **"The judge did not react" is
therefore about the judge**, and the leakage analysis' core assumption — that
untouched regions really are untouched — holds on real edits, not just on the
synthetic fixture.

*Design note:* `remove` s1 and s3 differ by 0.5 levels. The severity ladder is
effectively binary for `remove`; only `blur` is graded. Do not read a flat
remove-severity response as insensitivity to severity. Stage 4 collapses it —
see [`DECISIONS.md`](DECISIONS.md).

### The finding, three ways, all pointing the same direction

1. **The score usually does not move.** 53-80% of DAMAGED regions receive a score
   identical to their clean control. Deterministic decoding, so this is not
   sampling noise.
2. **When it moves, it is not the damaged region.** `target_unchanged` equals
   `other_unchanged` to three decimals in three of four cells (0.667/0.667,
   0.667/0.667, 0.533/0.533). The region we damaged is no more likely to change
   than one we did not touch.
3. **The judge revises the whole image at once.** Only **27%** of variants show
   some regions moving while others hold, against **67%** expected if regions
   moved independently at the same overall rate. Corroborated by leave-one-out
   redundancy **R^2 = 0.52-0.56**.

Together these say the per-region score behaves like **one whole-image judgement
replicated across region slots**, which is precisely the failure this audit was
built to detect. AUROC 0.45-0.47 is a *consequence*, and on its own it would have
been unreadable — AUROC is 0.5 both for a judge that never reacts and one that
reacts at random.

### Caveats, to state in the report

- **n = 5 photographs.** 90 rows per severity come from five images; the
  effective independent sample is 5. Suggestive, not reportable.
- Only `blur` and `remove` tested; one judge, one family.
- Regions average 4.5% of image area — the low end of the 2-25% band.

---

## Settled judge behaviour

Established 2026-08-25 (synthetic squares, real A.4.3 prompt, Qwen3-VL-8B):

- **The harness is correct.** Prompt parses, `background` / `overall_score`
  present, Equation (3) runs, images demonstrably reach the model (+396 tokens),
  vision path confirmed by a plain-question probe answering `red`/`blue`
  correctly. **Never re-debug the request path.**
- **The judge discriminates instruction-following.** Obeyed 25.0 vs ignored 0.0
  on the targeted region — the full width of the scale. An earlier all-5s result
  was caused entirely by the old placeholder prompt.

---

## RETIRED (2026-08-26): "semantic yes, photometric no" did NOT replicate

This was the top risk for a week. The pilot on real COCO edits closed it, and the
answer was no.

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

### The synthetic result that motivated the hypothesis (superseded)

Re-run on `mcvgpu2025s-0050` across all four corruptions. An earlier "the judge
ignores corruption entirely" had been drawn from `noise` alone and was too broad.
What held **on synthetic squares only**:

| corruption | region 0 succ/pres | region 1 (untouched) |
|---|---|---|
| clean | 25/25 | 25/25 |
| **remove** s1 | **20/15** | 25/25 |
| **remove** s2 | **20/15** | **20/15**  <- leakage |
| **remove** s3 | **20/15** | 25/25 |
| blur s1-s3 | 25/25 | 25/25 |
| jpeg s1-s3 | 25/25 | 25/25 |
| noise s1-s3 | 25/25 | 25/25 |

The judge appeared to track semantic change and not degradation: `remove` moved
it a full 10 points and moved `success` too (25 -> 20), while blur, JPEG and
noise returned a flat 25 at every severity.

The caveats flagged at the time were the right ones and they were what broke it:
flat textureless squares are far out of distribution, the `remove` severity
ladder was already flat (15/15/15), and the region-1 drop at s2 but not s1 or s3
was non-monotone — instability, not a spatial effect.

**Worth keeping as a methodological point for the report:** a synthetic sanity
check produced a clean, plausible, entirely wrong hypothesis, and only real data
caught it.

Every confound raised at the time has since been eliminated: the pilot ran on
real COCO edits; blur and remove were both tried and behave identically; and
`success` and `preserve` are reported separately, with neither localising. Do not
re-run any of it.

---

## NUISANCE AND EXPLOITABILITY SWEEP, 2026-09-15 — first measurement

`mcvgpu2025s-0043`, pilot profile: 5 bases, 16 regions, 80 variants,
Qwen3-VL-8B, greedy, plus a `T=0.7 n=5` floor run over the same variants. Eight
conditions; parse 100% on every greedy run, 99.5% on the floor.

Reference from the baseline condition: real damage moves the target's `reward`
by mean |delta| **0.141**, and 56% of damaged regions do not move at all. The
noise floor (median SD across samples) is **0.114** — nearly as large as the
damage effect itself.

### Exploitability

Mean gain over baseline. Split columns are clean controls only, "failed" =
baseline `sc_success` below 20 (`--failed-below`, the default). Failed cells
hold 9 target / 22 other regions, ok cells 7 / 14; **4 photographs per cell**.

| condition | `reward`, all regions | share rose | `phi`, failed edits (target / other) | `phi`, ok edits (target / other) |
|---|---|---|---|---|
| `noimg` | **+0.342** | 65% | **+19.9 / +19.4**, every region rose | -5.0 / -4.4 |
| `enhance` | -0.064 | 8% | 0.0 / 0.0 | -10.7 / -10.7 |
| `enhance_target` | -0.015 | 9% | +1.7 / +2.3 | -5.7 / -5.4 |

- **`noimg` is the one real exploit.** Without an image the judge falls to a
  high default (~20 `phi`): every region it had scored as a failed edit rose,
  and ok edits drifted down toward the same default. +0.34 `reward` is ~3x the
  noise floor. It is not an attack an editor can run directly — the editor
  controls pixels, not whether they are sent — but it shows the default is high,
  so any output that hides a failure from the judge inherits it. This replaces
  the earlier "text-only request returns all 25s" anecdote.
- **`enhance` lowered scores; the predicted AES exploit did not appear.** And it
  structurally cannot help a failed edit: Eq. (3) multiplies `phi` by AES, so a
  region at `phi = 0` stays at 0 however high PQ goes. Whether the drop is the
  preservation axis (sharpened edit differs more from source) or PQ itself has
  **not been checked** — compare `pq_naturalness`/`pq_artifacts` between
  `scores_enhance` and `scores_baseline`.
- **`enhance_target`: no local flattery.** On failed edits the lifted region
  gained less than the untouched ones; on ok edits both fell equally. Its gain is
  below the noise floor. Consistent with the pilot's whole-image judgement.

### Nuisance, one line each

`shuffle` moved `reward` by **1.55x** what real damage does (mean |delta|
0.219), `subset` 0.91x, `box` 0.72x. Under `shuffle`, list slot 3 averaged 0.03
against 0.26-0.33 for slots 0-2 — but slot 3 exists only on 4-region bases
(n=20), so position is confounded with base there.

### Caveats

Five photographs, four per exploit cell. Suggestive, not reportable, except
`noimg`, whose effect is large and unanimous on failed edits. A controls-only
rerun over all 150 bases (476 clean edits; `baseline`, `enhance`,
`enhance_target`, `noimg`, plus floor) is ~5h on one VM at the speed measured
here.

The report's tables behind every number above are committed in
[`results/nuisance_pilot_2026-09-15/`](../results/nuisance_pilot_2026-09-15/).
The raw score parquets are not (gitignored); they live on `0043` in
`out/nuisance/`.

---

## Instability at temperature 0.7

Across samples of an IDENTICAL input, the judge's score varied by **38% of the
scale** (SD 0.363 on a 0.959 range). A reward model that unstable is a problem
for RL training regardless of whether it localises.

Worth reporting separately, and worth a small deliberate `n=5 @ T=0.7` run on
~10 bases purely to characterise it. That run is also the noise floor —
see [`DECISIONS.md`](DECISIONS.md).

---

## The global AES factor, straight out of Equation (3)

The region reward is `sqrt(phi(IF_{i,r}) * AES_i)/C` where `AES_i = min(PQ)` is a
single *image-level* term multiplying every region of that image. **Part of each
"region" reward is global by construction**, before any judge behaviour is
measured. Within one image it cancels from region-to-region comparisons; across
variants it does not. Worth a paragraph in the report.

The `enhance` presentation tests it directly: a global cosmetic lift that
improves no edit should, if that reading is right, raise every region's reward
through `AES` alone.

---

## Instruction-family distribution, measured 2026-09-04

On the ~120 val2017 bases then on disk (360 regions, from
`cat data/bases/*/instruction.txt`):

| family | regions | share |
|---|---|---|
| recolour | 204 | **56.7%** |
| remove / erase | 57 | 15.8% |
| add sunglasses | 50 | 13.9% |
| make older | 49 | 13.6% |

Removal targets by category: cup 13, potted plant 11, bowl 11, book 6, bottle 5,
traffic light 3, parking meter 3, clock 2, fire hydrant 2, stop sign 1.
`person` is **27.5% of all regions** — COCO's commonest category — and half of
those drew "make the person look older", the vaguest instruction in the set and
the hardest to score on the success axis.

Two things this corrects:

- **Removal is ~16% of regions, not the ~1/3** implied by "10 of 29 categories".
  Categories are not equally frequent.
- **The colour monoculture already exists.** 57% of regions are "change this
  object's hue"; dropping removal would take it to ~67%. Removal is not what
  protects instruction diversity, so adding a material family was worth doing on
  its own merits, independent of the removal decision.

Caveat on both: those bases predate the category-uniqueness rule, which hits
`person` hardest (people almost never appear alone). Every proportion above will
shift. Treat it as the shape of the old design, not a prediction.

### After the fix — measured on the real `main` selection, 2026-09-04

150 train2017 bases, 476 regions, **3.17 regions/base**, under the uniqueness
rule and with the material family, the one-removal cap and the beard/moustache
templates all live:

| family | regions | share | was |
|---|---|---|---|
| material | 164 | **34.5%** | — |
| colour | 148 | **31.1%** | 56.7% |
| person attribute | 99 | 20.8% | 27.5% |
| remove / erase | 65 | 13.7% | 15.8% |

**The monoculture is gone**: no family is now more than about a third, against
colour's old 57%. Colour and material together are 65.6% — higher than colour
alone was — because removable categories over the cap fall back to material,
which moves mass out of `remove` and into `material` rather than out of the
image-editing task.

Three checks passed on the same run, all of which fail silently if broken:
**0 removal-cap violations** (65 removals over 150 bases, never 2 in one), **0
ambiguous clauses** (every region matches exactly one family — no region asked
to be both red and marble), and every base 3-5 regions.

Two predictions from the table above came true. `person` fell from 27.5% to
20.8%, which is the uniqueness rule hitting the category that almost never
appears alone. And `remove` barely moved, 15.8% -> 13.7%: most bases only ever
had one removable category, so the cap trimmed a tail rather than reshaping the
design — which is also why capping was the cheap fix and dropping removals
outright was not needed to buy diversity.

`regions/base` came out 3.17 against the 3.19 config.yaml had assumed from
val2017 pre-rule. The uniqueness rule cost 75% of the *images* and essentially
nothing per surviving image: a photo qualifies whole or not at all.

---

## Deviations from the paper, to state in any write-up

`src/judge_prompt.py` carries Appendix A.4.3 of arXiv:2606.26872 **verbatim**.
Three deviations belong in the report:

- **The PQ prompt is reconstructed, not verbatim.** The paper shows SFReward's PQ
  *output* (A.4.4) but never its PQ *prompt*; A.5.2's PQ prompt is the
  MultiEditBench/VIEScore one on 0-10 for GPT-4.1, a different purpose. Ours
  matches A.4.4's output shape. Marked in the file.
- **How the instruction and region list are appended is ours.** A.4.3 says only
  "You will be provided with pre-identified editing regions" and never shows the
  injection format.
- **SFReward is a fine-tuned model** (Qwen3-VL-8B + SFReward-14K); A.4.3 is the
  prompt that labelled that data with a Gemini-3-Pro teacher. We apply it to
  *base* Qwen3-VL-8B, so we audit the prompt-based protocol, not the released
  reward model.

Consequences of A.4.3 already implemented: the scale is **0-25, not 1-5**;
requests are **2 per variant** (one SC scoring every region at once, one
image-level PQ) rather than one per region — the protocol's own shape, and
cheaper than what we had; `max_tokens` is 1536, since A.4.3 demands per-region
`reasoning` before the scores; COCO `(x,y,w,h)` is converted to A.4.3's
`bbox_2d [x1,y1,x2,y2]`.

Judge output is constrained to a JSON schema. That fixes format only — every
score in 0-25 stays reachable — but it is a deviation from free generation and
should be stated. Without it the judge silently drops regions and covers only
~43% of what it was asked to score.

---

## Two things that must reach the report

- **Every pilot number rests on 5 photographs.** 90 rows per severity come from
  five images; the effective independent sample is 5, not 90. Say so explicitly
  wherever pilot numbers appear.
- **The temperature-0.7 instability is its own finding.** SD 0.363 on a 0.959
  range across samples of an identical input, independent of localisation.
