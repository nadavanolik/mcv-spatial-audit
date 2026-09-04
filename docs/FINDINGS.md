# Findings

What we have measured about the judge, and what belongs in the report. Harness
decisions live in [`DECISIONS.md`](DECISIONS.md); the plain-language version for
teammates is [`../TEAM_BRIEF.md`](../TEAM_BRIEF.md).

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

## Free exploitability data point

The judge scores an edit **it was shown no images of** — a text-only request
returns all 25s. Worth a line in the report.

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
