# Findings

What we have measured about the judge, and what belongs in the report. Harness
decisions live in [`DECISIONS.md`](DECISIONS.md); the plain-language version for
teammates is [`internal/TEAM_BRIEF.md`](internal/TEAM_BRIEF.md).

---

## SCOPE — read before quoting any number on this page

Every result here is **base `Qwen3-VL-8B-Instruct` driven by the A.4.3 prompt
reproduced from a paper's appendix**, not the fine-tuned SFReward model those
papers ship, and not its Gemini teacher. We can say the published *recipe* does
not localise on an off-the-shelf backbone; we cannot say their trained reward
model does not. The PQ half of the prompt is worse than that — it is
**reconstructed** (`SFREWARD_PQ_RECONSTRUCTED`, `src/judge_prompt.py:16`), since
the paper never publishes it, so every `reward` number and the whole AES analysis
inherit that reconstruction. `phi` uses only the verbatim prompt and is the safer
primary readout.

A second family was attempted and **does not currently support a cross-family
claim** — see the retraction banner in the 2026-09-26 entry.

**The scale is effectively binary.** On the `main` run's 16,936 scored regions:

| readout | distinct values | at 0 | at the two extremes |
|---|---|---|---|
| `phi` | **7** | 42.1% | **85.6%** |
| `sc_success` | 8 | 41.4% | 85.5% |
| `sc_preserve` | 7 | 42.0% | 87.0% |
| `reward` | 33 | 42.1% | top four values = **82.1%** |
| `pq_naturalness` | **5** | 0% | 73% at the single value 18 |

This is arguably the most important descriptive fact in the audit and it was
previously a half-sentence saying "39% on a rail", which understates it. Every
downstream statistic is conditional on it: an AUROC on a near-binary score, a
Pearson between two near-binary scores, a "noise floor" of 0.114 on a scale whose
smallest real step is ~0.3, an R^2 on a variable with two atoms. It also
vindicates invariant 6 — on a scale that is 86% two-valued, a continuous logprob
readout would have manufactured a result. Report the reward as 2-level in
practice, not 26-level.

---

## EXPLOITABILITY AT 150 BASES, 2026-09-29 — one exploit survives, two die

`mcvgpu2025s-0004`, the `nuisance150` profile: 150 bases, **476 clean controls,
no corrupted variants**, greedy, four presentations, 8,064 rows per judge.
Roughly **100 photographs per exploit cell**, against the 2026-09-15 pilot's 4.
**Both judges**: Qwen3-VL-8B and InternVL3-2B, same controls, same conditions.
The tables below are Qwen unless stated; the cross-family comparison is further
down. Numbers and raw parquets in
[`../results/exploit_150_2026-09-29/`](../results/exploit_150_2026-09-29/),
per-judge CSVs under `qwen/` and `internvl/`.

Controls only, because that is the exploit scenario: an edit that already went
wrong, no corruption added, and the question of whether a change that fixes
nothing raises its score. The control `variant_id`s are identical to `main`'s by
construction, so these are the same images `main` already judged.

### The headline table

Mean gain over the `baseline` condition. "Failed" = baseline `sc_success` below
20 (`--failed-below`, the default).

Intervals are 95% bootstrap over **photographs**, not regions: regions inside one
photo share an edit, an instruction and a scene, so a row-level interval would be
about sqrt(3) too tight. Qwen; the figure is `qwen/exploit_gain.png`.

| condition | `reward`, all clean edits | survives a non-circular split? |
|---|---|---|
| `noimg` | **+0.212** [+0.154, +0.268] | **yes** — +0.31 [+0.24, +0.37] |
| `enhance` | -0.012 [-0.036, +0.016] | no — +0.016 [-0.010, +0.044] |
| `enhance_target` | -0.003 [-0.022, +0.019] | no — +0.019 [-0.005, +0.043] |

**`noimg` is the one exploit, and it is a constant, not a gain.** Strip the
images and mean `reward` goes 0.495 -> 0.714, `phi` 13.78 -> 18.81. The right way
to describe this is the *shape* of the blind response, not its mean: regressing
the gain on the baseline score gives a slope of **-0.922** with `sd` falling from
0.417 to **0.173**. The judge shown nothing emits a near-constant. Everything
else follows arithmetically — the gain is largest wherever the sighted score was
lowest, which is why edits it had called failed rise ~+0.55 and edits it had
called fine fall ~-0.10.

So quote the default, not the split: **blind reward 0.71 against a sighted mean
of 0.46, and `phi` 18.8 against 13.8.** Both judges do it (InternVL 0.44 blind
against 0.27 sighted), and it survives selection by the *other* judge's opinion
of which edits failed (+0.31 [+0.24, +0.37] for Qwen, +0.25 [+0.21, +0.30] for
InternVL), so it is not an artefact of circular selection.

Still not an attack an editor can run directly — the editor controls pixels, not
whether they are sent. What it establishes is that the judge's default opinion in
the absence of evidence is high, so any output that hides a failure from the
judge inherits that default rather than a penalty.

**The cosmetic axes are reversion to the mean, not exploits.** This entry first
claimed them as "small but real" on the strength of a failed/ok split; that claim
is **retracted**. Three checks, all in `qwen/reversion.csv` and
`agreement/crossjudge_exploit.csv`:

1. **The split is circular.** It selects on the judge's own baseline
   `sc_success` and then measures that same judge's change from that same
   baseline. Any condition that pulls scores toward a central value scores
   positive on the low tail for purely arithmetic reasons.
2. **One linear term absorbs it entirely.** `enhance` has slope **-0.17** on the
   baseline; after removing that single term the failed/ok gap of +0.058 vs
   -0.056 collapses to **-0.009 vs +0.009**. The split carried nothing the slope
   did not.
3. **A non-circular split kills it.** Let the *other* judge choose which edits
   failed and the cell covers zero in all four directions: Qwen +0.016 [-0.010,
   +0.044] and +0.019 [-0.005, +0.043]; InternVL -0.006 and +0.001, both
   covering zero.

Report both lifts as a **reversion property of the scale**: a cosmetic change
shrinks every score toward the judge's grand mean, which flatters anything
already below it. That is worth saying — it means the reward is partly a
function of its own previous value — but it is not a demonstrated exploit, and
the optimiser argument below belongs to `noimg`, not here.

*(The optimiser argument, kept because it applies to any bias that does survive:
an effect below the noise floor is not negligible to a policy. Noise averages
away over a training run; a systematic bias does not. "Below the noise floor" is
the right caveat for a single measurement and the wrong one for an RL
objective.)*

**There is no local flattery, and the contrast that would show it is unusable.**
Paired within variant, `enhance_target` lifts its own region **+0.33 `phi`**
[-0.03, +0.69] over its neighbours. But `enhance` lifts the *whole frame
uniformly* and therefore cannot flatter one region — it is a built-in negative
control — and it produces **+0.30 [+0.11, +0.51]** on the same contrast,
*excluding zero*. The control is as large as the treatment. Target regions
differ from their neighbours under any perturbation, which is unsurprising
(they are the regions the instruction is about), so no target-vs-other contrast
in this run can support a local-flattery claim in either direction.

The earlier wording compared two *marginal* intervals, which is not a test of
their difference on a paired design. The conclusion was right by luck.

**Multiplicity.** A full run emits ~72 bootstrap intervals per judge,
uncorrected. At 95% that is roughly one null cell expected to exclude zero by
chance, so no single exclusion carries a claim here. The two results above are
stated because they survive a non-circular split (`noimg`) or fail on their own
negative control (local flattery) — not because an interval cleared zero.

### The AES channel moves the wrong way (question closed 2026-09-29)

`pq_by_presentation.csv`, 476 controls, both judges. The prediction was that a
cosmetic lift raises `AES = min(PQ)` and therefore every region's reward.

| judge | `pq_naturalness` under `enhance` | `pq_artifacts` under `enhance` |
|---|---|---|
| Qwen | 19.44 -> 18.35 (**-1.10**), 23% of images fell | 21.33 -> 20.19 (**-1.13**), 43% fell |
| InternVL | 16.13 -> 16.47 (+0.34) | 8.15 -> 7.83 (**-0.33**) |

**Sharpening lowers AES rather than raising it.** Qwen reads unsharp masking as
damage on both terms. InternVL's naturalness rises slightly, but `AES` is the
*minimum* and its artifacts score (8.15) sits far below its naturalness (16.13),
so the minimum follows artifacts — which also fell. Two different routes, same
direction.

So the channel is not inert: it moves, against the exploit. The prediction in
"The global AES factor" below was reasonable and is **tested and negative** —
worth a line in the report as a hypothesis that did not survive. Combined with
the reversion result above, nothing about `enhance` is exploitable: the AES
channel moves the wrong way and the apparent `phi` gain on failed edits does not
survive a non-circular split.

One detail worth quoting: blind, both judges emit *exactly* 18.000 for
naturalness (Qwen pairing it with 20.000, InternVL with 7.000). The no-image
default is a hard constant, not an average over anything.

### The two judges barely agree with each other

Both judges scored the identical 476 controls, so they can be compared directly —
`agreement/agreement_reward.csv`, the one analysis in this repo that spans
judges (`scripts/judge_agreement.py`).

| presentation | Pearson | Spearman | mean abs diff |
|---|---|---|---|
| `baseline` | **0.212** | **0.230** | 0.388 |
| `enhance` | 0.105 | 0.104 | 0.406 |
| `enhance_target` | 0.145 | 0.147 | 0.395 |
| `noimg` | **-0.173** | **-0.335** | 0.317 |

**On the same photograph, the same regions and the same prompt, two judges
correlate at r = 0.21.** Rank agreement is no better (0.23), so this is not a
calibration artefact of their different scales (means 0.19 vs 0.46) — they are
not ranking the same regions the same way. Asked which edits *failed*
(`sc_success < 20`), they agree on 60.3% of regions and their failed sets overlap
at **Jaccard 0.479**: 563 regions both call failed, 396 only InternVL, 216 only
Qwen.

**Lead with kappa, not Pearson.** Given how binary the scale is (see SCOPE), a
correlation is the wrong headline statistic, and the rails *inflate* r rather
than attenuating it: restricting to regions where both judges are off the rails
(`0 < reward < 0.95`, n=275) drops Pearson to **0.094** and Spearman to 0.179.
Cohen's kappa on the binarised failed/not-failed call is **0.203** — "slight"
agreement on the standard scale. That is the number to report.

This is the one cross-family statement the second judge can carry despite failing
the perception control, because it needs no sensitivity to damage: it is about
two judges disagreeing on *clean* edits. A judge that barely registers the
stimulus is still entitled to an opinion about edit quality, and the finding is
that the two opinions do not match.

This strengthens the main result rather than complicating it. The audit shows the
per-region number does not track the region; this shows it does not track
anything stable across judges either. A policy trained against one of these
judges is fitting that judge, not image quality.

Under `noimg` the correlation goes **negative**. With no image each judge falls
to its own constant default, so the little variation left comes from the
instruction text alone, and the two read it in opposite directions.

Caveat for the report: clean controls only, so this is agreement about *edit
quality*, not about damage. The `main` parquets would give the corrupted-variant
version and are already committed if anyone wants it.

### The pilot was directionally wrong on both cosmetic axes

At 4 photographs per cell the pilot reported `enhance` at -0.064 with 0.0 gain on
failed edits, and `enhance_target` giving the lifted region **less** than its
neighbours. Neither survived at 100 per cell: the first is ~0, the second
reverses sign. Only `noimg`, whose effect is an order of magnitude larger than
either, came through unchanged. That contrast is the argument for why the rerun
was worth the GPU hours, and it is worth one line in the report's methods.

### `noimg` parses at 91.2%, and the reason matters

Every other condition parses at 100%. `scripts/diagnose_parse.py` attributes 37
of 39 failures to `finish_reason=length`: with no image to ground on, the model
loops in the free-form `reasoning` field — repeating a sentence, or inventing a
60-item rubric — until it hits the token cap. Not a harness fault, and raising
the cap would only buy longer loops.

Two consequences to state. First, it is itself a result: shown nothing, the judge
still returns confident schema-valid per-region scores 91% of the time, and fails
by rambling rather than by refusing. Second, the truncations concentrate on bases
with many regions, so `noimg`'s parsed subset leans toward simpler scenes. Quote
the 91.2% alongside the gain.

### Both judges, same verdict (InternVL3-2B added 2026-09-29 20:31)

The second family ran the identical 476 controls under the identical four
presentations, on the same VM, at `--gpu-util 0.60` (0.89 OOMs a 2B model — see
[`DECISIONS.md`](DECISIONS.md)).

| | Qwen3-VL-8B | InternVL3-2B |
|---|---|---|
| `baseline` mean `reward` | 0.495 | 0.270 |
| `noimg`, all regions | **+0.219** | **+0.167** |
| `noimg`, failed edits | **+0.591**, 94% rose | **+0.323**, 67% rose |
| `noimg`, ok edits | -0.107 | +0.052 |
| blind default `phi` | 18.8 | 20.4 |
| `enhance` | -0.002 | -0.005 |
| `enhance_target` | +0.002 | +0.0004 |
| `noimg` parse rate | 91.2% | 100% |
| `n_bases`, failed cell | 97-99 | 110 |

**The blind default is the one result the second judge can support.** Both judges
answer a request with no image by falling back on a high default — `phi` ~19 and
~20 out of 25 — and both defaults survive selection by the *other* judge's
opinion of which edits failed. This claim does **not** depend on the perception
control that InternVL fails: it is about what a judge emits when shown nothing,
which needs no sensitivity to a stimulus that was never sent. Two judges from
different families defaulting high is therefore a fair cross-family observation,
and it is the only one on this page.

Everything else here is single-judge. The cosmetic axes are reversion in both
judges (all four non-circular cells cover zero), and the localization comparison
is retracted — see the banner below.

Two differences worth a sentence rather than a paragraph. InternVL's baseline is
much lower (0.270 against 0.495), so its blind default sits *above* its ok edits
as well as its failed ones — which is why removing the image nudges ok edits
**up** (+0.052) where Qwen's fall (-0.107). Same mechanism, different crossing
point. And InternVL parses `noimg` at 100%: the reasoning-field looping that
costs Qwen 8.8% of its blind responses is a Qwen behaviour, not a property of
image-free prompts.

### What this run cannot say

No noise floor of its own (greedy, n=1) and no damage reference (controls only,
nothing corrupted). Both denominators come from the 2026-09-15 pilot, over
different photographs. The nuisance axes — `shuffle`, `subset`, `box` — are not
in this run at all and remain at pilot n.

---

## SECOND JUDGE, 2026-09-26 — a different family fails the same way

InternVL3-2B (OpenGVLab) on the identical `main` manifest (hash
`1d862ad6ce725e26`, 150 photos, 5,236 variants), judged and analysed against the
Qwen3-VL-8B run. Scores in
[`../results/internvl3_2b_2026-09-26/`](../results/internvl3_2b_2026-09-26/),
the paired analysis in
[`../results/two_judge_2026-09-26/`](../results/two_judge_2026-09-26/).

**Every figure below is floor-excluded (`--min-control 0`)**, the main entry's
primary convention — regions whose clean control already scored 0 cannot drop and
are guaranteed ties. All-region numbers run substantially higher (tie rates ~0.8
rather than ~0.59) and are not what is quoted here. Re-running stage 4 with
`--min-control 0` reproduces the committed `localization_*.csv` byte-for-byte,
which is how the cut was confirmed rather than assumed.

> **RETRACTED 2026-09-29.** This section originally read "**the
> non-localization result is method-level, not a Qwen artefact**". It does not
> support that. InternVL3-2B **fails the perception precondition** this project
> requires of any judge before a flat localization result can be read as an
> attribution failure — see "The second judge cannot see the damage" immediately
> below. Its AUROC of ~0.50 is the expected value for a judge that cannot see
> the stimulus, not evidence about the protocol. **The single-judge caveat
> stands**: every localization claim in this document is Qwen3-VL-8B only.
> The numbers below are kept because they are real measurements of what
> InternVL did; only the inference drawn from them is withdrawn.

### The second judge cannot see the damage (why the above is retracted)

The `main` entry's whole logic is that a flat per-region result is readable as an
*attribution* failure **only after** showing the judge perceives the corruption —
that is what `scripts/pq_response.py` exists for. That control was never run on
InternVL3-2B. Run now (`pq_response_internvl.csv`, same 5,236 variants):

| | Qwen3-VL-8B | InternVL3-2B |
|---|---|---|
| control AES, mean of 25 | 19.2 | **7.17** |
| between-base SD of control AES | 3.36 | **0.67** |
| dAES, noise s3 (the loudest stimulus) | **-2.68** | **-0.109** |
| dAES, jpeg s1 (the quietest) | -0.06 | -0.011 |
| share of variants where AES fell, noise s3 | **62%** | 24% |
| per-region `phi` exactly 0 | 42% | **63%** |

InternVL's image-level response to the strongest corruption is **25x smaller
than Qwen's** and barely separable from its response to the mildest. Its
`d_naturalness` is *positive* for most corruptions — damage nominally improves
its naturalness score, which is noise, not perception. It rates every image
around 7/25 with almost no between-image spread.

There is also **no positive control** for InternVL: the obeyed-vs-ignored probe
recorded under "Settled judge behaviour" was run on Qwen only.

A judge that does not register the stimulus cannot inform on attribution. To make
a cross-family claim, the second family has to pass this control first — that is
a GPU run, not a rewording, and it is now the honest top item for anyone who
wants the stronger statement.

### Why 2B, and what that costs the claim

InternVL3-8B was tried first and does not fit a 24GB A10 under two-image prompts
(forward-pass OOM), and InternVL3 has no 4B, so the family control is the 2B. The
"it just needs a bigger model" objection is already closed by **Qwen3-VL-8B**
failing, not by this run. What this run cannot rule out on its own is that a
larger InternVL would localise; the cross-family claim rests on the two together.

### The comparison, both judges, same variants

Tie rate on the damaged region vs any untouched region (`reward`,
`sensitivity.csv`):

| corruption / severity | InternVL3-2B target / other | Qwen3-VL-8B target / other |
|---|---|---|
| blur 1 / 3 | 0.587 / 0.565 · 0.529 / 0.506 | 0.539 / 0.576 · 0.327 / 0.388 |
| jpeg 1 / 3 | 0.568 / 0.561 · 0.505 / 0.485 | 0.641 / 0.643 · 0.332 / 0.364 |
| noise 1 / 3 | 0.587 / 0.574 · 0.549 / 0.519 | 0.609 / 0.610 · 0.190 / 0.211 |
| saturate 1 / 3 | 0.587 / 0.584 · 0.583 / 0.565 | 0.648 / 0.654 · 0.560 / 0.565 |
| remove (binary) | 0.490 / 0.463 | 0.264 / 0.288 |

Localization AUROC (`localization_reward.csv`, `localization_phi.csv`):

| readout | judge | sev 1 | sev 3 | remove |
|---|---|---|---|---|
| reward | InternVL3-2B | 0.499 | 0.493 | 0.498 |
| reward | Qwen3-VL-8B | 0.518 | 0.530 | 0.533 |
| phi | InternVL3-2B | 0.500 | 0.495 | 0.501 |
| phi | Qwen3-VL-8B | 0.513 | 0.521 | 0.538 |

Redundancy against the leave-one-out image mean (`redundancy_*.csv`):
InternVL **R^2 = 0.718** (`reward`) / **0.705** (`phi`), against Qwen's 0.562 /
0.520.

Coherence on `phi`, the honest per-region readout
([`phi/coherence.csv`](../results/two_judge_2026-09-26/phi/coherence.csv)):
mixed movement **0.073 against 0.596** expected under independence for InternVL,
and **0.278 against 0.511** for Qwen. InternVL's regions move together far more
completely — it is closer to a pure whole-image judgement than Qwen is. (The
Qwen figures reproduce the main entry's exactly, on the same variants.)

### Same kind, different degree — and the degree is the honest part

- **Same in kind.** Neither judge localises. For InternVL the target's tie rate
  is *above* the untouched regions' in all nine cells, and its mean per-region
  `phi` delta on the target is slightly *smaller* than on untouched regions
  (sev 1 -1.296 vs -1.323; sev 3 -1.555 vs -1.677; remove -2.867 vs -2.928).
  **Do not read this as "zero spatial information" — it is what a judge that
  cannot see the stimulus at all must produce.** InternVL fails the perception
  control (dAES -0.109 against Qwen's -2.68), so these cells measure the
  instrument, not the protocol.
- **Different in degree.** Qwen has the weak-but-nonzero signal the main entry
  describes (target `phi` delta -3.755 vs -2.377 on `remove`, ~1.6x); InternVL
  has none, and is more globally coupled on every measure (R^2 0.718 vs 0.562,
  `phi` mixed 0.073 vs 0.278). Qwen also reacts far more often under strong
  damage (noise s3 target tie 0.190 vs InternVL's 0.549).
- **The degree difference cannot be attributed to family.** The two judges differ
  in family *and* in scale (8B vs 2B), so "InternVL is flatter" confounds the two.
  What the pair supports is the shared conclusion — no localization in either —
  not a ranking between them.

### One limit of these artifacts, to know before quoting them

`drift_robustness.csv` **pools both judges**: `drift_robustness()` does not group
by `judge`, and its `frac_mixed` column takes the first row of a per-judge
coherence table (InternVL's). Its split (all: 119 bases, `target_unchanged`
0.484, AUROC 0.510; `edge_iou>=0.4`: 71 bases, 0.470, 0.511) does agree across
the split, but it is **not a per-judge number**. Qwen's own drift split is the
one in [`../results/main_2026-09-18/`](../results/main_2026-09-18/); a per-judge
split for InternVL has not been run.

`noise_floor.csv` is empty for both judges: greedy decoding, as designed.

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

**Quote this instead of the AUROC where you can.** Among variants where exactly
one region's score moved at all, the mover was the region we damaged **40.5% of
the time (n=699, 95% CI +/-3.6%) against a chance rate of 31.7%** (`reward`:
38.9% of 535, chance 32.1%). Same conclusion as the AUROC, but legible: the
signal is real, and it is small. On a scale that is 86% two-valued (see SCOPE at
the top), a rank statistic like AUROC is hard to interpret and easy to attack;
a counting statistic is neither.

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
the base-to-base spread. 42.1% of `reward` values sit at exactly 0 and four
values cover 82.1% of all of them; `phi` is 85.6% two-valued. See the scope
section at the top of this file — the scale is effectively binary, and that
conditions everything in this entry.

### Caveats and scope for the report (state all of these)

- **Base Qwen3-VL-8B + the A.4.3 prompt, NOT the deployed reward model.** SFReward
  is Qwen3-VL-8B *fine-tuned* on 14K examples that a Gemini-3-Pro teacher labelled.
  We audit the prompt-based protocol on the base model. We cannot claim the
  fine-tuned SFReward, or the Gemini teacher, behaves this way — the fine-tuning
  exists precisely to shape this behaviour. This bounds the claim and must be
  loud.
- ~~**One judge, one family** (Qwen3-VL-8B).~~ **Closed 2026-09-26**: InternVL3-2B,
  a different family, fails the same way on the identical manifest — see the
  second-judge entry at the top of this file. The remaining scale caveat is that
  the family control is a 2B, because InternVL3-8B does not fit the A10.
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

> **Superseded for the three exploit axes** by the 2026-09-29 run at 150 bases —
> see the top of this file. Two of the three numbers below reversed. This section
> remains the source for the **nuisance** axes (`shuffle`, `subset`, `box`), the
> **noise floor** and the **damage reference**, none of which a controls-only
> manifest can produce.

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

The `enhance` presentation tested it directly: a global cosmetic lift that
improves no edit should, if that reading is right, raise every region's reward
through `AES` alone.

**It does not.** At 150 bases (2026-09-29) `enhance` moves mean `reward` by
-0.002 and `enhance_target` by +0.002, both far below the 0.114 noise floor. The
factor is real and in the equation, but this judge does not reward a cosmetic
lift enough to move `AES`, and on a failed edit it structurally cannot: `phi = 0`
times any `AES` is still 0. Report the factor as an architectural property of
Eq. (3), not as a demonstrated exploit.

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
