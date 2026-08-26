# Team brief

Updated 2026-08-26. Start here — this is the plain-language version.
Setup commands, repo structure and constraints live in [`README.md`](README.md).

---

## What we're doing, in one page

When an AI edits an image, someone has to score how well it did. Older methods
gave one number for the whole picture. That's a weak teaching signal: if the
instruction was "make the car red, remove the bottle, age the person" and the
model nails two and botches one, a single mediocre score doesn't say which.

The 2026 papers fix this by asking a vision-language model to score **each
region separately**. Four papers now build on that idea.

**Our question: is the per-region score really about that region?** Or is the
judge forming a general impression of the whole picture and writing more or less
the same number beside every region label?

**How we test it.** Take an edited image. Deliberately damage exactly one
region. Ask the judge to score all of them. We know where the damage is because
we put it there. Three things could happen:

- only the damaged region's score drops → the signal is real
- every region's score drops → it's a whole-image score in disguise
- the wrong region's score drops → the credit lands in the wrong place

Whichever we see is the result. Either way it's publishable: we confirm an
assumption four papers rest on, or we find a hole in it.

**Why the pipeline looks the way it does.** Our five VMs can't share files and
have small disks, so we can't pass gigabytes of damaged images around. But
damaging a region is *deterministic* — same image, same mask, same seed, same
bytes out. So each VM regenerates only its own share locally. Only ~300MB of
edited images ever moves between machines, once, at the start.

That only works if every machine produces **byte-identical** damage. Hence the
hash check below.

---

## Where we are

**The pipeline works end to end, and the go/no-go pilot said GO.**

All five stages run on real data: COCO filtering → editing with FLUX → damaging
one region → judging with Qwen3-VL-8B → analysis. 100% of judge responses parse
and every region gets scored.

### What the pilot found

On 5 images we damaged one region at a time and watched the scores.

**First we checked the damage was actually visible** — "the judge ignored it"
only means something if there was something to ignore. It was: the damaged
region changed by 8–36 brightness levels out of 255, while everything outside it
changed by ~0. Obvious damage, precisely confined.

**Then, three things that all say the same thing:**

1. **The score usually doesn't move at all.** 53–80% of damaged regions get a
   score *identical* to the undamaged version.
2. **When it does move, it isn't the damaged region.** A region we damaged is no
   more likely to change than one we never touched.
3. **The judge changes its mind about the whole image at once.** If it scored
   regions independently we'd expect ~67% of cases where some regions move and
   others hold. We see 27%.

Put together: **the "per-region" score looks like one whole-image judgement
copied into each region slot** — exactly the failure we set out to look for.

### The honest caveat

**This is five photographs.** It's internally consistent and it looks
convincing, but five images is a pilot, not a result. The full run uses 100
images and gives ~21× the data. Nobody should quote these numbers as final.

---

## YOUR MISSIONS

### Everyone: send Nadav your determinism hash

**~10 minutes. No GPU needed. Three of five VMs still owe this.**

Why it matters: if your machine's damaged images differ from mine by even one
pixel, your results can't be combined with mine — and nothing downstream would
reveal it.

```bash
ssh student@mcvgpu2025s-00XX          # your VM
cd ~ && git clone https://github.com/nadavanolik/mcv-spatial-audit.git
cd mcv-spatial-audit
python -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh judge           # or `core` if you only want the hash
```

Already cloned it? `cd ~/mcv-spatial-audit && git pull` instead of cloning.

**Send back two lines** from the output — the `numpy / cv2 / pillow` versions
and the `CROSS-VM FIXTURE HASH`. It should read:

```
numpy 1.26.4 cv2 4.11.0 pillow 10.4.0
CROSS-VM FIXTURE HASH: 776feeddd281fa726195bf504c7b19c8
```

- **Matches?** You're done, nothing else needed.
- **Different hash?** Don't fix it yourself — post it *with* the versions line.
  It's almost always a library that resolved differently, and we need to see
  which one.
- **A test failed?** Post the whole thing. That's a real bug worth finding.
- **"no virtualenv is active"?** You skipped the `source` line.

### Per-person roles

| Role | Where it stands | Next |
|---|---|---|
| **Editor VM** | stages 0+1 working, 5 of 100 images edited | Edit the remaining 95 (~5h), then `tar czf bases.tar.gz -C data bases` and upload. **Everyone else is waiting on this.** |
| **Judge harness** | working, 100% parse, real published prompt | Decide the sampling config — decision 1 below |
| **Corruption + manifest** | determinism confirmed on 2 of 5 VMs | Chase the other three. Own `config.yaml` |
| **Analysis** | stage 4 runs on real data | Start the figures. The tie-rate and coherence tables are the headline ones, not AUROC |
| **Second judge** | Qwen3-VL-4B downloaded and working | Pick a second *family*, not just a second size, and justify it |

---

## Decisions we need to make together

1. **Sampling config for the main run.** Right now we ask the judge 5 times at a
   random-ish temperature. That made our first pilot unreadable — the judge
   disagreed with *itself* by 38% of the scale on identical inputs. Switching to
   deterministic fixed it, and cut the run from 3.1h to 1.1h per VM. Likely
   answer: deterministic for the main run, plus a small sampled run to document
   the instability. **Decide before anyone starts the main run.**

2. **`remove` has no severity ladder.** Its "mild" and "severe" settings differ
   by 0.5 out of 35 — there's no gradient there at all. Either drop it from
   severity comparisons or redefine what severity means for it.

3. **"remove the X" instructions in stage 0.** They clash with `remove` also
   being one of our damage types, and "was this region preserved?" is
   meaningless for a region we told the editor to delete.

## Two things that must reach the report

- **Every pilot number rests on 5 photographs.** Say so wherever one appears.
- **The judge is unstable across repeated samples** — 38% of the scale on
  identical input. That's a problem for RL training on its own, separate from
  whether it localises.

---

## Gotchas if you're touching a VM

All handled in the code, but you'll hit the symptoms if you deviate:

- **`pip install torch` gives you a broken build.** It fetches CUDA 13; our
  driver caps at 12.8, and torch then claims there's no GPU at all. Use
  `--index-url https://download.pytorch.org/whl/cu128`.
- **Don't set `HF_HOME`.** A non-default path gives you a second empty cache and
  re-downloads models you already have until the disk fills.
- **Don't change `--gpu-util`.** 0.89 sits in a narrow window — both higher and
  lower fail — and all five VMs must use the same value.
- **Stage 1 needs sequential offload.** FLUX doesn't fit otherwise, and this is
  a VRAM limit, so freeing disk space won't help.

## A question we closed

We thought we'd found something: on synthetic test images the judge noticed when
an object was *deleted* but ignored it being *blurred*. That would have been a
clean result — the reward tracks meaning, not quality.

**It vanished on real photographs.** Blur and remove produce identical response
rates (20.0% vs 20.0% at mild, 26.7% vs 26.7% at severe). It was an artefact of
the fake images. Worth remembering: a synthetic sanity check produced a
plausible, self-consistent, completely wrong conclusion, and only real data
caught it.

---

## Timeline to 30.9

- **Week 1 — done.** Setup, pipeline verified, pilot, go/no-go. Verdict GO.
- **Week 2** — 100 images edited and shipped; manifest frozen; main run.
- **Week 3** — second judge family; nuisance + exploitability tests.
- **Week 4** — figures, LaTeX, repo cleanup.
- **Week 5** — buffer and the 5-minute talk. Don't plan work here.

The "Optimal Reward ∆" stretch goal is out of scope unless week 3 finishes
early.
