# Team brief — where we are and what's next

Short version: the pipeline is scaffolded and the hardware questions are
answered. What's left is running it.

## What we're actually testing (refresher)

RL methods for image editing used to score the whole edited image with one
number. That's a bad training signal when one instruction contains three jobs —
if the model nails the car and botches the sunglasses, a single mediocre score
tells it nothing about which part to fix.

The recent papers fix this by asking a VLM judge to score each region
separately. Our question: **is the per-region score actually about that region?**
Or is the judge forming a whole-image impression and writing roughly the same
number next to every region label?

Our test: corrupt exactly one region of an edited image, then ask the judge to
score every region. We know where the damage is because we put it there. Three
things can happen:

- only the corrupted region's score drops → the signal is real
- every region's score drops → it's the global score in disguise
- the wrong region's score drops → the credit is misplaced

Which one we observe is the result. Either answer is publishable: we either
validate an assumption four papers rest on, or find a hole in it.

## Hardware — the answers

Each of our five VMs has a **full A10 24GB**, ~440GB RAM, 36 cores, 90G of
writable disk, and **no sudo**. Important discoveries:

- The 8B judge fits in bf16 on one GPU. The project is viable as proposed.
- **We can't use fp8** — the A10 is too old an architecture for those kernels.
- **`/datashare` is read-only and there's no shared filesystem between VMs.**
  This shaped the whole design.
- `/mnt` is root-owned. `/dev/shm` (217GB, RAM-backed) is our scratch space.

## The design, in one idea

We can't pass gigabytes of images between five disconnected machines with 90G
disks each. But corrupting a region is a *deterministic* operation — same image,
same mask, same seed, same output bytes. So each VM regenerates only its own
share locally, into RAM.

What actually moves between machines: ~300MB of base edited images (once, ever)
plus a few MB of manifests and scores. That's it.

This only works if the corruption is byte-identical everywhere, so there's a
test suite enforcing it and a script that prints a hash we all need to match.

## What's already done

- Full pipeline scaffolded: COCO filter → editing → manifest → corruption →
  judging → analysis. Repo is runnable.
- Determinism test suite written **and passing** (repeatability, seed
  sensitivity, order independence, spatial locality, area monotonicity).
- **Our biggest stated risk is solved.** The proposal said no public benchmark
  has multi-region annotations — but COCO's instance segmentation gives us
  named, pixel-accurate, non-overlapping masks, and the category names generate
  the instructions too. That's an afternoon of filtering, not a week of
  annotation.
- **Caught a scoping error.** Our proposal claims ~1,000 variants, but the
  design matrix as written expands to 24,800 variants and ~99,200 judge calls —
  5× over budget. `config.yaml` now has a `main` profile sized correctly
  (~4,400 variants, ~1.2h per VM) plus a tiny `pilot` profile.

## Roles

| Role | First task |
|---|---|
| **Editor VM** | Download COCO, generate ~100 base edits, upload the tarball. **Everyone else is blocked on this — start first.** |
| **Judge harness** | Get vLLM + Qwen3-VL-8B running; replace the placeholder prompt with the real one from the paper. |
| **Corruption + manifest** | Confirm the determinism hash matches across all five VMs. Own `config.yaml`. |
| **Analysis** | Read `stage4_analyze.py`; sketch the figures before data exists. |
| **Second judge** | Pick the second VLM family and justify it. Cross-family agreement is itself a finding. |

Once base edits exist, everyone runs `SHARD=k bash scripts/run_shard.sh`.

## This week

1. Pick roles. Nothing parallelizes until this happens.
2. Everyone: clone the repo, `pip install -r requirements.txt`, run
   `bash scripts/verify_determinism.sh`, post your hash. **All five must match.**
3. Someone find the real SFReward prompt in arXiv:2606.26872's appendix. It's a
   hard dependency and currently a placeholder in the code.
4. **Run the pilot end to end** — 5 images, ~240 requests, minutes. Then look at
   the score histogram.

Point 4 is the one that matters. If the judge hands out 4/5 to every region
regardless of what we did to the image, there's no signal to measure and we need
to redesign — with four weeks left, not one. Everything else is plumbing.

## Timeline to 30.9

- **Week 1** — setup, pilot, judge histogram. Go/no-go decision.
- **Week 2** — 100 bases edited and shipped; manifest frozen; localization +
  redundancy results.
- **Week 3** — second judge family; nuisance + exploitability tests.
- **Week 4** — figures, LaTeX, repo cleanup.
- **Week 5** — buffer and the 5-minute talk. Don't plan work here.

The "Optimal Reward ∆" stretch goal from the proposal is out of scope unless
week 3 finishes early.
