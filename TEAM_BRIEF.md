# Team brief — where we are and what's next

Updated 2026-08-26.

Short version: **the pipeline runs end to end on real data.** All five stages,
100% parse rate, 100% region coverage, throughput understood and affordable.
What's left is running the experiment and writing it up.

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

## The design, in one idea

We can't pass gigabytes of images between five disconnected machines with 90G
disks each. But corrupting a region is a *deterministic* operation — same image,
same mask, same seed, same output bytes. So each VM regenerates only its own
share locally, into RAM.

What actually moves between machines: ~300MB of base edited images (once, ever)
plus a few MB of manifests and scores. That's it.

This only works if the corruption is byte-identical everywhere, so there's a
test suite enforcing it and a script that prints a hash we all need to match.
Two VMs have confirmed `776feeddd281fa726195bf504c7b19c8`. **Three still need
to.**

## What is done and verified

Everything below has actually been executed, not just written.

- **Stage 0 (COCO filter).** 187 of val2017's 5,000 images qualify (3–5
  distinct instructable categories at 2–25% area), mean 3.19 regions each. That
  covers `main`'s 100 bases with room to spare. `--survey` reports this from the
  annotations file alone, before you download any images.
- **Stage 1 (FLUX Kontext editing).** Works. 189s/image. The edit follows the
  instruction and mostly leaves other regions alone. 5 pilot bases are edited.
- **Stage 2 (corruption).** 75 pilot variants rendered from real edits in 3s.
- **Stage 3 (vLLM judging).** 100% parse rate, 100% region coverage,
  ~7.9s/request.
- **Stage 4 (analysis).** Migrated to the real score columns and verified
  against three synthetic judges with known behaviour.
- **The A.4.3 prompt is in, verbatim.** No longer a placeholder.

## What we learned the hard way (read this before touching the VMs)

Each of these cost real time and is now handled in code — but you'll hit the
symptoms if you deviate.

- **FLUX needs `--offload sequential`, not model-level.** Its transformer is
  23.8GB in bf16 and an A10-24Q leaves only 21.37GiB free. Model-level offload
  OOMs before step 1 of 28. This is VRAM, not disk — clearing the cache or
  unloading a judge changes nothing.
- **`torch` must be a cu128 build.** `pip install torch` resolves to a CUDA 13
  wheel, and our driver caps at 12.8; torch then reports "no CUDA device" and
  hides the reason in a warning. Use
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **`--gpu-util` has a narrow two-sided window, ~(0.861, 0.901).** Too high and
  vLLM won't start; too low and the KV cache is sized *negative*. The default is
  0.89. **All five VMs must use the same value** — it changes batch composition,
  which can perturb logits.
- **The judge's output must be schema-constrained.** Left free it drops regions
  and omits `background`, covering only ~43% of what we asked for — while
  reporting a healthy parse rate. It also falls into verbatim repetition loops
  that run to the token cap mid-JSON.
- **Never put `maxLength` in the schema.** It makes xgrammar count characters
  and costs **7.5×** for no quality gain.

## Roles

| Role | Status / next task |
|---|---|
| **Editor VM** | Stage 0 + 1 **working**; 5 of 100 bases edited. Next: edit the remaining 95 (~5h), tar `data/bases`, upload to HF Hub. **Everyone else is blocked on that tarball.** |
| **Judge harness** | vLLM + Qwen3-VL-8B **working**, prompt is the real A.4.3. Next: nothing blocking — help with the pilot. |
| **Corruption + manifest** | Determinism confirmed on 2 of 5 VMs. Next: get the other three to print the hash. Own `config.yaml`. |
| **Analysis** | `stage4_analyze.py` is migrated and tested. Next: sketch the figures against the synthetic fixtures in `tests/test_stage4.py`, which already produce every table. |
| **Second judge** | Qwen3-VL-4B is downloaded and runs. Next: pick a second *family* (not just scale) and justify it. |

## Setup on a fresh VM

```bash
git clone <repo> mcv-spatial-audit && cd mcv-spatial-audit
python -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh judge          # or editor / coco / core
```

Post the hash it prints. **All five must match.**

The editor VM needs its own venv (`.venv-editor`) because diffusers and vLLM
pin different torch builds — see README.

## What's next, in order

1. **Finish the base edits** (editor VM, ~5h for the remaining 95). Everything
   downstream waits on this.
2. **Run the pilot and look at the numbers.** `config.yaml`'s `pilot` profile is
   now `[none, blur, remove]` — `remove` is there because it is the only
   corruption the judge visibly reacted to on synthetic data, and `blur` is the
   contrast case. A pilot of only `blur` would have told us nothing.
3. **Cross-VM hash from the remaining three machines.**
4. **`main` run**: ~3.1h/VM, sharded five ways.
5. Second judge family; nuisance + exploitability tests; figures; LaTeX.

## The open scientific question

On synthetic images the judge reacted to `remove` (a 10-point drop) and **did
not move at all** for blur, JPEG or noise at any severity. If that holds on real
COCO edits, the finding is that the reward tracks *semantic* change but is blind
to *degradation* — which is a real result, and precisely the kind of hole this
project set out to look for.

It is not confirmed. Flat synthetic squares are far out of distribution, and the
pilot on real textured edits is what decides it. That is the single
highest-value thing left to learn.

## Timeline to 30.9

- **Week 1 — done.** Setup, pipeline verified end to end, harness bugs fixed.
- **Week 2** — 100 bases edited and shipped; manifest frozen; localization +
  redundancy on `main`.
- **Week 3** — second judge family; nuisance + exploitability tests.
- **Week 4** — figures, LaTeX, repo cleanup.
- **Week 5** — buffer and the 5-minute talk. Don't plan work here.

The "Optimal Reward ∆" stretch goal from the proposal is out of scope unless
week 3 finishes early.
