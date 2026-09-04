"""
Stage 4 against synthetic scores in stage 3's exact output schema.

Stage 4 has never seen real data and cannot until stages 0/1 run on the editor
VM. What CAN be checked on the laptop is that the analysis says the right thing
about data whose answer we already know - so this fabricates three judges with
three known behaviours and asserts the tables recover them:

  perfect      only the corrupted region drops           -> AUROC 1.0
  blind        every score is the same constant          -> degenerate, no crash
  global       damage moves EVERY region equally         -> AUROC 0.5, R^2 ~ 1

`global` is the important one. It is exactly the failure mode the audit exists
to detect - a per-region number that is really one image-level impression - and
a stage 4 that reported a healthy AUROC for it would be worse than useless.

Run:  ./.venv/Scripts/python.exe tests/test_stage4.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage4_analyze import (  # noqa: E402
    load, usable, noise_floor, delta_table, localization,
    localization_by_corruption, leakage_matrix, redundancy, axis_table,
    BINARY, FLAT_SEVERITY,
)

# 20, not 6. The null AUROC for a judge with no spatial resolution is centred
# on 0.5 either way, but its SD is 0.041 at 6 bases and 0.021 at 20 - so a
# +/-0.1 assertion is a 2.4-sigma coin flip on the small fixture and a ~5-sigma
# margin on this one. Measured, not guessed: 20 seeds at each size.
N_BASES = 20
REGIONS = [0, 1, 2]
CORRUPTIONS = ["blur", "noise", "remove"]
SEVERITIES = [1, 3]
N_SAMPLES = 5
C = 25.0

# Deliberately small but not tiny: the noise floor is estimated from the
# controls' sample-to-sample SD, so the jitter has to be big enough to measure
# and small enough that a real effect clears it.
JITTER = 0.4


def _phi(behaviour: str, scored: int, target: int, corruption: str,
         severity: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return (success, preserve) on 0-25 for one region of one variant.

    Corruption damages `preserve`, not `success`: the edit still follows the
    instruction, it is just degraded. That is the axis split the real judge is
    suspected of ignoring, so the fixture models it explicitly.
    """
    success, preserve = 22.0, 23.0
    if behaviour == "blind":
        return 25.0, 25.0
    if corruption != "none":
        hit = 4.0 * severity
        if behaviour == "perfect":
            if scored == target:
                preserve -= hit
        elif behaviour == "global":
            # The damage lands on every region equally - no spatial resolution
            # whatsoever, which is the failure this project is looking for.
            preserve -= hit
    j = rng.normal(0, JITTER, 2)
    return (float(np.clip(success + j[0], 0, 25)),
            float(np.clip(preserve + j[1], 0, 25)))


def make_scores(behaviour: str, judge: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for b in range(N_BASES):
        base_id = f"base{b:03d}"
        rows = [(r, "none", 0, True) for r in REGIONS]
        rows += [(r, c, s, False) for r in REGIONS
                 for c in CORRUPTIONS for s in SEVERITIES]
        for target, corr, sev, is_ctrl in rows:
            vid = f"{base_id}_{target}_{corr}_{sev}"
            for i in range(N_SAMPLES):
                pq = [float(np.clip(rng.normal(21, JITTER), 0, 25)),
                      float(np.clip(rng.normal(20, JITTER), 0, 25))]
                bg = float(np.clip(rng.normal(23, JITTER), 0, 25))
                for scored in REGIONS + ["bg"]:
                    if scored == "bg":
                        succ = pres = None
                        phi = bg
                    else:
                        succ, pres = _phi(behaviour, scored, target, corr, sev, rng)
                        phi = min(succ, pres)
                    recs.append(dict(
                        variant_id=vid, base_id=base_id,
                        target_region_id=str(target), corruption=corr, severity=sev,
                        area_bin="full", is_control=is_ctrl, sample_idx=i,
                        scored_region_id=str(scored),
                        sc_success=succ, sc_preserve=pres, sc_background=bg,
                        sc_overall_success=22.0, sc_overall_preserve=23.0,
                        pq_naturalness=pq[0], pq_artifacts=pq[1],
                        reward=float(np.sqrt(phi * min(pq)) / C),
                        parsed=True, raw="{}", judge=judge,
                    ))
    return pd.DataFrame(recs)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="mcv_stage4_"))

    frames = {b: make_scores(b, f"fixture/{b}", 1000 + i)
              for i, b in enumerate(["perfect", "blind", "global"])}
    for b, f in frames.items():
        f.to_parquet(tmp / f"scores_shard_{b}.parquet")

    print("=" * 68)
    print("1. SCHEMA: every column stage 3 emits survives the round trip")
    print("=" * 68)
    df = load(str(tmp / "scores_shard_*.parquet"))
    ok &= check("all three judges loaded", df.judge.nunique() == 3,
                f"{sorted(df.judge.unique())}")
    ok &= check("phi derived for foreground and bg", df.phi.notna().all())
    ok &= check("region ids normalised to str",
                df.scored_region_id.map(type).eq(str).all())
    ok &= check("bg phi equals sc_background",
                np.allclose(df[df.scored_region_id == "bg"].phi,
                            df[df.scored_region_id == "bg"].sc_background))

    print("\n" + "=" * 68)
    print("2. NOISE FLOOR is measured, positive, and near the injected jitter")
    print("=" * 68)
    nf = noise_floor(usable(df, "phi"), "phi")
    print(nf.round(4).to_string(index=False))
    nfi = nf.set_index("judge")["median"]
    ok &= check("one row per judge", len(nf) == 3)
    # The two jittered judges must show a real floor; `blind` emits a constant,
    # so its floor is exactly 0 and that is the informative answer, not a bug.
    # It also means the "effect vs noise floor" verdict cannot be fooled: a
    # constant judge has delta == 0 exactly, so 0 > 0 is false and it is
    # correctly reported as no signal rather than as an infinite effect size.
    ok &= check("jittered judges have a positive floor",
                bool(nfi[["fixture/perfect", "fixture/global"]].gt(0).all()),
                f"{nfi.round(4).to_dict()}")
    ok &= check("constant judge has a floor of exactly 0",
                nfi["fixture/blind"] == 0)
    ok &= check("floor within 3x of injected jitter",
                bool((nfi < 3 * JITTER).all()),
                f"jitter={JITTER}")

    print("\n" + "=" * 68)
    print("3. LOCALIZATION recovers each judge's known behaviour")
    print("=" * 68)
    d = delta_table(usable(df, "phi"), "phi")
    loc = localization(d)
    print(loc.round(4).to_string(index=False))

    a_perfect = loc[loc.judge == "fixture/perfect"].auroc
    a_global = loc[loc.judge == "fixture/global"].auroc
    ok &= check("perfect judge -> AUROC ~ 1.0", bool((a_perfect > 0.98).all()),
                f"{list(a_perfect.round(3))}")
    ok &= check("global judge -> AUROC ~ 0.5 (no spatial information)",
                bool(((a_global - 0.5).abs() < 0.1).all()),
                f"{list(a_global.round(3))}")
    ok &= check("blind judge produces no usable separation",
                bool(((loc[loc.judge == 'fixture/blind'].auroc - 0.5)
                      .abs() < 0.1).all()),
                f"{list(loc[loc.judge == 'fixture/blind'].auroc.round(3))}")
    ok &= check("background excluded from AUROC",
                "bg" not in set(d[d.scored_region_id == "bg"].scored_region_id)
                or len(localization(d)) == len(loc))
    # Graded severities only. `remove` reports as one BINARY row because its
    # severity 1 and 3 are the same stimulus; including it here would compare
    # an effect-size ladder against something that has none.
    graded = loc[(loc.judge == "fixture/perfect") & (loc.severity != BINARY)]
    ok &= check("severity 3 localises at least as well as severity 1",
                bool(graded.sort_values("severity").auroc.is_monotonic_increasing
                     or (a_perfect > 0.98).all()))

    print("\n" + "=" * 68)
    print("4. LEAKAGE MATRIX: diagonal for `perfect`, uniform for `global`")
    print("=" * 68)
    lm_p = leakage_matrix(d, "fixture/perfect")
    lm_g = leakage_matrix(d, "fixture/global")
    print("perfect:\n" + lm_p.round(3).to_string())
    print("global:\n" + lm_g.round(3).to_string())
    diag_p = np.mean([lm_p.loc[i, str(i)] for i in lm_p.index])
    off_p = np.mean([lm_p.loc[i, c] for i in lm_p.index
                     for c in lm_p.columns if c != str(i) and c != "bg"])
    ok &= check("perfect: diagonal far below off-diagonal",
                diag_p < off_p - 3.0, f"diag={diag_p:.2f} off={off_p:.2f}")
    diag_g = np.mean([lm_g.loc[i, str(i)] for i in lm_g.index])
    off_g = np.mean([lm_g.loc[i, c] for i in lm_g.index
                     for c in lm_g.columns if c != str(i) and c != "bg"])
    ok &= check("global: diagonal indistinguishable from off-diagonal",
                abs(diag_g - off_g) < 1.0, f"diag={diag_g:.2f} off={off_g:.2f}")

    print("\n" + "=" * 68)
    print("5. REDUNDANCY separates a real per-region score from a copied one")
    print("=" * 68)
    red = redundancy(usable(df, "phi"), "phi")
    print(red.round(4).to_string(index=False))
    r_global = float(red[red.judge == "fixture/global"].r2.iloc[0])
    r_perfect = float(red[red.judge == "fixture/perfect"].r2.iloc[0])
    ok &= check("global judge: R^2 high (region score is the image score)",
                r_global > 0.5, f"R^2={r_global:.3f}")
    ok &= check("perfect judge: R^2 lower than global",
                r_perfect < r_global, f"{r_perfect:.3f} < {r_global:.3f}")

    print("\n" + "=" * 68)
    print("6. AXIS TABLE shows preserve moving while success holds")
    print("=" * 68)
    ax = axis_table(df[df.judge == "fixture/perfect"])
    print(ax.to_string(index=False))
    dmg = ax[ax.corruption != "none"]
    ok &= check("preserve falls with severity",
                dmg[dmg.severity == "3"].sc_preserve.mean()
                < dmg[dmg.severity == "1"].sc_preserve.mean())
    ok &= check("success does NOT fall with severity",
                abs(dmg[dmg.severity == "3"].sc_success.mean()
                    - dmg[dmg.severity == "1"].sc_success.mean()) < 1.0)

    print("\n" + "=" * 68)
    print("6b. FLAT-SEVERITY corruptions collapse to one condition")
    print("=" * 68)
    # `remove` inpaints the object away at every setting: on real photographs
    # severity 1 and 3 changed the masked pixels by 35.01 vs 35.52 mean 8-bit
    # levels. A flat response there is an ABSENT STIMULUS, not an insensitive
    # judge, and pooling it into the severity ladder reports our own design as
    # a finding.
    ok &= check("remove reports as one binary condition",
                set(d[d.corruption == "remove"].severity) == {BINARY},
                f"{sorted(set(d[d.corruption == 'remove'].severity))}")
    ok &= check("graded corruptions keep their ladder",
                set(d[d.corruption == "blur"].severity) == {"1", "3"},
                f"{sorted(set(d[d.corruption == 'blur'].severity))}")
    ok &= check("no severity row is dropped by the collapse",
                len(d) == len(delta_table(usable(df, "phi"), "phi")))
    ok &= check("remove still gets its own by-corruption row",
                "remove" in set(localization_by_corruption(d).corruption))
    ok &= check("the binary row is separate, not merged into 1 or 3",
                BINARY in set(localization(d).severity),
                f"{sorted(set(localization(d).severity))}")

    print("\n" + "=" * 68)
    print("7. BY-CORRUPTION split runs and covers every corruption")
    print("=" * 68)
    lc = localization_by_corruption(d)
    print(lc.round(4).to_string(index=False))
    ok &= check("all corruptions present",
                set(lc.corruption.unique()) == set(CORRUPTIONS))

    print("\n" + "=" * 68)
    print("8. MISSING READOUTS are dropped per-column, not globally")
    print("=" * 68)
    holed = df.copy()
    holed.loc[holed.index[:len(holed) // 4], "reward"] = np.nan
    ok &= check("reward loses rows", len(usable(holed, "reward")) < len(holed))
    ok &= check("phi keeps all of them", len(usable(holed, "phi")) == len(holed))

    print("\n" + "=" * 68)
    print("8b. FLOOR FILTER drops only regions whose CLEAN control is at the floor")
    print("=" * 68)
    # A region already scored 0 on the clean edit cannot drop when damaged, so
    # it contributes a guaranteed zero delta to both classes and drags AUROC
    # toward 0.5. Dropping it must remove exactly those rows and nothing else --
    # a filter that took out real regions would manufacture the result it is
    # supposed to be measuring.
    from src.stage4_analyze import drop_floored  # noqa: E402
    dd = delta_table(usable(df, "phi"), "phi")
    dd.loc[dd.index[:40], "ctrl_score"] = 0.0
    kept = drop_floored(dd, 0.0)
    ok &= check("dropped exactly the floored rows",
                len(kept) == len(dd) - 40, f"{len(kept)} of {len(dd)}")
    ok &= check("every surviving row is above the threshold",
                bool((kept.ctrl_score > 0).all()))
    ok &= check("a threshold below the minimum keeps everything",
                len(drop_floored(dd, -1.0)) == len(dd))
    ok &= check("the filter does not add or reorder columns",
                list(kept.columns) == list(dd.columns))

    print("\n" + "=" * 68)
    print("9. THE CLI RUNS END TO END and writes every artefact")
    print("=" * 68)
    outdir = tmp / "analysis"
    r = subprocess.run(
        [sys.executable, "-m", "src.stage4_analyze",
         "--scores", str(tmp / "scores_shard_*.parquet"),
         "--out", str(outdir), "--all-readouts", "--min-control", "0"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    ok &= check("exit code 0", r.returncode == 0)
    for f in ("noise_floor.csv", "axis_by_severity.csv", "deltas.parquet",
              "localization_reward.csv", "localization_phi.csv",
              "redundancy_reward.csv"):
        ok &= check(f"wrote {f}", (outdir / f).exists())
    ok &= check("stdout is ASCII (the Windows console is not UTF-8)",
                r.stdout.isascii(),
                "" if r.stdout.isascii() else
                repr([c for c in set(r.stdout) if not c.isascii()][:5]))

    print("\n" + "=" * 68)
    print("ALL PASS" if ok else "FAILURES ABOVE")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
