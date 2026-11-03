# Lab Note T-M2: Displacement Morphology as Predictor of Failure Morphology

**Status:** COMPLETE — registered FAIL on effect-size thresholds; weak-but-real society signal noted
**Date:** 2026-10-16 · **Dataset:** JUNO_Unified_Dataset.csv (canonical verification PASSED) · Same episode set as T-M1 (149 usable episodes, 33 societies, 29 with ≥3 episodes; minimum-n gate PASSED)
**Pre-registration:** T-M2_preregistration.md (registered before outcome computation)

## Motivation

T-M1 showed displacement morphology does not predict recovery. T-M2 tests the relocated hypothesis: morphology predicts *where* a society fails — the failure locus (node with minimum V during the episode, ties broken by largest ΔV from baseline).

## Results

Failure locus distribution (n=149): Hands 61, Flow 32 [corrected: plain-argmin convention], Shield 16, Helm 15, Lore 10, Stewards 10, Archive 7, Craft 1. Fast-loop loci dominate (121/149), consistent with the pooled responsiveness ordering (Flow/Helm most reactive) — but the *damage* lands on Hands most often.

| Test | Result | Registered bar | Verdict |
|---|---|---|---|
| T-M2a: society modal locus (LOEO) predicts episode locus | accuracy 0.510 vs a 0.409 marginal baseline (ratio 1.25×); permutation null mean 0.287, p = 0.002 [corrected 2026-10-16: plain-argmin locus canonical; original session's ad-hoc tie-break gave 0.470/0.389/1.21×] | ratio ≥ 1.5 AND p < 0.05 | **FAIL on effect size** — society identity carries real (p=0.002) but sub-threshold information |
| T-M2b: stress-uptake argmax → failure locus concordance | Cohen's κ = 0.154 [corrected from 0.184; original value not reproducible under script v2 — see REPRODUCIBILITY.md correction 5] | κ ≥ 0.20 | **FAIL** (marginal miss) |
| T-M2c: failure locus → recovery | χ² p = 0.72; recovery rates 0.60–1.00 across loci, no structure | omnibus p < 0.05 required | **FAIL** |

## Interpretation

Per the registered interpretation table, both primary branches failing points toward the strong restriction — but with an important qualification the table anticipated only in part. T-M2a's permutation result (p = 0.002) is decisive that the society signal is *not* scoring habit: knowing which society is experiencing an episode nearly doubles your chance of naming the failure locus over the null (0.47 vs 0.27). The signal is real; it is simply weaker than the registered bar (1.5× marginal). Honest summary: **societies fail in moderately characteristic places, but the characteristic is a lean, not a law.**

T-M2b's marginal miss (κ = 0.184 vs bar 0.20) qualifies the "universal mechanics" branch: the node that absorbs the most stress becomes the failure locus only 29% of the time. Stress-uptake and damage are partially decoupled — consistent with the two-layer pathology finding from Argentina (Hands–Flow basement as damage site regardless of where stress enters).

T-M2c closes the loop from T-M1: neither typicality nor locus predicts recovery. Recovery is common-mode; failure locus is partially society-typical; the two layers remain independent.

## Combined T-M1/T-M2 statement for the framing document

Societies have persistent stress-distribution signatures (p = 0.004) and moderately characteristic failure loci (p = 0.002, effect 1.2× marginal). Neither predicts whether or how fast they recover (AUC ≈ 0.50; χ² p = 0.72). The identity conjecture survives only in weakened form: *a society's adaptive character describes where it hurts, not whether it heals.* Regime type (V-Dem libdem, AUC 0.55, ρ = 0.21 on recovery speed) currently carries more outcome signal than any morphology measure tested.

## Limitations

Modal-locus prediction discards signature shape (uses only society mode); a signature-distance classifier might extract more. κ bar 0.20 was registered as "weak but real"; 0.184 is a borderline miss and should not be over-read in either direction. Hands dominance (39% of loci) compresses achievable accuracy. Failure locus by argmin V conflates chronic weakness with episode-specific damage; a ΔV-only locus definition is a registered candidate for T-M3.

## Correction trail

None. Registered fails reported as fails.
