# Lab Note T-M1: Displacement Morphology as Predictor of Episode Outcomes

**Status:** COMPLETE — registered FAIL on primary hypothesis
**Date:** 2026-10-16
**Dataset:** JUNO_Unified_Dataset.csv (KaliBond/wintermute, juno/), 48 societies; canonical verification PASSED (V = C+K−S+0.5A recomputed, max drift 3.6e-15). External rival: V-Dem core v16 from same repo (`v2x_libdem` liberal democracy index).
**Pre-registration:** T-M1_preregistration.md (registered before outcome computation). Two amendments logged below.

## Motivation

The 2026-10-16 reframing of CAMS as complex-adaptive-systems science proposed that a society's identity-in-adaptation is its persistent way of distributing stress across the eight functions (displacement morphology), and that this should predict how adaptive episodes resolve — better than regime type, ideology or aggregate state. FAC (Aug 2026) established morphology is society-level and stable but never tested it as an outcome predictor against rivals. T-M1 is that test.

## Methods

Stress episodes: within-society z-score of mean Stress; onset at first year z_S ≥ 1.0 after ≥2 years below; end at z_S < 0.5. 201 episodes extracted from 48 societies; 158 / 42 societies after censoring rules; 149 / 33 after LOEO typicality construction. [corrected: 42 was the post-censoring count] Displacement signature: per-node ΔS (peak year minus 2-year pre-onset mean), L1-normalised 8-vector. Society morphology: leave-one-episode-out mean signature; typicality = cosine(episode signature, LOEO mean). Primary outcome (as amended): full regain of trough V̄ loss within 5 years of episode end (base rate 0.73). Censored episodes excluded. Rivals: (R1) V-Dem `v2x_libdem` at onset; (R3) baseline V̄ and peak stress amplitude. Evaluation: logistic regression, 10× repeated stratified 5-fold CV. Nulls: 500-permutation society-label shuffle for stability gap; 200-permutation shuffle for morphology AUC.

**Amendment 1:** registered outcome (ratio ≥ 0.9 of baseline) produced base rate 0.006 — ratio undefined when baseline V̄ ≈ 0 or negative. Defect corrected to absolute criterion.
**Amendment 2:** absolute criterion set at full regain of peak-to-trough loss (≡ max post-window V̄ ≥ pre-onset baseline), base rate 0.73. Sensitivity grid reported.

## Results

| Test | Result | Verdict |
|---|---|---|
| T-M1c (stability gate) | within-society signature cos 0.849 vs between 0.823; gap 0.027, permutation p = 0.004 | PASS (gate open) |
| T-M1a (morphology predicts recovery) | CV AUC = 0.492 ± 0.018 | **FAIL** (needs ≥0.65 and ≥+0.05 over rivals) |
| T-M1b (adds beyond rivals) | full model 0.548 vs rivals-only 0.555: ΔAUC = −0.007 | **FAIL** |
| T-M1d (rivals beat morphology) | libdem AUC 0.547; baseline state 0.553; both > morphology 0.492 | **TRUE — conjecture fails as stated** |

Null check: in-sample morphology AUC 0.497 exceeds the 95th percentile (0.441) of the society-shuffle null (p = 0.005) — but in the *wrong* (anti-predictive) direction, and it does not survive cross-validation. Typicality vs years-to-recovery: Spearman ρ = 0.06, p = 0.54 (nil); libdem vs years-to-recovery: ρ = 0.21. Sensitivity grid (thresholds 0.9–1.1 × baseline, windows 3–7 yr): morphology AUC 0.50–0.52 throughout; libdem inverts (0.38–0.45 at short windows, 0.55 at 5 yr).

## Interpretation

The conjecture fails in its strong form. Displacement morphology is real (T-M1c passes: societies do have characteristic stress-absorption signatures, replicating FAC at dataset scale) but it is **descriptive character, not predictive machinery** — at least for recovery outcomes at annual resolution. Whether a society regains its losses within five years is carried weakly by regime type (V-Dem liberal democracy, AUC ≈ 0.55) and aggregate state, not by *which nodes absorb the stress*.

This is consistent with the earlier FAC partition (morphology society-level, recovery episode-level, ICC ≈ 0.09) now shown to have teeth: the two levels genuinely do not connect through this outcome. It also aligns with the null-model finding that the common mode (PC1) does the crisis-detection work: recovery is a common-mode phenomenon; morphology lives in the residual layer.

Note also the stability gap is small (0.849 vs 0.823): all societies' signatures are broadly similar (stress concentrates in fast-loop nodes everywhere), and the society-specific deviation is a modest overlay.

## What survives

- Morphology as *identity/character*: societies have reproducible stress-distribution signatures (p = 0.004). Useful for classification, blind identification, and historical description.
- Regime-type signal in recovery speed: V-Dem libdem predicts 5-yr recovery and years-to-recovery better than any CAMS morphology measure tested here. Worth a registered follow-up on mechanism (is it K_Stewards? Flow capacity?).

## Limitations

Annual resolution ceiling applies; integer-scale inputs limit signature precision; episode extraction is convention-gated (z_S thresholds); recovery outcome discards transformation cases; V-Dem coverage dropped 11 episodes; horse race used linear models only. The morphology measure is the *mean* signature — distributional morphology (variance of signatures) untested.

## Correction trail

Amendments 1–2 above; registered interpretation table applied: "T-M1a FAIL, rivals win → conjecture fails as stated; morphology is descriptive character, not predictive machinery. Reported plainly." Done.
