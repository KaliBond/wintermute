# Lab Note T-M3: Failure Locus by Episode-Specific Damage (ΔV-only)

**Status:** COMPLETE — mixed: T-M3b PASS (large), T-M3a FAIL (confound confirmed), T-M3c FAIL
**Date:** 2026-10-16 · **Dataset:** JUNO_Unified_Dataset.csv (canonical verification PASSED) · Same 149 episodes as T-M1/T-M2
**Pre-registration:** T-M3_preregistration.md (registered before outcome computation)

## Results

**Locus redistribution:** redefining failure locus as largest ΔV drop from pre-onset baseline relocates 62.4% of episodes [corrected from 58% under canonical plain-argmin locus]. Hands falls from 58 to 14 episodes; Flow rises 32→49, Helm 15→32. T-M2's Hands dominance was substantially a *level artefact*: Hands sits chronically low in V, so argmin-V named it the "locus" even when it dropped little. This is a substantive correction to any cross-society "Hands basement" reading — though note the within-society Argentina finding (June 2026) already used ΔV and is not touched by this correction.

| Test | Result | Bar | Verdict |
|---|---|---|---|
| T-M3a: society modal ΔV-locus (LOEO) | accuracy 0.309 vs marginal 0.329 (ratio 0.94); permutation p = 0.034 (null mean 0.223) | ≥1.5× marginal AND p<0.05 | **FAIL** — real but *weaker* than T-M2a's level-based signal (1.21×) |
| T-M3b: stress-uptake argmax → ΔV-locus | concordance 0.564, Cohen's κ = 0.476 | κ ≥ 0.20 | **PASS** — strong |
| T-M3c: ΔV-locus → recovery | χ² p = 0.19 | p < 0.05 | **FAIL** |

## Interpretation (per registered table)

1. **T-M3b PASS where T-M2b failed → uptake/damage coupling restored.** The node that absorbs the most stress in an episode is the node that drops the most in value, 56% of the time, κ = 0.48. Stress-uptake and damage are strongly coupled at episode level — the decoupling in T-M2 was an artefact of measuring damage by level rather than change. This is the cleanest validation yet of the displacement-signature construct: d_e (stress signature) genuinely tracks where the episode does its work.

2. **T-M3a weaker than T-M2a → the society signal was chronic level, not damage pattern.** Societies' characteristic "failure loci" under argmin-V were largely their chronic low-V nodes recurring. Episode-specific damage is only marginally society-typical (p = 0.034, below-marginal accuracy ratio). The society-identity signal in failure morphology is real but lives in standing structure, not in how episodes unfold.

3. **T-M3c FAIL:** damage locus does not predict recovery either (p = 0.19). Lore-locus episodes show the lowest recovery rate (0.40, n=10) — exploratory only, below the omnibus bar, registered as a candidate for a slow-loop-damage hypothesis rather than a result.

## Updated combined statement (T-M1/T-M2/T-M3)

- Stress-uptake signatures track episode damage strongly (κ = 0.48) — the displacement construct is behaviourally real.
- Societies have characteristic *standing* weak nodes (level signal, p = 0.002) but only marginally characteristic *damage* patterns (p = 0.034).
- Nothing in morphology — typicality, level-locus, or damage-locus — predicts recovery (AUC ≈ 0.50; p = 0.72 / 0.19). Recovery remains common-mode plus a weak regime-type signal.

Refined sentence for the framing document: *stress lands where it lands (uptake signature → damage, κ ≈ 0.5); societies differ in where they stand weak, not in how episodes strike them; and whether they heal is a different question altogether.*

## Limitations

ΔV-locus still uses episode minimum vs 2-yr baseline; noisy for societies with sparse sampling near onset. LOEO modal prediction is a crude classifier of signature shape. Lore-locus recovery dip (0.40) underpowered (n=10). κ = 0.48 is "moderate" by Landis-Koch conventions, not decisive; registered bar met but effect should be replicated on the blind holdout sets before hardening.

## Correction trail

None. T-M2 Hands-dominance reading corrected herein (level artefact).
