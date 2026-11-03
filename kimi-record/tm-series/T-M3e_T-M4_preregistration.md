# Pre-Registration T-M3e: Hardened Lag Test & T-M4: Blind Replication of Stress-Excluded Coupling

**Date:** 2026-10-16 · **Status:** REGISTERED before computation

## T-M3e — Lag safeguards (same data as T-M3d-2)

Registered safeguards demanded before the lag result carries weight:
1. **Society-year fixed effects** (absorb common shocks) in addition to node FE and own-lag controls.
2. **Reverse-time placebo:** ΔS_t → ΔC_{t−1}, ΔK_{t−1}. A genuine forward process should show |forward coef| > |reverse coef|; comparable magnitudes indicate scoring smear.
3. Clustering by society (not node), respecting cross-node dependence within society.
4. Record-count reconciliation note (40,944 panel obs vs 43,400 file rows; user's cited 32,361 does not match this file version — flagged, not reconciled).

Bars: forward coefficients negative, p < 0.01, AND |forward| ≥ 1.5× |reverse placebo|. If placebo comparable → lag result demoted to "consistent with scoring smear", temporal claim withdrawn.

## T-M4 — Blind replication

**Frozen estimator (exactly):** episode extraction (z_S within society, onset z≥1.0 after ≥2 yrs <1.0, end z<0.5); trough-aligned deltas at node V-minimum year vs 2-yr pre-onset mean; stress-excluded residual R_i = ΔC_i + ΔK_i + 0.5ΔA_i; concordance of argmax ΔS_i with argmin R_i; Cohen's κ; 1000 within-episode node-shuffle permutation null.

**Frozen comparison:** in-sample benchmark κ = 0.249 (threshold bar 0.20). Replication pass: blind κ ≥ 0.20 AND permutation p < 0.05. Frozen minimum-n: ≥30 episodes across ≥6 societies in the blind corpus.

**Data:** JUNO_Blind_Set_{1,2,3,5,6,7}.csv (repo blind-test/), previously untouched by this programme's morphology pipeline. T_Offset schema; canonical verification (V drift check) run first, drift flagged.

## Registered interpretations

- T-M4 PASS → stress-excluded coupling generalises to independently constructed score series; proceed to external damage-record validation.
- T-M4 FAIL (κ < 0.20 or p ≥ 0.05) → in-sample result is corpus-specific; displacement mechanism claim restricted to unified-dataset societies; report plainly.
- T-M3e placebo fail → T-M3d-2 temporal claim withdrawn from the manuscript sentence; contemporaneous association retained.
