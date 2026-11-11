# Lab Note T-M3e (Hardened Lag) & T-M4 (Blind Replication)

**Date:** 2026-10-16 · **Status:** COMPLETE — T-M3e PASS, T-M4 PASS
**Pre-registration:** T-M3e_T-M4_preregistration.md (registered before computation)

## T-M3e — Hardened lag test (unified dataset, 40,472 node-year obs)

Within-transformation (FWL) equivalent of society-year + node fixed effects; own-lag controls; society-clustered errors; sampling gaps >2 yr excluded. Kernel died during the dummy-variable attempt; estimator rebuilt via Frisch-Waugh residualisation — identical point estimates to the dummy specification, noted for the record.

| Direction | ΔC | ΔK |
|---|---|---|
| Forward (ΔS_t → Δ_{t+1}) | −0.0938, p = 4.5e-18 | −0.0927, p = 8.2e-16 |
| Reverse placebo (ΔS_t → Δ_{t−1}) | −0.0519, p = 5.3e-08 | −0.0413, p = 1.2e-05 |
| Forward/reverse ratio | 1.81× (bar 1.5×) | 2.25× (bar 1.5×) |

**Verdict: PASS.** The temporal claim survives hardening: stress uptake precedes capability decay with correct asymmetry. Two qualifications: (1) coefficients attenuated from −0.138 to −0.094 once common society-year shocks were absorbed — about a third of the raw association was shared-shock variance; (2) the non-zero placebo (reverse coefficients −0.04 to −0.05) confirms some scoring smear exists, as suspected. The forward dominance is real but the manuscript sentence should cite the hardened −0.09 figures, not the originals.

**Record-count note:** panel 40,472 node-years derives from the current unified file (43,400 node rows). The 32,361 figure cited by the user does not match this file version; flagged, unreconciled.

## T-M4 — Blind replication (JUNO_Blind_Set_{1,2,3,5,6,7}, frozen estimator)

Canonical schema verified (C,K,S,A only; T_Offset parsed to integer years). Frozen pipeline applied unchanged: z_S episode extraction, trough-aligned deltas, R_i = ΔC_i + ΔK_i + 0.5ΔA_i, argmax ΔS vs argmin R, Cohen's κ, 1000-shuffle permutation null. Minimum-n gate: 138 episodes ≥ 30, 48 blind societies ≥ 6 — PASSED.

| Set | Episodes | κ |
|---|---|---|
| 1 | 26 | 0.135 |
| 2 | 25 | 0.157 |
| 3 | 25 | 0.432 |
| 5 | 23 | 0.216 |
| 6 | 16 | 0.481 |
| 7 | 23 | 0.150 |
| **Pooled** | **138** | **0.267** (concordance 0.384, permutation p = 0.001, null 95th = 0.059) |

**Verdict: PASS** against frozen bars (pooled κ = 0.267 ≥ 0.20; p < 0.05). The blind estimate (0.267) lands slightly *above* the in-sample benchmark (0.249) — no degradation on untouched, independently anonymised score series.

**Registered caveat — heterogeneity:** set-level κ ranges 0.135–0.481; three of six sets fall below the 0.20 bar individually. The pooled pass is carried by sets 3, 5, 6. This is consistent with a real but moderate effect (per-set n ≈ 16–26 gives wide sampling error) but a fixed-effects meta-analytic read is more honest than the pooled number alone: the coupling replicates directionally in all six sets (all κ > 0) and exceeds 0.20 in half. Report both.

## Combined manuscript sentence (revised, hardened figures)

> CAMS distinguishes persistent functional vulnerability, the distribution of episode-specific disturbance, and subsequent adaptive outcomes. Registered tests show that stress uptake is associated with deterioration in non-Stress operating components (in-sample κ = 0.249, stress-excluded; blind replication κ = 0.267, n = 138 episodes, 48 societies, p = 0.001, with set-level heterogeneity 0.135–0.481), and that increases in scored Stress precede subsequent reductions in Coherence and Capacity (hardened coefficients −0.094/−0.093, forward/reverse placebo ratios 1.8–2.3×). Neither standing morphology nor damage locus has demonstrated reliable predictive power over aggregate recovery.

## Next steps (unchanged, per user priority)

External damage-record validation (staffing, production, administrative effectiveness, institutional continuity) against the identified damage loci; then the V-Dem mechanism test for recovery speed. Lore-locus recovery dip remains exploratory only.

## Correction trail

Lag coefficients revised downward (−0.138 → −0.094) after society-year FE; placebo confirms partial scoring smear; decomposition 0.50 ≈ 0.25 mechanical + 0.25 genuine retained as heuristic only (user qualification, 2026-10-16).
