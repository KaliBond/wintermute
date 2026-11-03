# Lab Note T-M3d: Mechanical-Coupling Audit of T-M3b

**Status:** COMPLETE — T-M3d-1 PASS (κ = 0.249 ≥ 0.20 bar, p = 0.001); mechanical share quantified; T-M3d-2 PASS
**Date:** 2026-10-16 · Same 149 episodes · Dataset canonical-verified
**Pre-registration:** T-M3d_preregistration.md (registered before computation)

## Motivation

T-M3b (κ = 0.476) compared argmax ΔS_i with argmin ΔV_i, but ΔV = ΔC + ΔK − ΔS + 0.5ΔA: the −ΔS term guarantees partial concordance by construction. T-M3d isolates the non-mechanical component before any blind replication.

## Methods

Per node per episode: trough-aligned deltas (t_i* = node's own V-minimum year; ΔX = X(t_i*) − 2-yr pre-onset mean, X ∈ {C,K,S,A,V}). Stress-excluded residual damage R_i = ΔC_i + ΔK_i + 0.5ΔA_i (identity R = ΔV + ΔS verified to 1e-9). Nulls: 1000 within-episode node-shuffles of R (alignment null) and of ΔS (pure-algebra floor). Lag test: node-level panel, ΔC_{t+1}/ΔK_{t+1} ~ ΔS_t + own-lag + node fixed effects, node-clustered errors, 40,944 node-year observations (distinct from the 40,472-obs society-year-FE panel in T-M3e), sampling gaps >2 yr excluded.

## Results

| Quantity | κ | Interpretation |
|---|---|---|
| T-M3b re-run (trough-aligned) | 0.499 | total concordance, stress-inclusive |
| Pure-algebra floor (shuffled ΔS) | 0.252 | concordance the −ΔS term generates even when stress pattern is *unrelated* to R |
| **T-M3d-1: argmax ΔS vs argmin R** | **0.249** | genuine non-stress deterioration alignment (null mean 0.022, 95th 0.073, p = 0.001) |

Decomposition: of T-M3b's κ ≈ 0.50, roughly half is the algebraic identity term and half is real coordination — the node absorbing the most stress also deteriorates most in Coherence, Capacity and Abstraction. The registered bar (κ ≥ 0.20) is met by the stress-excluded measure alone.

**T-M3d-2 (lag):** ΔS_t → ΔC_{t+1}: coefficient −0.138, p = 1.8e-16. ΔS_t → ΔK_{t+1}: −0.138, p = 1.0e-21. Both with own-lag controls and node fixed effects. Stress uptake systematically *precedes* non-stress deterioration — the temporal ordering required for the weakest causal reading.

## Interpretation (per registered table)

- T-M3d-1 κ = 0.249 ≥ 0.20 → **displacement mechanism survives the audit.** The displacement signature is not a redescription of the scoring equation. Cleared for blind replication.
- T-M3d-2 significant → stress→capability-decay ordering supported, with the registered caveat: scorers assigning all four vitals in one pass may smear judgements contemporaneously; annual-resolution lead claims remain barred, and the panel includes non-episode years (the lag is a general property, not episode-specific).
- Mechanical share ≈ 50%: any future use of stress-inclusive damage loci must report the algebraic floor (≈ 0.25 κ) alongside, or use R-based damage.

## Revised verdict on T-M3b

T-M3b stands but re-described: total κ 0.50 = ~0.25 algebraic identity + ~0.25 genuine organisational coupling. The genuine component is what goes forward to blind replication.

## Updated synthesis (guarded form, per user's 2026-10-16 qualification)

A society's standing functional configuration contains modest historical regularities (chronic weak-node recurrence, p = 0.002) but does not reliably determine where an episode's greatest *new* damage occurs (T-M3a: 0.94× marginal). Stress uptake aligns with calculated functional damage, and the independent (non-mechanical) component of that alignment is now established at κ ≈ 0.25 with correct temporal ordering. Neither standing morphology nor damage locus predicts recovery (T-M1, T-M3c). Structural condition, disturbance and adaptation are distinct, interacting processes.

## Limitations

Trough-aligned measurement lets each node pick its own worst year (slight concordance inflation vs society-peak alignment; T-M3 original alignment gave κ 0.476, consistent). Lag test uses all years, not episodes only. Scorer contemporaneity smearing cannot be excluded from within the data. The R-measure still derives from CAMS scores — the user's final safeguard (external damage records) remains open and is the correct next step after blind replication.

## Correction trail

T-M3b re-described (mechanical share quantified); no registered fails rescinded.
