# Pre-Registration T-M3d: Mechanical-Coupling Audit of T-M3b

**Date:** 2026-10-16 · **Status:** REGISTERED before computation · Same 149 episodes as T-M1/T-M2/T-M3 · Dataset canonical-verified.

## Rationale

T-M3b (κ = 0.476) compared argmax ΔS_i with argmin ΔV_i. But V = C + K − S + 0.5A, so ΔV_i = ΔC_i + ΔK_i − ΔS_i + 0.5ΔA_i: if C, K, A were constant, concordance would be 100% by construction. κ = 0.476 therefore bounds, but does not establish, an independent organisational relationship. T-M3d isolates the non-mechanical component.

## Operational definitions

- Per node per episode: t_i* = year of node V minimum within the episode window. ΔX_i = X_i(t_i*) − mean X_i(2 yrs pre-onset), for X ∈ {C, K, S, A, V}. (Aligned-at-trough convention: both uptake and damage measured at the node's own trough year. Secondary check with society-peak-year uptake per T-M3 original.)
- **Stress-excluded residual damage:** R_i = ΔV_i + ΔS_i = ΔC_i + ΔK_i + 0.5ΔA_i.
- **Primary test T-M3d-1:** concordance of argmax ΔS_i with argmin R_i (does the stress-absorbing node also deteriorate most in its non-stress components?). Cohen's κ vs chance; bar κ ≥ 0.20 carried from T-M3b. Permutation null: shuffle R across nodes within episode (preserves marginal distributions, breaks node alignment), 1000 shuffles.
- **Mechanical share estimate:** recompute T-M3b concordance under pure-algebra simulation: hold ΔC, ΔK, ΔA at observed values, replace ΔS with episode-preserving node-shuffled ΔS; the drop in κ quantifies how much of T-M3b was the −ΔS term.
- **T-M3d-2 (lag):** does node-level ΔS_i(year t) predict ΔC_i, ΔK_i (t+1) beyond autocorrelation? Panel regression with node fixed effects, society-clustered errors; bar: coefficient on ΔS_t significant at p < 0.01 with sign negative for ΔC_{t+1}, ΔK_{t+1}.

## Registered interpretations

- T-M3d-1 κ ≥ 0.20 → stress uptake tracks genuine non-stress deterioration; displacement mechanism survives the audit; proceed to blind replication.
- κ < 0.20 → T-M3b was predominantly the algebraic identity; displacement signature demoted to redescription of the scoring equation at episode level; T-M3b verdict revised to FAIL-mechanical.
- T-M3d-2 significant → temporal precedence supports causal ordering stress→capability decay (weakest form; scoring-smearing caveat registered).
