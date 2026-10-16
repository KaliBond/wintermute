# Pre-Registration T-M3: Failure Locus by Episode-Specific Damage (ΔV-only)

**Date:** 2026-10-16 · **Status:** REGISTERED before outcome computation · Dataset: JUNO_Unified_Dataset.csv (canonical verification PASSED) · Same episode set as T-M1/T-M2 (149 usable episodes, 33 societies).

## Rationale

T-M2 defined failure locus as argmin V during the episode — conflating chronic weakness with episode-specific damage (a node can be chronically lowest and drop little). Registered candidate fix: define locus by largest ΔV drop from the 2-year pre-onset baseline. T-M3 re-runs the T-M2 battery under this definition.

## Operational definitions

- **Failure locus (ΔV-only):** node with largest drop ΔV_i = V_i(episode minimum) − V_i(pre-onset 2-yr mean). If a node's episode minimum exceeds its baseline (no drop), it cannot be a locus.
- All other definitions (episodes, signatures, LOEO modal locus, uptake argmax, recovery outcome) identical to T-M1/T-M2.

## Hypotheses & bars (carried from T-M2)

| ID | Test | Bar |
|---|---|---|
| T-M3a | Society modal locus (LOEO) accuracy | ≥1.5× marginal base rate AND permutation p < 0.05 (500 perms) |
| T-M3b | Stress-uptake argmax → ΔV-locus concordance | Cohen's κ ≥ 0.20 |
| T-M3c | ΔV-locus → recovery | omnibus χ² p < 0.05 |

## Registered interpretations

- T-M3a PASS where T-M2a failed → the chronic-weakness confound was masking society signal; morphology-as-differential-diagnosis strengthened.
- T-M3a effect ≈ T-M2a effect → society locus signal was already damage-specific; confound hypothesis fails.
- T-M3b PASS where T-M2b failed → stress-uptake does drive damage (not level); uptake/damage coupling restored.
- Locus distribution shift away from Hands → T-M2's Hands dominance was a level artefact (Hands chronically low V), a substantive correction to the "basement" interpretation.
