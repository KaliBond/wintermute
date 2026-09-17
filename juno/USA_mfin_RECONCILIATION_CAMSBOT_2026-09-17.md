# USA MFIN reconciliation — CAMSBOT verify (2026-09-17)

**Spec:** `CAMNATIONSMFIN_methodology.txt` FINAL aggregation  
**Files:** `USA_CAMS_mfin_2015-2026_block{1,2}.csv`  
**Tolerance:** ±0.175 on Node Value (1-dp means)

## Rules tested

1. `V = C_mean + K_mean − S_mean + 0.5·A_mean`
2. Block1 `Node Value` inside Block2 `[V_min, V_max]`
3. Society Bond Strength = mean over C(8,2)=28 pairs of  
   `B_ij = [0.6·C_i·C_j + 0.4·A_i·A_j] · exp(−(S_i+S_j)/20)`

## Results (independently recomputed)

| Rule | Result |
|------|--------|
| 1 Node Value | **60/96** cells violate beyond ±0.175 |
| 2 Envelope containment | **12/96** outside `[V_min, V_max]` (all slightly above V_max, ≤+0.4) |
| 3 Bond Strength | **12/12** years fail; delivered/spec ratio **0.317–0.437** |

Worst V deltas (delivered − spec): Archive 2026 **+3.0** (4.5 vs 1.5); Lore late years **+2.0**; Helm 2026 **+1.6**. Hands carries **+1.0** in most years. Mean delivered−spec gap grows with Stress (~0 early → **+1.27** by 2026).

Bond: delivered 2015 **9.051** vs spec **~28.5** (ratio 0.317); 2026 **5.341** vs spec **~12.22** (ratio 0.437). Same shape, different scale and weaker stress damping than `/20`.

## Trajectory robustness

Rank order and phase story are **stable under FINAL V**: 2026 weak→strong still Archive → Helm → Lore → Flow → Hands → Craft → Stewards → Shield. Stewards & Shield 2026 match spec exactly. Absolute V/BS levels are **not** FINAL-compliant — do not mix with future FINAL-spec runs without re-aggregation from raw scorer CSVs.

## Companion CSVs

- `reconciliation_nodevalue_all.csv`
- `reconciliation_nodevalue_violations.csv`
- `reconciliation_envelope_violations.csv`
- `reconciliation_bond.csv`
