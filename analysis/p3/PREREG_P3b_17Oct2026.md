# P3b frozen specification: CAMS aggregate vs external baselines (17 Oct 2026, before any P3b outcome was computed)

Inputs as in PREREG_P3_17Oct2026.md: FROZEN_CAMS.csv, INDEPENDENT_EVENTS.csv (Powell & Thyne coups V2026.08.29), plus vdem_core_v16.csv (SHA-256 prefix d4d4de4a8c7ddd7f).

Target: any coup attempt in (t, t+h]. h = 3 primary; h = 5 sensitivity.

## Features (all measured at year t, using information up to t only)
- B1 coup history: `coup10` = 1 if the register records a coup attempt in years t−9…t; `ysl` = years since the most recent coup attempt at or before t, capped at 30 (30 if none since the start of coverage). The register starts in 1950, so history is left-censored for early years; this is documented, not corrected.
- B2 V-Dem: `v2x_libdem(t)` and `v2x_libdem(t) − v2x_libdem(t−1)`. The electoral democracy index (`v2x_polyarchy`) and GDP per capita are **not in vdem_core_v16.csv**, so they are omitted (no optional GDP extension).
- B3 = B1 + B2.
- C = the 4 CAMS dimension means.
- C+B3 = C + B1 + B2.

## Common sample
Society-years in the P3 primary set (horizon-eligible, complete eight-node) that also have non-missing V-Dem libdem at t and t−1. No imputation. All models are scored on this identical sample; C is re-reported on it.

## Protocol
Leave-one-society-out; the same pipeline as analysis/p3_compare.py `fit_predict` (median imputer, standard scaler, L2 logistic C = 0.1), with preprocessing fitted on training societies only. Pooled out-of-fold AUC and Brier. Society-cluster bootstrap 95% CI for paired differences, 2000 replicates, seed 20261017.

Comparisons: C vs B3; **C+B3 vs B3 (key)**; C vs B1.

## Calendar-wave robustness
Repeat all models with decade indicator variables added to every model's features.

## Frozen success criterion
C+B3 beats B3 at h = 3 (primary, no decade control) if the ΔAUC 95% CI excludes 0 (positive) OR ΔBrier < 0 with its 95% CI excluding 0. Sensitivity and robustness runs are reported but do not change the verdict.
