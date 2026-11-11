# Verification of T-M series script v2 — 16 October 2026

Script `tm_series_analysis.py` (v2) was run unmodified in a fresh Python virtual environment (pandas, numpy, scipy, scikit-learn, statsmodels) on 16 October 2026. It completed without error in about 45 seconds. All eight input CSVs matched the SHA-256 prefixes in `REPRODUCIBILITY.md`. The regenerated intermediates matched the archived ones (identical shapes, columns and text values; numeric differences ≤ 3e-15; `T-M4_per_set_kappa.csv` byte-identical). `episodes.csv` was not modified by the run.

## Input data not duplicated here

The eight input CSVs are identical (by SHA-256) to files already in this repository at commit `446928c`, so they are not copied:
`juno/JUNO_Unified_Dataset.csv`, `juno/vdem_core_v16.csv`, `blind-test/JUNO_Blind_Set_{1,2,3,5,6,7}.csv`. Copy them into this directory before running the script. The nested `T-M_series.zip` from the archive (which bundles these same files) is also omitted.

## Rerun versus reported

| Figure | Reported (v2 log) | Rerun |
|---|---|---|
| Episodes extracted / censored / analysis | 201/48 → 158/42 → 149/33 | Same |
| T-M1 morphology AUC (society-peak, canonical) | 0.492 | 0.492 |
| T-M1 V-Dem / baseline AUC | 0.547 / 0.553 | Same |
| T-M1b full / rivals only | 0.548 / 0.555 | Same |
| T-M1c within / between / gap / p | 0.849 / 0.823 / 0.0267 / 0.004 | Same |
| T-M1 trough-aligned variant | 0.497 / 0.024 | Not computed by v2 (matches the earlier v1 run) |
| T-M2a accuracy / marginal / ratio / null / p | 0.510 / 0.409 / 1.25× / 0.287 / 0.002 | Same |
| T-M2 Hands count | 61 | 61 |
| T-M2b κ | 0.154 | 0.154 |
| T-M2b tie-break variant κ | 0.162 | Not computed by v2 |
| T-M3 relocated share | 62.4% | 0.624 |
| T-M3a ratio / p | 0.94× / 0.034 | Same |
| T-M3b κ | 0.476 | 0.476 |
| T-M3c χ² p | 0.19 | 0.1931 |
| T-M3d, 149 set (primary): κ / κ_R / null 95th / p / floor | 0.499 / 0.249 / 0.073 / 0.001 / 0.252 | Same |
| T-M3d, 158 set (sensitivity): κ / κ_R / floor | 0.507 / 0.265 / 0.254 | Same |
| T-M3d-2 node-FE panel n / coefficients | 40,944 / −0.138 | 40,944 / −0.1382, −0.1381 |
| Placebo-column panel | 40,608 / −0.138 | Not printed by v2 |
| T-M3e hardened panel n | 40,472 | 40,472 |
| T-M3e forward C / K | −0.0938 / −0.0927 | Same |
| T-M3e reverse C / K | −0.0519 / −0.0413 | Same |
| T-M4 pooled n / κ / concordance / p | 138 / 0.267 / 0.384 / 0.001 | Same |
| T-M4 per-set κ (1,2,3,5,6,7) | 0.135, 0.157, 0.432, 0.216, 0.481, 0.150 | Same |

## Open items for Kimi

1. **`low_memory` explanation for T-M2b withdrawn.** Loading without `low_memory=False` changes only the types of `Decay_Index_Calc`, `Phase_Calc` and `ESCH_Calc`, none of which the analysis uses. A full rerun without it still gives κ = 0.154 and leaves every other figure unchanged. The original 0.184 remains unexplained.
2. **Three variants missing from the script:** the trough-aligned T-M1 variant (0.497 / 0.024), the T-M2b tie-break variant (0.162) and the 40,608 placebo-column panel. Either add them to the script output or stop citing them.
3. **Stale first correction log** in `REPRODUCIBILITY.md` still states 0.470 vs 0.389 (1.21×), null 0.269 and 201/42, contradicting the v2 log. Remove it or mark it superseded.
4. **Duplicate entries:** v2 correction items 7 and 8 record the same society-count correction.
5. **Runtime:** `REPRODUCIBILITY.md` says about 10 minutes; the script runs in about 45 seconds.
