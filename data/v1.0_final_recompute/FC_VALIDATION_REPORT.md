# CAMS v1.0-Final Falsification-Criteria Validation Report

**Date:** 2026-07-29 20:02
**Primary series:** 34
**Society-years (in-range):** 3,860
**Node-observations:** 30,880

## Summary

| Status | Count |
|--------|------:|
| PASS | 3 |
| PARTIAL FAIL | 1 |
| FAIL | 2 |
| SKIP | 0 |

## FC1 — Stress-Capacity Anti-Correlation

**Status:** PASS

| Metric | Value |
|--------|------:|
| median_rho_per_year | -0.5549 |
| mean_rho_per_year | -0.4668 |
| pct_negative | 82.8500 |
| ttest_vs_minus0.5_t | nan |
| ttest_vs_minus0.5_p | nan |
| pooled_rho | -0.4070 |
| pooled_p | 0.0000 |
| n_society_years | 3860 |
| threshold | median ρ < -0.5 |

## FC2 — Three-Tier Crisis Composite (proxy)

**Status:** PASS
**Note:** JUNO_v2.1 secondary tier omitted; using ΔV̄<-12% proxy

| Metric | Value |
|--------|------:|
| primary_Vmin_lt4_hit_pct | 95.4800 |
| primary_Vmin_lt4_fp_pct | 0.0000 |
| secondary_dVmean_lt12_hit_pct | 27.6800 |
| secondary_dVmean_lt12_fp_pct | 1.5500 |
| acute_sigma_rate_gt0.5_hit_pct | 20.8800 |
| acute_sigma_rate_gt0.5_fp_pct | 13.6900 |
| composite_any_tier_hit_pct | 98.4300 |
| composite_any_tier_fp_pct | 14.2600 |
| n_crisis_years | 1528 |
| n_stable_years | 1227 |

## FC3 — Common-Latent-Factor (PCA)

**Status:** FAIL

| Metric | Value |
|--------|------:|
| pc1_variance_explained | 0.8328 |
| pc2_variance_explained | 0.0466 |
| pc3_variance_explained | 0.0430 |
| kaiser_effective_dim | 1 |
| pc1_all_positive | False |
| pc1_min_loading | -0.3688 |
| n_obs | 3860 |

## FC4 — Bond-Health Coupling

**Status:** PASS

| Metric | Value |
|--------|------:|
| pooled_per_node_r | 0.9291 |
| pooled_per_node_p | 0.0000 |
| system_level_r | 0.9564 |
| system_level_p | 0.0000 |
| median_per_society_r | 0.9767 |
| n_societies_tested | 34 |
| threshold | ρ > 0.50 |

## FC5 — Shield Ranking

**Status:** FAIL

| Metric | Value |
|--------|------:|
| median_rank_in_stable_years | 6.0000 |
| pct_top_half | 30.9700 |
| shield_vs_system_B_r | 0.9305 |
| shield_vs_system_B_p | 0.0000 |
| n_stable_years_tested | 1227 |

## FC6 — λ₂ Degradation Prior to Crisis

**Status:** PARTIAL FAIL
**Note:** Threshold: >33% of crises show pre-decline (v3.2-R achieved 1/3)

| Metric | Value |
|--------|------:|
| pct_declining_3yr_pre_crisis | 0.0000 |
| n_crisis_onsets_tested | 144 |
| median_mk_tau | 0.0000 |

---

## Interpretation Notes

- **FC1**: The Stress-Capacity anti-correlation is the bedrock thermodynamic signature. A median ρ < −0.5 across society-years is required for the framework to claim internal consistency.
- **FC2**: JUNO_v2.1 character classification is not available in the raw recomputed series; the secondary tier uses ΔV̄ < −12% as a proxy. For full validation, run JUNO v1.2 regime classifier on the same corpus.
- **FC3**: PCA dimensionality should be ≤3 for the 8-node architecture to be justified against a 1-factor null model (per Memory 45).
- **FC4**: Bond-Health coupling > 0.50 is the minimum threshold; v3.2-R achieved >0.60.
- **FC5**: Shield ranking tests whether the security node maintains structural prominence during stable periods. Collapse to bottom-half rank indicates Praetorian inversion.
- **FC6**: λ₂ degradation is the weakest empirical signal (v3.2-R: 1/3 crises). It is retained as a falsification criterion but downgraded to secondary status.

**Next step:** If any criterion FAILs, inspect the underlying data for (a) scoring drift, (b) node-mapping errors, or (c) formula implementation bugs.
