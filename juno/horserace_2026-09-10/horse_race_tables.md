# Horse-race tables — Epiphenomenon (10 Sep 2026 Sydney)

Primary sample: **n=17**, years **2009–2025** (`horse_race_data.csv`).

Predictors standardized (z) in OLS for coefficient comparability.

Language: exploratory / comparative fit — **not** validated causal claims.


## 1. Univariate Spearman / Pearson vs sino_ppm

| Predictor | n | Spearman r | p | Pearson r | p |
|---|---:|---:|---:|---:|---:|
| tau_L2 | 17 | -0.211 | 0.4167 | -0.219 | 0.3992 |
| BS_L2 | 17 | -0.255 | 0.3235 | -0.289 | 0.2613 |
| Hands_L2 | 17 | -0.625 | 0.0073 | -0.404 | 0.1078 |
| tau_L3 | 17 | -0.419 | 0.0940 | -0.273 | 0.2888 |
| BS_L3 | 17 | -0.441 | 0.0763 | -0.298 | 0.2457 |
| Hands_L3 | 17 | -0.601 | 0.0107 | -0.303 | 0.2369 |
| china_export_share_pct | 16 | 0.288 | 0.2790 | 0.401 | 0.1233 |
| prc_actions_count | 17 | 0.633 | 0.0064 | 0.580 | 0.0146 |
| prc_sanctions_active | 17 | 0.708 | 0.0015 | 0.785 | 0.0002 |
| us_alliance_pressure | 17 | 0.195 | 0.4528 | 0.218 | 0.3998 |
| election_year | 17 | 0.000 | 1.0000 | -0.200 | 0.4404 |
| labor_gov | 17 | -0.144 | 0.5805 | -0.278 | 0.2792 |
| security_announce_any | 17 | 0.566 | 0.0178 | 0.623 | 0.0076 |

**Best external by |Spearman|:** `prc_sanctions_active` (r=0.708). **Second:** `prc_actions_count` (r=0.633).


## 2. Univariate z-OLS (sino_ppm ~ z(predictor))

| Predictor | n | coef (z) | SE | p | R² | adj-R² |
|---|---:|---:|---:|---:|---:|---:|
| tau_L2 | 17 | -2.435 | 2.806 | 0.3992 | 0.048 | -0.016 |
| BS_L2 | 17 | -3.215 | 2.754 | 0.2613 | 0.083 | 0.022 |
| Hands_L2 | 17 | -4.500 | 2.631 | 0.1078 | 0.163 | 0.107 |
| prc_sanctions_active | 17 | 8.748 | 1.780 | 0.0002 | 0.617 | 0.591 |
| prc_actions_count | 17 | 6.464 | 2.342 | 0.0146 | 0.337 | 0.293 |
| china_export_share_pct | 16 | 4.596 | 2.803 | 0.1233 | 0.161 | 0.101 |
| us_alliance_pressure | 17 | 2.432 | 2.807 | 0.3998 | 0.048 | -0.016 |
| labor_gov | 17 | -3.101 | 2.762 | 0.2792 | 0.078 | 0.016 |
| security_announce_any | 17 | 6.936 | 2.250 | 0.0076 | 0.388 | 0.347 |

## 3. Nested / sparse multivariate z-OLS


### `externals_top2` — n=17, R²=0.697, adj-R²=0.654, F=16.118, F-p=0.0002

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| prc_sanctions_active | 7.330 | 1.796 | 0.0011 | 1.20 |
| prc_actions_count | 3.462 | 1.796 | 0.0744 | 1.20 |

### `cams_L2_only` — n=17, R²=0.351, adj-R²=0.202, F=2.347, F-p=0.1202

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | 30.488 | 18.483 | 0.1230 | 55.18 |
| BS_L2 | -26.985 | 21.527 | 0.2321 | 74.86 |
| Hands_L2 | -6.909 | 7.039 | 0.3443 | 8.01 |

### `cams_tau_BS_L2` — n=17, R²=0.303, adj-R²=0.204, F=3.047, F-p=0.0797

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | 36.566 | 17.391 | 0.0541 | 48.99 |
| BS_L2 | -39.406 | 17.391 | 0.0398 | 48.99 |

### `nested_tau_L2_plus_best_ext` — n=17, R²=0.618, adj-R²=0.563, F=11.305, F-p=0.0012

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | 0.323 | 1.937 | 0.8699 | 1.11 |
| prc_sanctions_active | 8.849 | 1.937 | 0.0004 | 1.11 |

### `nested_BS_L2_plus_best_ext` — n=17, R²=0.617, adj-R²=0.562, F=11.278, F-p=0.0012

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| BS_L2 | 0.171 | 1.995 | 0.9328 | 1.17 |
| prc_sanctions_active | 8.814 | 1.995 | 0.0006 | 1.17 |

### `nested_Hands_L2_plus_best_ext` — n=17, R²=0.619, adj-R²=0.565, F=11.383, F-p=0.0012

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| Hands_L2 | -0.613 | 2.068 | 0.7712 | 1.27 |
| prc_sanctions_active | 8.466 | 2.068 | 0.0011 | 1.27 |

### `nested_tau_L2_plus_top2_ext` — n=17, R²=0.700, adj-R²=0.631, F=10.135, F-p=0.0010

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | -0.703 | 1.860 | 0.7114 | 1.21 |
| prc_sanctions_active | 7.023 | 2.023 | 0.0041 | 1.43 |
| prc_actions_count | 3.675 | 1.937 | 0.0803 | 1.31 |

### `full_cams_L2_plus_best_ext` — n=17, R²=0.646, adj-R²=0.528, F=5.472, F-p=0.0096

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | 6.804 | 16.069 | 0.6795 | 70.53 |
| BS_L2 | -3.021 | 18.210 | 0.8710 | 90.58 |
| Hands_L2 | -4.261 | 5.478 | 0.4517 | 8.20 |
| prc_sanctions_active | 7.752 | 2.454 | 0.0082 | 1.64 |

### `sparse_tau_prc_trade` — n=16, R²=0.695, adj-R²=0.619, F=9.118, F-p=0.0020

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | 1.027 | 1.950 | 0.6080 | 1.14 |
| prc_sanctions_active | 8.970 | 2.010 | 0.0008 | 1.21 |
| china_export_share_pct | 2.354 | 1.891 | 0.2370 | 1.07 |

### `sparse_tau_us_trade` — n=16, R²=0.227, adj-R²=0.034, F=1.176, F-p=0.3596

| Predictor | coef (z) | SE | p | VIF |
|---|---:|---:|---:|---:|
| tau_L2 | -0.874 | 3.219 | 0.7906 | 1.23 |
| us_alliance_pressure | 2.472 | 3.201 | 0.4549 | 1.21 |
| china_export_share_pct | 4.551 | 2.929 | 0.1462 | 1.02 |

## 4. Horse-race summary (who wins on adj-R²)

- Univariate winner (among CAMS L2 + top2 externals): **`univ_z_prc_sanctions_active`** (adj-R²=0.591).

- CAMS L2 block (`tau_L2+BS_L2+Hands_L2`) adj-R²=0.202 (note multicollinearity; VIFs high).

- Externals top2 (`prc_sanctions_active`+`prc_actions_count`) adj-R²=0.654.

- Nested `tau_L2` + best external adj-R²=0.563.

## 5. Sensitivity — `hansard_master` τ/BS scale (different from `horse_race_data`)

`hansard_master.csv` Bond Strength is on a ~13–23 scale; `horse_race_data.csv` BS is ~1.8–2.6 (Hands-aligned). Grok Report v3 lag claims (r≈−0.74) track the **hansard_master** scale.

| Predictor | Spearman r | p | z-OLS coef | p | R² | adj-R² |
|---|---:|---:|---:|---:|---:|---:|
| tau_L2_hm | −0.707 | 0.0015 | −5.585 | 0.040 | 0.251 | 0.202 |
| BS_L2_hm | −0.698 | 0.0018 | −5.477 | 0.045 | 0.242 | 0.191 |

Nested vs best external (`prc_sanctions_active`):

| Model | adj-R² | CAMS z-coef (p) | sanctions z-coef (p) |
|---|---:|---|---|
| tau_L2_hm + sanctions | 0.575 | −1.387 (0.526) | +8.022 (0.002) |
| BS_L2_hm + sanctions | 0.580 | −1.579 (0.458) | +7.976 (0.002) |

**Reading:** On the Report-aligned scale, lagged τ/BS are strong **univariate** associates of `sino_ppm`, but lose independent significance once the contemporaneous PRC sanctions regime dummy enters — sanctions retain p≈0.002.

## 6. Lagged-externals robustness (honesty check)

Primary race compares **CAMS L2 (leading)** to **contemporaneous** externals.

| Lagged external | n | Spearman vs sino_ppm | p |
|---|---:|---:|---:|
| prc_sanctions_L2 | 15 | +0.349 | 0.202 |
| prc_actions_L2 | 15 | +0.693 | 0.004 |
| us_alliance_L2 | 15 | +0.094 | 0.738 |
| china_export_share_L2 | 15 | +0.307 | 0.266 |

Sanctions as a **same-year regime** dominate; as an L2 lead they weaken. Discrete PRC actions retain a strong L2 Spearman (+0.693).
