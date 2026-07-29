# US Interest Burden × CAMS Stress — Study Report

**Kari Freyr McKern · July 2026 · 30-year panel (1996–2025)**
**Sources: FRED (OMB FYOINT / FYFR) × USA_cam5_1996_2026_cleaned.csv**

---

## 1. Design

Test variable: **Interest payments as % of federal revenue (IPR)** = FYOINT / FYFR × 100.

Three-part test:
1. Contemporaneous Pearson correlations between CAMS annual panel variables and IPR
2. Lead-lag analysis (lags −3 to +3): does CAMS Stress lead the fiscal cycle or follow it?
3. AR(1) baseline vs CAMS-augmented models; walk-forward OOS test (train ≤2020, test 2021–2025)

Panel: 30 year-observations, 1996–2025. CAMS data from five-scorer ensemble (USA_cam5_1996_2026_cleaned.csv). Fiscal data from US OMB via FRED (last updated April 2026). CAMS 2026 exists but FRED data ends at FY2025, so the panel is 30 rows.

---

## 2. The IPR Series

IPR peaked at 16.6% in 1996, fell to a trough near 6.9% (2015), then re-accelerated:

| Year | IPR (%) | CAMS Mean Stress | CAMS Bond Strength |
|------|---------|-----------------|-------------------|
| 1996 | 16.6 | 3.5 | 31.6 |
| 2000 | 11.0 | 5.3 | 22.7 |
| 2008 | 10.0 | 8.2 | 14.0 |
| 2015 | 6.9 | 5.9 | 20.7 |
| 2020 | 10.1 | 9.3 | 11.0 |
| 2023 | 14.8 | 6.1 | 21.0 |
| 2025 | 18.5 | 7.9 | 13.4 |

The 2025 IPR (18.5%) now exceeds the 1996 opening level. The 2022–2025 rise is the steepest three-year acceleration in the panel.

---

## 3. Contemporaneous Correlations

| Variable | r | p | Significant? |
|----------|---|---|-------------|
| Shield Node Value | +0.380 | 0.038 | * |
| Mean Bond Strength | +0.372 | 0.043 | * |
| Hands Node Value | +0.317 | 0.087 | — |
| Mean Capacity | +0.308 | 0.098 | — |
| Mean Node Value | +0.277 | 0.139 | — |
| Stewards Node Value | +0.280 | 0.134 | — |
| Mean Stress | −0.303 | 0.104 | — |
| Archive Stress | −0.178 | 0.346 | — |
| Archive Node Value | +0.128 | 0.502 | — |
| Stress Dispersion | −0.111 | 0.559 | — |
| Reactivity Ratio | −0.026 | 0.891 | — |

**Key finding:** Mean Stress correlates *negatively* with IPR (r = −0.303). Higher CAMS stress goes with lower interest burden, not higher. This is the opposite of the naive hypothesis. The explanation: the 1996–2000 period had high IPR and low CAMS stress (Clinton surplus era); the 2008–2010 period had moderate IPR and peak CAMS stress. Only in 2022–2025 are both high simultaneously.

Bond Strength and Shield Node Value are the only statistically significant correlates — both positive. Institutional coupling and military-security dominance track the fiscal pre-commitment cycle more than stress does.

---

## 4. Lead-Lag Analysis

Lag convention: positive lag = CAMS leads IPR (CAMS year t predicts IPR at t+lag). Negative lag = fiscal leads CAMS.

| Variable | lag −3 | lag −2 | lag −1 | lag 0 | lag +1 | lag +2 | lag +3 |
|----------|--------|--------|--------|-------|--------|--------|--------|
| Mean Stress | −0.503* | −0.346 | −0.326 | −0.303 | −0.338 | −0.144 | +0.190 |
| Mean Bond Strength | +0.705* | +0.475* | +0.422* | +0.372* | +0.379* | +0.181 | −0.118 |
| Mean Node Value | +0.625* | +0.397* | +0.321 | +0.277 | +0.311 | +0.117 | −0.207 |
| Archive Node Value | +0.693* | +0.332 | +0.192 | +0.128 | +0.197 | +0.099 | −0.176 |
| Stewards Node Value | +0.330 | +0.290 | +0.284 | +0.280 | +0.282 | +0.115 | −0.129 |

**The direction of causality runs from fiscal to CAMS, not the reverse.** The strongest correlations for Bond Strength, Node Value, and Archive are all at lag −3 (fiscal leads by 3 years). High IPR periods are followed 3 years later by higher CAMS structural coherence. CAMS stress variables do not lead the interest-burden cycle.

---

## 5. Predictive Models

### In-sample (1996–2025)

| Model | R² | ΔR² | RMSE | β(CAMS) |
|-------|-----|-----|------|---------|
| AR(1) baseline | 0.764 | — | 1.431 | — |
| AR(1) + Mean_BondStrength | 0.783 | +0.020 | 1.370 | −0.098 |
| AR(1) + Archive_V | 0.779 | +0.015 | 1.385 | −0.122 |
| AR(1) + Mean_Stress | 0.774 | +0.010 | 1.400 | +0.267 |
| AR(1) + Stewards_V | 0.767 | +0.003 | 1.421 | −0.053 |
| AR(1) + Reactivity_Ratio | 0.764 | 0.000 | 1.430 | +0.359 |

### Out-of-sample walk-forward (train 1996–2020, test 2021–2025)

| Model | OOS RMSE | vs AR(1) |
|-------|----------|---------|
| AR(1) baseline | 3.416 | — |
| AR(1) + Archive_V | 3.347 | **−0.069** |
| AR(1) + Mean_BondStrength | 3.385 | **−0.030** |
| AR(1) + Mean_Stress | 3.447 | +0.031 |
| AR(1) + Stewards_V | 3.525 | +0.110 |

Archive_V and Bond Strength provide modest OOS improvement. Stress variables worsen OOS performance.

---

## 6. Interpretation

CAMS does not lead the interest-burden cycle. The lead-lag structure is reversed from what the oligarchic-capture hypothesis would predict if it operated through a simple CAMS Stress → fiscal pre-commitment channel.

The finding is structurally interesting rather than a null result. It suggests:

1. **CAMS measures institutional capacity that enables debt to be sustained**, not the point at which it generates visible stress. High fiscal burden (1996, 2023–2025) coexists with or follows periods of CAMS structural coherence — the system is coherent enough to keep carrying the load, not yet broken by it.

2. **The stress → fiscal mechanism may operate at a longer lag than 3 years**, or may manifest in legitimacy/conflict indicators rather than in fiscal variables. The IPR series is partly mechanical (interest rate × debt stock); CAMS may be measuring the political economy layer that determines how long the burden can be sustained before triggering legitimacy crises.

3. **Shield node significance is theoretically important**: the positive correlation of Shield Node Value with IPR (r = +0.38*) is consistent with the military-financier coalition argument. Periods of high defence-security institutional dominance track periods of high interest burden. This is a structural coupling, not a causal claim from this test alone.

---

## 7. Jobs Worth Doing

- **Debt-to-GDP test**: monotonic rise since 2001 may show cleaner correlation with CAMS stress than IPR does (run against GFDEGDQ188S)
- **Stewards × Archive interaction term**: test divergence between governance capacity and legal-administrative encoding
- **Regime-conditional analysis**: split panel by JUNO regime and run separate correlations
- **Cross-national control**: run same IPR × CAMS test for Germany and Australia — if CAMS leads fiscal in those cases, the US lag pattern is structurally distinctive
- **Legitimacy/conflict indicators next**: Gallup institutional trust, electoral volatility, Polity scores — these are the predicted downstream effects where CAMS Stress should lead

---

## 8. Files

- Panel CSV: `analysis/us_interest_burden_cams_panel.csv` (30 rows × 18 columns)
- HTML report: `analysis/us_interest_burden_cams_study.html`
- This document: `analysis/us_interest_burden_cams_study.md`
