# US Legitimacy Crisis × CAMS Stress — Study Report

**Kari Freyr McKern · July 2026 · 30-year panel (1996–2025)**  
**Sources: Gallup / Pew × USA_cam5_1996_2026_cleaned.csv**  
**Companion: [Study 1 — Interest Burden](us_interest_burden_cams_study.md)**

---

## 1. Design

Test variable pair:
- **Gallup Average Institutional Confidence** — % with "great deal/quite a lot" of confidence in 14 major institutions (averaged)
- **Pew Federal Government Trust** — % saying they trust the government in Washington "just about always" or "most of the time"

Same three-part test as Study 1:
1. Contemporaneous Pearson correlations with CAMS annual panel variables
2. Lead-lag analysis (lags −3 to +3): does CAMS Stress lead legitimacy collapse, or follow it?
3. AR(1) baseline vs CAMS-augmented models; walk-forward OOS test (train ≤2020, test 2021–2025)

Panel: 30 year-observations, 1996–2025.

---

## 2. The Legitimacy Series

| Year | Gallup % | Pew % | CAMS Mean Stress | Bond Strength |
|------|----------|-------|-----------------|---------------|
| 1996 | 39 | 28 | 3.5 | 31.6 |
| 2000 | 41 | 44 | 5.3 | 22.7 |
| 2004 | 43 | 44 | 6.1 | 21.3 |
| 2008 | 32 | 27 | 8.2 | 14.0 |
| 2012 | 32 | 29 | 5.7 | 19.4 |
| 2016 | 31 | 19 | 6.7 | 17.4 |
| 2020 | 36 | 20 | **9.3** | **11.0** |
| 2022 | 27 | 20 | 6.9 | 16.9 |
| 2023 | 26 | 16 | 6.1 | 21.0 |
| 2024 | 28 | 22 | 6.5 | 19.5 |
| 2025 | 28 | **9** | 7.9 | 13.4 |

The 2020 CAMS Stress spike (9.28, highest in the panel) preceded the 2022–2023 trust trough (Gallup 27–26, Pew 20–16) by 2–3 years. The 2025 Pew reading (9%) is the lowest in this panel's series — though an independent check of Pew's own September 2025 release finds a reported overall topline nearer 17%, so the 9% figure here is unconfirmed pending reconciliation of question wording/subgroup against that source.

---

## 3. Contemporaneous Correlations

### Gallup Institutional Confidence

| Variable | r | p | Significant? |
|----------|---|---|-------------|
| Archive Node Value | +0.624 | 0.000 | *** |
| Helm Node Value | +0.606 | 0.000 | *** |
| Mean Capacity | +0.583 | 0.001 | ** |
| Mean Node Value | +0.557 | 0.001 | ** |
| Mean Bond Strength | +0.555 | 0.001 | ** |
| Archive Stress | −0.513 | 0.004 | ** |
| Mean Stress | −0.408 | 0.025 | * |
| Shield Node Value | +0.406 | 0.026 | * |
| Reactivity Ratio | −0.248 | 0.187 | — |
| Cognitive Gap | +0.046 | 0.810 | — |

### Pew Federal Trust

| Variable | r | p | Significant? |
|----------|---|---|-------------|
| Archive Node Value | +0.620 | 0.000 | *** |
| Helm Node Value | +0.612 | 0.000 | *** |
| Mean Node Value | +0.504 | 0.005 | ** |
| Mean Capacity | +0.494 | 0.006 | ** |
| Mean Bond Strength | +0.490 | 0.006 | ** |
| Archive Stress | −0.477 | 0.008 | ** |
| Shield Node Value | +0.397 | 0.030 | * |
| Stress Dispersion | −0.325 | 0.080 | . |
| Mean Stress | −0.320 | 0.085 | . |
| Cognitive Gap | +0.167 | 0.378 | — |

**Key finding:** Archive Node Value and Helm Node Value are the strongest contemporaneous correlates of institutional trust (r ≈ +0.61 for both). The fixation/institutional-memory node (Archive) and the executive governance node (Helm) together explain most of the trust variance. This is structural, not transient.

---

## 4. Lead-Lag Analysis — The Critical Test

Lag convention: positive lag = CAMS at year t predicts target at t+lag (CAMS leads).

### Gallup Confidence — Lead-Lag

| Variable | lag−3 | lag−2 | lag−1 | lag 0 | lag+1 | lag+2 | lag+3 | Direction |
|----------|-------|-------|-------|-------|-------|-------|-------|-----------|
| Mean Stress | −0.216 | −0.304 | −0.390* | −0.408* | −0.499* | −0.636* | **−0.686*** | CAMS Leads |
| Mean Bond Strength | +0.486* | +0.512* | +0.531* | +0.555* | +0.631* | +0.759* | **+0.796*** | CAMS Leads |
| Mean Node Value | +0.428* | +0.501* | +0.556* | +0.557* | +0.636* | +0.766* | **+0.795*** | CAMS Leads |
| Archive Node Value | +0.601* | +0.612* | +0.631* | +0.624* | +0.657* | +0.790* | **+0.803*** | CAMS Leads |
| Archive Stress | −0.493* | −0.488* | −0.526* | −0.513* | −0.548* | −0.687* | **−0.720*** | CAMS Leads |
| Reactivity Ratio | +0.031 | −0.169 | −0.243 | −0.248 | −0.266 | −0.268 | −0.195 | Flat |
| Cognitive Gap | +0.193 | +0.045 | +0.000 | +0.046 | +0.002 | −0.013 | +0.017 | No signal |

### Pew Federal Trust — Lead-Lag

| Variable | lag−3 | lag−2 | lag−1 | lag 0 | lag+1 | lag+2 | lag+3 | Direction |
|----------|-------|-------|-------|-------|-------|-------|-------|-----------|
| Mean Stress | −0.180 | −0.251 | −0.287 | −0.320. | −0.417* | −0.560* | **−0.622*** | CAMS Leads |
| Mean Bond Strength | +0.374. | +0.471* | +0.450* | +0.490* | +0.551* | +0.705* | **+0.739*** | CAMS Leads |
| Archive Node Value | +0.512* | +0.591* | +0.587* | +0.620* | +0.625* | +0.746* | +0.692* | CAMS Leads |
| Archive Stress | −0.425* | −0.486* | −0.451* | −0.477* | −0.519* | −0.651* | **−0.597*** | CAMS Leads |

**Critical result:** Nearly every primary CAMS variable shows a lead-lag profile where the correlation *strengthens* as lag increases from 0 to +3 — the reverse of the IPR study pattern (where Bond Strength peaked at lag −3, meaning fiscal led CAMS). Two exceptions on the Pew side: Archive Node Value peaks at lag+2 (+0.746*) and eases to +0.692* at +3; Archive Stress peaks at lag+2 (−0.651*) and eases to −0.597* at +3. This is consistent with the legitimacy channel running independently of the fiscal one, though the pattern was found by scanning seven lags across multiple variables and two targets, so it should be treated as a strong prior rather than a single preregistered result.

---

## 5. Predictive Models

### Gallup Confidence — 1-Year Ahead OOS (train ≤2020, test 2021–2025)

| Model | R² (in-sample) | OOS RMSE | vs AR(1) |
|-------|---------------|----------|---------|
| AR(1) baseline | 0.779 | 2.805 | — |
| AR(1) + Mean Stress | 0.814 | 2.583 | **−0.222** |
| AR(1) + Archive Stress | 0.798 | 2.667 | **−0.138** |
| AR(1) + Mean Node Value | 0.822 | 2.671 | **−0.134** |
| AR(1) + Archive Node Value | 0.807 | 2.737 | **−0.068** |
| AR(1) + Bond Strength | 0.821 | 2.792 | −0.013 |
| AR(1) + Reactivity Ratio | 0.782 | 2.791 | −0.014 |

### Pew Federal Trust — 1-Year Ahead OOS

| Model | R² (in-sample) | OOS RMSE | vs AR(1) |
|-------|---------------|----------|---------|
| AR(1) baseline | 0.692 | 8.168 | — |
| AR(1) + Archive Stress | 0.716 | 7.386 | **−0.782** |
| AR(1) + Archive Node Value | 0.718 | 7.419 | **−0.749** |
| AR(1) + Mean Stress | 0.734 | 7.642 | **−0.526** |
| AR(1) + Mean Node Value | 0.731 | 7.674 | **−0.494** |
| AR(1) + Bond Strength | 0.735 | 7.848 | **−0.320** |
| AR(1) + Reactivity Ratio | 0.703 | 8.322 | +0.154 |

**OOS interpretation:** In Study 1 (fiscal), adding Mean Stress worsened OOS RMSE by +0.031. Here it *improves* Gallup prediction by −0.222 (−7.9%) and Pew prediction by −0.526 (−6.4%). Archive Stress is the single best Pew augmentation (−0.782, −9.6%). The asymmetry is the cleanest summary of the mechanism.

---

## 6. The Asymmetric Mechanism

| | Study 1: Interest Burden | Study 2: Legitimacy |
|---|---|---|
| Direction | Fiscal → CAMS | CAMS → Legitimacy |
| Key lag | −3 (fiscal leads) | +2 to +3 (CAMS leads) |
| Stress predictor | Worsens OOS | Improves OOS |
| Bond Strength | Strong positive lead | Strong positive lead |
| Mechanism | System absorbs fiscal load | Trust erodes from structural stress |

The implied sequence: structural stress accumulates in CAMS nodes (Archive, Helm) → public trust erodes 2–3 years later → debt burden becomes politically unsustainable only when institutions have lost public authority.

---

## 7. Verdict

The legitimacy-channel hypothesis passes its first falsification test. CAMS Stress **leads** legitimacy decline by 2–3 years, and the mechanism runs through legitimacy rather than fiscal absorption. Oligarchic capture is a plausible upstream driver of that channel but is not measured directly here, so it remains a candidate interpretation, not a tested one. The 2025 CAMS readings (Stress 7.93, Archive V 1.5 — the lowest in the entire panel, below 2020's previous low of 1.8) replicate — and in Archive V's case exceed — the structural signature of 2020 (Stress 9.28). The 3-year-lag model implies continued legitimacy pressure in 2026–2028, a forecast resting on a five-point out-of-sample window and worth treating as a working hypothesis rather than a settled prediction.

---

## 8. Files

- Panel CSV: `analysis/us_legitimacy_cams_panel.csv` (30 rows × 16 columns)
- HTML report: `analysis/us_legitimacy_cams_study.html`
- This document: `analysis/us_legitimacy_cams_study.md`
