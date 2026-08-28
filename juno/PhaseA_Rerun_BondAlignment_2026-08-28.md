# Phase A re-run after Bond_Strength_Calc node alignment

**Date:** 2026-08-28 (Australia/Sydney)  
**Repo commit this re-run used:** `9103057` (PR #1 merged — aligned `Bond_Strength_Calc`, legacy column retained)  
**Original report:** [`PhaseA_Validation_Report_2026-08-16.md`](PhaseA_Validation_Report_2026-08-16.md)  
**Script:** [`rerun_phase_a_bond_alignment.py`](rerun_phase_a_bond_alignment.py)  
**Numbers:** [`phase_a_bond_rerun_results_2026-08-28.csv`](phase_a_bond_rerun_results_2026-08-28.csv)

This addendum does **not** rewrite the 2026-08-16 conclusions. It records what a node-labelled re-run actually produces on the post-fix panel versus the published figures and versus `Bond_Strength_Calc_legacy`.

---

## 1. What was re-run

Phase A tests that **slice `Bond_Strength_Calc` by Node label**:

| Test | Slices Bond by Node? | Re-run? |
|---|---|---|
| T1 Shield × SIPRI / World Bank milex % GDP | Yes (Shield) | Yes, aligned **and** legacy |
| T3a Australia Shield × V-Dem | Yes (Shield) | Yes |
| T3c cross-national Shield × V-Dem | Yes (Shield) | Yes |
| T3c per-society Shield × `v2x_libdem` | Yes (Shield) | Yes |
| T2 Hands rank / H–S bond | No (Node_Value rank; pairwise \(B_{ij}\) from C/K/S/A) | Sanity-check only |
| T3b λ₂ by CAMS Phase | No (`Lambda2_Calc` is year-level) | Sanity-check only |
| T3c `v2x_libdem` ANOVA by Phase; λ₂ × libdem | No | Sanity-check only |
| T4 Maddison × Phase transitions | No | Not re-run (unaffected) |
| T5 France Archive; Denmark HSC triangle | No (Node_Value / raw pairwise) | Sanity-check (NV / ranks) |

Panel restriction: **original JUNO_36** (the 12 societies added 2026-08-20 are excluded). Current 36-society slice: 32,264 node-rows, 4,033 society-years. Phase A vintage was 32,361 rows / 4,048 society-years.

No Phase A pipeline script was in the repo (commit `6a84a2a` added only the report, the results CSV, and `vdem_core_v16.csv`). Tests were reconstructed from the published protocol.

---

## 2. Alignment fact that changes the interpretation

The 2026-08-28 fix note said node-labelled Phase A work (especially Shield × SIPRI) had to be re-run because `Bond_Strength_Calc` had been misaligned to Node.

Direct comparison of the **Phase A vintage file** (`juno/JUNO_Unified_Dataset.csv` at `4289140`, 36 societies, the file Phase A actually read) against the current columns:

| Comparison (32,264 overlapping node-rows) | Result |
|---|---|
| Phase A vintage vs current **aligned** `Bond_Strength_Calc` | max \|diff\| = 5.00×10⁻⁵ (rounding); **0%** of cells differ by >10⁻⁴ |
| Phase A vintage vs current **legacy** | **88.8%** of cells (92.2% of Shield rows) differ by >10⁻⁴ |

So:

- Published Phase A node-labelled statistics were computed on a panel that was **already node-aligned**.
- Current `Bond_Strength_Calc` restores that Phase A column.
- `Bond_Strength_Calc_legacy` is the scramble introduced on **2026-08-20** when 12 countries were integrated (`a95f5c8`), not the column Phase A published. It is a counterfactual of the 8-day misaligned window, not a reconstruction of 2026-08-16.

---

## 3. T1 — Shield × military expenditure % GDP

### 3a. World Bank proxy (`MS.MIL.XPND.GD.ZS`, 1960–2024)

Hong Kong has no WB series. N and societies match the original report exactly.

| Statistic | Published 2026-08-16 | Aligned (current) | Legacy (Aug 20 scramble) |
|---|---|---|---|
| Pearson r (cross-section) | **−0.1279** | **−0.1279** | −0.1647 |
| p | 1.3×10⁻⁷ | 1.30×10⁻⁷ | 9.20×10⁻¹² |
| Within-society demeaned r | **+0.0383** | **+0.0383** | −0.0185 |
| p (within) | 0.115 (n.s.) | 0.1148 (n.s.) | 0.4459 (n.s.) |
| N | 1,693 | 1,693 | 1,693 |
| Societies | 35 | 35 | 35 |

**Verdict vs published:** Aligned **reproduces the published WB result to the reported precision**. The published RESPECIFY (null within-society, negative cross-section driven by composition) **holds**. Legacy would have made the cross-section more negative and left within-society null; that is not what was published.

### 3b. SIPRI Share of GDP (official Excel, 1949–2025)

The Drive file named in the original report (`sipri_milex_1949_2025.xlsx`) and the processed `_shield_sipri_merged.csv` are **not in the repo**. This re-run used the public SIPRI Military Expenditure Database Excel currently posted at sipri.org (`SIPRI-Milex-data-1949-2025_v1.2.xlsx`, Share of GDP sheet; same numbers via OWID). Inner join of Shield society-years to SIPRI (Hong Kong unmatched):

| Statistic | Published 2026-08-16 (Drive SIPRI) | Aligned, public SIPRI v1.2 | Legacy, public SIPRI v1.2 |
|---|---|---|---|
| Pearson r (cross-section) | **+0.251** | **−0.1153** | −0.1552 |
| p | 4.3×10⁻³⁵ | 7.90×10⁻⁷ | 2.69×10⁻¹¹ |
| Within-society demeaned r | **+0.121** | **+0.0546** | +0.0067 |
| p (within) | 3.9×10⁻⁹ | 0.0197 | 0.775 (n.s.) |
| N | 2,360 | 1,823 | 1,823 |
| Societies | 33 | 35 | 35 |
| Years | 1949–2025 | 1949–2025 | 1949–2025 |

OWID’s SIPRI milex/GDP packaging gives the same n, r, and p as the official Excel Share of GDP sheet.

Constant-(2024)-US$ milex (not the published % GDP claim) inner-joins to **n = 1,725 / 33 societies**, aligned r = +0.022 (n.s.) — still not the published +0.251 / n = 2,360.

Linear interpolation / forward-fill of sparse JUNO series (USA, India, Turkey, Canada, Netherlands, Thailand are 5-year) raises n toward ~2,196–2,213 but **does not flip the Share-of-GDP sign** (aligned ffill r ≈ −0.115).

**Verdict vs published:** The published SIPRI-real revision (**r = +0.251 cross / +0.121 within, CONDITIONAL PASS**) is **not reproduced** from in-repo JUNO + public SIPRI v1.2. N, society count, and **sign** all differ. This discrepancy is **not caused by the bond-alignment fix**: Phase A already used aligned Shield bonds (WB matches; vintage column matches). Likely causes: the missing Drive extract, a different SIPRI vintage/sheet, or a merge that is not an exact society-year inner join. Until that extract is recovered, **do not treat the published SIPRI-real figure as confirmed**. The WB proxy result (reproduced) remains the only T1 number that can be checked.

---

## 4. T3 — V-Dem × Shield

External file is in-repo: `juno/vdem_core_v16.csv` (the Phase A extract). Cross-national tables below use the published complete-case panel: rows with non-missing `v2x_libdem`, **N = 3,806**, all 36 societies, 1790–2025.

### 4a. Cross-section and within-society (Shield `Bond_Strength_Calc`)

| Indicator | Published r (cross / within) | Aligned r (cross / within) | Legacy r (cross / within) |
|---|---|---|---|
| `v2x_rule` | **+0.417** / **+0.127** (n=3,806) | **+0.4167** / **+0.1272** (n=3,806; p=8.41×10⁻¹⁶⁰ / 3.36×10⁻¹⁵) | +0.4531 / +0.1486 |
| `v2x_neopat` | **−0.397** / **−0.059** (n=3,706) | **−0.3972** / **−0.0594** (n=3,706; p=2.79×10⁻¹⁴⁰ / 2.99×10⁻⁴) | −0.4298 / −0.0883 |
| `v2x_liberal` | **+0.333** / **+0.103** (n=3,806) | **+0.3333** / **+0.1028** | +0.3851 / +0.1468 |
| `v2x_civlib` | **+0.324** (n=3,806) | **+0.3244** / +0.1099 | +0.3720 / +0.1598 |
| `v2x_libdem` | **+0.304** / **+0.098** (n=3,806) | **+0.3036** / **+0.0983** (p=5.40×10⁻⁸² / 1.21×10⁻⁹) | +0.3542 / +0.1464 |

### 4b. Per-society Shield × `v2x_libdem` (aligned = published list)

| Metric | Published | Aligned | Legacy |
|---|---|---|---|
| Median within-society r | **+0.18** | **+0.180** | +0.193 |
| Fraction positive | **63.9%** | **63.9%** | 69.4% |
| Fraction p<0.05 | **61.1%** | **61.1%** | 66.7% |
| Developmental (+) | Singapore +0.69, India +0.68, Denmark +0.64, Norway +0.51, Poland +0.51, UAE +0.48, Chile +0.44 | **same seven, same r to 2 d.p.** | India +0.86, Singapore +0.72, Denmark +0.71, Turkey +0.71, … |
| Post-imperial (−) | Israel −0.78, Netherlands −0.68, UK −0.60, Russia −0.34, Ukraine −0.36 | **same five, same r to 2 d.p.** | Israel −0.81, Netherlands −0.61, UK −0.48, USA −0.40, South Africa −0.31 |

### 4c. Australia depth (T3a) — not reproduced

| V-Dem indicator | Published (n=131) | Aligned (n=150) | Legacy (n=150) |
|---|---|---|---|
| `v2x_liberal` | **+0.812** | +0.236 | +0.122 |
| `v2x_neopat` | **−0.792** | −0.233 | −0.124 |
| `v2x_rule` | **+0.786** | +0.300 | +0.205 |

No year-window of the in-repo Australia Shield × `vdem_core_v16` series yields r ≈ 0.81 (n=131 windows sit around r ≈ 0.27–0.34). The original T3a text points to an Australia-only V-Dem extract and a Lasso; that file is not in the repo. **T3a published coefficients are not reproduced here.** T3c (full panel) is the confirmatory Shield result and **does** reproduce.

### 4d. Year-level tests (unaffected; reproduced)

| Statistic | Published | Re-run (current panel) |
|---|---|---|
| λ₂ by Phase ANOVA F | 1,568.5 | **1,568.5** (n=4,033) |
| λ₂ Phase 1 / 6 means (n) | 0.736 (343) / 2.756 (1,658) | **0.736 (343) / 2.756 (1,658)** |
| `v2x_libdem` by Phase ANOVA F | 91.7, p=2.5×10⁻⁹¹ | **91.7**, p=2.46×10⁻⁹¹, n=3,806 |
| Phase 1 / 6 mean libdem (n) | 0.164 (315) / 0.443 (1,608) | **0.164 (315) / 0.443 (1,608)** |
| λ₂ × libdem within-society r | +0.202, p=3.3×10⁻³⁶ | **+0.2016**, p=3.28×10⁻³⁶, n=3,806 |

**Verdict vs published:** T3c Shield × V-Dem **holds** under current aligned `Bond_Strength_Calc` (it is the same column Phase A used). STRONG PASS stands. Legacy would have inflated every Shield–V-Dem r; that inflation was never published. T3a Australia remains an unreproduced single-country pilot. λ₂ / Phase tests were never in scope for the bond-column bug.

---

## 5. What can stay as-is

| Item | Why |
|---|---|
| `SBD_Calc` / `Lambda2_Calc` year-level work | Columns were not rewritten; they already matched recomputation |
| T3b λ₂ by Phase; T3c Phase ANOVA; λ₂ × libdem | Year-level; reproduced exactly |
| T4 Maddison × CAMS transitions | Uses `Phase_Calc` / `Regime_Label_Calc`, not Bond-by-node |
| T2 Hands peripherality (rank 2.17 / 8) | `Node_Value_Calc` rank; re-run 2.171 (Stewards 4.411); n=1,123 vs published 1,127 |
| T5 France Archive NV premium / Denmark Helm–Craft–Stewards ranks | Node_Value: France premium 0.645 vs 0.644; DK Helm 17.21 / rank 3.15, Craft 17.49 / 3.11, Stewards 16.40 / 4.73 — match |
| T2 H–S pairwise \(B_{ij}\) from C/K/S/A | Not the per-node mean column; v1.2 pairwise / SBD = 0.894 vs published 0.88×. The published 20.54 → 19.05 trend is a different scale and was not recovered as v1.2 \(B_{ij}\) (~0.27) or Node_Value (~7.7) |
| T5 Denmark triangle means ~47–51 | Published “raw formula” is **not** v1.2 \(B_{ij}\) (0–1). v1.2 HSC mean = 0.685 vs labour 0.652. Do not treat 48.6 as a Bond_Strength_Calc-by-node result |

---

## 6. Do published Phase A conclusions hold, reverse, or shrink?

| Test | Published verdict | After this re-run |
|---|---|---|
| T1 WB proxy | RESPECIFY (then superseded by SIPRI-real) | **Holds** as a number: r=−0.1279, within n.s., n=1,693 |
| T1 SIPRI-real | CONDITIONAL PASS, r=+0.251 / +0.121, n=2,360 | **Not reproduced** from public SIPRI v1.2 (sign is negative, n=1,823). Bond alignment is not the reason. Treat as **unrecovered extract**, not as a reversed conclusion of the fix |
| T2 Hands rank | PASS (internal) | **Holds** |
| T3 V-Dem Shield / Phase / λ₂ | STRONG PASS | **Holds** for T3c and T3b. T3a Australia coefficients **not reproduced** |
| T4 Maddison | PASS (consistency) | **Stays as-is** (not re-run; no Bond-by-node) |
| T5 FR/DK | PASS | NV / ranks **hold**; triangle raw-formula scale not v1.2 Bond |

**Overall:** Because Phase A already used node-aligned Shield bonds, the 2026-08-28 alignment fix **does not change the published T3c or T1-WB figures**. The legacy column is a counterfactual of the 20–28 Aug scramble: it would have inflated T3 Shield–V-Dem correlations and made T1 WB more negative, but those numbers were never the 16 Aug report.

The open hole is T1 SIPRI-real (Drive workbook missing) and T3a Australia (n=131 extract missing). Year-level SBD / λ₂ analyses are unaffected.

---

## 7. Blockers / missing files

- `sipri_milex_1949_2025.xlsx` (Drive) and `_shield_sipri_merged.csv` — not in repo. Public SIPRI v1.2 was used instead.
- `JUNO_36_PhaseA_Dataset_Mapping.md` — cited as protocol source, not in repo.
- Australia-only V-Dem file used for T3a (n=131) — not in repo; `vdem_core_v16.csv` gives n=150–151 and different r.
- ILO STAT (T2 external) — still unavailable; not in scope.
- Original Phase A Python/R pipeline — never committed; this addendum’s script reconstructs the published protocol.

Public downloads used: [SIPRI Milex Excel](https://www.sipri.org/sites/default/files/SIPRI-Milex-data-1949-2025_v1.2.xlsx), World Bank `MS.MIL.XPND.GD.ZS`.

---

*Re-run 2026-08-28. Do not silently edit `PhaseA_Validation_Report_2026-08-16.md` conclusions; this file is the record of the bond-alignment check.*
