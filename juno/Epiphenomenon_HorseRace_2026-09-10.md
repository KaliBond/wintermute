# Epiphenomenon Horse-Race Note — CAMS L2 vs externals → Hansard `sino_ppm`

**Date:** 10 September 2026 (Sydney)  
**Purpose:** Implements the **highest-value next step** named in *Epiphenomenon_Trove_Grok_Report_v3* (14 May 2026), §8 *Limitations and Next Steps*: a formal horse-race of lagged CAMS metrics against independent external variables (PRC actions, US alliance pressure, Australian trade exposure to China, domestic political variables) predicting annual Hansard Sino threat discourse (`sino_ppm`).

**Pointer:** Place next to the OOS freeze at `wintermute/juno/OOS_PREDICTIONS_2026-2028.md`. Optional one-line pointer in `CANONICAL_STATUS.md` only if kept minimal. **Rollback = revert the PR** that adds this note.

---

## Data provenance

| Item | Path |
|------|------|
| Nitra / box working set | `/workspace/epiphenomenon/` |
| Primary CAMS + outcome panel | `horse_race_data.csv` (2009–2025, n=17; Hands-aligned BS scale) |
| Outcome extension | `hansard_master.csv` (2006–2008); `hansard_raw_counts.csv` (2026 partial) |
| CAMS annual (lags for 2006–08) | `aus_annual.csv`, `hands_grievance_hansard.csv` |
| Report text | `Grok_Report_v3.txt` / `.pdf` |
| This run outputs | `juno/horserace_2026-09-10/` |
| Reproducible script | `juno/horserace_2026-09-10/run_horse_race.py` |

**Not used:** `australia_econ_controls.csv` (ends ~1954). No new 8-node Trove harvest. No second-coder packs scored.

---

## Method

1. **Outcome:** annual House Hansard `sino_ppm` (threat-coded China discourse). Primary sample **2009–2025** from `horse_race_data.csv` (matches prior Hands horse-race n≈17). Exploratory extensions: 2006–2008 (`hansard_master`); 2026 partial (`sino_raw/total_words` → 10.03 ppm).
2. **CAMS predictors (lagged):** `tau_L2`, `BS_L2`, `Hands_L2` (+ optional L3) from `horse_race_data`. Sensitivity: `tau_L2`/`BS_L2` rebuilt from `hansard_master` (different BS scale ~13–23).
3. **Externals (2006–2025):** built from public official / WITS–Comtrade series and a transparent event list (`horserace_2026-09-10/event_coding.md`, `SOURCES.md`). **No invented numeric values.** Missing cells labelled MISSING.
4. **Design (small-n honest):** univariate Spearman/Pearson; nested z-OLS (externals only / CAMS L2 only / CAMS + externals); few predictors; report coef, SE, p, R², adj-R², n, VIF. Softened language throughout.

---

## Results (quoted from this run only)

### Who wins (primary panel, `horse_race_data`, n=17)

**Contemporaneous PRC sanctions regime (`prc_sanctions_active`, 2020–2023) wins the univariate horse-race** on this Hands-aligned panel:

| Predictor | Spearman r | p | z-OLS adj-R² | z-coef (p) |
|-----------|------------:|---:|-------------:|------------|
| `prc_sanctions_active` | **+0.708** | 0.0015 | **0.591** | +8.748 (0.0002) |
| `prc_actions_count` | +0.633 | 0.0064 | 0.293 | +6.464 (0.015) |
| `security_announce_any` | +0.566 | 0.018 | 0.347 | +6.936 (0.008) |
| `Hands_L2` | **−0.625** | 0.0073 | 0.107 | −4.500 (0.108) |
| `BS_L2` | −0.255 | 0.32 | 0.022 | −3.215 (0.26) |
| `tau_L2` | −0.211 | 0.42 | −0.016 | −2.435 (0.40) |
| `china_export_share_pct` | +0.288 | 0.28 | 0.101 | +4.596 (0.12) |
| `us_alliance_pressure` | +0.195 | 0.45 | −0.016 | +2.431 (0.40) |
| `labor_gov` / `election_year` | ~0 | n.s. | ~0 | n.s. |

**Nested / block models (z-OLS, n=17):**

| Model | R² | adj-R² | Notes |
|-------|---:|-------:|-------|
| Externals top2 (`prc_sanctions_active` + `prc_actions_count`) | 0.697 | **0.654** | sanctions p=0.001; actions p=0.074 |
| CAMS L2 block (`tau_L2`+`BS_L2`+`Hands_L2`) | 0.351 | 0.202 | high VIF / multicollinearity |
| `tau_L2` + `prc_sanctions_active` | 0.618 | 0.563 | τ n.s. (p=0.87); sanctions p=0.0004 |
| Sparse `tau_L2` + sanctions + trade share | 0.695 | 0.619 | n=16; sanctions p=0.0008; τ & trade n.s. |

**Comparative fit reading (not causal validation):** On the primary Hands-aligned panel, **external PRC coercion measures dominate adj-R²**. Among CAMS L2 terms, **Hands_L2** carries the only clear univariate Spearman signal (r=−0.625); τ_L2/BS_L2 are weak on this scale.

### Sensitivity — Report-aligned `hansard_master` τ/BS scale

| Predictor | Spearman r | p | z-OLS adj-R² |
|-----------|------------:|---:|-------------:|
| `tau_L2` (hansard_master) | **−0.707** | 0.0015 | 0.202 |
| `BS_L2` (hansard_master) | **−0.698** | 0.0018 | 0.191 |

Nested with sanctions: adj-R²≈0.58; **CAMS coefs become n.s.** (τ p=0.53; BS p=0.46) while **sanctions remain p≈0.002**. So even where lagged τ/BS match the May 2026 Report’s univariate lag story, they do **not** retain independent explanatory power once the PRC sanctions dummy is in the model.

### Lag asymmetry (honesty)

Lagging sanctions two years (`prc_sanctions_L2`) weakens the external (Spearman +0.35, p=0.20, n=15). Discrete `prc_actions_L2` stays strong (+0.693, p=0.004). The primary “sanctions win” result is partly **same-year co-movement** with discourse, not a pure lead.

---

## What was missing / incomplete

- `china_export_share_pct` for **2025 and 2026** (WITS/Comtrade not retrieved) → MISSING  
- `australia_econ_controls.csv` not usable for 2006+  
- 2026 `sino_ppm` is **partial-year** (raw counts through available files)  
- No second-coder event scoring; single transparent table in `event_coding.md`  
- DFAT pivot XLSX / Composition-of-Trade PDFs returned HTTP 403 from this box (Akamai); trade series therefore WITS + Comtrade API  

---

## Limits (Validation & Limits honesty)

- **Small n (17)** — coefficients unstable; prefer sparse models; do **not** claim validated causal identification.  
- **Multicollinearity** among CAMS L2 (and between sanctions and security-announce years).  
- **Scale dependence:** Hands-aligned `horse_race_data` BS ≠ `hansard_master` BS; Report lag magnitudes reproduce only on the latter.  
- **No new Trove harvest; no second coder.**  
- Contemporaneous externals vs lagged CAMS is an asymmetric race; see lag robustness above.  
- Exploratory comparative fit only.

### Shield depression lifts (cite-only; out of scope for this race)

This note does **not** estimate Shield×depression lifts. If cited elsewhere, label both registered figures: **0.89 = registered 3-title**; **0.84 = 6-title extension**. Do **not** quote pooled PR-GAP-2 AUC 0.563 here (this run does not touch that paper).

---

## Artefacts

| File | Role |
|------|------|
| `horserace_2026-09-10/external_series.csv` | Year panel + competitors + source flags |
| `horserace_2026-09-10/event_coding.md` / `SOURCES.md` | Transparent coding + URLs |
| `horserace_2026-09-10/horse_race_results.json` | Machine-readable coefs / R² / n |
| `horserace_2026-09-10/horse_race_tables.md` | Readable tables |
| `horserace_2026-09-10/horse_race_adjr2.png` | Optional adj-R² bar chart |
| `horserace_2026-09-10/run_horse_race.py` | Reproducible script |

**Rollback:** revert the PR that lands this note beside `juno/OOS_PREDICTIONS_2026-2028.md`.
