# Pre-Registration T-M1: Displacement Morphology vs Rival Classifications
## Does a society's characteristic stress-response signature predict episode outcomes better than regime type or aggregate state?

**Date:** 2026-10-16 · **Status:** REGISTERED before any outcome computation · **Analyst:** K. McKern / Kimi
**Dataset:** JUNO_Unified_Dataset.csv (github.com/KaliBond/wintermute, juno/), canonical verification PASSED (max |V drift| = 3.6e-15, 48 societies). External rival: V-Dem core v16 (repo `juno/vdem_core_v16.csv`) for electoral-democracy index as ideology/regime-type rival.

## 1. Epistemic position

Conjecture under test (from the 2026-10-16 framing discussion): a society's identity-in-adaptation is its historically persistent way of distributing stress across the eight functions (displacement morphology), and this predicts how episodes resolve better than regime type, ideology, or aggregate state. FAC refinement (Aug 2026) found morphology is society-level and stable (median C_V ≈ 0.131), recovery episode-level (ICC ≈ 0.09) — but that finding did NOT test morphology as an outcome predictor against rivals. This test does.

## 2. Operational definitions (canonical operators only)

- **Stress episodes:** society mean Stress S̄(t), z-scored within society over its full series. Onset = first year with z_S ≥ 1.0 following ≥2 consecutive years z_S < 1.0. Episode ends at first year z_S < 0.5. Episodes separated by ≥3 years; merge otherwise.
- **Displacement signature** d_e: per-node ΔS_i = S_i(episode peak year) − mean S_i(2 years pre-onset), normalised to unit L1 norm (8-vector).
- **Society morphology** M_s: mean signature across that society's OTHER episodes (leave-one-episode-out, no leakage).
- **Morphology typicality** of episode e: cosine similarity cos(d_e, M_s^LOEO).
- **Outcome (primary):** RECOVERY = V̄ reaches ≥90% of pre-onset 2-year baseline within 5 years of episode end. Episodes ending within 5 years of series end are censored (excluded).
- **Outcome (secondary):** years-to-recovery (censored at series end), and transformation (end-state V̄ > baseline +5%).
- **Rival predictors:** (R1) V-Dem electoral democracy index (v2x_polyarchy) at onset year; (R2) CAMS_v1_Regime / Regime_Label_Calc at onset (categorical); (R3) aggregate state: baseline V̄ and z_S peak amplitude.

## 3. Hypothesis ladder & pass/fail

| ID | Hypothesis | Test | Pass condition |
|---|---|---|---|
| T-M1a | Morphology typicality predicts recovery | Logistic regression, AUC via 10× repeated 5-fold CV stratified by society | AUC ≥ 0.65 AND exceeds R1 and R3 univariate AUCs by ≥0.05 |
| T-M1b | Morphology adds information beyond rivals | ΔAUC of full model vs rivals-only model | ΔAUC ≥ 0.03, DeLong-style permutation p < 0.05 |
| T-M1c | Morphology stability precondition | Within-society signature cosine similarity vs between-society | within > between, permutation p < 0.05 (sanity gate; if FAIL, T-M1a/b uninterpretable as stated) |
| T-M1d | Regime/ideology rival beats morphology | R1 or R2 AUC ≥ morphology AUC | If TRUE → conjecture FAILS as stated |

**Null model:** 1000 permutations shuffling society labels on episodes (destroys morphology-outcome link, preserves marginal rates). Morphology AUC must exceed 95th percentile of null.

**Minimum-n gate:** proceed only if ≥60 censored-adjusted episodes across ≥20 societies; else report as exploratory.

## 4. Registered interpretations

- T-M1a PASS + T-M1d FAIL → morphology carries transferable predictive structure; supports "coordination-of-adaptation as identity" conjecture; candidate for holdout replication.
- T-M1a FAIL, rivals win → conjecture fails as stated; morphology is descriptive character, not predictive machinery. Reported plainly.
- All FAIL incl. T-M1c → morphology not stable in this dataset at this resolution; FAC result does not generalise; scope restriction required.

## 5. Known limitations (registered)

Annual resolution ceiling (lead/lag claims barred); V-Dem covers most but not all society-years (missing → episode dropped for R1 comparisons only); z-score episode extraction is convention, not theory; recovery threshold 90%/5yr is a registered convention, sensitivity at 80–100% / 3–7yr reported.
