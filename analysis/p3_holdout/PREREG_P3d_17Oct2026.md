# P3d frozen specification: holdout replication of the curated P3b finding (17 Oct 2026)

Written and committed before any P3d outcome was computed.

## Holdout set (30 societies)
These are the societies in the 47-society P3 panel (JUNO at 446928c, Hong Kong excluded) that are not among the 18 curated P3c societies:

Argentina, Australia, Brazil, Canada, Colombia, Denmark, India, Indonesia, Iraq, Israel, Italy, Japan, Latvia, Lebanon, Netherlands, Pakistan, Philippines, Poland, Saudi Arabia, Singapore, South Africa, Sweden, Syria, Thailand, Turkey, UAE, UK, USA, Ukraine, Venezuela.

(Eritrea is curated-only and was never in the 47-society panel. The 17 other curated societies are removed.)

**Not untouched data.** All 30 were part of the earlier 47-society P3/P3b runs, whose results are known. This is a holdout from the curated finding (P3c), not a fresh sample.

## Data and models
- Panel: `FROZEN_CAMS.csv` (JUNO scores) restricted to the 30 holdout societies.
- Events: `INDEPENDENT_EVENTS.csv` (Powell & Thyne V2026.08.29 coup attempts).
- V-Dem: `vdem_core_v16`.
- Features and code: `p3b_benchmark.py` logic, unchanged.
  - B1: coup in t−9…t; years since last coup, capped at 30.
  - B2: v2x_libdem, Δv2x_libdem.
  - B3 = B1 + B2.
  - C: 4 CAMS means.
  - C+B3.
- Common sample: rows with V-Dem at t and t−1; no imputation.
- Leave-one-society-out within the 30 holdout societies only (training uses holdout societies only). Fit settings are the same as in `p3_compare.fit_predict`.
- Pooled out-of-fold AUC and Brier. Society-bootstrap 95% CI, 2000 replicates, seed 20261017.

## H1 (primary)
C+B3 beats B3 at horizon 3.

**Frozen success rule:** the ΔAUC (C+B3 − B3) society-bootstrap 95% CI excludes 0 on the positive side. ΔBrier is reported as secondary and does not decide the result.

Also reported: C vs B3, C vs B1. Horizon 5 is a sensitivity analysis and does not change the verdict.

## Secondary (exploratory)
Prediction from P3c: the gain is larger where History + V-Dem is weaker.
- Compute each holdout society's own B3 out-of-fold AUC at h=3. This is defined only for societies with both outcome classes in their own common-sample rows.
- Median-split those societies into "B3-weak" (AUC ≤ median) and "B3-strong".
- Report pooled ΔAUC (C+B3 − B3) with bootstrap CI within each half, and the difference between halves (weak − strong) with a bootstrap CI that resamples societies within each half.
- Societies with only one class are excluded from the split; their number is reported.
- This is labelled exploratory; it is not part of the success rule.
