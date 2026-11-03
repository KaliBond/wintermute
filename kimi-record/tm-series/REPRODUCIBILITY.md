# Reproducibility Record — T-M Series (2026-10-16)

## Data provenance

All data retrieved 2026-10-16 from `github.com/KaliBond/wintermute`, branch `main`, commit `446928c5285b` (2026-10-14T02:55:49Z).

| File | Path in repo | SHA-256 (first 16) |
|---|---|---|
| JUNO_Unified_Dataset.csv | `juno/` | `49ee7b173abc5856` |
| vdem_core_v16.csv | `juno/` | `d4d4de4a8c7ddd7f` |
| JUNO_Blind_Set_1.csv | `blind-test/` | `be1c3ac58ad3cbdd` |
| JUNO_Blind_Set_2.csv | `blind-test/` | `513d717265ff1e7b` |
| JUNO_Blind_Set_3.csv | `blind-test/` | `fc9b1a087a4ec1ce` |
| JUNO_Blind_Set_5.csv | `blind-test/` | `3ae22e0d7eef5225` |
| JUNO_Blind_Set_6.csv | `blind-test/` | `682383b2a78866f2` |
| JUNO_Blind_Set_7.csv | `blind-test/` | `5470cc70f70984c6` |

Canonical verification: V = C + K − S + 0.5A recomputed against stored `Node_Value_Calc`, max |drift| = 3.6e-15 (pass bar 1e-9).

## Blind set 4

No `JUNO_Blind_Set_4.csv` exists in the repository at commit 446928c5 (full tree listing checked: sets present are 1, 2, 3, 5, 6, 7). The numbering gap predates this analysis; its reason is not documented in the repo files inspected. This is a dataset-provenance question for the corpus maintainer, not an artefact of the T-M pipeline.

## Blind series key

`JUNO_Answer_Key.csv` and `JUNO_Answer_Key_B.csv` exist in the repo root at the same commit. They were NOT read or used in any T-M analysis. Blind-set identities remain unblinded-to-no-one: the T-M4 pipeline treated `Society_ID` labels as anonymous throughout. Unblinding for T-M5 external matching is a pending user decision (recorded in T-M5 coverage assessment, open decision 1).

## Registered random seeds

| Test | Procedure | Seed |
|---|---|---|
| T-M1 | CV (RepeatedStratifiedKFold 5×10) | 0 |
| T-M1c | within/between permutation (500) | 42 |
| T-M1 | morphology AUC null (200) | 7 |
| T-M2a | society-label permutation (500) | 11 |
| T-M3a | society-label permutation (500) | 23 |
| T-M3d-1 | R node-shuffle null (1000) | 5 |
| T-M3d | mechanical floor simulation (1000) | 9 |
| T-M4 | permutation null (1000) | 31 |

Environment: Python 3.12, pandas / numpy / scipy / scikit-learn / statsmodels (sandbox builds, 2026-10). All generators `numpy.random.default_rng(seed)`.

## Corrections applied 2026-10-16 (post-audit)

1. T-M3e panel = 40,472 obs (society-year FE, FWL). 40,944 was the T-M3d node-FE panel. Both figures now attributed correctly.
2. T-M2a stated as: accuracy 0.470 vs 0.389 marginal baseline (1.21×); permutation null mean 0.269, p = 0.002. The 1.21× is against the marginal baseline, not the null.
3. Episode accounting: 201 episodes / 42 societies extracted → 158 / 42 after censoring rules (`episodes.csv`) → 149 / 33 after LOEO typicality (T-M1 analysis set). The 158-row file is the pre-typicality set; eligibility now traceable via `tm_series_analysis.py`.

## Correction log v2 (2026-10-16, post clean-machine rerun by K. McKern)

Root causes identified and resolved; script v2 is canonical.

1. **pivots() unused argument** — bug, fixed. Script failed at the T-M1 horse-race line until removed.
2. **T-M3d analysis set** — lab notes reported the 149-episode analysis set (κ 0.499 / κ_R 0.249); script v1 ran the 158 censored set (0.507 / 0.265). Script v2 computes both: 149 = primary (as reported), 158 = sensitivity. No verdict changes.
3. **Displacement-signature alignment** — the original session built typicality signatures at the society stress-peak year; script v1 used trough-aligned ΔS. Canonical = society-peak (0.492 AUC; gap 0.0267). The trough-aligned variant (0.497 AUC; gap 0.024) is retained in script v2 as a labelled variant. Verdicts unchanged.
4. **Failure-locus tie-break** — the original session broke argmin-V ties by largest ΔV (ad hoc, undocumented in v1). Script v2 uses plain deterministic argmin. Canonical T-M2a: 0.510 vs 0.409 marginal (1.25×), null 0.287, p = 0.002; Hands count 61; T-M3 relocated share 62.4%. Verdicts unchanged (T-M2a still fails the 1.5× bar).
5. **T-M2b original κ 0.184 not reproducible** — under plain argmin the value is 0.154 (tie-broken variant 0.162). Residual attributed to first-session dtype handling (initial CSV load without low_memory=False, mixed-type columns). Corrected to 0.154 in the notes; verdict unchanged (FAIL vs 0.20 bar).
6. **Panel sizes** — three specs now labelled: T-M3d-2 original node-FE forward-gap-only panel n = 40,944 (coefficients −0.138); with placebo columns required n = 40,608 (same coefficients); T-M3e hardened panel n = 40,472 (−0.094/−0.093). The 40,944/40,472 confusion was a spec difference, not an error in either number.
7. **Episode accounting** — extraction 201 episodes / 48 societies (notes previously said 42, which was the post-censoring count); eligible 158 / 42; analysis set 149 / 33.
8. **Society count in extraction** — 48, not 42; the 42 figure belonged to the censored set. Corrected throughout.

## Script

`tm_series_analysis.py` — single consolidated script reproducing every headline figure from the raw CSVs with the seeds above. Expected values annotated per line. Run time ≈ 10 min.
