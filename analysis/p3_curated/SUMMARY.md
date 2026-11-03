# P3c: curated ensemble panel (17 Oct 2026)

Spec frozen first: PREREG_P3c_17Oct2026.md (SHA-256 40131b9e…). Scripts: inventory.py → select_runs.py → build_curated_inputs.py → p3c_run.py.

## Curation
- 70 candidate ENS/envelope runs inventoried (run_eligibility.csv). Selected: 18 societies (selected_runs.csv): Cambodia, Chile, China, DRC, Egypt, Eritrea, Ethiopia, France, Germany, Greece, Iran, Laos, Mauritania, Mongolia, New Zealand, Nigeria, Norway, Russia.
- 11 envelopes are templates, not scorer disagreement. One identical SD sequence is shared by the m5n block2 files for Australia, China, Russia, UK and USA (2000–2026), for Belgium, Morocco, Portugal and Sri Lanka (1900–2026), and by both Japan 1850–2026 envelopes, including `cleaned_datasets/Japan_ENV_1850_2026_cleaned.csv`. `scripts/check_envelopes.py` does not detect this.
- Degenerate envelopes (>10% of rows with all SD = 0): Argentina 72%, Poland 81%, UK 80%, Italy 73%, Colombia 59%, Australia 12.5%; also Canada, Sweden, Thailand, Türkiye and USA (which are additionally non-annual).
- CURATED_CAMS.csv: 19,320 node-rows, 2,415 society-years (1800–2026); 1,277 society-years overlap the coup register, with 39 coup-years.
- For the 17 societies shared with JUNO, the curated scores are identical to JUNO on all 19,048 shared node-rows. Curation therefore selects a subset; it does not change any score. Eritrea is the only addition.
- λ2 was recomputed from the canonical bond formula and matches JUNO `Lambda2_Calc` within 5e-5.

## Results (pooled LOSO, society bootstrap 95% CI). Power is low: 18 societies, 93 positive windows at h=3.
| Test | h | n / soc / pos | AUC | ΔAUC (CI) | ΔBrier (CI) |
|---|---|---|---|---|---|
| P3 node − agg | 3 | 1235 / 18 / 93 | 0.778 agg, 0.753 node | −0.025 (−0.095, +0.048) | +0.0017 (−0.0029, +0.0068) |
| P3 node − agg | 5 | 1199 / 18 / 135 | 0.786, 0.750 | −0.036 (−0.099, +0.027) | +0.0064 (−0.0031, +0.0172) |
| P3b C+B3 − B3 | 3 | 1233 / 18 / 93 | 0.785 vs 0.728 | +0.058 (+0.004, +0.131) | +0.0030 (−0.0032, +0.0126) |
| P3b C+B3 − B3 | 5 | 1197 / 18 / 135 | 0.790 vs 0.711 | +0.079 (+0.016, +0.162) | +0.0038 (−0.0079, +0.0221) |
| P3b C − B3 | 3 | | 0.778 vs 0.728 | +0.050 (−0.033, +0.155) | +0.0051 (−0.0045, +0.0189) |
| P3b C − B1 | 3 | | 0.778 vs 0.623 | +0.155 (+0.053, +0.310) | +0.0051 (−0.0046, +0.0194) |
| P7 agg+λ2 − agg | 3 | 1235 / 18 / 93 | 0.780 vs 0.778 | +0.001 (−0.008, +0.008) | −0.0002 (−0.0008, +0.0004) |
| P7 agg+λ2 − agg | 5 | 1199 / 18 / 135 | 0.787 vs 0.786 | +0.001 (−0.005, +0.007) | −0.0002 (−0.0011, +0.0006) |

JUNO restricted to the same society-years (17 societies) gives essentially the same numbers (P3b h=3 ΔAUC +0.055, CI +0.003 to +0.135). Full detail: results/ALL.json.

## Verdict
- P3 and P7 remain null.
- P3b nominally meets the frozen ΔAUC criterion on the curated subset, but Brier is worse, the CIs are wide (ΔAUC CI width ≈ 0.13), and the external baseline is much weaker in this subset (B3 AUC 0.728 vs 0.802 on the 47 societies). The same result appears on JUNO restricted to these societies, so it reflects which societies are included, not better scoring. It does not overturn the 47-society null. It is a hypothesis for a pre-registered test on new societies.
