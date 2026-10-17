# P3b: CAMS aggregate vs external coup baselines (17 Oct 2026)

Spec: PREREG_P3b_17Oct2026.md (frozen first; SHA-256 55456e4c…). Script: ../../p3b_benchmark.py. Common sample 2,879 of the 2,885 P3 society-years (6 dropped for missing V-Dem; no imputation), 47 societies. Pooled LOSO; society-bootstrap 95% CI (2000 reps). Electoral democracy and GDP are not in vdem_core_v16, so B2 = libdem + 1-yr change only.

| Run | positives / base rate | AUC B1 / B2 / B3 / C / C+B3 | C+B3 − B3: ΔAUC (CI); ΔBrier (CI) | C − B3: ΔAUC (CI) | C − B1: ΔAUC (CI) |
|---|---|---|---|---|---|
| **Primary h=3** | 234 / 0.081 | 0.700 / 0.666 / 0.802 / 0.721 / 0.816 | +0.015 (−0.001, +0.038); +0.00007 (−0.0016, +0.0018) | −0.080 (−0.164, −0.005) | +0.022 (−0.073, +0.129) |
| S1 h=5 | 324 / 0.116 | 0.665 / 0.669 / 0.787 / 0.717 / 0.801 | +0.014 (−0.004, +0.038); +0.0002 (−0.0022, +0.0028) | −0.070 (−0.159, +0.009) | +0.052 (−0.050, +0.167) |
| R1 h=3 + decade FE | 234 / 0.081 | 0.799 / 0.752 / 0.824 / 0.779 / 0.837 | +0.013 (+0.0003, +0.030); −0.0005 (−0.0030, +0.0019) | −0.046 (−0.090, +0.0004) | −0.020 (−0.066, +0.034) |
| R1 h=5 + decade FE | 324 / 0.116 | 0.787 / 0.756 / 0.815 / 0.778 / 0.827 | +0.012 (−0.002, +0.032); −0.0005 (−0.0036, +0.0030) | −0.037 (−0.082, +0.010) | −0.009 (−0.058, +0.046) |

Brier (primary): B1 0.0662, B2 0.0726, B3 0.0667, C 0.0718, C+B3 0.0667.

Verdict: fails the frozen criterion. CAMS adds a small, consistently positive but non-significant AUC increment over external baselines (+0.012 to +0.015), with no Brier gain. Alone, CAMS is significantly worse than coup history + V-Dem at h=3.
