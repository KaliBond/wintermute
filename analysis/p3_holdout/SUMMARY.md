# P3d holdout replication (17 Oct 2026)
Prereg committed first: git 4199cc2e (2026-10-17T18:22:10+11:00), PREREG sha256 94dc4862…

30 societies; common sample at h=3: 1,676 society-years, 141 positive windows (base rate 0.084). At h=5: 1,629 society-years, 189 positive windows.

| h | AUC B1 / B2 / B3 / C / C+B3 | C+B3 − B3: ΔAUC (CI); ΔBrier (CI) | C − B3 ΔAUC (CI) | C − B1 ΔAUC (CI) |
|---|---|---|---|---|
| 3 | 0.740 / 0.700 / 0.834 / 0.691 / 0.840 | +0.006 (−0.015, +0.031); +0.0009 (−0.0032, +0.0059) | −0.143 (−0.287, −0.046) | −0.049 (−0.176, +0.070) |
| 5 | 0.712 / 0.698 / 0.814 / 0.676 / 0.814 | −0.0004 (−0.019, +0.022); +0.0028 (−0.0027, +0.0098) | −0.138 (−0.282, −0.030) | −0.035 (−0.174, +0.108) |

H1: NOT supported (the CI includes 0).

Exploratory split (h=3; 14 societies with both classes, 16 excluded as single-class, median B3 AUC 0.608):
- B3-weak (7 societies): ΔAUC −0.003 (−0.054, +0.057).
- B3-strong (7 societies): ΔAUC −0.028 (−0.067, +0.014).
- Weak − strong: +0.025 (−0.046, +0.094).
- The predicted moderation is not detected.
