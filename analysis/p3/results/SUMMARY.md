# P3 result, 17 Oct 2026

Spec: ../PREREG_P3_17Oct2026.md (frozen before outcomes; SHA-256 87a854a7…). Script analysis/p3_compare.py unchanged; pooled metrics and society-cluster bootstrap (2000 reps, seed 20261017) via ../p3_pooled.py.

| Run | n society-years | societies | positive windows | base rate | AUC agg / node | Brier agg / node | ΔAUC (95% CI) | ΔBrier (95% CI) |
|---|---|---|---|---|---|---|---|---|
| Primary: coups, h=3 | 2885 | 47 | 236 | 0.082 | 0.720 / 0.720 | 0.0722 / 0.0733 | +0.0001 (−0.038, +0.045) | +0.0011 (−0.0027, +0.0051) |
| S1: coups, h=5 | 2804 | 47 | 326 | 0.116 | 0.716 / 0.717 | 0.0975 / 0.0980 | +0.0009 (−0.037, +0.042) | +0.0005 (−0.0053, +0.0062) |
| S2: V-Dem libdem drop ≥0.05, h=3 | 4732 | 47 | 169 | 0.036 | 0.673 / 0.667 | 0.0344 / 0.0361 | −0.0059 (−0.054, +0.041) | +0.0017 (−0.0002, +0.0049) |
| P7 exploratory: agg + λ2 vs agg, coups h=3 | 2885 | 47 | 236 | 0.082 | 0.720 / 0.718 | 0.0722 / 0.0721 | −0.0018 (−0.007, +0.003) | −0.0001 (−0.0003, +0.0002) |

Verdict: P3 fails its frozen criterion. The 32 node scores add nothing over the 4 aggregate means for externally coded events. λ2 (r = 0.87 with mean node value) adds nothing either.
