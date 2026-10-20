# Erratum, 20 October 2026: bond scale, "guarantees", and episode labels

We don't edit the past, we qualify it. The files named below stay exactly as they were. This note records what in them should not be relied on.

## 1. Bond-scale inconsistency

**What the documents say**

- `CAMS_OOS_Predictions_2027_2028.pdf` (12 July 2026) defines q_i = (0.6C_i + 0.4A_i)/10 and B_ij = √(q_i q_j) · 2^(−(S_i+S_j)/10). On 0–10 scores q_i ≤ 1 and the decay factor ≤ 1, so B_ij ≤ 1. Its regime table uses that scale (Stable Adaptive needs B̄ > 0.30; Systemic Crisis B̄ < 0.20; Freeze/Collapse B̄ < 0.15).
- The same memo, China entry, says edge weights "consistently average above 30.0". That cannot happen under its own formula.
- `prospective-tests/CAMS_USA_Prospective_Test_2026-2028.md` (sealed 28 April 2026) reports B̄ = 8.986 for 2026 (line 20), a 2020–2026 series from 11.786 to 8.986 (lines 78–86), and P1 and P7 thresholds of 13.0, 10.0, 8.0 and 7.0.

**Code in the repo**

- `cams_soil_gauge.py` lines 14–21 implement the memo's kernel: bounded to [0, 1].
- `cams-telescope/cams-compute.js` lines 23–45 use a different kernel: a Gaussian RBF on the distance between node (C, K, S, A) vectors, W = exp(−d²/2σ²) with σ the median pairwise distance. Also bounded to (0, 1]. So the two repo kernels disagree with each other in form, but both are on a 0–1 scale.

**Which scale produced 8.986**

8.986 is exactly the mean of the eight per-node B values in the USA document's node table (lines 31–38: 7.180, 7.401, 10.072, 8.779, 8.407, 8.779, 8.779, 12.493). Neither repo kernel can produce those values, because both are bounded by 1. The per-node formula is **undetermined** from the repo. It is attributed to a CAMS v3.0 `formulation3.0.txt` that is not in the repository. One clue from the table: Stewards (K = 4.4) and Craft and Hands (K = 5.0) share B = 8.779 with the same C, S and A, so that B does not depend on K.

**Qualification**

- The memo's B̄ thresholds and the USA document's B̄ values are on different, incompatible scales. Neither should be compared with the other.
- "Average above 30.0" for China is withdrawn as unsupported by any formula in the repo.
- USA P1 and P7 are qualified as resting on an undocumented bond scale. They are retired from the event-graded register (`prospective-tests/CAMS_Prospective_Register_v1_DRAFT.md`) in any case, as self-scored.
- `juno/OOS_PREDICTIONS_2026-2028.md` already notes the JUNO v1.2 bond matrix is rank-1 and its network visuals illustrative. That stands.

## 2. "Guarantees" withdrawn

The memo's United States entry (pdftotext line 75) says "The derivative of the structural decay curve guarantees that the Helm node will breach the critical V_min < 4.0 threshold by mid-2027". The word "guarantees" is withdrawn. It was a forecast, not a certainty, and it should read as one. Separately, JUNO v3 records USA Helm V = 8.0 in 2026, well above 4.0.

## 3. `JUNO_B_EP_Predictions.csv` relabelled

The file has 197 rows covering 7 societies and years 1801 to 2022 (none after 2022). Each row labels an inflection already in the record (61 SHARP_RECOVERY, 43 NODE_FAILURE_CROSSING, 43 COLLAPSE, 35 SYSTEMIC_CRISIS_ENTRY, 11 HOLLOW_HIGH_V, 4 PEAK_STABILITY) with an EP class and confidence. These are **retrospective episode labels**, not predictions, despite the file name and the `EP_Pred` column. They must not be counted as forecasts or hits.

## Files qualified, not edited

- `CAMS_OOS_Predictions_2027_2028.pdf`
- `prospective-tests/CAMS_USA_Prospective_Test_2026-2028.md`
- `JUNO_B_EP_Predictions.csv`
