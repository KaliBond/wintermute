## Best-two-scorer sensitivity (Claude agent + Gemini)

**This is a supplementary, post-hoc primary-threshold check. It does not change the primary pool or its 6 PASS / 9 FAIL headline.**

The canonical pair is **Claude agent (A−) + Gemini (B)**: the highest overall locked grade, then the first scorer among the B tier with sound calibration. Both runs are weighted equally after within-run averaging, using the same primary thresholds and central V/Bond calculations as pool (a).

| Item | Primary (7 runs) | Claude agent + Gemini | Comparison |
|---|---|---|---|
| P1 | PASS | **PASS** | same |
| P2 | FAIL | **FAIL** | same |
| P3 | FAIL | **FAIL** | same |
| P4 | PASS | **PASS** | same |
| P5 | PASS | **PASS** | same |
| P6 | FAIL | **FAIL** | same |
| P7 | FAIL | **FAIL** | same |
| P8 | FAIL | **FAIL** | same |
| P9 | FAIL | **FAIL** | same |
| P10 | PASS | **PASS** | same |
| P11 | FAIL | **FAIL** | same |
| P12 | PASS | **PASS** | same |
| C1 | FAIL | **FAIL** | same |
| C2 | PASS | **PASS** | same |
| C3 | FAIL | **FAIL** | same |
| FC5 | NOT TRIGGERED | **NOT TRIGGERED** | same |

**Outcome: 6 PASS / 9 FAIL, with no indeterminates; FC5 NOT TRIGGERED.** The passes are P1, P4, P5, P10, P12 and C2; the fails are P2, P3, P6, P7, P8, P9, P11, C1 and C3, exactly matching the primary scorecard. **P6 is FAIL at 13 satisfied / 3 not satisfied.** **P8 is FAIL at rounded ratios 5.00 (Vanke) / 5.91 (Longfor), limit 5.0:** the unrounded Vanke ratio is 4.998818728522337, so it does not clear the sealed ≥5.0 rule. `best2_scorers_sensitivity.csv` retains the display-rounded 5.0, and `51_score.py` shows the unrounded sealed comparison; the knife-edge must not be called a pass.

If the second seat were **Kimi web** instead of Gemini, P6 would flip to PASS (14/16), giving 7 PASS / 8 FAIL; P8 would remain FAIL at 4.30/5.23, and FC5 would remain untriggered. This alternative does not change the canonical pair or the primary headline.

Numeric inputs and metrics are preserved in `best2_scorers_sensitivity.csv`.
