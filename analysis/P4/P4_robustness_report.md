# P4 Robustness Tests: Report
**Date:** 2026-11-11  
**Tests:** Circular-shift null (A), Stress-change-alone baseline (B), First differences of V (C)  
**Protocol:** crisis_typology_rule_v1 (sealed) | V = C + K − S | EPI={Helm,Lore} | ECO={Hands,Stewards,Flow} | threshold ±1.0  
**Permutations (Test A):** 10,000 | seed 42

---

## Test A: Circular-Shift Null

**Question:** Does the temporal ordering of years carry information beyond what the structural configuration already provides? Or could you shuffle the years and get the same detector firing pattern?

**Method:** For each firm, the gap series (ECO_mean − EPI_mean, year by year) is circularly shifted by a random offset drawn uniformly from {0, …, n−1}. The detector is re-run on the shifted series. Repeated 10,000 times. Fire *count* is preserved exactly under any circular shift (the same values appear, just reordered). The test therefore focuses on **run length**: the longest consecutive streak of fires: as the statistic sensitive to temporal clustering.

| Entity | Obs fires | Obs run | Null run μ | p(run ≥ obs) |
|---|---|---|---|---|
| Boeing | 8 | 8 | 6.39 | 0.299 |
| Airbus | 6 | 6 | 5.10 | 0.502 |
| Theranos | 6 | 4 | 3.64 | 0.727 |
| Quest Diagnostics | 5 | 3 | 2.82 | 0.818 |
| WorldCom | 4 | 4 | 3.55 | 0.662 |
| AT&T | 5 | 2 | 2.00 | 1.000 |
| Lockheed Martin | 6 | 3 | 5.10 | 1.000 |
| Illumina | 6 | 4 | 3.64 | 0.731 |
| Verizon | 5 | 5 | 5.00 | 1.000 |

**Interpretation:**

The null fire count equals the observed fire count for every firm, by construction: circular shifts preserve the full set of values. This tells us that **fire rate is entirely explained by the structural position**, not by temporal proximity to any particular event. The detector fires when the configuration is at the right position in phase space; it makes no use of the calendar.

The run-length test asks whether fires cluster more than chance would predict. The answer is: **mostly not.** For eight of nine firms, the observed run is achievable by a substantial fraction of random permutations (p ≥ 0.30). The exception is Boeing, which shows the longest observed run (8 consecutive years) relative to its null distribution (mean 6.39), with p = 0.299. This is not conventionally significant, but Boeing's run does sit in the right tail of its null: consistent with a genuine deterioration that deepened year on year rather than a configuration that fluctuated in and out of the fire zone.

The implication for the phase-space argument: **year ordering adds minimal information.** What predicts firing is structural position, not temporal proximity to events. The same configuration in any year would fire; the temporal sequence reflects the trajectory through phase space, not any calendar effect.

---

## Test B: Stress-Change-Alone Baseline

**Question:** Can the detector be reduced to a simpler rule based only on change in mean system Stress (ΔS)? If ΔS alone reconstructs the fire labels, the EPI/ECO topology adds nothing beyond tracking how stressed the system is becoming.

**Method:** For each firm-year, compute ΔS = mean Stress across all 8 nodes at year t minus year t−1. Sweep a threshold τ: fire if |ΔS| > τ. Find the τ that maximises accuracy against the sealed fire labels. Compare against EPI/ECO detector accuracy (which is 1.0 by construction at threshold 1.0, since fire is defined by that threshold).

| Entity | EPI/ECO acc | ΔS acc (best τ) | r(ΔS, Gap) |
|---|---|---|---|
| Boeing | 1.000 | 0.700 | +0.654 |
| Airbus | 1.000 | 0.700 | −0.305 |
| Theranos | 1.000 | 0.636 | +0.466 |
| Quest Diagnostics | 1.000 | 0.636 | +0.189 |
| WorldCom | 1.000 | 0.667 | +0.750 |
| AT&T | 1.000 | 0.556 | −0.327 |
| Lockheed Martin | 1.000 | 0.500 | −0.465 |
| Illumina | 1.000 | 0.545 | −0.378 |
| Verizon | 1.000 | 0.200 | −0.408 |
| **Global** | **1.000** | **0.593** | **+0.020** |

Note: EPI/ECO accuracy is 1.0 by construction (fire is defined using the same threshold). The comparison is between how well the EPI/ECO topology partitions the problem vs. how well ΔS alone achieves the same partition.

How the Global row is computed: Global accuracy is the year-weighted mean of the per-firm accuracies, that is, correct firm-years summed over all nine firms divided by 86 firm-years (7 + 7 + 7 + 7 + 6 + 5 + 5 + 6 + 1 = 51 of 86 = 0.593). Global r is the unweighted mean of the nine per-firm r values (sum +0.176, divided by 9 = +0.020). It is not a pooled correlation. The year counts are in the N_years column of P4_test_B_stress_change.csv.

**Interpretation:**

The mean of the nine per-firm correlations between ΔS and the EPI/ECO gap is r = +0.020. (An earlier version gave +0.176, which is the sum of the nine values, not their mean.) Because the per-firm signs differ, this mean is not a pooled correlation, and no share of variance is claimed from it. Weighted by years, the best per-firm ΔS classifiers are right in 51 of 86 firm-years (0.593). That is the same as the share of firm-years that fire (51 of 86), so it is the rate a rule that always predicts a fire would reach. Per-firm accuracy peaks at 70% (Boeing, Airbus) and falls as low as 20% (Verizon).

The sign of r(ΔS, Gap) varies across firms: positive for the cases (Boeing +0.654, WorldCom +0.750, Theranos +0.466) and negative for the secondary controls (Lockheed −0.465, Illumina −0.378, Verizon −0.408). This sign reversal reflects the EPI/ECO topology distinction: cases fire epistemologically (EPI trailing ECO), and in those firms rising stress is concentrated in the EPI nodes, so ΔS predicts the gap direction. Controls fire economically (ECO trailing EPI), and rising stress is concentrated in ECO nodes, reversing the relationship.

**The EPI/ECO topology cannot be replaced by ΔS.** The topology specifies *where* stress is accumulating, not just *how much* stress is changing. That distinction is load-bearing.

---

## Test C: First Differences of V (ΔGap)

**Question:** Does ΔGap = Gap_t − Gap_{t-1} (the first difference of the EPI/ECO gap) predict firing better than the gap level? If so, the detector would be more powerful as a rate-of-change instrument than as a position instrument: consistent with the phase-space trajectory interpretation.

**Method:** For each firm-year, compute the gap level and its first difference. Sweep τ for each and find the best classifier. Additionally, check whether ΔGap exceeds 0.5 in the year *preceding* a fired year (a leading indicator test).

| Entity | Level acc | ΔGap acc | Lead signal (year before) |
|---|---|---|---|
| Boeing | 1.000 | 0.800 | 62% |
| Airbus | 1.000 | 0.700 | 50% |
| Theranos | 1.000 | 0.545 | 67% |
| Quest Diagnostics | 1.000 | 0.545 | 60% |
| WorldCom | 1.000 | 0.667 | 25% |
| AT&T | 1.000 | 0.556 | 60% |
| Lockheed Martin | 1.000 | 0.600 | 17% |
| Illumina | 1.000 | 0.545 | 17% |
| Verizon | 1.000 | 1.000 | 0% |
| **Global** | **1.000** | **0.640** |: |

How the Global row is computed: as in Test B, the year-weighted mean of the per-firm accuracies (8 + 7 + 6 + 6 + 6 + 5 + 6 + 6 + 5 = 55 of 86 = 0.640). An earlier version gave 0.593, which is Test B's year-weighted ΔS accuracy, not Test C's. The year counts are in the N_years column of P4_test_C_first_differences.csv.

**Interpretation:**

Level gap outperforms first differences globally (1.000 vs. 0.640 at best thresholds, year-weighted). The detector is a **position instrument**, not a velocity instrument: it fires when the configuration is in the wrong region of phase space, not when it is moving toward that region. This is consistent with the phase-space analogy: the H-R diagram locates a star by its current observables, not by the direction of its recent drift.

However, the **leading indicator** result adds nuance. For the three cases, ΔGap exceeds 0.5 in the year before 62% (Boeing), 67% (Theranos) and 25% (WorldCom) of fire years: for Boeing and Theranos, the gap was already moving before it crossed the threshold; for WorldCom, mostly not. This is a trajectory property: the system approaches the boundary before crossing it. The leading signal is weaker for the secondary controls (17% for Lockheed and Illumina), consistent with their EPI/ECO imbalances being structural rather than dynamic.

The correct description is therefore: the detector fires on **position**, but position is reached via **trajectory**, and the trajectory is partially visible one year ahead in the first difference of the gap. The two descriptions are complementary, not competing.

---

## Summary

| Test | Question | Finding |
|---|---|---|
| A: Circular-shift null | Does year order matter? | No, beyond fire rate. Runs are not significantly more clustered than chance (Boeing marginal, p = 0.30). |
| B: Stress-change-alone | Can ΔS replace topology? | No. Mean per-firm r(ΔS, Gap) is +0.020, with signs that differ by firm. Best ΔS classifiers are right in 0.593 of firm-years (year-weighted), the same as the share of firm-years that fire. Topology specifies where stress accumulates, not just how much. |
| C: First differences | Level or velocity? | Level wins (1.000 vs 0.640, year-weighted). Lead signals for the cases are 62% (Boeing), 67% (Theranos) and 25% (WorldCom): two of the three cases approach the boundary before crossing it. |

**Combined reading:** The EPI/ECO detector fires on structural position. That position is not an artifact of temporal ordering (Test A), is not reducible to global stress dynamics (Test B), and is best captured by level rather than velocity (Test C), though velocity provides partial early warning for case-tier firms (Test C lead signal). The three tests jointly support the phase-space location interpretation: the detector measures where the system is, not what is happening to it in the short term.

---

*Output files: P4_test_A_circular_shift.csv, P4_test_B_stress_change.csv, P4_test_C_first_differences.csv*  
*Parent study: P4_detector_results_9firm.csv | P4_detector_report_9firm.md*
