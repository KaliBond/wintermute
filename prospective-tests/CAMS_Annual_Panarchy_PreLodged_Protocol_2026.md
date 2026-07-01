# Pre-Lodged Experimental Protocol

## Testing Panarchy and Holling Adaptive-Cycle Dynamics in Annual CAMS Data

**Project:** CAMS / JUNO Annual Panarchy Test
**Prepared by:** Kari McKern
**Protocol status:** Pre-lodged primary analysis plan
**Purpose:** To test whether annual CAMS society-node data exhibit empirically detectable Holling-cycle phase dynamics and cross-scale panarchic “revolt” cascades under pre-specified criteria.

---

## 1. Research Question

Do annual CAMS datasets for Russia, China, the United States and Germany exhibit panarchic adaptive-cycle behaviour consistent with Holling's (r \rightarrow K \rightarrow \Omega \rightarrow \alpha) model?

The experiment distinguishes two levels of claim:

1. **Phase-kinematic support:** societies move through identifiable adaptive-cycle regions in CAMS phase space.
2. **Panarchic cross-scale support:** fresh node-level viability failures precede system-level release events more often than expected under within-society temporal shuffle nulls.

The experiment is explicitly falsification-oriented. Failure to detect the pre-specified revolt mechanism will count against strong panarchic causality, even if weaker adaptive-cycle phase structure remains present.

---

## 2. Data

### 2.1 Primary datasets

The primary analysis will use annual CAMS ensemble-mean data for:

* Russia
* China
* United States
* Germany

Each society must have yearly observations containing the eight CAMS nodes and the four canonical node variables:

[
C_i,\ K_i,\ S_i,\ A_i
]

where:

* (C_i) = coherence
* (K_i) = capacity / knowledge / capability
* (S_i) = stress
* (A_i) = abstraction / activation

The analysis will use the annual ensemble mean files as the primary source. Envelope files may be used only for uncertainty reporting, not for changing phase calls or primary thresholds.

### 2.2 Inclusion criteria

A society-year is admissible if all eight CAMS nodes are present and have valid (C,K,S,A) values.

A society is admissible if it contains enough annual observations to identify multi-year phase runs and release events. Missing years will not be interpolated for the primary analysis. Sensitivity runs may use interpolation, but must be reported separately.

---

## 3. Locked Derived Variables

### 3.1 Node viability

Node viability is computed using the locked CAMS/JUNO viability formula already specified for the annual ensemble files.

For each node (i) in year (t):

[
V_{i,t}
]

The following system-level summaries are then computed:

[
\bar{V}*t = \frac{1}{8}\sum*{i=1}^{8} V_{i,t}
]

[
V_{\min,t} = \min_i(V_{i,t})
]

A node is treated as critically stressed when:

[
V_{i,t} < 4
]

This threshold is locked before analysis.

---

### 3.2 Fresh node crossing

A **fresh node crossing** occurs when a node crosses newly below the viability threshold:

[
V_{i,t} < 4
]

and

[
V_{i,t-1} \geq 4
]

Chronic low viability does not count as a fresh crossing. The revolt test is therefore aimed at newly propagating node failures, not persistent background weakness.

---

### 3.3 Bond strength verification and recomputation

Legacy bond-strength columns will be checked before analysis.

If the maximum recorded bond value satisfies:

[
BS_{\max} \leq 1.5
]

then the existing bond values may be used.

If:

[
BS_{\max} > 1.5
]

then all bond strengths must be recomputed using the canonical annual formula:

[
B_{ij,t} = \sqrt{q_{i,t}q_{j,t}} \cdot 2^{-(S_{i,t}+S_{j,t})/10}
]

where (q_i) is the locked canonical node-quality term used in the CAMS annual bond script.

The mean bond strength is:

[
\bar{B}*t = \frac{1}{N_B}\sum*{i<j} B_{ij,t}
]

where (N_B = 28) for the complete eight-node undirected graph.

---

### 3.4 Network fragmentation

For each society-year, the weighted bond matrix is used to construct a graph Laplacian. Algebraic connectivity is computed as:

[
\lambda_{2,t}
]

where (\lambda_2) is the second-smallest eigenvalue of the weighted Laplacian.

Falling (\bar{B}*t) indicates weakening average institutional coupling. Falling (\lambda*{2,t}) indicates increasing graph fragmentation.

---

## 4. Within-Society Standardisation

To avoid cross-polity magnitude contamination, each society is standardised against its own historical series.

For each society:

[
P_t = z(\bar{V}_t)
]

[
C_{x,t} = z(\bar{B}_t)
]

where (z(\cdot)) is calculated using the full available time series for that society.

Thus:

* (P_t) represents relative systemic viability within that society's own history.
* (C_{x,t}) represents relative institutional coupling within that society's own history.

Annual rates of change are computed as true one-year first differences:

[
\Delta P_t = P_t - P_{t-1}
]

[
\Delta C_{x,t} = C_{x,t} - C_{x,t-1}
]

No five-year differencing will be used in the primary annual test.

---

## 5. Phase Classification

The analysis uses the previously locked CAMS adaptive-cycle classifier. The same level-and-rate thresholds used in the earlier phase-space analysis are retained.

No thresholds may be adjusted after inspecting the annual results.

Each society-year is classified into one of four Holling phases:

### 5.1 (r) phase — exploitation / expansion

A year is classified as (r) when the system is in a rebuilding or expansionary state, with improving viability and/or coupling from a relatively low or moderate base.

### 5.2 (K) phase — conservation / rigidity

A year is classified as (K) when the system shows high accumulated viability and coupling, but declining flexibility, increasing stress load, or reduced adaptive margin.

### 5.3 (\Omega) phase — release

A year is classified as (\Omega) when the system undergoes a rapid loss of viability and/or coupling.

For this protocol, a system-level (\Omega)-release must satisfy the locked classifier's release criteria and must also show at least one of the following non-circular fragmentation conditions:

[
\Delta C_{x,t} < T_{\Delta C}^{\Omega}
]

or

[
\Delta \lambda_{2,t} < T_{\Delta \lambda}^{\Omega}
]

where (T_{\Delta C}^{\Omega}) and (T_{\Delta \lambda}^{\Omega}) are the pre-existing release thresholds from the CAMS classifier.

This prevents the release outcome from being defined solely by the same node-level crossing used as the revolt predictor.

### 5.4 (\alpha) phase — reorganisation

A year is classified as (\alpha) when the system remains degraded or disordered after release but shows stabilisation, recombination, or renewed upward movement in (P_t) or (C_{x,t}).

---

## 6. Primary Hypotheses

### H1 — Adaptive-cycle ordering

If CAMS captures Holling-cycle dynamics, phase transitions should not be randomly ordered. The observed sequence should show excess movement in the direction:

[
r \rightarrow K \rightarrow \Omega \rightarrow \alpha \rightarrow r
]

relative to alternative unordered or reverse transitions.

**Primary support criterion:**
At least three of the four societies must show more forward-cycle transitions than reverse-cycle transitions.

**Falsification criterion:**
If forward-cycle ordering is not greater than reverse or unordered transition frequency in at least three societies, the claim of general Holling-cycle phase kinematics is not supported.

---

### H2 — Conservation-trap dwell predicts release depth

If (K)-phase rigidity represents a conservation trap, longer (K)-dwell periods should be associated with deeper subsequent (\Omega)-release events.

A (K)-dwell is defined as a consecutive run of years classified as (K).

Primary rule:

[
K\text{-dwell length} = \text{number of consecutive annual observations classified as }K
]

No gap tolerance is allowed in the primary analysis. A sensitivity analysis may allow a one-year excursion if immediately followed by a return to (K), but this must be labelled supplementary.

Release depth is measured as the subsequent maximum decline from the final (K)-year to the release trough within the next ten years:

[
D_{\Omega} = \max_{1 \leq h \leq 10} \left(P_{t_K} - P_{t_K+h}\right)
]

A parallel release-depth measure will be calculated for coupling:

[
D_{C} = \max_{1 \leq h \leq 10} \left(C_{x,t_K} - C_{x,t_K+h}\right)
]

**Primary statistical test:**
Pearson correlation between (K)-dwell length and subsequent release depth.

**Primary support criterion:**
The correlation between (K)-dwell length and (D_{\Omega}) must be positive in at least three of the four societies, with pooled reporting across all observed (K)-dwell events.

**Strong support criterion:**
Positive correlation for both (D_{\Omega}) and (D_C), with the effect present in at least three societies.

**Falsification criterion:**
No positive relationship between (K)-dwell length and subsequent release depth in the majority of societies.

---

### H3 — Revolt cascade test

If CAMS captures panarchic cross-scale revolt, then fresh node-level viability failures should occur within a pre-release window more often than expected by chance.

A revolt precursor is defined as:

[
V_{i,t} < 4
]

with:

[
V_{i,t-1} \geq 4
]

occurring within ten years before a system-level (\Omega)-release:

[
1 \leq t_{\Omega} - t_{\text{cross}} \leq 10
]

The primary revolt statistic for each society is:

[
R_{\text{obs}} =
\frac{
\text{number of } \Omega \text{ releases preceded by at least one fresh node crossing within 10 years}
}{
\text{number of } \Omega \text{ releases}
}
]

The result is then compared with a within-society shuffle null.

---

## 7. Null Model

For each society, release years are randomly permuted 500 times within that society's own observed annual time range.

The node-crossing series is held fixed.

For each permutation, compute:

[
R_{\text{null},k}
]

for (k = 1,\dots,500).

The empirical one-sided p-value is:

[
p =
\frac{
1 + #(R_{\text{null},k} \geq R_{\text{obs}})
}{
1 + 500
}
]

This null model tests whether observed fresh node crossings precede release years more often than expected under that society's own temporal structure.

---

## 8. Primary Decision Rules

### 8.1 No support for panarchic revolt

The panarchic revolt mechanism is not supported if fewer than two of the four societies show observed revolt-preceding frequency above the 95th percentile of the within-society shuffle null.

### 8.2 Partial panarchic support

The panarchic revolt mechanism receives partial support if at least two of the four societies satisfy:

[
R_{\text{obs}} > P_{95}(R_{\text{null}})
]

and the observed effect is substantively interpretable in historical sequence.

### 8.3 Strong panarchic support

The panarchic revolt mechanism receives strong support if at least three of the four societies satisfy:

[
R_{\text{obs}} > P_{95}(R_{\text{null}})
]

and the revolt sequence includes:

1. fresh node crossing,
2. subsequent bond weakening or (\lambda_2) decline,
3. system-level (\Omega)-release.

### 8.4 Falsification condition

If fresh node crossings do not precede releases above the shuffle-null expectation in at least two societies, the strong CAMS claim of cross-scale panarchic revolt is rejected for this dataset.

In that case, CAMS may still retain support for phase kinematics, conservation-trap dynamics, or stress-modulated institutional weakening, but not for the stronger nested-cascade interpretation.

---

## 9. Sensitivity Analyses

The following analyses are explicitly secondary and may not override the primary decision rules.

### 9.1 Revolt window sensitivity

Repeat the revolt test using:

[
7\text{-year window}
]

and

[
12\text{-year window}
]

The primary window remains ten years.

### 9.2 One-year (K)-dwell tolerance

Repeat the (K)-dwell analysis allowing a single non-(K) excursion inside an otherwise continuous (K)-run.

This will test whether brief classifier noise obscures longer conservation traps.

### 9.3 Bond recomputation sensitivity

If legacy bond columns pass the (BS_{\max} \leq 1.5) check, report results using both retained and recomputed bond strengths where feasible.

The primary result remains the version determined by the pre-check rule.

### 9.4 Envelope uncertainty

Where ensemble envelope files are available, repeat the phase and revolt calls at upper and lower envelope bounds.

Report whether primary findings are robust, weakened, or reversed under uncertainty.

Envelope analysis is uncertainty reporting only. It does not alter the pre-lodged primary decision rule.

---

## 10. Reporting Requirements

For each society, report:

1. years covered;
2. number of valid annual observations;
3. number of (r,K,\Omega,\alpha) years;
4. observed phase-transition matrix;
5. number and timing of (K)-dwell periods;
6. subsequent release depth after each (K)-dwell;
7. all fresh node crossings below (V_i < 4);
8. all system-level (\Omega)-release years;
9. observed revolt statistic (R_{\text{obs}});
10. shuffle-null distribution and percentile position;
11. whether the society meets no, partial, or strong support criteria.

The final report must explicitly distinguish:

* phase-kinematic support;
* conservation-trap support;
* revolt-cascade support;
* unsupported or falsified claims.

Any resolution-corrected change from earlier five-year results must be flagged as a "resolution flip".

---

## 11. Interpretation Rules

The experiment may support three different conclusions.

### Outcome A — Full panarchic support

If phase ordering, (K)-dwell release depth, and revolt-cascade tests all pass, CAMS supports a strong panarchic interpretation: societies behave as nested adaptive systems in which node-level failures can propagate into system-level release.

### Outcome B — Partial panarchic support

If phase ordering and (K)-dwell tests pass but the revolt cascade is inconsistent, CAMS supports Holling-style adaptive-cycle kinematics but not strong cross-scale revolt causality.

### Outcome C — Falsification of strong panarchy

If the revolt test fails across the majority of societies, the strong claim that fresh node failures predict system release is rejected for this annual dataset.

This does not invalidate CAMS as a phase-space or institutional-stress model. It narrows the claim: CAMS may describe adaptive-cycle movement without proving nested panarchic causation.

---

## 12. Pre-Lodged Commitment

No classifier thresholds, release definitions, viability thresholds, revolt windows, or bond-recalculation rules may be altered after inspection of the annual results for the primary analysis.

Any later changes must be labelled exploratory.

The central falsification commitment is:

> CAMS will only claim empirical support for panarchic revolt if fresh node-level viability crossings precede system-level release events above within-society shuffle-null expectation in at least two of the four annual society series.

This protocol therefore tests whether CAMS captures not merely historical patterning, but a measurable cross-scale mechanism of stress-modulated institutional release and reorganisation.
