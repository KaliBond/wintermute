# Stress × Flow interaction — test report

*22 September 2026 — K. McKern / ComplexityWorkz*

## Question

Does Flow Capacity moderate the effect of Stress on subsequent Coherence change?
Hypothesis as stated in the essay: "the association between high Stress and
subsequent loss of Coherence should weaken as Flow Capacity rises."

## Data

`JUNO_Unified_Dataset.csv` — 36 societies, 32,361 node-year rows. After
aggregating to society-years (mean over 8 nodes for Stress and Coherence; Flow
node Capacity for relief), 4,047 society-years; 3,705 with a successor year
(h = 1), 3,751 with a year + 5.

## Models

- Signed change: `ΔC = C(t+h) − C(t) ~ Stress(t) × K_flow(t)`, pooled and with
  society fixed effects; SEs clustered by society.
- Volatility: `|ΔC| ~ Stress(t) × K_flow(t)` (relief should shrink swings in
  either direction).
- Horizons h = 1 and h = 5 years.

## Results

**Signed change.** The naive hypothesis fails in sign, decisively:
high-Stress society-years are followed by Coherence *gains*, not losses
(β_Stress = +0.076/yr pooled, p < 0.001; +0.056 with society FE). The panel is
mean-reverting at annual scale — stress spikes are scored as resolved, and/or
carry-forward smoothing imposes oscillation. The interaction is **negative and
significant at h = 1** (β = −0.008, p = 0.007 pooled; p = 0.019 with FE):
the Stress→Coherence coupling is roughly halved between the 25th and 75th
percentile of Flow Capacity (simple slope 0.034 → 0.016 pooled; 0.019 → 0.003
with FE). At h = 5 the interaction is negative but not robust under fixed
effects (p = 0.28).

**Volatility.** Stress raises the magnitude of subsequent Coherence swings
(β = +0.07 to +0.30 across specifications, all p < 0.01), and high Flow
Capacity damps that effect — interaction negative, significant at h = 5
(β = −0.019, p = 0.010 FE; p = 0.004 pooled). Not significant at h = 1 (p ≈ 0.3–0.9).

## Interpretation

1. The literal claim (stress → coherence *loss*, cushioned by flow) is not
   supported. On average, stress is followed by recovery.
2. The deeper claim — that Flow is a pressure-relief organ — is supported in a
   revised form: **Flow Capacity decouples Stress from subsequent Coherence
   dynamics.** Where flow is open, stress leaves little footprint; where flow is
   blocked, stress propagates into visible coherence movement. This is the
   Calhoun–cage signature in panel form: same load, different coupling,
   depending on exits.
3. The volatility result is the cleaner match to the relief story: stress makes
   societies turbulent; flow throttles the turbulence, significantly over a
   5-year horizon.

## Caveats

- Scores are partly carry-forward smoothed (see per-society scoring notes);
  consecutive society-years are not independent. Clustered SEs are reported,
  but effective sample size is smaller than n suggests.
- Observational association; no causal identification. Reverse paths (coherence
  breakdowns raising measured stress; scorers resolving stress retrospectively)
  are plausible.
- Society fixed effects absorb level differences; results describe within-society
  dynamics.

## Reproduce

```
python stress_flow_interaction.py
```

Outputs: `stress_flow_interaction_results.csv` (all coefficients),
`stress_flow_interaction.png` (raw data with binned trends by Flow tercile).


---

## Addendum — 25 September 2026: carry-forward audit and de-smoothed replication

**Why.** The blind-panel audit (see `audit_supported_conclusions_report.md`)
showed the 12-society blind motif data carry scores forward on only 0–19.7% of
society-years, while the unified panel is heavily carry-forward smoothed in
places. The interaction results above were therefore re-tested on the clean
36-society unified panel (`JUNO_Unified_Dataset.csv`, 4,047 society-years)
with a per-society carry-forward audit, so that the essay's tested claim and
the blind-panel claims rest on equally examined data.

**Carry-forward audit** (fraction of society-years identical to the previous
year across all 8 nodes × C,K,S,A; flag at >30%):

| Society | repeat % | Society | repeat % |
|---|---|---|---|
| Israel | 87.5 | Syria | 53.8 |
| Lebanon | 86.7 | Saudi Arabia | 44.9 |
| Indonesia | 85.9 | Japan | 44.7 |
| Pakistan | 82.3 | Norway | 36.1 |
| Italy | 79.0 | Poland | 34.2 |
| Denmark | 72.8 | Australia | 34.2 |
| Iraq | 70.3 | France | 33.9 |
| Hong Kong | 68.1 | | |

Fifteen of 36 societies exceed the flag. Dropping every carry-forward repeat
society-year removes 1,200 of 4,047 society-years (30%), leaving 2,847
("no_repeat" rows in `stress_flow_interaction_results.csv`). The de-smoothed
panel is dominated by the 21 cleaner societies; for the flagged societies the
time-series variation is thin, which is itself a finding about where the
instrument is currently least informative.

**Full vs de-smoothed comparison** (interaction β, p):

| Test | full | no_repeat |
|---|---|---|
| Signed h = 1, pooled | −0.0083, p = 0.007 | −0.0129, p = 0.0001 |
| Signed h = 1, society FE | −0.0073, p = 0.019 | −0.0126, p = 0.0006 |
| Signed h = 5, society FE | −0.0133, p = 0.277 | −0.0177, p = 0.052 |
| Volatility h = 5, society FE | −0.0192, p = 0.010 | −0.0204, p = 0.0003 |

Simple-slope coupling (Stress effect on ΔCoherence at fixed Flow percentile,
society FE, h = 1): full panel 0.019 (p25) → 0.003 (p75); de-smoothed
0.033 → 0.003.

**What changed and what did not.**

1. The decoupling result **replicates and strengthens** on the de-smoothed
   panel at h = 1. The essay's claim is more defensible, not less.
2. The rejected sign **persists**: the Stress main effect on subsequent
   Coherence change remains positive everywhere (+0.11/yr pooled, de-smoothed
   h = 1). The literal "stress → loss" hypothesis stays rejected.
3. The volatility damping at h = 5 strengthens (p = 0.010 → 0.0003, FE).
4. The signed h = 5 interaction remains **not robust under fixed effects**
   (p = 0.28 full, 0.052 de-smoothed). Do not claim a five-year decoupling in
   signed terms; claim it only for volatility.
5. Carry-forward smoothing was *damping* the estimated interaction, not
   creating it — reassuring against the artefact worry, though with 15 flagged
   societies the audit flags exactly where re-scoring would most improve the
   panel.
