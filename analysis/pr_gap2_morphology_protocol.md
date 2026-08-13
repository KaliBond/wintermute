# PR-GAP-2/3 External Rupture-Morphology Coding Protocol

*Draft — July 2026. For use by independent coders classifying the PR-GAP-2 rupture chronology prior to PR-GAP-3 replication.*

---

## 0. Purpose and scope

This protocol specifies how to classify each of the 126 PR-GAP-2 rupture events into one of five morphological types using **only externally sourced, non-CAMS indicators**. The output is a blinded external classification that will be compared, after coding is complete, against the CAMS-derived deficit-crisis / coordination-transition classification (Section 6) to test whether the AUC = 0.740 discrimination reported in PR-GAP-2 replicates under an independent rupture typology.

Coders must be able to complete classification using only the indicator sources listed in Section 3. No CAMS node scores, no P_Type or Rupture_Type labels, and no PR-GAP-2 analysis output of any kind may be consulted at any point during classification.

---

## 1. Data-access embargo

**This is a hard constraint on the coding process, not a suggestion.**

- Coders receive only `pr_gap2_ruptures.csv` (Society, Rupture_Year — the bare event list). The `Type` column in that file is a rough descriptive tag from the original source chronology (e.g., "coup," "civil_war") and may be shown to coders as a starting pointer to the event, but it is not a substitute for independent classification and must not be treated as the answer.
- `pr_gap2_ruptures_classified.csv` (contains CAMS P_Type) and `pr_gap2_panel_derived.csv` (contains node-level and derived CAMS variables) are **embargoed**. These files must not be opened, queried, or referenced by coders during classification.
- These two files should be stored outside the coders' working directory (e.g., held by the study coordinator, not committed to any branch or shared folder the coders have access to) until both coders have submitted final classifications and inter-rater agreement has been computed.
- If a coder is inadvertently exposed to either embargoed file before completing classification, that coder's classifications for all events must be discarded and redone by a fresh coder.
- The study coordinator (not the coders) performs the comparison in Section 6, after classification is locked.

---

## 2. The five morphological types

Each rupture event is assigned to exactly one type.

### 2.1 Degradation-driven breakdown

Institutional capacity declined over an extended period before the rupture; successor arrangements were absent or actively disputed at the moment of rupture; state functions contracted or fragmented rather than transferring intact.

- **Positive example — Germany, 1933.** The Nazi seizure of power followed roughly four years of Weimar institutional degradation (hyperinflation legacy, Depression-era unemployment, a succession of short-lived cabinets governing by emergency decree under Article 48). No stable successor arrangement existed independent of the rupture itself; the Enabling Act that followed was extracted under duress rather than reflecting a preexisting negotiated framework.
- **Positive example — Iraq, 2003.** The Ba'athist state had experienced over a decade of institutional contraction under sanctions (1990–2003) before the invasion. Successor governance was undefined at the moment of collapse and remained contested for years afterward; core administrative and security functions fragmented rather than transferring.
- **Negative example — Estonia, 1991.** Although preceded by decades of Soviet rule, the moment of independence involved a preexisting Estonian government-in-exile framework and rapid, largely intact establishment of successor institutions — this is a coordinated transition (2.2), not a degradation-driven breakdown, despite superficially "long buildup" optics.

### 2.2 Coordinated transition

Power transferred under preexisting constitutional or negotiated arrangements; successor institutions were in place before the rupture event; state capacity was largely preserved through the transition.

- **Positive example — Poland, 1989.** The Round Table Agreements negotiated a transition between the incumbent government and Solidarity, an organisation with preexisting internal governance structures, prior to the transfer of power. State administrative capacity continued largely uninterrupted.
- **Positive example — Estonia, 1991.** Independence was achieved through a recognised legal/diplomatic process with a preexisting government structure (continuity government-in-exile tradition) ready to assume authority; core state functions were established rapidly and with broad international recognition rather than through internal institutional collapse.
- **Negative example — Chile, 1973.** No negotiated framework existed; Allende's government was removed by force with no preexisting successor arrangement — this is a fast-onset elite seizure (2.3), not a coordinated transition.

### 2.3 Fast-onset elite seizure

Rupture executed in days to weeks; prior institutional degradation absent or minimal; the key mechanism was elite/military network action rather than structural collapse.

- **Positive example — Chile, 1973.** The coup was executed within a single day (11 September); while political polarisation preceded it, core state administrative capacity was intact immediately before the event. The mechanism was military elite action, not a multi-year erosion of institutional capacity.
- **Positive example — Thailand, 2014.** Consistent with Thailand's recurring pattern (also 1932, 2006), the military's seizure of power was executed within days of the declaration of martial law, with no extended prior degradation of administrative or judicial capacity.
- **Negative example — Iraq, 2003.** Although the military phase was fast, the institutional collapse was the product of a preceding decade-plus of sanctions-driven degradation and was primarily externally caused (see 2.4) — this is not a fast-onset elite seizure in the intended sense.

### 2.4 Externally imposed rupture

The rupture was primarily caused by external military or political force rather than internal institutional dynamics.

- **Positive example — Poland, 1939.** The German and Soviet invasions directly caused state collapse; the rupture originated entirely from external military action, not from internal institutional dynamics.
- **Positive example — Estonia, 1940.** Soviet occupation and annexation was imposed externally; internal institutional conditions were not the proximate cause.
- **Negative example — Egypt, 2011.** The uprising and Mubarak's departure were driven by internal mass mobilisation and elite defection, not external force — this is a degradation-driven breakdown or coordinated transition depending on further indicators, not an externally imposed rupture.

### 2.5 Mixed / indeterminate

The event does not clearly fit a single type, or the evidence available from external sources is insufficient to classify confidently.

- **Positive example — Russia, 1991.** The Soviet collapse simultaneously exhibits degradation-driven breakdown features at the federal/union level (prolonged economic and administrative decline through the 1980s, disputed successor arrangements for the union government) and coordinated-transition features at the level of several constituent republics (preexisting republican governing institutions, negotiated transfers in some cases). Coders should default to this category rather than forcing a single-level judgment.
- **Positive example — Egypt, 2011–2013.** The 2011 revolution and 2013 coup are coded as separate events in the chronology but are causally entangled (the 2013 event is partly a reversal of the 2011 transition). Evidence available externally is genuinely mixed on whether 2013 reflects elite seizure, degradation of the post-2011 transitional institutions, or both. Code as mixed/indeterminate unless the decision tree in Section 4 resolves it cleanly using that event's own indicator values.

---

## 3. Coding indicators (preregistered, observable without CAMS)

Coders record each indicator below for every event, using only the listed source classes. Record "insufficient evidence" rather than guessing if a source does not clearly resolve an indicator.

| # | Indicator | Operationalisation | Primary source(s) |
|---|---|---|---|
| I1 | Transition duration | Days from triggering event to institutional replacement (successor holding effective authority) | Cline Center event records |
| I2 | Constitutional continuity | Whether the new regime operated under a modified version of the prior constitutional framework, vs. wholesale replacement | V-Dem `v2exdfcbhs` and ERT episode data |
| I3 | Successor institutions preformed | Whether the successor government, party, or constitutional body existed and had functioning internal structure before the rupture | Historical records; V-Dem party/legislature data |
| I4 | Elite agreement | Whether a negotiated transfer or inter-elite agreement preceded the rupture, as opposed to unilateral force | Archival records; Polity IV notes |
| I5 | Foreign intervention | Whether external military or political force was the primary proximate cause | Cline Center; UCDP external support codes |
| I6 | State capacity continuity | Whether core administrative, judicial, and security functions continued after the rupture without major interruption | V-Dem `v2svstterr`; state fragility indices |
| I7 | Pre-rupture crisis duration | Number of years of documented political, economic, or social crisis preceding the rupture | Reinhart/Rogoff; UCDP conflict onset data |

Operational cut points used by the decision tree (Section 4):
- **Fast** transition duration: I1 ≤ 30 days. **Slow**: I1 > 30 days.
- **Extended** pre-rupture crisis: I7 ≥ 2 years of documented crisis. **Minimal**: I7 < 2 years.
- **High** state capacity continuity: I6 = core functions uninterrupted. **Low**: I6 = major interruption/fragmentation.

---

## 4. Classification decision tree

Preregistered before any event is coded. Apply in strict order; stop at the first branch that resolves.

```
Q1. Is I5 (foreign intervention) the primary proximate cause?
    YES → EXTERNALLY IMPOSED RUPTURE
    NO  → continue to Q2

Q2. Are I2 (constitutional continuity), I3 (successor institutions
    preformed), and I4 (elite agreement) ALL affirmative?
    YES → COORDINATED TRANSITION
    NO  → continue to Q3

Q3. Is I1 fast (≤30 days) AND I7 minimal (<2 years) AND
    I6 high (state capacity continuity maintained)?
    YES → FAST-ONSET ELITE SEIZURE
    NO  → continue to Q4

Q4. Is I7 extended (≥2 years) AND I6 low (major interruption/
    fragmentation) AND (I3 or I4 is negative — no preformed
    successor / no elite agreement)?
    YES → DEGRADATION-DRIVEN BREAKDOWN
    NO  → MIXED / INDETERMINATE
```

### Tie-breaking rule

If an event satisfies the criteria for more than one type (e.g., Q2 and Q3 both resolve YES on their respective indicators, or indicators genuinely conflict across sub-national levels as in Russia 1991), do not force a choice — assign **MIXED / INDETERMINATE**. If a coder is uncertain whether an indicator is affirmative or negative because sources conflict or are silent, record "insufficient evidence" for that indicator; if this makes two or more branches unresolvable, the event defaults to MIXED / INDETERMINATE. The decision tree does not permit coder discretion to override a branch outcome based on general impression.

---

## 5. Inter-rater agreement procedure

1. Two coders classify all 126 events independently, each applying Section 4 without consulting the other or any embargoed file.
2. Compute Cohen's κ across the two coders' five-way classifications for the full event set.
3. For events where the two coders disagree:
   a. Both coders jointly re-apply the decision tree, indicator by indicator, to identify which indicator value produced the divergent branch.
   b. If re-application under strict adherence to Section 4 resolves the disagreement, adopt the resolved classification.
   c. If disagreement persists after step (a), assign the event to **MIXED / INDETERMINATE**.
4. Report final Cohen's κ (pre-resolution, on the two independent classification sets) as the primary reliability statistic.
5. List all originally disagreeing events, their coder-assigned types, indicator values, and resolution path in an appendix table for qualitative review.

---

## 6. Comparison design (performed by study coordinator after coding is locked)

Only after both coders have submitted final classifications and Section 5 is complete does the coordinator unseal `pr_gap2_ruptures_classified.csv` and perform the following:

1. Cross-tabulate external morphology type (five categories) against CAMS P_Type (deficit_crisis / coordination_transition / unknown).
2. Report Cohen's κ specifically between the external "degradation-driven breakdown" category and CAMS "deficit_crisis" classification, treating this as the key convergent-validity test.
3. Re-run LOSO cross-validation (same specification as PR-GAP-2 Section 2.7) on the subset of events externally classified as degradation-driven breakdown, and compare the resulting AUC to the CAMS-classified deficit-crisis AUC of 0.740.
4. Identify and tabulate all disagreement cases (external type vs. CAMS P_Type) for qualitative review, with particular attention to fast-onset elite seizure and externally imposed rupture events, which the CAMS model is not expected to detect (per PR-GAP-2 Section 4.4 scope conditions).
5. Structure the interpretation section around the following five-outcome framework:

| Outcome | Definition |
|---|---|
| Strong agreement | External degradation-driven breakdown and CAMS deficit_crisis agree on ≥80% of jointly classifiable events, and externally-classified-only LOSO AUC is within 0.05 of 0.740 |
| Moderate agreement | 60–79% agreement, or externally-classified-only AUC within 0.05–0.10 of 0.740 |
| Weak — external predicts, CAMS does not add | External classification alone achieves comparable AUC; CAMS structural variables add negligible discrimination once external morphology is known |
| Weak — neither predicts well | Both external- and CAMS-classified subsets fail to exceed chance-level discrimination at conventional significance |
| CAMS outperforms external | CAMS-classified deficit-crisis AUC materially exceeds externally-classified-only AUC, suggesting CAMS captures degradation dynamics external categorical coding misses |

---

## 7. Deliverable

The completed protocol, applied output (both coders' independent classifications, agreement statistics, resolution table), and the Section 6 comparison constitute the PR-GAP-3 data package. This document alone, together with `pr_gap2_ruptures.csv`, must be sufficient to brief two new coders with no further instruction.
