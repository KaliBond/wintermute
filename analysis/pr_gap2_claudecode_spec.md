# PR-GAP-2 Revision Spec for Claude Code

## Context

PR-GAP-2 is a cross-national CAMS panel study. The manuscript is at `analysis/pr_gap2_manuscript_draft.md`. All data CSVs are in `analysis/`. The CLAUDE.md in the repo root describes the CAMS framework and data shapes.

The study has received a methodological review. The review accepted the study's core findings but identified two required changes:

1. **Framing revision** (do now): The Abstract, title, and contemporary-warning language overstate the result as prospective prediction. They need to be revised to correctly characterise the study as a retrospective validation study with preliminary discrimination results.

2. **Coding protocol** (draft now, execute later): The 0.740 LOSO AUC is promising but not yet independently validated because the deficit-crisis outcome is defined using CAMS-derived variables (sign of expressed cognition P). The decisive next step is to classify the 126 rupture events using external, non-CAMS criteria, blinded to CAMS values.

Both tasks are described below. Do them in order.

---

## Task 1: Manuscript framing revision

Read `analysis/pr_gap2_manuscript_draft.md` in full before making any changes.

### 1a. Title

Current title (approximately): "Structural Coordination and Societal Rupture: A Cross-National Panel Study Using CAMS (PR-GAP-2)"

Replace with language that signals retrospective validation and construct stability rather than prospective prediction. Something like:

> **Structural Coordination and Rupture Morphology: A Retrospective Panel Validation of CAMS Across 44 Societies, 1900–2025 (PR-GAP-2)**

The key shift: "prediction" → "validation"; "rupture" alone → "rupture morphology" (signals the morphology-discrimination finding rather than a general early-warning claim).

### 1b. Abstract

The Abstract must be rewritten to reflect the three-layer framing agreed with the reviewer:

**Layer 1 — Construct stability:** Panel-wide T1/T2 results confirm that CAMS stress is consistently associated with reduced expressed cognition (41/41 evaluable societies after excluding a defective India series) and reduced canonical bond strength (42/42 societies). Because these quantities share components within the JUNO operators, this result should be interpreted as evidence of cross-context construct invariance rather than independent causal identification.

**Layer 2 — Correctly null pooled result:** For the undifferentiated rupture outcome (all 126 coded events), the model discriminates pre-rupture from non-rupture years only marginally (AUC = 0.563, p = .087). This null result is substantively important: it confirms that rupture is not a unitary structural phenomenon and that CAMS does not generically predict coups, civil wars, independence events, and financial crises under a single model.

**Layer 3 — Promising but not yet independent discrimination:** Classifying ruptures by the sign of pre-event expressed cognition (76 deficit crises, 41 coordination transitions) and restricting the outcome to deficit crises yields AUC = 0.740 (t(28) = 7.39, p < .001, 26/29 LOSO folds positive). This result is promising but not yet independently validated: the rupture subtype is defined using CAMS-derived information, creating potential leakage between predictor and outcome. The essential next step is to replicate this discrimination using an external, CAMS-blinded rupture-morphology classification.

The Abstract should end with something like:

> These results establish PR-GAP-2 as a substantial retrospective validation study supporting the CAMS framework as a promising instrument for distinguishing degradation-driven institutional collapse from coordinated political reorganisation. Independent validation of the morphology discrimination requires an externally classified rupture taxonomy, which is the subject of the planned PR-GAP-3 study.

Keep the Abstract under 300 words.

### 1c. Contemporary warnings section (Discussion 4.7 or wherever it appears)

The manuscript currently describes South Africa, Venezuela, and the United States as being at historical stress maxima and uses language that implies elevated crisis probability. This must be revised.

Replace any language implying crisis probability with "structural alert" framing. The specific change:

- Remove any implication that high stress = elevated near-term crisis probability
- Add a clear statement that calibrated probabilities, false-alarm rates, and detection thresholds are not yet available from this study
- Use language like: "These readings represent structural alerts — configurations in which the CAMS framework has historically been associated with higher rates of deficit crisis — not calibrated near-term collapse probabilities."
- Add a sentence noting that societies at historical stress maxima have historically resolved their conditions through institutional reform as well as through rupture, and that the model cannot determine which path a given society will take

---

## Task 2: External rupture-morphology coding protocol

After completing Task 1, create a new file `analysis/pr_gap2_morphology_protocol.md`.

This document specifies how to independently classify the 126 PR-GAP-2 rupture events by morphological type, using only external (non-CAMS) criteria, for use in PR-GAP-3.

### 2a. Purpose

The protocol must produce a blinded external classification of each rupture event along a primary dimension (degradation-driven breakdown vs. coordinated transition vs. fast-onset elite seizure vs. externally imposed rupture vs. mixed/indeterminate) without access to CAMS variable values at any point in the classification process.

### 2b. Primary classification dimensions

The protocol should operationalise the following five types. Write a clear definition and at least two positive and one negative example for each:

1. **Degradation-driven breakdown** — Institutional capacity declined over an extended period before the rupture event; successor arrangements were absent or disputed; state functions contracted or fragmented
2. **Coordinated transition** — Power transferred under preexisting constitutional or negotiated arrangements; successor institutions were in place before the rupture; state capacity was largely preserved through the event
3. **Fast-onset elite seizure** — Rupture was executed in days to weeks; prior institutional degradation was absent or minimal; the key mechanism was elite network action rather than structural collapse
4. **Externally imposed rupture** — The rupture was primarily caused by external military or political force rather than internal institutional dynamics
5. **Mixed / indeterminate** — The event does not clearly fit a single type, or evidence is insufficient

### 2c. Coding indicators (preregistered, observable without CAMS)

For each event, coders record the following indicators from external sources only. List sources alongside each indicator:

| Indicator | Operationalization | Primary source |
|---|---|---|
| Transition duration | Days from triggering event to institutional replacement | Cline Center event records |
| Constitutional continuity | Whether the new regime operated under a modified version of prior constitutional framework | V-Dem `v2exdfcbhs` and episode data |
| Successor institutions preformed | Whether the successor government, party, or constitutional body existed before the rupture | Historical records; V-Dem party/legislature data |
| Elite agreement | Whether a negotiated transfer or inter-elite agreement preceded the rupture | Archival records; Polity IV notes |
| Foreign intervention | Whether external military or political force was the primary proximate cause | Cline Center; UCDP external support codes |
| State capacity continuity | Whether core administrative, judicial, and security functions continued after the rupture without major interruption | V-Dem `v2svstterr`, state fragility indices |
| Pre-rupture crisis duration | Number of years of documented political, economic, or social crisis preceding the rupture | Reinhart/Rogoff; UCDP conflict onset |

### 2d. Classification rules

Write a decision tree using the above indicators that assigns each event to one of the five types. The decision tree must be:
- Fully specified before coders see any event (preregistered)
- Operable without reference to CAMS data
- Accompanied by a tie-breaking rule for ambiguous cases

### 2e. Inter-rater agreement procedure

Two coders classify all 126 events independently. Disagreements are resolved by:
1. Applying the decision tree strictly
2. If still unresolved, assigning the event to "mixed / indeterminate"
Cohen's κ should be reported. Events with κ-level disagreement should be listed for qualitative adjudication.

### 2f. Comparison design

After external classification is complete, the protocol should specify how to compare it to the CAMS-derived P-type classification:

- Cross-tabulate external morphology type against CAMS P_Type (deficit_crisis / coordination_transition / unknown)
- Report Cohen's κ between external "degradation-driven" and CAMS "deficit_crisis"
- Run LOSO separately on externally classified degradation events and compare AUC to the CAMS-classified 0.740
- Identify and list disagreement cases for qualitative review
- The reviewer's five-outcome comparison table (strong agreement / moderate / weak-external-predicts / weak-neither / CAMS-outperforms) should structure the interpretation section

### 2g. Data sources available in this repo

The following files are available for reference when writing the protocol:
- `analysis/pr_gap2_ruptures.csv` — 126 events with Society, Rupture_Year, Type
- `analysis/pr_gap2_ruptures_classified.csv` — same events with CAMS P_Type; **coders must not see this file**
- `analysis/pr_gap2_panel_derived.csv` — society-year panel; **coders must not see this file**

The protocol document should include a data-access instruction: the CAMS classification files are embargoed from coders during the external classification phase and should be stored separately.

---

## Success criteria

Task 1 is complete when:
- The title no longer contains prediction-language
- The Abstract uses the three-layer framing and ends with the PR-GAP-3 forward reference
- The contemporary-warnings language uses "structural alert" and explicitly disclaims calibrated probabilities
- No other sections of the manuscript have been changed

Task 2 is complete when:
- `analysis/pr_gap2_morphology_protocol.md` exists
- It contains definitions, examples, and coding indicators for all five morphological types
- The decision tree is fully specified
- The inter-rater agreement procedure is described
- The comparison design section matches the reviewer's five-outcome framework
- The document can be handed to two independent coders with no further instruction

---

## Do not do

- Do not rerun any analysis or modify any CSV files
- Do not revise Results or Discussion sections beyond the contemporary-warning framing fix
- Do not add new citations
- Do not change the Methods section
