# Zenodo Deposit Manifest

One record per artefact, in the order they should be deposited. Each
section is metadata to paste into a new Zenodo upload — title, description,
upload type, keywords, licence, and how it should link to the others via
`related_identifiers`. None of these have been deposited yet; DOIs below are
literally `PENDING` until each record is actually created.

Every deposit should relate back to the concept DOI of this repository
(`isSupplementTo` or `isDerivedFrom`, once minted) and to
`https://neuralnations.org` (`isSupplementTo`).

---

## 1. Formal Paper I — Introducing CAMS

**File:** `CAMS_Paper1_Introducing_CAMS.pdf`

**Title:** Human Societies as Complex Adaptive Meta-Systems: Introducing the CAMS Framework

**Description (Zenodo, ~180 words):**
> This paper introduces CAMS, a framework for analysing human societies as
> complex adaptive systems composed of eight recurring functional nodes
> (governance, security, knowledge, resource stewardship, production,
> labour, memory/legitimacy, and exchange), each scored annually on
> coherence, capacity, stress, and abstraction. Node scores combine into a
> node-viability metric and an inter-node bond-strength measure, giving a
> longitudinal structural signature for a society that can be compared
> across historical periods and across different societies. The paper sets
> out the founding premise — that societal crisis and adaptation leave
> recoverable structural traces — introduces five longitudinal validation
> datasets, and states six falsification criteria the framework commits to
> in advance. It is the first in a four-paper sequence; later papers extend
> the functional taxonomy, formalise the coordination mathematics, and test
> specific collapse signatures. Intended audience: researchers in
> comparative historical sociology, complex systems, and computational
> social science who have not previously encountered CAMS.

**Upload type:** publication / preprint
**Keywords:** complex adaptive systems, civilisational analysis, institutional coordination, computational social science, CAMS
**Licence:** CC-BY-4.0
**Related identifiers:** `isPartOf`/sequence-linked to Papers II–IV below; `isSupplementTo` this repository's concept DOI and neuralnations.org.

---

## 2. Formal Paper II — Functional Taxonomy

**File:** `CAMS_Paper2_Functional_Taxonomy.pdf`

**Title:** A Functional Taxonomy of Human Social Organisation

**Description (Zenodo, ~170 words):**
> This paper extends the CAMS framework's eight-node architecture into a
> functional taxonomy of human social organisation, arguing that the nodes
> represent recurring coordination problems every sufficiently complex
> society must solve, rather than institutions specific to any one culture
> or era. It examines how geography and material constraint shape which
> node carries the greatest coordination burden in a given society, drawing
> on a panel of 229 polities. The taxonomy is presented as a classification
> tool independent of any predictive claim: it places a society within a
> structural family without asserting what will happen to it next. Intended
> audience: comparative politics and historical-sociology researchers
> interested in cross-cultural institutional classification.

**Upload type:** publication / preprint
**Keywords:** functional taxonomy, institutional coordination, comparative politics, CAMS
**Licence:** CC-BY-4.0
**Related identifiers:** sequence-linked to Papers I, III, IV; `isSupplementTo` this repository and neuralnations.org.

---

## 3. Formal Paper III — Coordination Formalism

**File:** `CAMS_Paper3_EES_Coordination_Formalism.pdf`

**Title:** CAMS and the Extended Evolutionary Synthesis

**Description (Zenodo, ~170 words):**
> This paper formalises the CAMS coordination model in thermodynamic terms,
> situating it relative to the Extended Evolutionary Synthesis in
> biology. It develops phase-space diagnostics for a society's coordination
> state and derives the mathematical conditions under which the model
> predicts a coordination phase transition. This is the most mathematically
> technical paper in the four-paper sequence and is intended for readers
> already familiar with Paper I's node architecture who want the full
> operator derivations rather than the applied results in Papers II and IV.
> Intended audience: complexity scientists and theoretical biologists
> interested in coordination-dynamics formalisms transferable across
> domains.

**Upload type:** publication / preprint
**Keywords:** coordination formalism, extended evolutionary synthesis, phase-space diagnostics, thermodynamics, CAMS
**Licence:** CC-BY-4.0
**Related identifiers:** sequence-linked to Papers I, II, IV; `isSupplementTo` this repository and neuralnations.org.

---

## 4. Formal Paper IV — Entropy Flows and Collapse Signatures

**File:** `CAMS_Paper4_Entropy_Flows_Collapse_Signatures.pdf`

**Title:** Entropy Flows and Collapse Signatures

**Description (Zenodo, ~160 words):**
> This paper applies the CAMS coordination formalism to three historical
> cases — Germany 1910–1930, the late Western Roman Empire (380–450 CE),
> and the United Kingdom as a control case — to identify a specific
> early-warning signature: a breach of the "Stewards" (resource-stewardship)
> node's bond-strength floor preceding structural collapse in the crisis
> cases but not in the control. It is the most applied and narrative of the
> four papers, intended as a worked demonstration of what the formalism in
> Paper III detects in practice. Intended audience: historians and
> political scientists interested in early-warning indicators for
> institutional collapse, and readers who want a concrete before/after
> account of the framework rather than its abstract derivation.

**Upload type:** publication / preprint
**Keywords:** collapse signatures, entropy flows, early warning, institutional coordination, CAMS
**Licence:** CC-BY-4.0
**Related identifiers:** sequence-linked to Papers I–III; `isSupplementTo` this repository and neuralnations.org.

---

## 5. Working Papers

**Scope:** the non-formal-sequence research output — PR-GAP-2 (rupture
morphology across 44 societies), the Operator Portability synthesis, the
CAMS Blind Experiment Report, and related working papers under `analysis/`
and `research/`.

**Title:** CAMS/JUNO Working Papers Collection

**Description (Zenodo, ~190 words):**
> This is a collection of working papers extending and stress-testing the
> CAMS/JUNO framework beyond its four formal papers: retrospective
> validation of the framework as a leading indicator of political rupture
> across a 44-society, 125-year panel (PR-GAP-2); a synthesis of three
> operator-portability studies testing whether the framework's core
> operators are structurally portable across corporate and national
> domains; and a blind-identification experiment in which the framework's
> outputs, stripped of society labels, were tested for whether they could
> still be correctly classified. **Flagged: this collection spans papers at
> different stages of completion and internal review** — some have been
> through independent computational audit, at least one companion analysis
> (η_soil / Norway) is explicitly paused pending rework, and none has been
> through external peer review. This should be stated on the Zenodo record
> itself, not left implicit. Intended audience: readers who want to see the
> framework's stress-testing process, not only its finished claims.

**Upload type:** publication / working paper
**Keywords:** rupture morphology, operator portability, blind validation, CAMS, JUNO
**Licence:** CC-BY-4.0
**Related identifiers:** `isSupplementTo` this repository's concept DOI; `references` the four formal papers above.

---

## 6. The Dataset

**Resolved 2026-08-20.** The author confirmed the intended deposit by
pointing to `neuralnations.org/datasets`, which describes the collection as
"41,519 CAMS records across 43 societies and **50 historical series**" —
i.e. "the 50-society dataset" was shorthand for the 50-*series* count,
referring to `cleaned_datasets/`, not the separate JUNO unified panel
(`juno/JUNO_Unified_Dataset.csv`, 48 societies / 43,400 records — a
different, newer collection). Note "series" ≠ "societies" here: several
societies have more than one series (different scorer, different time
window, or ensemble-mean vs envelope), which is why 50 series spans only
43 societies.

**Title:** CAMS Cleaned Dataset Collection — 43 Societies, 50 Historical Series

**Description (Zenodo, ~180 words):**
> This dataset collection contains 41,519 CAMS node-score records across
> 50 historical series spanning 43 societies. Each series records one of
> eight functional nodes (Helm, Shield, Lore, Archive, Stewards, Craft,
> Hands, Flow) per year, scored on four dimensions — Coherence, Capacity,
> Stress, Abstraction — by a multi-model LLM ensemble (GPT, Grok, Gemini).
> Several societies have more than one series in this collection (distinct
> scorer runs, time windows, or ensemble-mean vs envelope/uncertainty
> pairs), which is why 50 series covers 43 rather than 50 societies. Node
> Value and Bond Strength are derived quantities computed from the raw
> scores. This is a research dataset produced by AI-assisted historical and
> institutional scoring, not primary archival data, and should be
> understood and cited as such. Intended audience: researchers wanting to
> reuse, replicate, or extend the CAMS scoring corpus.

**Upload type:** dataset
**Keywords:** node scores, longitudinal panel, institutional coordination, CAMS
**Licence:** CC-BY-4.0
**Related identifiers:** `isSupplementTo` this repository and
`neuralnations.org/datasets`; `isReferencedBy` the working papers and
formal papers that use it. Consider a separate, later deposit for the
newer JUNO unified panel (48 societies / 43,400 records) once it has its
own stable release point, rather than conflating the two collections.

---

## 7. Scoring Rubrics

**Files:** `juno/JUNO_v1.2-Final_Formalism.md` (locked operator spec),
`cams-diy-kit/` (the public scorer prompt/instructions distribution).

**Title:** CAMS/JUNO Scoring Protocol and Rubric (v1.2-Final)

**Description (Zenodo, ~180 words):**
> This deposit contains the scoring rubric and protocol used to produce
> CAMS/JUNO node scores: the exact prompt given to language-model scorers,
> the node definitions and scoring conventions (1–10 scale, positive-only
> stress), the locked mathematical operators (node viability, bond
> strength, algebraic connectivity), the six-regime classifier with its
> exact thresholds and evaluation order, and the numerical-precision policy
> that prevents floating-point round-trips from silently changing a
> society-year's assigned regime. Depositing this separately from the
> dataset itself is deliberate: it lets a reader or replicator verify
> exactly how a score was produced, independent of any specific scoring
> run, and lets the protocol be cited on its own when critiquing or
> extending the method rather than the data.
> Intended audience: anyone attempting to reproduce, audit, or extend
> CAMS/JUNO scoring.

**Upload type:** publication / other (methodology)
**Keywords:** scoring protocol, rubric, reproducibility, CAMS, JUNO
**Licence:** CC-BY-4.0
**Related identifiers:** `isSupplementTo` this repository; `isReferencedBy` the dataset deposit above.

---

## 8. Code Release

**Scope:** this repository itself, tagged `v1.0.0` (see the unpublished
release draft). This is not a separate manual Zenodo upload — publishing
the GitHub release triggers the existing GitHub–Zenodo integration, which
archives the repository and mints the **concept DOI** that every other
deposit above should reference.

**Title:** (from `CITATION.cff` / `.zenodo.json` — kept identical to both,
per the instruction that they must not contradict each other)

**Upload type:** software
**Keywords:** (as in `.zenodo.json`)
**Licence:** MIT (code) — note this differs from the CC-BY-4.0 covering
every other deposit in this manifest; the Zenodo record for the code
release should state MIT explicitly rather than inherit CC-BY-4.0 by
default.
**Related identifiers:** every deposit above should reference this one as
`isSupplementTo`; this one should not need to reference them individually
(the repository is the root of the citation graph).
