# CAMS Lab Note — 2026-11-24

## Title
Close reading of van der Leeuw and Dirks (2025) against the CAMS–Shannon correspondence: additional insights, cautions, and one open audit item

**Status:** Working lab note — interpretive reading record, not pre-registered; no numeric tests run

**Sources read:**

1. van der Leeuw, S., & Dirks, G. (2025). Society building as collaborative selection of information-processing dimensions? *Global Perspectives*, 6(1), 143744. Read from the uploaded PDF (`globalperspectives_2025_6_1_143744.pdf`, UC Press open access, downloaded 24 November 2026). Full coverage: sections 1–9, Box 2, author biographies, references.
2. GPT (OpenAI), "How information theory informs CAMS" (audit text, 24 November 2026). Read from the uploaded copy (`user_pasted_clipboard_long_content_as_file_How information theo.txt`, 199 lines), complete.

**Checked against:** `CAMS_Informational_Perspective_Research_Note.md` (post-audit edition; KaliBond/wintermute `papers/`, commit `094c190`, blob `e3d26c6d`, 34,075 bytes, byte-identical local copy) and the Shannon Laboratory v2.2 (website version `260fa7b`).

**Canonical formulas:** Not exercised. This note runs no computation on any dataset and re-derives nothing; no canonical operator is altered or extended by anything below.

**Provenance:** Reading, mapping and prose by Kimi (Moonshot AI), commissioned by K. F. McKern, 24 November 2026. Epistemic status is labelled per item: **[paper]** the authors' own claim, quoted verbatim; **[mapping]** my correspondence onto CAMS; **[speculation]** conjecture that would require a registered test before any reliance.

## 1. Motivation

The research note cites van der Leeuw and Dirks (2025) once, in its section 2, for a single claim: societies "grow by adding dimensions along which they process information, and crises narrow that dimensionality." The paper carries considerably more than that one sentence uses. The purpose of this note is twofold: to put the additional material on record with its epistemic status separated, and to document the reading itself, so that the claim of a close read is checkable rather than asserted.

## 2. Method of reading (coverage record)

- Pass 1 (earlier the same day): the PDF's front matter through the opening of section 6, plus the references tail. That pass truncated mid-sentence in section 6, cutting off at "observations that have not (yet?)be…".
- Pass 2 (this session): the PDF re-read from page 4 through the end of the references, recovering the truncated sentence in full: "In our opinion, it is more productive to define 'noise' as observations that have not (yet?) been subsumed in a theoretical and cognitive perspective that gives them sense in the society that is confronted with them" (§6).
- Pass 3 (this session): the audit text read complete.
- Cross-check: a grep of the research note located every existing use of the paper (one in-text sentence in §2, one reference-list entry), confirming that the items in section 3 below are additions, not duplicates.

Section anchors used in this note (paper's own numbering):

- §1–§2 (titles not transcribed): subject–object non-separation; the observer as part of the observed; limits of human perception; categorisation preamble.
- §3 How do societies collectively create meaning? Covers habitus, mindscapes, "metadata".
- §4 Institution building as multilevel collaboration.
- §5 From groups to classes. Covers the Tversky and Gati open-to-closed transition; "tools for thought and action"; the death of collaborations.
- §6 Cognized and uncognized dimensions. Covers the noise redefinition; the metrics aphorism.
- §7 Questions concerning the collaborative process. Covers resonance; values; the four-dimension claim; Sievers et al. (in press); Dávid-Barrett and Dunbar (2013); von Rueden et al. (2014); the "logjam"; Box 2.
- §8 No collaboration or institution is forever.
- §9 Summary.

All quotations below were transcribed from the uploaded PDF. Quotations are verbatim, including the original punctuation.

## 3. Findings

### 3.1 Noise redefined: stress is relational to the category set

**[paper]** Section 6 rejects the Kahneman, Sibony and Sunstein (2021) framing of noise as measurement error: "We take a different view. In our opinion, it is more productive to define 'noise' as observations that have not (yet?) been subsumed in a theoretical and cognitive perspective that gives them sense in the society that is confronted with them." Information processing is "the process of transforming 'noise' into interpretable signals"; closing a category discards what does not fit "back into the category 'noise.'"

**[mapping]** This makes Stress relational. S is not a property of the world alone but of the world relative to the society's category set. A society whose categories have over-closed reclassifies disturbance as noise and stops registering it: measured stress can fall because the framework went blind, not because the world calmed. That is a third signature beside regression (C holds, A falls) and masking (A holds, C falls): apparent calm.

**[mapping]** The point converges with the audit from the opposite direction. The audit holds that no information measure on CAMS scores can separate adaptation from conformity without an outside referent series (harvests, prices, mortality, battle outcomes). Van der Leeuw and Dirks hold that the category system decides what counts as signal at all. Two independent arguments for the same instrument requirement.

### 3.2 Open-to-closed categories: abstraction as a lifecycle, not a level

**[paper]** The paper's core mechanism is Tversky and Gati's (1978) two-step: observations are first grouped into open, exploratory, similarity-based categories, then closed into defined classes. "The transition from open groups to closed classes is fundamental to the dynamic of collaboration" (§5). Closure "formalizes those ideas that are the acknowledged result of the collaboration, ensuring a transition from comprehension of socio-environmental dynamics to competency in dealing with them (without the need for comprehension)" (§5; footnoted as "the society's version of Kahneman's (2013) type 1 information processing"). In the summary: "the dimensionality of the categories is reduced, so that processing the information becomes cheaper in terms of energy" (§9).

**[mapping]** This is the note's information-bottleneck reading of A with a time axis added. Closure is compression; reopening is the expensive move a crisis demands. Crisis in this frame is the environment forcing category reopening faster than the society can pay the energetic cost. The coding fork acquires an energetic reading: under stress the cheap path is closure (repetition, redundancy, the regression route), and masking is the failure to close properly, complex output continuing through a channel that cannot carry it. The paper also states the bottleneck's selection criterion in plain words: what matters is which dimensions are dropped.

### 3.3 Institutions as frozen closures; decay endogenous to success

**[paper]** Closed categories are "instantiated and where possible anchored in particular 'tools for thought and action' (inventions) such as institutions, laws, technologies, implicit or explicit algorithms... intended to perdure beyond the period of active collaboration" (§5). But the discarded dimensions persist in the world: "The ideas not shared later become the seeds for institutional—and ultimately societal—instability. Thus, transient stability institutionalizes" (§5). And in §8: "The decay of institutions (technologies, etc.) is inherent in the dynamic that is responsible for creating them! Solutions create problems (van der Leeuw 2012)!"

**[mapping]** This supplies the content theory that the audit's memory result lacked. The audit showed that memory's worth equals the environment's predictable structure (a record worth 0.22 bits in a recurring environment, nothing in a memoryless one). Van der Leeuw and Dirks explain the mechanism: what Archive stores is closed categories, and closed categories decay exactly when the context they were compressed from shifts.

**[speculation]** The next crisis grows in the dimensions the last solution discarded. If so, the locus of future stress should be predictable from what a society's dominant framework currently ignores. Operationalising that would need a registered measure of "ignored dimensions," which is not obviously available from scored C/K/S/A; flagged as candidate material for a pre-registration, nothing more.

### 3.4 The authority echo has empirical anchors

**[paper]** Section 7 quotes Sievers et al. (in press, fMRI conversation experiments): "participants perceived as high in social status spoke more, gave more orders, and implicitly rejected others' proposals. Despite being rated more influential by their groups, these high-status participants were less neurally influential, raising the possibility that their conversation behaviors produced public compliance without private acceptance." By contrast, high-centrality participants "encouraged others to speak, producing more equal turn-taking," with greater neural alignment.

**[paper]** Two further anchors in the same section. Dávid-Barrett and Dunbar (2013), as summarised by the authors: "any tendency for a group to develop a resource (in our case information) asymmetry quickly leads to emergence of a more highly differentiated network in which elites emerge around a dominant individual." And von Rueden et al. (2014) on Tsimane' forager-horticulturalists: leaders' "ranks on physical dominance, kin support, and trustworthiness predict how well their groups perform, but only where group members have a history of collaborative interaction."

**[mapping]** The laboratory's robust post-audit finding is the authority echo: raising Helm's authority adds 0.161 bits of agreement beyond the shared environment while costing Helm 0.015 bits of world-knowledge (30 of 30 seeds), and silencing Helm's broadcast improves Helm's accuracy in every seed. Sievers et al. are that phenomenon observed in human groups: dominance raises surface compliance while lowering actual alignment. Von Rueden et al. add that executive effectiveness is conditional on pre-existing collaborative history, an empirical nod to the multiplicative bond structure. These are citation candidates for the note's authority-echo paragraph. Caveat: Sievers et al. is in press; the passages above are the paper's quotations, not verified against the Sievers manuscript.

### 3.5 A tension to name: the four-dimension claim

**[paper]** "Four dimensions are commonly the scope for most collaborations among individuals and are generally the limit for collective human thinking" (§7).

**[mapping]** If that holds, no real collective holds eight nodes at once, and the eight-node architecture is usable by groups only because LAPH chunks it into base, loop and pairs. This is the authors' assertion from practice, not a demonstrated result, and it sits in visible tension with the cognitive-limit argument for eight nodes. Named here per the anomalies-named-plainly rule. Its constructive reading: the layered architecture is load-bearing, not cosmetic.

### 3.6 Metrics become values: a reflexive hazard

**[paper]** "In fact, as the saying goes, 'we cannot measure what we like, so we like what we can measure'" (§6). Earlier, the paper notes that objectified, uncategorised phenomena are "poly-interpretable and in a sense no different from noise—signals waiting for an appropriate interpretative framework" (§2 region).

**[mapping]** The framework decides what is signal before any measurement begins, and the available metrics then become the values pursued. Applied reflexively: C, K, S and A are the analyst's closed categories, and any scored panel will drift toward optimising the measurable. This is the retrieval rule turned on CAMS itself. A single sentence in the note's hazards section would cover it.

### 3.7 The logjam and redundancy dominance

**[paper]** Mature organisational ecosystems converge on shared infrastructure until "such convergence often creates a 'logjam'—a situation in which the organizations are so entangled that it is impossible to initiate any major changes." The remedy is for organisations to "open up enough closed categories to adapt to the dynamics of the convergence" (§7).

**[mapping]** The audit found the toy redundancy-dominated at every setting (O-information +3.7 to +4.4 bits) and treated that as construction-forced, since weighted averaging copies information but cannot create joint-only information. Van der Leeuw and Dirks predict mature real systems trend the same way in fact: entangled, rigid, redundancy-heavy. So positive O-information on the panel could serve as a maturity or entanglement measure. In the toy it remains forced by construction and proves nothing; any such claim must run on data.

### 3.8 Open item carried over from the audit

**[mapping]** The audit's one suggestion touching a canonical operator was never scheduled: Φ = Σ KᵢSᵢ multiplies capacity by stress, while channel theory works with the margin (K − S) or the ratio (S/K); the audit proposed testing Θ = Φ/Ψ against a margin-based variant on the panel. Recorded here so it is not lost. Note the standing red line on within-node recombinations (FC3, σ, PAS null): such a variant is description by default and would need gate conditions and holdout before any predictive claim.

## 4. Interpretation

Nothing above overturns anything in the research note or the laboratory. The additions are of three kinds. First, sharpenings of positions the note already holds: the relational definition of noise (3.1) strengthens the case for an external referent series, and the category-lifecycle mechanism (3.2, 3.3) gives the audit's memory result its missing content theory. Second, empirical anchors: Sievers et al., Dávid-Barrett and Dunbar, and von Rueden et al. (3.4) give the authority echo a human-subjects literature to sit beside the toy. Third, cautions: the four-dimension claim (3.5) and the metrics aphorism (3.6) cut against careless use of the framework itself.

If any of this is folded into the note, the natural edits are small: three citations in the authority-echo paragraph, one sentence in the hazards section, and the "apparent calm" signature named beside the masking fork. None is urgent; the note stands without them.

## 5. Limitations and cautions

- The paper's quantum-mechanical material (decoherence, Everett, subject–object analogies) is analogy, not mechanism. It is recorded here as read but is not cited as support for anything.
- Box 2's settlers/pioneers/navigators/anarchists typology is practitioner lore drawn from the business literature; not load-bearing.
- The four-dimension claim (3.5) and much of §7 rest on the authors' practical experience, explicitly offered as "a space of investigation," not as demonstrated results.
- Sievers et al. is in press; quoted passages are the paper's own quotations.
- No numeric tests were run for this note. Items labelled [mapping] are interpretive correspondences; the one item labelled [speculation] (3.3, predicting crisis locus from ignored dimensions) has no operationalisation yet.
- The audit text was re-read for completeness; its substantive content was already incorporated into the note and laboratory v2.2 earlier on 24 November 2026, except the open item at 3.8.

## 6. Addenda

None as of first issue.

---

**Dated 24 November 2026, Australia/Sydney (AEDT).** Reading record and prose by Kimi (Moonshot AI). Sources: uploaded PDF of van der Leeuw and Dirks (2025) and uploaded audit text, both read in full as documented in section 2. This note is a reading record, not a tested result; nothing in it modifies canonical operators or the research note.
