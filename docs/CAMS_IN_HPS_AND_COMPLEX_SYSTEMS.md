# CAMS in the History and Philosophy of Science and of Complex Systems

**Kari McKern** · Neural Nations / ComplexityWorkz · Sydney · Draft of 5 November 2026

> **Status of this draft.** Drafted by Claude (Sonnet 5.5) from the repository `KaliBond/wintermute` and from the brief set out below, for the author to revise. It reports **no new analysis**. Every empirical figure quoted is quoted from a file or commit in the repository, with its path or short SHA, and none has been re-run for this paper. Every personality, origin and cognition claim is a **hypothesis with a stated defeat condition**, not a result. Citations marked **[verified]** were checked against bibliographic records in this session; those marked **[not re-verified]** are given from memory and need checking before submission.

---

## Abstract

CAMS reads a meta-human system through a fixed instrument: eight functions (the Paladins: Helm, Shield, Lore, Archive, Craft, Flow, Stewards, Hands), each scored on four metrics (Coherence C, Capacity K, Stress S, Abstraction A). The eight-by-four array is the state. What the instrument measures is the arrangement, that is, where each function sits relative to the other seven. This paper places that instrument in the history of the sciences it borrows from: biological taxonomy and Darwin's descent-based classification; Popper, Kuhn and Lakatos on how theories are held and lost; Prigogine on structure maintained by dissipation; Holling's adaptive cycle and panarchy; the Santa Fe programme on complex adaptive systems; cybernetics; Ibn Khaldun; and Turchin's cliodynamics.

Against that background the paper develops four ideas. (1) **Social cognition emerges per node.** Define *focus* F = A × C and *available energy* E = K − S. When S > K, E is negative, the node is dissipating faster than it can maintain itself, and its signed activation is negative: it reads everything negatively. (2) **Every meta-human system has a personality**, namely the persistent profile of its eight (F, E) pairs and their coupling, which is what is informally recognised as national character. (3) **Personality has thermodynamic origins**: lineages that descend from mobile, place-independent forms differ from lineages bound to river systems and geography in which nodes carry the load. (4) **Lore is the collective operating system**, and mythology is node-level Lore instantiated in individual brains.

Ideas (2) and (3) are offered as hypotheses, each with conditions under which it would be abandoned. The paper also sets out what the repository's own audits already count against these ideas.

---

## 1. The fixed object

The instrument is not under discussion in this paper. It is the given.

- **Eight functions** (Paladins): Helm (direction), Shield (boundary and coercion), Lore (meaning), Archive (memory), Craft (technical reproduction), Flow (circulation), Stewards (allocation of assets), Hands (applied labour). Functional glosses follow `functional_taxonomy.html` and `backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`.
- **Four metrics** per function, per period: Coherence (internal alignment and coordination clarity), Capacity (demonstrated ability to perform the function), Stress (rate of entropy production within the function, not the severity of an external challenge) and Abstraction (the degree to which symbolic activity is operationally integrated with the function). Definitions follow `canonical-definitions.html` (20 October 2026), which takes precedence over earlier wording.
- **The state** of a meta-system at time *t* is the array X(t) ∈ ℝ^(8×4). Aggregates exist (mean node value, mean bond strength), but the claim of the instrument is that two systems can share an aggregate while differing in arrangement, and that the arrangement carries the information (`backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, section 2).

"Relative position" is the operative phrase. A node's score means little alone; it means something against the other seven. Throughout, a claim about a node is a claim about its place in the configuration.

I do not discuss other decompositions here. The instrument's history, including its earlier node sets, is in `docs/CAMS_LAB_DIARY.md`, section 13.

---

## 2. Where this sits: a short intellectual genealogy

### 2.1 Taxonomy: from Linnaeus to Darwin, and the question of descent

A classification is a theory about what matters. Linnaeus (1735) arranged organisms by shared characters in nested ranks, a scheme that needed no theory of how the pattern arose. Darwin (1859, chapter 13) argued that the natural system is genealogical: groups are nested because of descent with modification, and the characters we weight are the ones that track ancestry. The later split in the discipline between classification by overall similarity (numerical taxonomy; Sokal and Sneath, 1963) and classification by descent (phylogenetic systematics; Hennig, 1966) is, for our purposes, a split between two questions that CAMS also has to keep apart:

1. **Function.** Which coordination problems does a system solve, and how are its eight solutions arranged? This is similarity of configuration, and the repository's taxonomy paper (`functional_taxonomy.html`, May 2026) builds it from the problems upward and recovers eight clusters in 229 polities by Ward clustering, which is a numerical-taxonomic procedure in Sokal and Sneath's sense.
2. **Descent.** Which institutional lineage does a system continue? The same paper argues for path dependency: a polity "can change its formal political system, its dominant ideology, even its religion, and remain in the same coordination morphology" (section 4).

Biology's lesson is that these can come apart. Similar forms arise by convergence (analogy) as well as by descent (homology), and the classification that serves one question will mislead on the other. The origin hypothesis in section 6 is a hypothesis about **descent**; the cluster result is a statement about **form**. Neither establishes the other.

A useful tool from palaeontology is the morphospace, a space of possible forms in which actual forms occupy a small region and the empty regions are as informative as the occupied ones (Raup, 1966 **[not re-verified]**). The repository already uses the term (the Morphospace Atlas, `7dff1af`). The eight-by-four array defines a morphospace for meta-systems, and a personality (section 5) is a region of it.

### 2.2 Popper, Kuhn, Lakatos: how the programme should hold its claims

**Popper** (1934/1959) demands that a theory forbid something. The practical translation in this repository is the pre-registration habit: the 1 July 2026 panarchy protocol was lodged before the analysis (`b0668b2`), the 2026 to 2028 holdout list is frozen (`aad6218`), and failures are kept (DT-H4, the Paladin audit). Popper also warned against historicism, the claim to predict the course of history from laws of development (1957). The repository's own position, in the version history, has narrowed from "predicts civilisational resilience or collapse" to "does not predict the future" (`docs/VERSION_HISTORY.md`, "Where Claims Narrowed"). This paper keeps to the narrower position.

**Kuhn** (1962) describes normal science within a paradigm and, before one exists, a pre-paradigmatic phase of competing schools without agreed fundamentals. Comparative historical social science is plausibly pre-paradigmatic in that sense. CAMS does not claim to be a paradigm. A modest reading of the repository's history is that it is one such school, with an instrument, a corpus and a habit of publishing its retractions.

**Lakatos** (1970) is the most useful lens, because he describes a research programme as a hard core protected by a negative heuristic and a protective belt of auxiliary hypotheses that absorb refutation. A programme is progressive if its belt adjustments predict novel facts, and degenerating if they merely accommodate. Applied to CAMS:

- **Hard core** (fixed by decision, as the brief for this paper says): eight functions by four metrics, with the configuration as the state.
- **Protective belt**: the node-value formula, the bond form, the regime classifier and its thresholds, the scoring protocol, the derived quantities F and E.

The repository's own history reads as a long sequence of belt adjustments: the v2.3 bond retired and frozen (`1083b2c`); the clamp on the cognition metric removed; an unreachable classifier branch deleted and a coverage gap closed (`464222f`); a misaligned bond column repaired (`898b0d6`); bond thresholds on incompatible scales withdrawn in an erratum (`2c9750e`). Each of these is a sound repair on Lakatos's account only if it is followed by a novel, tested prediction. **That test has not been passed on the record.** The 2026 to 2028 holdout is frozen and unscored, and `juno/CANONICAL_STATUS.md` says plainly that JUNO v1.2 "is not externally validated until this holdout is scored".

Two consequences follow for how claims in this paper are framed. First, because the hard core is protected by decision, **defeat conditions must be placed on the belt and on the auxiliary claims about what the core describes**, since refutation of those is the only refutation the programme can register without abandoning its object. Second, the programme as a whole should be judged by Lakatos's criterion: does it, over time, anticipate facts it did not already contain? The honest present answer is "not yet shown".

### 2.3 Thermodynamics and Prigogine

Schrödinger (1944) posed the problem of how a living system maintains order against the second law: by exporting entropy to its surroundings. Prigogine and colleagues generalised it: far from equilibrium, open systems driven by a flux of energy can self-organise into **dissipative structures**, ordered configurations that exist only so long as they dissipate (Glansdorff and Prigogine, 1971 **[not re-verified]**; Prigogine and Nicolis, 1977 **[not re-verified]**; Prigogine and Stengers, 1984 **[not re-verified]**). In the social register, White (1943) tied cultural advance to energy harnessed per capita, Tainter (1988) tied the fate of complex societies to the returns on the energy cost of complexity, Chaisson (2001) proposed energy rate density as a common measure across cosmic evolution, and England (2013) argued that driven systems tend to adapt so as to dissipate more of the energy they absorb **[White, Tainter, Chaisson, England: not re-verified]**.

The CAMS reading, which is an analogy and not a derivation, is the following. Stress is defined as the rate of entropy production within a function. Capacity is the demonstrated ability to perform it. Their difference, E = K − S, is the margin by which the function can still maintain itself against its own dissipation. A dissipative structure with entropy production exceeding what it can export is not maintained. The repository's version history records that the thermodynamic framing replaced the neural-network framing in December 2025 (`f4c1ec1`). The word "thermodynamic" in this repository therefore names a modelling choice, and the formal link is weak: K and S are rubric scores on a 1 to 10 scale, not quantities in joules and watts, and the rev-6-October paper itself says a numerical zero in K − S "is the equality of two rubric scores" and "not yet an independently calibrated boundary" (section 2). The analogy is a source of hypotheses, not a warrant.

### 2.4 Holling, panarchy and near-decomposability

Holling (1973) separated engineering resilience (speed of return to equilibrium) from ecological resilience (how much disturbance a system absorbs before it flips into another regime). The adaptive cycle (Gunderson and Holling, 2002) describes four phases of growth, conservation, release and reorganisation, and the panarchy is a nest of such cycles at different scales, in which a fast small cycle can trigger release in a slower larger one ("revolt"), and a slow large cycle can supply stored resources to a small one that is reorganising ("remember").

Three connections to CAMS, and one hazard.

- **Slow and fast loops.** The taxonomy paper divides the eight functions into a slow loop (Archive, Lore, Stewards) and a fast loop (Helm, Shield, Flow, Hands, Craft), and says that "many civilisational transitions occur not when all nodes weaken simultaneously but when slow-loop coherence and fast-loop reaction decouple" (section 5.5). That is a panarchy claim about cross-scale coupling in CAMS terms. Simon's near-decomposability (1962) supplies the background: complex systems are typically organised as subsystems that interact strongly within and weakly between, with different characteristic rates.
- **Pre-lodged test.** The annual panarchy and Holling-cycle test for Russia, China, the USA and Germany was lodged before analysis (`b0668b2`). I have not read its result. **[not checked]**
- **Release as a node phenomenon.** In the panarchy picture, release follows a build-up of connectedness and stored capital with falling flexibility. The CAMS reading is that a node running with high F and shrinking E is in a pre-release state. That is a testable restatement (hypothesis H-E3 below).
- **A notation hazard.** Holling's cycle uses K for the *conservation* phase and r for growth. CAMS uses K for **Capacity**. The two have nothing to do with each other. Any figure that places both on one page must say so.

### 2.5 The Santa Fe programme

The Santa Fe Institute made "complex adaptive system" a working term: many interacting agents that adapt, with aggregate behaviour that is not read off any one of them (Holland, 1992, 1995; Gell-Mann, 1994; Kauffman, 1993; Waldrop, 1992; Mitchell, 2009; Miller and Page, 2007 **[all not re-verified]**). The name CAMS inherits that term. But the instrument is not an agent-based model. It is a **coarse-grained state description**: the agents (people, firms, officials) are integrated into eight functions, and the model does not simulate them. This matters for what can be claimed. A coarse-grained description can show that a configuration persists or shifts. It cannot show *why* at the level of the agents, and a coupling between two functions is an estimated association of their scores, not a demonstrated interaction (`backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, section 3).

The repository also contains a direct warning of this kind. The bond strength in JUNO v1.2 is **algebraically rank-1**: B_ij = w_i w_j with w_i = √q_i · 2^(−S_i/10), so it holds no pair-specific information beyond the per-node scalars (`juno/CANONICAL_STATUS.md`; audit `2e713ba`). Anything in this paper that speaks of "coupling" must therefore be read as coupling estimated from temporal co-movement of scores, not as the bond matrix, until a successor with genuine pair terms is calibrated.

### 2.6 Cybernetics

Wiener (1948) founded the study of control and communication in animal and machine. Ashby (1956) supplied the law of requisite variety: only variety can absorb variety, so a regulator must be able to produce at least as many distinct responses as the disturbances it faces. Conant and Ashby (1970) argued that every good regulator of a system must be a model of that system **[verified; note that the title overstates the result, which concerns the simplest among optimal regulators and a homomorphism, not an identity]**.

For CAMS the useful consequences are these. Helm is the regulator whose job is direction under uncertainty; by the good-regulator result it needs a model of the system it directs, and Archive (memory) and Lore (meaning) are where a society's models are kept. A Helm whose model is out of date, or whose Lore and Archive have decoupled from the operating nodes, would be predicted to regulate badly. The October 2026 signature pack contains a related pattern: *narrative–material gap* (high Abstraction in Lore and Archive while Flow, Craft and Hands capacity falls), and *Helm Incompetence* (Helm capacity falls and stress rises while Craft and Flow hold) (`docs/theory/CAMS_Organisational_Pathologies_Thesis.md`, section 5). Both are cybernetic failures in the sense that the regulator is not tracking the regulated. The thesis reports their window-concentration p-values; I do not reproduce or endorse them here.

### 2.7 Ibn Khaldun

The *Muqaddimah* (1377; trans. Rosenthal, 1958 **[not re-verified]**) is the nearest ancestor of the origin hypothesis in section 6. Ibn Khaldun distinguished the desert, mobile life (*badawa*) from settled, urban life (*hadara*) and argued that the former generates stronger group solidarity (*'asabiyya*) because its members depend on one another for defence and survival. Dynasties founded on solidarity conquer settled states, then decay as success brings luxury, taxation and dependence on mercenaries, and are in turn displaced; he gave a dynasty a life of about three generations or roughly a hundred and twenty years **[number from memory; not re-verified]**. (For an intellectual biography, Irwin, 2018 **[not re-verified]**.)

The mapping is suggestive, not exact. *'Asabiyya* is a social-cohesion variable; in CAMS terms the nearest candidate is Coherence in Helm, Shield and Lore, where mobile groups are predicted to be strong. Decay through luxury is a rise in settled-function stress with a decline in the mobilising functions. What Ibn Khaldun provides is the **direction of the argument**: a contrast of mobile and settled lineages, with institutional character derived from a mode of subsistence and not from regime type. It is also a warning. His account is a cycle theory, and cycle theories invite fitting.

### 2.8 Turchin and cliodynamics

Turchin (2003, 2006) set out to make history an analytical, predictive science, building on population dynamics and on Ibn Khaldun's *'asabiyya*, which he formalised as *asabiya* and linked to the metaethnic frontier. Turchin and Nefedov (2009) studied secular cycles in agrarian states, and Turchin (2016) elite overproduction in the United States **[all not re-verified]**. The Seshat databank (Turchin and colleagues, 2015 **[not re-verified]**) coded social-complexity variables across many societies, and Turchin and colleagues (2018, PNAS 115(2): E144 to E151) reported that a single principal component accounts for roughly three-quarters of the variation across 414 societies in 51 variables **[verified]**.

That last result bears on CAMS directly, in two opposed ways. It supports the use of aggregate scores as meaningful (a dominant common axis exists in an independent dataset). It also warns that a dominant common axis can swamp configuration: if most of the variance in node scores is one common factor, the "relative position" of the functions may be a small residual. The repository's own panels show something similar. The rev-6-October paper reports "a dominant common field in the panels and limited separation between some higher-order structure and scorer disagreement" (section 8), and the node-dynamics audit of 15 August 2026 found node response coefficients 69.1% common-mode and 30.9% relational (`2e713ba`). A Seshat cross-validation page exists in the repository (`7c12ca0`, updated `4660b28`); I have not read its results. **[not checked]**

---

## 3. Social cognition emerges per node

### 3.1 The definitions

For node *i* at time *t*:

- **Focus** F_i = A_i × C_i. Abstraction is the degree to which symbolic activity is operationally integrated with the function; Coherence is internal alignment. Their product is high only when the node is both aligned and grounded in its own conceptual work. (The repository calls the same quantity P_i = C_i A_i in `backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, and the "dream" d_i = A_i C_i / 100 in `paladin-validation.html`. "Focus" is used here for its plainness. The factor of 100 is a scale convention only.)
- **Available energy** E_i = K_i − S_i. Capacity minus the rate of entropy production in the function.
- **Signed activation** σ_i = F_i · E_i (JUNO's form also divides F by 100, and v1.2 uses a sign-preserving clip so that near-zero E does not erase sign: `juno/JUNO_v1.2-Final_Formalism.md`, section 1).

The first appearance of F and E side by side in this repository is the explorer change of 18 April 2026: "radar = A×C, bar = K vs S" (`974d564`). Both are *derived* from the canonical four. They are not extra dimensions (`canonical-definitions.html`).

### 3.2 The claim

**Cognition is local first.** Each function of a meta-system is a semi-autonomous processor with its own focus and its own budget. The system-level cognitive state (what a society attends to, fears, and thinks possible) is what results when those local states are integrated, and it can differ sharply from any single node's. The same idea has appeared in the repository as "sub-brains" (Subbrain Activation, `324ed89`) and in the ESCH framework as semi-autonomous nodes. The unmerged branch `crisis-typology-esch-cognition-5nov2026` (`c0d8798`, `d51e14e`) draws five USA cognitive profiles from this. I treat that work as exploratory.

**A node with S > K consumes its own body.** When E_i < 0, the function's entropy production exceeds its demonstrated ability. By the thermodynamic analogy in 2.3, it is spending structure to meet current demand. By the sign of σ_i, its activation is negative whatever its focus: a high-focus node with negative energy is *more* strongly negative, not less. This is the formal content of "sees everything negatively": the sign of E multiplies the focus. A node does not become dim when it runs a deficit; it keeps its attention and attaches a negative valence to it.

The prevalence is not rare. JUNO v1.2's own change log records that "31.8% of node-years carry negative activation" in the 3,002-society-year blind corpus (`juno/JUNO_v1.2-Final_Formalism.md`, section 7). That is a statement about the panel the formalism was calibrated on, which consists of model-scored societies.

### 3.3 What the repository already says against the strong version

This section matters more than the one before it.

1. **Three of four Paladin renderings are relabelled extrema.** The 20 July 2026 adversarial audit found that "dreams through" (argmax of d), "can act through" (argmax of e) and "the burden falls upon" (argmin of e) carry no information beyond the ranking that produced them (`paladin-validation.html`; commit `4289140`). Anything said here about *which node* has the most focus or the deepest deficit is a statement of a ranking, not a discovery.
2. **The one rendering that survived is non-predictive.** Alignment, the correlation across the eight nodes between d and e, was non-redundant and stable, but the audit found its predictive AUC to be about 0.48 (same source). It describes a persistent trait, not a warning.
3. **The cognition split has a post-hoc history.** The deficit-crisis AUC of 0.740 in PR-GAP-2 arises from splitting 126 ruptures by the sign of the JUNO expressed-cognition measure; the unsplit leave-one-society-out AUC is 0.563 and not significant (`39370b7`; also reported in `backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, section 4, as mean LOSO AUC across 37 societies, t(36) = 1.76, p = .087). The commit message describes the split as lifting the AUC; it does not say whether the split was planned before the pooled result was seen. **[unclear]**
4. **The scores are produced by language models reading the historical record.** A node's "focus" is, in practice, a scorer's judgement. The rev-6-October paper states the consequence: agreement between scorers "establishes consistency under specified conditions; it does not establish historical truth" (section 8).

### 3.4 Hypotheses and defeat conditions (cognition)

| ID | Hypothesis | Would be abandoned if |
|---|---|---|
| **H-C1** | Node-level E = K − S carries information about outcomes within that node's own domain that the node's V = C + K − S + A/2 does not. | Across the corpus, adding F and E to V gives no out-of-sample improvement in predicting an independently dated, node-specific outcome (for example, Shield E against externally recorded military expenditure shocks) beyond V alone, using leave-one-society-out evaluation with the evaluation design registered in advance. |
| **H-C2** | Negative E (S > K) produces **negatively valenced output** from that node's domain: documents, speech and reporting from the function read as more negative than from the same function in positive-E years. | Independently coded valence of text from the relevant domain (for instance, parliamentary speech for Helm, press for Flow) shows no difference between negative-E and positive-E node-years after controlling for the era's baseline sentiment and for scorer exposure to the same text. The repository holds an exploratory Hansard note (`juno/Epiphenomenon_HorseRace_2026-09-10.md`); I have not read it and it is not evidence for H-C2. |
| **H-C3** | A node with high F and falling E is in a pre-release state: its subsequent drop in V is larger than for nodes with falling E and low F. | The F-dependence disappears under matched-E comparison; or timing shows V falling before E, which would make E a lagging indicator. (The unmerged branch notes E depletion lags F loss in the USA series and proposes but has not sealed a leading-indicator rule.) |
| **H-C4** | System-level cognition is an integration of node-level cognition, so that a system whose nodes' (F, E) are *misaligned* is persistently different from one where they are aligned. | Alignment (the correlation of d and e) shows no stable between-society difference, or differs only through its component extrema. The Paladin audit has already found stability but no prediction; H-C4 is restricted to the descriptive claim. |

---

## 4. Every meta-human system has a personality

### 4.1 Definition

The **personality** of a meta-human system is the persistent component of its configuration: the time-averaged profile of the eight (F_i, E_i) pairs, with its dispersion across years, together with the persistent pattern of co-movement among the functions, assessed over a window long enough to span changes of government. It is a region of the morphospace in 2.1 that the system returns to.

This is what ordinary language calls **national character**, when ordinary language is being careful. The point of a definition in these terms is that it can be wrong. The rev-6-October paper's first hypothesis already says it in structural terms: "estimated coupling patterns are more persistent across changes of government, dynasty or ideology than a regime-centred account would predict" (H1, section 5).

### 4.2 The informal characters, and what each would predict

These descriptions are the author's, and they are the sort of thing that is easy to say and hard to test. Each is translated here into a prediction about the profile.

| System | Informal character (author's) | Predicted profile signature | Specific defeat condition |
|---|---|---|---|
| **Australia** | Laid back, conservative, attached to records | Persistently low Stress across nodes; high and stable Archive Coherence and Capacity; low variance in all nodes across decades; slow-changing profile. | Archive is not higher than the cross-corpus median for Archive after controlling for era and scorer; or Australia's across-decade profile variance is not below the corpus median. |
| **Germany** | Engineering-obsessed, tightly coupled, tension building slowly | Persistently high Craft Capacity and Coherence; high co-movement among functions (*coupling* as an estimated association, see 2.5); a long, slow rise in Stress in selected nodes with delayed release. | Craft is not above the corpus median; or the build-up of stress, where it appears, is no slower than in matched societies; or co-movement is not above the corpus median once the common factor is removed. |
| **China** | A dynastic state over about 1.4 billion highly localised people, with Lore and Craft bound to geography | A strongly centralised Helm and Shield profile at national level combined with Lore, Craft and Hands whose scores are **regionally heterogeneous**: national-level scores for these are averages over very different regional states. | Sub-national scoring (provinces or regions) shows no more between-region variance in Lore and Craft than in other large states; or national scoring shows no Helm and Shield centralisation signature relative to the corpus. |

Two cautions on terms.

- **"Tightly coupled."** Gelfand and colleagues (2011) use "tight" for cultures with strong norms and low tolerance of deviance **[verified]**. That sense and the CAMS sense (strong co-movement among functions) are different things. They may be related, but the paper should use "tight" only in Gelfand's sense and "coupled" only in the estimated-association sense.
- **"Dynastic."** For China, the claim that Lore and Craft are "bound to geography" is a claim about locality, not about dynasty. The taxonomy paper places the PRC in the cluster it calls Imperial mode, with Rome, the Ottoman Empire, the British Empire and others, and says the riverine-bureaucratic-imperial signature persists across the Ming, Qing and the PRC (`functional_taxonomy.html`, sections 4 and 7). That is consistent with the character above; it does not test it.

### 4.3 The strongest evidence against, which any test must answer

Terracciano and colleagues (2005) compared national-character ratings with measured personality in 49 cultures and found that the ratings were reliable but did not converge with the assessed traits (median correlation .04); the ratings looked like stereotypes **[verified]**. This is a direct defeat risk. If CAMS profiles "recover" Australian laidbackness and German engineering, the question is whether the instrument has measured anything about these societies or has read the stereotype back from the text its scorers were trained on.

The risk is concrete here because scoring is done by language models. The repository's deep-time work acknowledges that "recognisable events can reveal identities and outcomes" even under name-masking (`backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, section 7). So the test of personality cannot be "the profile matches the stereotype".

### 4.4 Hypotheses and defeat conditions (personality)

| ID | Hypothesis | Would be abandoned if |
|---|---|---|
| **H-P1** | Profiles are persistent within a society across changes of government, and more so than a regime-centred account predicts. | Regime category explains between-period profile variation as well as or better than society identity, after controlling for common trends, scoring artefacts and temporal dependence. (This restates H1 of the rev-6-October paper.) |
| **H-P2** | Profiles carry information **beyond the informal stereotype**. | In a blind-identification design in which the profile is shown without names, a stereotype-only baseline (for example, a rater given only each society's popular descriptors, or a model prompted with the stereotype and no data) identifies societies as well as or better than the profile. |
| **H-P3** | Profile differences predict specified, independently observed behaviours. | They do not predict anything from an independent source (for example, differences in recorded responses to a shared shock) beyond cross-sectional controls. |
| **H-P4** | The profile is a property of the system and not of the scorer. | Scorer-family differences (Claude, Grok, GPT, Kimi) in profile exceed the between-society differences, or the profile reverses under a different scorer family. The repository's envelope files record scorer variance, and the deep-time study already found one scorer family leaving Lore and Archive unscored in non-literate societies where others scored them (`backstory/`; DT-2 adoption `e5be317`). |

---

## 5. Thermodynamic origins of personality

### 5.1 The hypothesis

A meta-system's persistent profile reflects the **energy regime** of the lineage from which its institutions descend, because the mode of subsistence sets which coordination problems are acute (the taxonomy paper's argument, `functional_taxonomy.html`, section 3) and the solutions to those problems harden into path-dependent architecture (section 4 of the same paper).

Two stylised lineages:

- **Mobile lineages** (hunting and herding bands, nomadic and pastoral confederations). Energy is captured by moving to the gradient, not by tying infrastructure to it. Wealth is carried or herded. Institutions must travel: kinship, oral law, genealogy, a war-leader, rapid decision. In CAMS terms the predicted weight falls on **Helm, Shield and Lore** (portable direction, defence and meaning), with **Archive structurally weak** (the taxonomy paper's steppe entry says "Archive structurally weak" and gives Shield and Flow primacy). Ibn Khaldun's desert groups (2.7) are the classical case.
- **Place-bound lineages** (river systems and irrigated geography). Energy is captured by building and maintaining infrastructure that cannot move. This demands multi-year planning, water-rights adjudication, granaries and records. The taxonomy paper's riverine entry reads: "Annual flooding cycles require multi-year planning, water-rights adjudication, and granary management. Egypt, Mesopotamia, early China → Stewards–Archive primacy." Here the predicted weight falls on **Stewards and Archive**, and, for the claim at issue, on **Lore and Craft that are tied to place**: ritual calendars, local cults, local technique.

The author's framing is that **Western systems descend from mobile, non-place-bound forms, while Eastern ones such as Thailand and China are bound to river systems and geography.**

### 5.2 Why this is a hypothesis with real difficulties

I think it is worth stating this hypothesis because it can be tested, and equally because it is easily misread as a claim it is not.

1. **"Western" and "Eastern" are not natural kinds.** Egypt and Mesopotamia, the riverine origin of much of the Western institutional inheritance (writing, law, record-keeping, bureaucracy), are in the region the taxonomy paper cites as the *riverine* case. A hypothesis that Western systems descend from mobile forms must say which Western lineage, and why the riverine ancestors do not count.
2. **Ancestry in Europe is a mixture.** Palaeogenomic work shows that a major component of present-day European ancestry arrived from the Pontic–Caspian steppe about 4,500 years ago, alongside earlier Neolithic farmer and hunter-gatherer components (Haak and colleagues, 2015 **[verified]**; Anthony, 2007 **[not re-verified]**). That is evidence that mobile pastoral lineages contributed to European *populations and languages*. It is not evidence about *institutions* of the modern states, and the step from ancestry to institutional form is exactly what the hypothesis would need and does not have.
3. **Place-bound is not a property of the East.** The Chao Phraya and the Yellow River are river systems; so are the Nile and the Tigris and Euphrates. Mobile pastoralism is also a feature of large parts of Eastern Eurasia. And Scott (2009) argues that upland Southeast Asian peoples organised themselves *to avoid* state incorporation, which is neither river-bound nor Western **[not re-verified]**.
4. **The classical bridge is contested.** The link between irrigation and centralised despotism is Wittfogel's (1957), and it has been widely criticised; Carneiro's circumscription theory (1970) offers a different route from environment to the state **[both not re-verified]**. Scott (2017) and Graeber and Wengrow (2021) complicate the whole picture of a single route from foraging to the state **[not re-verified]**.
5. **Essentialism.** The hypothesis is about energy regime and the institutions that grew around it. It is not a claim about peoples, and it cannot be allowed to become one. Statements about "East" and "West" in this paper carry that restriction.

### 5.3 Restating it in a form that can fail

To make the hypothesis testable, replace East and West with a measurable variable and a measurable outcome.

- **Lineage variable.** For each society or period, code *independently of CAMS scores* the dominant mode of energy capture of the lineage whose institutions the polity continues: *mobile* (pastoral, nomadic, maritime), *place-bound hydraulic* (river-valley irrigation), *place-bound rain-fed*, or *mixed*. The coding rule, evidence and codings must be registered before any profile is inspected. The rev-6-October paper's H6 makes the same demand: define conditions independently, "rather than assigning broad categories after inspecting the scores".
- **Outcome.** The persistent profile of section 4, restricted to pre-registered contrasts: Stewards and Archive weight against Helm, Shield and Lore weight; and, for place-bound lineages, within-system regional heterogeneity of Lore and Craft.

### 5.4 Hypotheses and defeat conditions (origins)

| ID | Hypothesis | Would be abandoned if |
|---|---|---|
| **H-O1** | Lineages coded mobile show a persistent profile weighted to Helm, Shield and Lore, and lineages coded place-bound hydraulic show a persistent profile weighted to Stewards and Archive. | On held-out societies, lineage coding adds no predictive information on the pre-registered contrasts beyond simpler baselines (current region, development level, era). Shared history, technological change and contact between societies are controlled, as H6 of the rev-6-October paper demands. |
| **H-O2** | The contrast survives *within* the cultural West and *within* the cultural East. | The East–West label predicts profile better than lineage coding does. Under the hypothesis the expected result is the reverse: Western riverine-origin systems (for example, those continuing Nile or Mesopotamian institutional lineages) should resemble Eastern riverine systems, and Eastern pastoral systems should resemble Western pastoral ones, so that the East–West label becomes redundant once lineage is known. |
| **H-O3** | Place-bound lineages show greater regional heterogeneity of Lore and Craft than mobile lineages at comparable scale. | Sub-national scoring (which the taxonomy paper lists as an out-of-sample test) shows no such difference. |
| **H-O4** | The profile of a system continues through changes in subsistence: a polity that stops being pastoral keeps the institutional weighting for a long time (path dependency). | Profiles realign with the new energy regime within a generation or two, so that lineage adds nothing once current regime is known. |

H-O2 is deliberately phrased so that the most natural reading of the author's claim, the East–West claim, is the one the data could embarrass.

### 5.5 A word on "thermodynamic"

I use the word for the origin story in the weak sense of 2.3: a mode of energy capture sets a pattern of stresses and capacities, and the pattern persists. There is no calculation in this paper of an energy budget for any society. Energy rate density (Chaisson) or energy capture per capita (White) would be the right quantities for H-O1, and neither is currently in the corpus. **[not checked]** whether Seshat variables could serve as a proxy.

---

## 6. Lore is the collective operating system

### 6.1 The claim

**Lore** is the function that sustains shared meanings, identities and justifications. The strong form is that Lore is the society's *operating system*: the layer that lets heterogeneous parts run together. In the computing analogy, an operating system (a) provides a common interface between hardware and applications, (b) schedules shared resources, (c) enforces permissions, and (d) is replaceable, with the applications continuing to run if the interface is preserved. The Lore analogues are (a) a common vocabulary of obligation and identity that every other function addresses, (b) calendars, rituals and festivals, which schedule collective attention, (c) taboo, norm and the distinction between sacred and profane, and (d) conversion and revolution, in which one operating system is replaced and economic and military functions continue under a new one.

**Mythology** is then the *instantiation* of node Lore in individual human brains. Lore as a function is distributed and has no location; mythology is what it looks like when it is stored in a mind, rehearsed in a story and transmitted by telling. The distinction is the one between a protocol and the packets it travels in.

This is a reading of an old tradition, not a novelty. Durkheim (1912) argued that religion is society representing itself to itself, and that collective representations bind the group. Anderson (1983) argued that nations are imagined communities held together by shared narrative technology, and Hobsbawm and Ranger (1983) that many traditions are recent inventions. Henrich (2016) treats culturally transmitted knowledge as an adaptive inheritance. Dawkins (1976) coined "meme" for a cultural replicator, a framing that has been contested **[all not re-verified]**.

### 6.2 Where CAMS adds something, and where it doesn't

What CAMS adds is the position of Lore *among the other seven*. It supplies Lore's four scores, and therefore a claim about how well the operating system is aligned (C), how well it works (K), how much entropy it produces (S), and how far it is operationally integrated with the function it names (A). In CAMS terms the most striking hypothesis is that the operating system is **a hub**: Lore's Coherence should predict the co-movement of the other functions more strongly than any other node's Coherence does.

What it does not add is any claim about the content of myth. The instrument scores the *function's performance*; it does not interpret stories. The rev-6-October paper is careful on a related point: "The Zeitgeist is the proposed connection between these arrangements and lived public meaning... That connection requires evidence of its own. It cannot be established by giving a network a suggestive mythic name" (section 3). The same applies here. The "operating system" language is a model of the function; it should not be allowed to carry evidential weight of its own.

### 6.3 Lore and Archive: the case of oral societies

A direct test of whether Lore is distinct from Archive comes from the deep-time work. GPT, in the 2 October 2026 study, reportedly left Lore and Archive unscored for societies without writing, while Claude and Grok assigned substantial scores, and the DT-2 amendment now requires oral evidence to be recognised and "function demonstrably absent" to be separated from "evidence insufficient" (`backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md`, section 6; adoption in `e5be317`). On the operating-system reading, this is the signature one would expect: in a literate society, memory has a medium outside the brain (Archive) and meaning has a separate one (Lore); in an oral society, **both run on the same hardware, human memory**, and the story that carries the law, the genealogy and the identity is one object. Mythology as instantiated Lore is then also the instantiated Archive. This is a testable inference, not a finding.

### 6.4 Hypotheses and defeat conditions (Lore)

| ID | Hypothesis | Would be abandoned if |
|---|---|---|
| **H-L1** | Lore's Coherence is a hub: changes in Lore C are more strongly associated with subsequent changes in other nodes' C and K than changes in any other single node's C are. | Using lagged changes with temporal-dependence-preserving nulls, Lore does not rank above the median of the other nodes, or its rank is reproduced by a Lore-shuffled null. Note the bond matrix cannot be used for this test, since it is rank-1 (2.5). |
| **H-L2** | In non-literate societies, Lore and Archive scores are more strongly correlated than in literate societies, because both are stored in the same medium. | Under DT-2 scoring, the Lore–Archive association in oral societies is no stronger than in literate ones across the registered cases, or the difference is produced by scorer-family differences. |
| **H-L3** | Replacement of the dominant Lore (for example, by conversion, ideological revolution or conquest) is followed by a Lore Coherence trough and a rebound while Craft, Flow and Hands capacities persist. | Craft, Flow and Hands capacities fall with Lore through the transition, or Lore Coherence shows no trough relative to matched non-transition periods. |
| **H-L4** | The signature *narrative–material gap* (very high Abstraction in Lore and Archive while material capacity falls) precedes large declines, as an operating system detached from its applications. | The repository's own frozen-rule test on the signature pack fails to replicate on the held-out set, or fails to beat simpler detectors of existing stress. (The thesis reports that this signature leans in the expected direction at raw p = .072, and sits above 0.05 as a raw p-value; `docs/theory/CAMS_Organisational_Pathologies_Thesis.md`, section 6.) |

---

## 7. What counts against the whole

Setting the section-specific conditions aside, there are five ways in which the programme as described could fail altogether, each already acknowledged in the repository's own writing.

1. **The common factor.** If almost all variance in the eight-by-four array is a single common field, then "relative position" is a small residual, and personality (a configuration claim) has little to attach to. Evidence in the repository points both ways (2.8).
2. **Scorer-dependence.** All the profiles are model-scored. Ensemble tightness "certifies precision, not ground truth; shared model bias remains an unquantified confound" (`juno/JUNO_v1.2-Final_Formalism.md`, section 6).
3. **Circularity of the cognition quantities.** F and E are arithmetic on rubric scores. A claim that they measure cognition and energy needs evidence outside the scores. The same objection was made to λ₂ and mean bond strength (correlation 0.972; same document, section 5).
4. **No prospective success yet.** The only prospective record is a frozen, unscored list (`juno/OOS_PREDICTIONS_2026-2028.md`). By Lakatos's criterion (2.2) the programme is not yet shown to be progressive.
5. **Narrative fit.** Each of Ibn Khaldun's cycles, Holling's phases, Turchin's secular waves and this paper's personalities is a story that can fit data it was not tested on. A pre-registered, blind, sub-national or out-of-sample test is the only protection, and it is the protection the repository already requires of itself.

---

## 8. What is claimed and what is not

**Claimed.** (i) The eight-by-four instrument has an intellectual lineage in several sciences, and its placement among them clarifies what kind of claim it can support. (ii) Focus and available energy are well-defined derived quantities on the canonical array, and a sign convention in the existing activation metric already formalises "negative energy yields a negative reading". (iii) The idea that a meta-system has a persistent profile is a precise, testable restatement of national character. (iv) The origin and Lore ideas are hypotheses with defeat conditions listed above.

**Not claimed.** That Australia, Germany or China *has* the profile described in 4.2. That Western and Eastern institutions descend from the lineages described in 5.1. That mythology has been shown to be instantiated Lore. That any of F, E or σ measures cognition or energy in a physical sense. That any of this predicts the future.

**Style note.** The paper avoids evaluative terms for systems and for the instrument (for example, any rating of a society as sound or unsound). Claims about welfare, justice or the standing of a regime require evidence beyond coordination scores, as the rev-6-October paper states (section 4).

---

## References

**Verification key.** **[verified]** = bibliographic details (author, year, venue, volume, pages) confirmed against records in this session. **[not re-verified]** = given from memory; the work exists to my knowledge, but details need checking before any submission. No reference here is invented; if a detail cannot be confirmed it should be removed, not corrected from memory.

*Philosophy and history of science; taxonomy*

- Darwin, C. (1859). *On the Origin of Species by Means of Natural Selection*. London: John Murray. (Chapter 13 on classification.) [not re-verified]
- Hennig, W. (1966). *Phylogenetic Systematics*. Urbana: University of Illinois Press. [not re-verified]
- Kuhn, T. S. (1962). *The Structure of Scientific Revolutions*. Chicago: University of Chicago Press. [not re-verified]
- Lakatos, I. (1970). Falsification and the methodology of scientific research programmes. In I. Lakatos and A. Musgrave (eds), *Criticism and the Growth of Knowledge*, pp. 91 to 196. Cambridge: Cambridge University Press. [not re-verified]
- Linnaeus, C. (1735). *Systema Naturae*. Leiden. [not re-verified]
- Popper, K. R. (1959). *The Logic of Scientific Discovery*. London: Hutchinson. (Original: *Logik der Forschung*, Vienna, 1934.) [not re-verified]
- Popper, K. R. (1957). *The Poverty of Historicism*. London: Routledge and Kegan Paul. [not re-verified]
- Raup, D. M. (1966). Geometric analysis of shell coiling: general problems. *Journal of Paleontology*, 40, 1178 to 1190. [not re-verified]
- Sokal, R. R. and Sneath, P. H. A. (1963). *Principles of Numerical Taxonomy*. San Francisco: W. H. Freeman. [not re-verified]

*Thermodynamics and energy*

- Chaisson, E. J. (2001). *Cosmic Evolution: The Rise of Complexity in Nature*. Cambridge, MA: Harvard University Press. [not re-verified]
- England, J. L. (2013). Statistical physics of self-replication. *Journal of Chemical Physics*, 139, 121923. [not re-verified]
- Glansdorff, P. and Prigogine, I. (1971). *Thermodynamic Theory of Structure, Stability and Fluctuations*. London: Wiley-Interscience. [not re-verified]
- Prigogine, I. and Nicolis, G. (1977). *Self-Organization in Nonequilibrium Systems*. New York: Wiley. [not re-verified]
- Prigogine, I. and Stengers, I. (1984). *Order Out of Chaos*. New York: Bantam. [not re-verified]
- Schrödinger, E. (1944). *What Is Life?* Cambridge: Cambridge University Press. [not re-verified]
- Tainter, J. A. (1988). *The Collapse of Complex Societies*. Cambridge: Cambridge University Press. [not re-verified]
- White, L. A. (1943). Energy and the evolution of culture. *American Anthropologist*, 45(3), 335 to 356. [not re-verified]

*Resilience, panarchy, complexity*

- Gell-Mann, M. (1994). *The Quark and the Jaguar*. New York: W. H. Freeman. [not re-verified]
- Gunderson, L. H. and Holling, C. S. (eds) (2002). *Panarchy: Understanding Transformations in Human and Natural Systems*. Washington, DC: Island Press. [not re-verified]
- Holland, J. H. (1992). Complex adaptive systems. *Daedalus*, 121(1), 17 to 30. [not re-verified]
- Holland, J. H. (1995). *Hidden Order: How Adaptation Builds Complexity*. Reading, MA: Addison-Wesley. [not re-verified]
- Holling, C. S. (1973). Resilience and stability of ecological systems. *Annual Review of Ecology and Systematics*, 4, 1 to 23. [not re-verified]
- Kauffman, S. A. (1993). *The Origins of Order*. New York: Oxford University Press. [not re-verified]
- Miller, J. H. and Page, S. E. (2007). *Complex Adaptive Systems: An Introduction to Computational Models of Social Life*. Princeton: Princeton University Press. [not re-verified]
- Mitchell, M. (2009). *Complexity: A Guided Tour*. Oxford: Oxford University Press. [not re-verified]
- Simon, H. A. (1962). The architecture of complexity. *Proceedings of the American Philosophical Society*, 106(6), 467 to 482. [not re-verified]
- Waldrop, M. M. (1992). *Complexity: The Emerging Science at the Edge of Order and Chaos*. New York: Simon and Schuster. [not re-verified]

*Cybernetics*

- Ashby, W. R. (1956). *An Introduction to Cybernetics*. London: Chapman and Hall. [not re-verified]
- Conant, R. C. and Ashby, W. R. (1970). Every good regulator of a system must be a model of that system. *International Journal of Systems Science*, 1(2), 89 to 97. [verified]
- Wiener, N. (1948). *Cybernetics: Or Control and Communication in the Animal and the Machine*. Paris: Hermann; New York: Wiley. [not re-verified]

*History, cycles, cliodynamics*

- Ibn Khaldun (1377). *Muqaddimah*. Trans. F. Rosenthal (1958), *The Muqaddimah: An Introduction to History*, 3 vols. New York: Pantheon. [not re-verified]
- Irwin, R. (2018). *Ibn Khaldun: An Intellectual Biography*. Princeton: Princeton University Press. [not re-verified]
- Turchin, P. (2003). *Historical Dynamics: Why States Rise and Fall*. Princeton: Princeton University Press. [not re-verified]
- Turchin, P. (2006). *War and Peace and War: The Rise and Fall of Empires*. New York: Pi Press. [not re-verified]
- Turchin, P. (2016). *Ages of Discord*. Chaplin, CT: Beresta Books. [not re-verified]
- Turchin, P. and Nefedov, S. A. (2009). *Secular Cycles*. Princeton: Princeton University Press. [not re-verified]
- Turchin, P. et al. (2015). Seshat: the Global History Databank. *Cliodynamics*, 6(1), 77 to 107. [not re-verified; page range in particular]
- Turchin, P. et al. (2018). Quantitative historical analysis uncovers a single dimension of complexity that structures global variation in human social organization. *Proceedings of the National Academy of Sciences*, 115(2), E144 to E151. [verified]

*Origins, geography, subsistence*

- Anthony, D. W. (2007). *The Horse, the Wheel, and Language*. Princeton: Princeton University Press. [not re-verified]
- Carneiro, R. L. (1970). A theory of the origin of the state. *Science*, 169, 733 to 738. [not re-verified]
- Graeber, D. and Wengrow, D. (2021). *The Dawn of Everything*. London: Allen Lane. [not re-verified]
- Haak, W. et al. (2015). Massive migration from the steppe was a source for Indo-European languages in Europe. *Nature*, 522(7555), 207 to 211. [verified]
- Scott, J. C. (2009). *The Art of Not Being Governed*. New Haven: Yale University Press. [not re-verified]
- Scott, J. C. (2017). *Against the Grain*. New Haven: Yale University Press. [not re-verified]
- Wittfogel, K. A. (1957). *Oriental Despotism*. New Haven: Yale University Press. [not re-verified]

*Culture, meaning, national character*

- Anderson, B. (1983). *Imagined Communities*. London: Verso. [not re-verified]
- Dawkins, R. (1976). *The Selfish Gene*. Oxford: Oxford University Press. [not re-verified]
- Durkheim, E. (1912). *Les formes élémentaires de la vie religieuse*. Paris: Alcan. [not re-verified]
- Gelfand, M. J. et al. (2011). Differences between tight and loose cultures: a 33-nation study. *Science*, 332, 1100 to 1104. [verified]
- Henrich, J. (2016). *The Secret of Our Success*. Princeton: Princeton University Press. [not re-verified]
- Hobsbawm, E. and Ranger, T. (eds) (1983). *The Invention of Tradition*. Cambridge: Cambridge University Press. [not re-verified]
- Terracciano, A. et al. (2005). National character does not reflect mean personality trait levels in 49 cultures. *Science*, 310(5745), 96 to 100. [verified]

*Repository sources (primary)*

- `docs/CAMS_LAB_DIARY.md` (this branch): history of nodes, metrics, formulas and names.
- `docs/VERSION_HISTORY.md`; `docs/errata/ERRATUM_2026-10-20_bond_scale_and_labels.md`.
- `canonical-definitions.html` (20 October 2026).
- `juno/JUNO_v1.2-Final_Formalism.md`; `juno/CANONICAL_STATUS.md`.
- `functional_taxonomy.html` (NeuralNations Working Paper 3, May 2026).
- `paladin-validation.html` (20 July 2026; commit `4289140`).
- `backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md` (commit `4bd2381`).
- `docs/theory/CAMS_Organisational_Pathologies_Thesis.md`.
- Commits cited by short SHA in the text; all resolve in this repository.
