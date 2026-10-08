# the experiment CAMS: A framework hypothesis

### One representation for every society, including our own.

*Kari McKern · Neural Nations / ComplexityWorkz*  
*Original draft: 4 October 2026 · Revised draft: 6 October 2026*

---

## Abstract

Human organisation allows primates who will never meet to share skills, distribute surpluses, enforce decisions and maintain a common picture of the world. The Complex Adaptive Model of Societies (CAMS) proposes a common representation of that coordination. It reads societies through eight functions—the Paladins: Helm, Shield, Lore, Archive, Craft, Flow, Stewards and Hands—scored for Coherence, Capacity, Stress and Abstraction.

This paper proposes two derived coordinates: symbolic organisation, P = CA, and functional headroom, H = K − S. It asks whether the relationships among those functions reveal persistent structures, changes under strain and differences between collapse and recovery. These coordinates are proposed indices; their interpretation and measurement validity remain open to testing.

Six hypotheses concern the persistence of coupling, constriction before coercion, hysteresis, collapse pathways, narrative alignment and ecological setting. A deep-time study dated 2 October 2026 supplies exploratory observations, including a failed debt-relief hypothesis from that study’s separate register. It does not confirm the hypotheses proposed here. The framework, its scoring instrument and its interpretations remain answerable to independent evidence, alternative representations and prospective tests.

## 1. The comparative move

A society is a way for strangers to act together.

People who will never meet must agree on prices, honour obligations, recognise decisions and trust that grain handed over this season will still be accounted for next season. They coordinate through institutions, habits, threats, records and stories. CAMS begins by asking whether those arrangements can be compared through a common set of functions.

The natural-history premise is simple. Human societies belong within the world we study. Their members are primates; their institutions depend on material flows, accumulated knowledge and the management of cooperation and aggression. Their own accounts of their uniqueness are part of the evidence.

Societies are pods. The metaphor directs attention to learned cooperation and adaptation to a setting. It does not establish that a society is a single organism, that its members share one interest, or that every institution benefits the group. An arrangement can persist because it benefits a powerful minority, because alternatives are suppressed, or because changing it is costly. Persistence alone does not demonstrate optimal adaptation.

River management, maritime exchange, continental distances and mobile pastoralism present different coordination problems. They may favour different relationships among authority, force, production, exchange and memory. Those relationships are propositions to investigate. Geography does not write a society’s constitution in advance.

Money, citizenship, property, corporations, nations, rank, law and debt are shared abstractions with material consequences. A border cannot be picked up, but a fence can be built along it and a person imprisoned for crossing it. The collective arrangement constrains the people who reproduce it. Its effects do not require it to possess a mind or a purpose.

The comparative move costs us an exemption. Our own society becomes a case alongside others. Neither exceptional virtue nor exceptional guilt places it outside inquiry. Comparison does not imply moral equivalence: it requires us to specify the similarities and preserve the differences that matter.

## 2. The proposed instrument

CAMS divides societal coordination into eight functions. The Paladins give those functions memorable faces; they are neither social classes nor individual people. One institution may perform several functions, and one function may be distributed across many institutions.

| Function | Image | Coordination role |
| --- | --- | --- |
| Helm | The Throne | Sets collective direction and makes binding decisions. |
| Shield | The Battlefield | Organises coercion, defence and enforcement. |
| Lore | The Temple | Sustains shared meanings, identities and justifications. |
| Archive | The Library | Preserves and transmits records, procedures and social memory, including oral traditions. |
| Craft | The Workshop | Maintains skilled production and turns knowledge into workable things. |
| Flow | The Agora | Coordinates exchange and circulation. |
| Stewards | The Manor | Controls and allocates assets, stores and resources. |
| Hands | The Fields | Supplies the applied labour on which collective activity depends. |

Giving each function a place in the representation prevents leadership or warfare from standing in for the whole society. It does not establish that the functions contribute equally to every outcome. Equal representation is a design choice whose effects must be examined.

Each function i receives four scores under a specified rubric: Coherence Cᵢ, Capacity Kᵢ, Stress Sᵢ and Abstraction Aᵢ. In the source study these use a 1–10 scale. Rubric versions and historical adapters must accompany every dataset; scores produced under different versions cannot simply be assumed interchangeable.

The proposed coordinates are:

$$
P_i = C_i A_i, \qquad H_i = K_i - S_i.
$$

P is intended to represent shared symbolic organisation: the conjunction of coherence and abstraction within a function. H is intended to represent functional headroom: capacity relative to stress. Neither label is a direct observation. Both are interpretations of arithmetic applied to rubric scores.

The summed symbolic coordinate is:

$$
\Psi = \sum_{i=1}^{8} C_i A_i.
$$

Two societies can have the same Ψ while distributing their scores very differently. CAMS therefore investigates the arrangement of the eight functions as well as any aggregate.

These calculations introduce assumptions. Multiplication assumes that the combination of Coherence and Abstraction is meaningfully represented by their product. Subtraction assumes that Capacity and Stress are sufficiently commensurable for their difference to be useful. A numerical zero in H is the equality of two rubric scores; it is not yet an independently calibrated boundary between viable and non-viable organisation. Sensitivity to rescaling, alternative combinations and aggregation must be reported. A percentage change in an index must not be presented as the same percentage change in a society’s actual capacity.

## 3. From scores to relationships

The central question concerns relationships: which functions change together, which diverge, and whether those patterns persist through disruption.

Let Wᵢⱼ denote an estimated association between changes in functions i and j. In this paper, “coupling” means that estimated association unless a causal relationship is separately established. Shared movement can arise from interaction, a common external shock, overlapping rubric definitions or a scorer’s tendency to move several scores together.

A reproducible coupling analysis must specify which variables enter the estimate, whether it uses levels or changes, the observation window, minimum series length, treatment of irregular intervals and missing values, and its uncertainty. Short or sparse historical series may support descriptions of trajectories without supporting reliable network estimates. A snapshot of eight scores is not, by itself, an observed network.

The further proposition that headroom changes the strength of relationships can be written schematically as:

$$
W^{*}_{ij}(t) = W_{ij}\,g\bigl(H_i(t),H_j(t)\bigr).
$$

This is a model proposal, not an implemented estimator. The gain function g and the interpretation of W must be specified before a confirmatory test. A relationship built into g cannot subsequently be reported as a discovery from the data.

The working conjecture is that sustained strain sometimes concentrates coordination around Helm, Shield and Lore: direction, coercion and collective meaning. The system may retain considerable organisation while narrowing the range of responses it can sustain. This paper calls that pattern constriction.

Another conjecture is that relationships reorganised during a crisis can persist after the immediate strain eases. If the arrangement differs on entry to and exit from a crisis at comparable headroom, that is evidence consistent with hysteresis. It is distinct from recovery merely taking longer than decline.

The Zeitgeist is the proposed connection between these arrangements and lived public meaning: what people fear, justify, celebrate and consider possible. That connection requires evidence of its own. It cannot be established by giving a network a suggestive mythic name.

## 4. What the numbers can claim

CAMS is a proposed instrument for comparing societal coordination. Its measurement validity and predictive usefulness remain under test.

A high score does not confer legitimacy or virtue. Coherent organisation can serve conquest, exploitation or mass violence. Conversely, disagreement can accompany the expansion of freedom. Claims about welfare, democratic capacity or justice require evidence beyond the coordination scores.

The source manuscript reports a pooled predictive AUC of 0.563 (mean leave-one-society-out AUC across 37 societies, all 126 coded ruptures; t(36) = 1.76, p = .087), only modestly above the chance reference of 0.5. That figure should travel with its outcome definition, sample, evaluation design and uncertainty; it is not a general accuracy rating for CAMS. Descriptive structure, early warning and prediction of particular events are separate claims.

The number and boundaries of the functions are provisional. So are their weights, scales and derived indices. The representation should be compared with simpler summaries, alternative groupings and models based on independently observed variables. Complexity earns its place only when it contributes something reproducible and useful.

## 5. Six hypotheses

The following are research hypotheses with proposed tests, not a completed preregistration. Before new confirmatory analysis, each requires a dated protocol specifying eligible cases, exclusions, estimators, effect sizes of interest, uncertainty and decision rules. A study with insufficient precision is inconclusive; a failed significance test alone does not establish absence.

### H1. Coupling persists across regime change

Within a society, estimated coupling patterns are more persistent across changes of government, dynasty or ideology than a regime-centred account would predict.

Test this by comparing changes across independently identified regime transitions with changes during other intervals and differences between societies, while controlling for common trends, scoring artefacts and temporal dependence. Compare the society-based explanation directly with regime-based alternatives.

The hypothesis is weakened if within-society persistence disappears under those controls, or if regime categories explain the patterns as well as or better than society identity.

### H2. Constriction precedes large-scale coercion

Sustained low headroom is followed by an increased concentration of coupling around Helm, Shield and Lore before an independently dated escalation of coercion or war.

The test must define duration, concentration, event severity and lead time in advance. It must include strained societies that do not escalate, so that false alarms count alongside apparent warnings. Event timing must come from evidence outside the CAMS scores.

The hypothesis is weakened if concentration occurs only during or after escalation, or adds no useful information beyond simpler measures of existing conflict and stress. Temporal precedence would support an early-warning interpretation. Establishing a mechanism would require additional causal evidence.

### H3. Recovery follows a different path

A society’s estimated coupling pattern differs between deterioration and recovery at comparable levels of headroom.

Compare the two phases using a predefined matching rule and topology-distance measure. Specify whether matching concerns aggregate headroom or the full eight-function profile: the same aggregate can conceal very different distributions. Account for elapsed time, external conditions and uncertainty.

The hypothesis is weakened if recovery retraces deterioration within the registered tolerance. Recovery speed is a separate outcome and must not substitute for evidence of hysteresis.

### H4. Collapse pathways differ in recovery

Collapses beginning with declining coherence in coordinating functions, while material capacity remains relatively intact, recover differently from collapses involving severe capacity loss and destruction of institutions that preserve knowledge. The specific directional prediction is faster recovery in the former group.

Classify pathways without using subsequent recovery to assign the category. Use external evidence of material destruction and continuity of knowledge institutions, alongside the scored trajectories. Define recovery consistently and test whether the classification adds information beyond collapse severity, duration and external conditions.

Candidate cases include Ur III, the Shang–Zhou transition, the Late Bronze Age Levant and the post-Harappan Indus, subject to evidence adequacy. The Egypt–Aegean contrast that helped formulate this hypothesis is exploratory and cannot serve as its independent confirmation.

The hypothesis is weakened if adequately observed cases show no meaningful difference, the opposite difference, or no contribution beyond simpler explanations.

### H5. Narrative aligns with coupling

Independently coded dominant narratives align with the functional relationships estimated from CAMS scores.

Coders who have not seen the scores should classify texts under a separate, predefined scheme. Test their classifications against the coupling estimates and against plausible alternatives, including stable genre conventions and common responses to events. Wherever possible, separate the evidence used for narrative coding from that supplied to scorers.

The hypothesis is weakened if alignment does not exceed those baselines. Alignment alone would not establish that myth follows coupling: stories can also organise action and reshape relationships. A directional claim requires a separate temporal test.

### H6. Setting helps predict coupling

Ecological and economic conditions help predict functional relationships across societies beyond the familiar cases used to formulate the framework.

Define the relevant conditions independently—such as dependence on irrigation, maritime exchange, mobility or territorial distances—rather than assigning broad categories after inspecting the scores. Test their contribution on held-out cases and account for shared history, technological change and contact between societies.

The hypothesis is weakened if these variables add no useful information beyond simpler baselines or if apparent effects depend on a few famous examples. The claim is probabilistic, not a geographical destiny.

## 6. First look: the deep-time study

The study dated 2 October 2026, Reading Societies Without Their Stories, is reported in the source manuscript as a blind, preregistered examination of ancient collapse and recovery. The account describes twenty society-windows and 22 passes per prompt: 15 from Claude, four from Grok and three from GPT, using a deep-time adapter to the v1.2-OPT rubric. It reports that passes breaching blindness were excluded.

These are study-reported details and results, not independently verified findings in this redraft. A publication-ready methods account must reconcile the manuscript’s separate reference to “ten time windows”, identify model versions and independent runs, and document exclusions. The two counts are consistent: Prompt A scored 12 societies at one anchor year each, and Prompt B scored ten time windows (eight added windows plus the two collapses rescored as overlap anchors), giving 20 distinct society-windows. The study page names the scorers as Claude (Opus 5.5), Grok 4.7 and GPT (ChatGPT Work/Codex, version unverified), and states that the model behind the third Claude run is inferred from its scores, not confirmed. Unequal pass counts must also be reflected in any pooling scheme; repetitions within a model family are not independent historical witnesses.

The reported Egyptian series shows mean Coherence for Helm and Lore declining across two windows before the First Intermediate Period: from 8.1 to 5.1 under Claude, 8.0 to 5.4 under Grok, and 8.0 to 6.5 under GPT. In the Aegean series, the corresponding measure reportedly remained comparatively stable until breakdown.

That contrast helped generate the proposed distinction between collapse pathways. It does not establish that distinction. A decline in model-scored coherence cannot, by itself, identify the historical cause of a collapse.

The source also reports a sharper Aegean decline and slower return towards earlier score levels. Exact rates previously labelled “headroom per century” are reported here as mean H, pending recomputation, because the manuscript also states that the series have not yet been recomputed as P and H. The underlying calculations must establish whether those rates describe K − S, Node Value or another quantity. The study’s own definitions and scripts settle that question: it defines Energy as Capacity − Stress, and its per-century rates are changes in mean Energy across the eight nodes (nodes left NA are excluded from the mean). Because H = K − S, these are rates of mean H. As reported, and pending recomputation from raw scores, mean H changed per century as follows (Claude/Grok/GPT ensembles): Egypt, fall −2.86/−2.02/−1.75 and recovery +2.56/+2.01/+1.39; Aegean, fall −7.27/−6.19/−4.17 and recovery +1.95/+1.27/+1.59. Each fall runs from the last pre-collapse window to the breakdown window, and each recovery from the breakdown window to the first recovery window. P (C × A) and per-node coupling have not yet been computed for these series. Recovery dates likewise require a specified baseline and threshold before becoming comparative evidence.

Reported patterns include relatively persistent Lore scores, low Hands scores in many state windows, and a smaller Hands gap in the sampled Aboriginal Australian and Māori cases. These observations invite investigation of whose capacities a polity sustains and whose strain it transfers. They remain dependent on case selection, rubric interpretation and the surviving record.

A separate hypothesis in the original study’s register concerned whether Mesopotamian debt-relief edicts were associated with lower labour stress over the specified comparisons. Call this DT-H4 to distinguish it from the new collapse-pathway H4. The reported result failed the registered prediction under all three model families: Hands Stress was lower than the comparator in none of the three windows.

That failure stays in the record. The possibility that edicts were responses to already elevated debt strain limits the design’s causal interpretation; it does not turn the failed test into a successful one. A redesigned test must be registered as a new test.

The study also exposed an instrument problem. GPT reportedly treated the absence of writing as grounds to leave Lore and Archive unscored in societies without written records, while Claude and Grok assigned substantial scores. Oral tradition can preserve law, genealogy, ecological knowledge and procedures. A rubric that equates writing with memory risks measuring a documentary preference of the scorer.

The proposed DT-2 revision explicitly recognises oral evidence, separates “function demonstrably absent” from “evidence insufficient”, and clarifies how skilled but coerced labour enters the assessment. Scores produced under that revision will need versioned comparison with earlier scores. A correction improves the instrument; it does not retroactively validate its previous readings.

## 7. The next test

The proposed next study spans six settings: a mobile forager society; Liangzhu in the lower Yangtze; Old Kingdom Egypt; Mycenaean Pylos; Qin and Han China; and a maritime Polynesian society. Broad categories must become bounded cases and time windows before registration. Egypt and Pylos are returning cases; the remaining cases are new to the reported study, but none should be described as historically unknown to a language model.

The order of work is part of the protection.

First, freeze the rubric, evidence rules, case selection, analysis code and hypotheses. Identify which comparisons are exploratory and which are confirmatory. Register the coupling estimator and null models before inspecting the new scores. For time-series claims, the nulls must preserve relevant temporal dependence rather than merely shuffle observations.

Second, obtain independent scoring runs with the hypotheses and expected pairings withheld. Record model versions, prompts, evidence access, missingness and exclusions. Concealing names is useful but insufficient: recognisable events can reveal identities and outcomes. Specialist scoring, where feasible, should also precede exposure to model scores.

Third, seal the raw scores and calculate the registered quantities. Propagate scoring uncertainty into the derived indices and relationships. Report sensitivity to model family, rubric choices and alternative representations. Separate well-supported trajectory descriptions from network estimates the series cannot sustain.

Fourth, compare results with independent evidence and simpler baselines. Adding another model family tests sensitivity to the scorer; it does not by itself supply independent historical validation. Specialist judgements are valuable, but should also disclose their sources and uncertainty.

Only then reconstruct what it might have felt like to stand in each society. Those voices may communicate an interpretation. They remain outside the evidence used to test it.

The decisive question is whether the representation contributes reproducible information beyond what its scoring instructions and familiar historical narratives already supply.

## 8. Limits and standing
The observer belongs inside the inquiry. Language models inherit uneven archives, familiar narratives and shared training influences. Historians also work through incomplete records and interpretive traditions. Neither source of judgement is exempt from scrutiny.

Agreement between scorers establishes consistency under specified conditions; it does not establish historical truth. A shared rubric can produce shared errors. Apparent coupling can arise from common movement across scores, and apparent differences between societies can arise from differences in documentation. Missing evidence must not silently become low capacity.

The source manuscript reports a dominant common field in the panels and limited separation between some higher-order structure and scorer disagreement. Those observations make it especially important to test whether P and H contribute distinguishable information and whether the eight-function representation improves on simpler alternatives. The answer remains open.

The natural-history premise motivates comparison. It cannot certify a particular division into eight functions, an arithmetic transformation or a causal account of collapse. Each must earn its standing through the work it performs.

CAMS proposes a common representation of societal coordination. Its functions, scoring scales and derived coordinates are provisional choices whose usefulness must be demonstrated. Failures of individual hypotheses require revision of the corresponding claims. Repeated failure to recover reproducible structure, agreement with independent evidence or useful performance against simpler alternatives would count against the representation itself.

The source manuscript identifies the backstory folder of the KaliBond/wintermute repository as the location of the deep-time data, prompts and code. Publication should cite a fixed version of that record, with the relevant registrations and exclusions, so that the claims on this page can be checked against the work that produced them.

No society receives an exemption from comparison; no part of the instrument receives an exemption from revision.
