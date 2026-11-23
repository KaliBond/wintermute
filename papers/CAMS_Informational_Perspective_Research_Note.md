# CAMS from an Informational Perspective

**A research note on the Shannon layer: what the corpus holds, what the laboratory shows, and what is ready for pre-registration**

Status: exploratory synthesis. No empirical claims beyond the cited literature and the cited CAMS corpus; all simulation findings are conditional on a constructed toy.
Provenance: extends the CAMS Shannon Laboratory v2.0 (version `d0080d0`) and v2.1 (`7231113`), built on the CAMS Information Laboratory v1.0. Corpus anchors: KaliBond/wintermute, notably *Entropy Flows and Collapse Signatures* (Paper 4, corrected final internal draft, 16 May 2026) and the *Appendix on the Eight-Node Optimum*.
Date: 2026-11-24.

---

## 1. Where the corpus already stands

The CAMS programme has carried an informational vocabulary from early on, but in two distinct registers that have never been formally joined.

The first register is **thermodynamic**. Paper 4 frames societies as far-from-equilibrium dissipative structures: fast-loop nodes (Shield, Craft, Flow, Hands) export entropy while slow-loop nodes (Helm, Lore, Archive, Stewards) generate and store negentropy, with the Library Attractor ℒ = (B_Lore × B_Archive)/S_Hands as the coupling measure. Its falsifiable yield is the Stewards Bond Strength floor breach — the one collapse signature present in Germany 1910–1930 and Late Rome 380–450 CE and absent in the UK control — plus the disciplined finding that an ℒ gradient is shock-response, not diagnosis.

The second register is **compression-theoretic**. The eight-node appendix grounds the architecture in near-decomposability (Simon 1962), synchronisation theory, and model-order selection: the node count as a bias–variance optimum under MDL/AIC/BIC-style penalties, with coupling channels scaling as N(N−1)/2. The Compression Theorem — eight nodes as a regime-dependent coarse-grained basis rather than natural kinds — lives in this register.

What neither register supplied, until now, is the **operational Shannon layer**: bits, measured. Not entropy as metaphor or as penalty term, but entropy, mutual information and transfer entropy estimated from belief or score trajectories. The Shannon Laboratory (v2.0, v2.1) built that layer on a constructed toy. This note reviews the intellectual lineage the layer inherits, states the correspondence with CAMS quantities and its limits, reports what the toy demonstrates, and extracts the hypotheses that are ready for pre-registration against the unified panel.

## 2. The lineage

Four strands converge on the claim that a society can be studied as an information-processing system without metaphorical licence.

**Entropy as uncertainty.** Shannon (1948) defined information as reduction of uncertainty, a functional of probability distributions with no necessary physical referent; Wiener (1948) made communication and control the joint grammar of machines and organisms; Jaynes (1957) showed maximum-entropy reasoning is principled inference, not physics envy. This strand licenses the use of "entropy" in social modelling at the operational layer — the appendix's three-layer defence (analogue, operational, validation) is exactly this move, and it survives reviewer scepticism because the mathematics does not depend on the metaphor.

**Economies and organisations as information processors.** Hayek (1945) posed the knowledge problem: the relevant facts of a society exist only dispersed among millions of actors, and the central problem is their transmission and use — the intellectual ancestor of the laboratory's upward-feedback gate. Galbraith (1974) formalised organisation design as matching information-processing capacity to task uncertainty. North (1990) read institutions as uncertainty-reducing structure. In this strand, what CAMS calls Capacity is close to channel capacity, and what it calls Stress is noise on the channel.

**Societies selecting their information-processing dimensions.** Van der Leeuw's long programme — from van der Leeuw, Lane and Read (2009) on information processing in the emergence of social organisation to van der Leeuw and Dirks (2025), *Society Building as Collaborative Selection of Information-Processing Dimensions?* — argues that societies grow by adding dimensions along which they process information, and that crises involve the collapse or narrowing of that dimensionality. This is the closest external relative of the Compression Theorem, and the laboratory's stated conceptual companion. Turchin et al. (2018), finding a single principal dimension structuring global variation in social complexity across the Seshat record, supply the quantitative-historical foil: one strong common mode against which any multi-node information structure must prove itself. Tainter (1988) adds the collapse-side cost accounting: complexity is problem-solving capacity purchased at rising marginal cost, which in information terms is a claim about diminishing bits per unit of coordination effort.

**Measured information flow.** Schreiber (2000) introduced transfer entropy to distinguish driving from responding elements by conditioning on the target's own past; Lizier, Prokopenko and Zomaya (2008) localised it as a spatiotemporal filter for distributed computation; Watanabe (1960) had earlier given total correlation as the multivariate generalisation of mutual information — the formal name of the laboratory's redundancy measure. Two results in this strand discipline everything below. Barnett, Barrett and Seth (2009) proved transfer entropy and Granger causality are equivalent for Gaussian variables — on approximately Gaussian scored series, the autoregressive proxy is not a shortcut but the same quantity, cheap to estimate. And James, Barnett and Crutchfield (2016) showed that transfer entropy is not, in general, a measure of mechanistic information *flow*: it quantifies predictive information transfer, and conflating the two is a category error. Flack (2017) completes the frame from the collective-computation side: coarse-graining as a downward causation mechanism — Abstraction, in CAMS terms, is what a system does when it compresses microlevel variation into slow, predictive aggregate variables that then constrain the components.

## 3. The correspondence, stated with its limits

The laboratory's working dictionary, kept deliberately conservative:

| CAMS quantity | Shannon reading | Status |
|---|---|---|
| Coherence C | Redundancy: total correlation across node streams (Watanabe 1960) | Computed in lab |
| Capacity K | Channel capacity: bits movable per unit time | Illustrative model only |
| Stress S | Channel noise: crossover probability, entropy injection | Illustrative model only |
| Abstraction A | Compression: bits retained per observation; coarse-graining (Flack 2017; Rissanen 1978) | Not yet modelled |
| Bond strength B_ij | Mutual information between node streams | Computed in lab |
| η_loop (memory feedback) | Directed transfer entropy toward Helm and Lore | Computed in lab |
| λ₂ | Mixing rate: how fast local news becomes global | Not yet modelled |
| Node value V_i | None. Value is not a quantity of information | Deliberately unmapped |

Three disciplines attach to the table. First, it is a correspondence to test, not an identity; the CAMS record shows recombinations of its own quantities failing four times when they merely redescribed V (FC3, the σ demotion, the static-responsiveness retraction, the PAS null), and the Shannon layer is held to the same bar: bits must predict something C, K, S and A do not. Second, the James–Barnett–Crutchfield (2016) critique is accepted in advance: transfer entropy here means predictive information transfer under a stated estimator, never a claim that a node "sends bits" in a mechanistic sense. Third, Lore and Archive are treated as distinct functions throughout — they are co-vulnerable, not interchangeable, and the laboratory's signature table shows them informationally separable.

## 4. What the laboratory demonstrates

The toy is a constructed mechanism demonstration: eight belief-holding functions, directed message passing, a randomly reversing environment, paired societies differing only in the upward-feedback gate. Within those assumptions, the Shannon layer shows:

**The asymmetry of executive information.** Upward transfer entropy (Hands/Craft/Flow → Helm) exceeds downward (Helm → operators) by roughly an order of magnitude in both societies (0.108 vs 0.017 bits in A; 0.156 vs 0.011 in B, shuffle-null p ≈ 0.02), and opening the gate raises both the upward flow and the bits Helm carries about the true environment (0.170 → 0.248). Broadcast weight is not information flow: a loud executive is not an informed one, and the operators' knowledge of reality does not need the executive's help. This is Hayek's knowledge problem rendered measurable inside a toy.

**Functions have information-processing characters.** Hands is a pure sensor (highest environmental mutual information, 0.526 bits; informs widely, informed by nobody). Archive is the system's informational sink (highest transfer entropy in, 0.159; near-zero out, 0.012; lowest entropy rate). Helm is an integrator: informed by all, originator of little. Average-linkage clustering on the five-feature signatures recovers 6/8 of the LAPH layer assignment — and the mismatches instruct: this sluggish, low-evidence Helm clusters with the cognitive layer, not the executive one.

**Memory can be a drag.** Silencing Hands costs Helm the most environmental information (0.034 bits in B). But silencing Archive *gains* Helm 0.031 bits: the memory channel anchors the executive to stale states. η_loop measures whether memory holds society together; the silencing sign measures whether memory holds the executive back. Both effects are real, and their tension — continuity versus conservatism — is exactly the tension Paper 4's recovering-Germany versus terminal-Rome contrast poses in another vocabulary.

**Redundancy is dominated by common drive.** Total correlation runs 5.5–5.8 bits of a possible 8 in both societies, barely moving with the gate, because shared observations and the common broadcast lock the functions together. This is the toy's version of the panel's known structure — PC1 of ΔV carries 71.7% of shared motion — and it warns that any redundancy measure on real CAMS data will mostly measure the common mode unless the FSLS surrogate discipline is applied.

None of this validates CAMS. It validates a measurement programme: signatures, emergence checks, and interventions, each portable to data where the parameters are not imposed.

## 5. Hazards on the path to data

Four hazards are registered in advance, because each has burned this programme or its neighbours before.

*Estimator bias and resolution.* Plug-in entropies are biased downward at short series lengths; the unified panel's irregular sampling (annual to >5-year strata) makes raw series lengths short and heterogeneous. Mitigation: Miller–Madow or shuffle-corrected estimators, society-normalised statistics, stratification by sampling gap, and the Gaussian shortcut where justified — Barnett et al. (2009) makes Granger causality an efficient estimator of the same quantity, with both computed and concordance reported.

*The common mode.* With PC1 carrying ~72% of shared motion, mutual information and total correlation between node series are guaranteed to be large and uninteresting. The FSLS surrogate null (registered 2026-08-14, with its two subspace repairs) exists precisely to destroy cross-node alignment while preserving marginals, autocorrelation and the common mode. No cross-node information claim should be reported except against FSLS.

*The within-node trap.* New arithmetic on C, K, S, A combinations has failed four times. The informational layer computes over *trajectories of the validated component levels*, not over new recombinations; any candidate that reduces to a redescription of V is to be discarded at the gate, as the PAS amendment ruling (2026-10-19) requires.

*Interpretation drift.* Transfer entropy without the James–Barnett–Crutchfield caveat, "entropy" without the analogue/operational/validation separation, and layer emergence without the toy's honesty clause (in a toy the parameters wrote the signatures) are the three rhetorical failure modes. All three are barred from the manuscript register.

## 6. Pre-registration candidates

Five hypotheses, stated with operational definitions, pass/fail bars and nulls, ready to move into a protocol document. All use the unified panel (36 societies, ~4,000 society-year transitions), canonical V_i series computed from raw C, K, S, A (bond-drift check first), society-normalised pooling, and the FSLS surrogate null.

**INFO-1 — Nodes have stable informational signatures.** Compute, per society and node, the four-feature signature over V_i trajectories: entropy, entropy rate, mean transfer entropy in, mean transfer entropy out (quantised ternary symbols; Granger proxy reported alongside). Within-node cross-society signature similarity exceeds between-node similarity. Bar: median within-node Spearman ρ of signature vectors across societies exceeds the between-node median with permutation p < 0.01; blind sets 1, 2, 3, 5, 6, 7 held out for replication. Fail condition: the inequality reverses or the null is not cleared.

**INFO-2 — The executive is informationally downstream, and the asymmetry attenuates before crisis.** Upward transfer entropy (Craft, Hands, Flow → Helm, Lore) exceeds downward transfer entropy in quiescent windows across the panel; the asymmetry index attenuates in the five years preceding the 92 registered crisis onsets. Bar: pre-registered directions, permutation p < 0.01 for both contrasts, society-normalised. Registered hedge: at annual resolution, lead/lag claims are limited to "among the leading indicators", never "the" leader.

**INFO-3 — Crisis is a redundancy collapse.** Total correlation across the eight V_i series drops during crisis observations while mean entropy rate rises — desynchronisation (ω↑, λ₂↓) expressed in bits. Bar: AUC ≥ 0.60 against the blind-ablation crisis-flag convention, society-normalised, and the effect must shrink but remain signed under FSLS surrogates; a result that vanishes under FSLS is reported as common-mode artefact.

**INFO-4 — Layers emerge from information structure.** Average-linkage clustering of the INFO-1 signatures recovers the LAPH layer assignment (cognitive: Archive, Lore, Stewards; executive: Helm, Shield; material: Craft, Flow, Hands) at ≥ 6/8 nodes in at least two-thirds of eligible societies, against the distribution of matches under FSLS surrogates. Bar: observed match rate above the 95th percentile of the surrogate distribution. Any pass is reported with the toy's honesty clause inverted: parameters are not imposed here, so emergence is evidence — about the scoring grammar's information structure, not yet about the world.

**INFO-5 — The silencing sign on data.** The panel cannot be intervened on, but the laboratory's silencing has a statistical analogue: conditional transfer entropy, removing each node's contribution by projecting it out of all other series (or equivalently, Granger tests with node k conditioned away). Hypothesis: in episodes flagged by the ΔV-based failure-locus convention, removing the locus node's informational contribution reduces Helm's predictability most; and the sign of the Lore/Archive removal effect distinguishes homeostatic from transformative recoveries — negative (removal helps) marking conservatism drag. Bar: ranking concordance with the failure-locus distribution at κ ≥ 0.20, the T-M3d bar, with the mechanical-coupling audit applied since locus and outcome share scoring inputs.

## 7. What this note is not

It does not claim the eight-node architecture is minimal or optimal — the appendix's own framing makes eight an adjudicable model-order choice, and the blind ablation retained it as a well-motivated working grammar. It does not claim nodes are organs or agents; they are functional grammar, and the informational signatures are properties of the grammar's trajectories. It does not revise any canonical operator. And it does not promote the toy's numbers to evidence: the toy calibrates intuition and shames vague vocabulary; only the panel can confirm or kill the candidates above.

## References

- Barnett, L., Barrett, A. B., & Seth, A. K. (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters*, 103(23), 238701.
- Flack, J. C. (2017). Coarse-graining as a downward causation mechanism. *Philosophical Transactions of the Royal Society A*, 375(2109), 20160338.
- Galbraith, J. R. (1974). Organization design: An information processing view. *Interfaces*, 4(3), 28–36.
- Hayek, F. A. (1945). The use of knowledge in society. *American Economic Review*, 35(4), 519–530.
- James, R. G., Barnett, N., & Crutchfield, J. P. (2016). Information flows? A critique of transfer entropies. *Physical Review Letters*, 116(23), 238701.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620–630.
- Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2008). Local information transfer as a spatiotemporal filter for complex systems. *Physical Review E*, 77(2), 026110.
- McKern, K. F. (2026). *Entropy flows and collapse signatures* (CAMS Paper 4, corrected final internal draft). KaliBond/wintermute.
- McKern, K. F. (2026). *Appendix on the eight-node optimum in multi-timescale coordination networks*. KaliBond/wintermute.
- North, D. C. (1990). *Institutions, institutional change and economic performance*. Cambridge University Press.
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465–471.
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461–464.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423; 27(4), 623–656.
- Simon, H. A. (1962). The architecture of complexity. *Proceedings of the American Philosophical Society*, 106(6), 467–482.
- Tainter, J. A. (1988). *The collapse of complex societies*. Cambridge University Press.
- Turchin, P., Currie, T. E., Whitehouse, H., François, P., Feeney, K., et al. (2018). Quantitative historical analysis uncovers a single dimension of complexity that structures global variation in human social organization. *PNAS*, 115(2), E144–E151.
- van der Leeuw, S., & Dirks, G. (2025). Society building as collaborative selection of information-processing dimensions? *Global Perspectives*, 6(1), 143744.
- van der Leeuw, S., Lane, D., & Read, D. (2009). The long-term evolution of social organization. In D. Lane et al. (Eds.), *Complexity perspectives in innovation and social change*. Springer.
- Watanabe, S. (1960). Information theoretical analysis of multivariate correlation. *IBM Journal of Research and Development*, 4(1), 66–82.
- Wiener, N. (1948). *Cybernetics: or control and communication in the animal and the machine*. MIT Press.
