# CAMS from an Informational Perspective

**A research note on the Shannon layer: what the corpus holds, what the laboratory shows after audit, and what is ready for pre-registration**

Status: exploratory synthesis, second edition. No empirical claims beyond the cited literature and the cited CAMS corpus; all simulation findings are conditional on a constructed toy.
Provenance: extends the CAMS Shannon Laboratory v2.0 (version `d0080d0`), v2.1 (`7231113`) and v2.2 (`b865d82`), built on the CAMS Information Laboratory v1.0. Corpus anchors: KaliBond/wintermute, notably *Entropy Flows and Collapse Signatures* (Paper 4, corrected final internal draft, 16 May 2026) and the *Appendix on the Eight-Node Optimum*. This edition incorporates an independent audit of the laboratory and of the first edition (24 November 2026), whose scripts reproduce every default figure and test every headline claim against a control that could break it.
Date: 2026-11-24 (v2, same day, post-audit).

---

## 1. Where the corpus already stands

The CAMS programme has carried an informational vocabulary from early on, in registers that have never been formally joined.

The first register is **thermodynamic**. Paper 4 frames societies as far-from-equilibrium dissipative structures: fast-loop nodes export entropy while slow-loop nodes generate and store negentropy, with the Library Attractor η_loop = (B_Lore × B_Archive)/S_Hands as the coupling measure. Its falsifiable yield is the Stewards Bond Strength floor breach, the one collapse signature present in Germany 1910–1930 and Late Rome 380–450 CE and absent in the UK control, plus the disciplined finding that an η gradient is shock-response, not diagnosis.

The second register is **compression-theoretic**. The eight-node appendix grounds the architecture in near-decomposability (Simon 1962), synchronisation theory, and model-order selection under MDL/AIC/BIC-style penalties. The Compression Theorem, eight nodes as a regime-dependent coarse-grained basis rather than natural kinds, lives in this register.

The first edition of this note claimed that neither register supplied an operational Shannon layer, and the audit correctly flagged the claim as wrong about the repo. The repository already contains **dDIG** (`src/ddig_analysis.py`): conditional mutual information between one node's change and the other nodes' next change, given regime, shock size and decade. Conditioning on a shared-shock proxy is precisely the common-drive control the laboratory needed. The repo also carries a Shannon evenness index for the stress distribution (**ESD**, `src/cams_analyzer.py`). What dDIG and ESD do not supply is the interactive estimator bench: a place where plug-in entropies, transfer entropies, intervention baselines and nulls can be varied and watched. The Shannon Laboratory is that bench, and dDIG is the natural starting estimator for the panel tests in Section 7, subject to the two repairs registered there.

## 2. The lineage

Four strands converge on the claim that a society can be studied as an information-processing system without metaphorical licence. The audit adds a fifth, and it is the one that carries a theorem.

**Entropy as uncertainty.** Shannon (1948) defined information as reduction of uncertainty, a functional of probability distributions with no necessary physical referent; Wiener (1948) made communication and control the joint grammar of machines and organisms; Jaynes (1957) showed maximum-entropy reasoning is principled inference, not physics envy. This strand licenses "entropy" in social modelling at the operational layer: the appendix's three-layer defence (analogue, operational, validation) is exactly this move, and it survives reviewer scepticism because the mathematics does not depend on the metaphor. The cost, named by the audit, is that entropy is narrative-blind: it cannot tell truth from shared error. A society agreeing on something false shows the same redundancy as one agreeing on something true. Truth enters only through mutual information with an outside referent: harvests, prices, mortality, battle outcomes. Without such a series, no information measure on CAMS scores can separate adaptation from shared delusion.

**The coding condition.** Shannon's channel coding theorem says reliable transmission is possible only while the source's entropy rate is below the channel's capacity; above it, errors are unavoidable. Read K as a node's capacity and S as entropy arriving unprocessed, and the sign of K − S is that threshold. Ashby's (1956) law of requisite variety is the same statement from cybernetics, and van der Leeuw's definition of crisis as a temporary incapacity of a society's information processing is the same statement from archaeology. The craftsman who starts cutting corners when stress exceeds capacity is the theorem in action: above capacity, information must be dropped. And the coder facing overload has exactly two exits, both already named in the doctrine. Lower the rate and add redundancy: repeat, simplify, use fewer symbols, and C holds while A falls (the regression to Helm, Shield and Lore, the textbook response to a noisy channel rather than a failure of nerve). Or hold the rate and lose agreement: receivers decode different messages, C falls while A holds, and Γ = A/C rises (masking abstraction). The testable fork: after S overtakes K, which exit does a node take?

**Compression and what it keeps.** The information bottleneck (Tishby, Pereira and Bialek 1999) asks how to compress an input while keeping what matters about a target. Abstraction is a move along that trade-off: fewer bits kept, ideally the relevant ones. Healthy abstraction compresses while keeping information about the environment; masking abstraction keeps the code shared across nodes while environmental information drains. Rissanen (1978) supplies the model-order discipline; Flack (2017) the collective-computation frame, coarse-graining as downward causation; Bialek, Nemenman and Tishby (2001) the quantity that disciplines memory: predictive information, the bits the past holds about the future beyond the present. A memoryless environment offers none, which is why the laboratory's original Archive could only ever be a drag.

**Economies, organisations and societies as information processors.** Hayek (1945) posed the knowledge problem: the relevant facts exist only dispersed, and the problem is their transmission and use. Galbraith (1974) formalised organisation design as matching information-processing capacity to task uncertainty. North (1990) read institutions as uncertainty-reducing structure. Van der Leeuw's long programme (van der Leeuw, Lane and Read 2009; van der Leeuw and Dirks 2025) argues societies grow by adding dimensions along which they process information, and crises narrow that dimensionality. Turchin et al. (2018) supply the quantitative-historical foil: one strong common mode structuring global variation in social complexity, against which any multi-node information structure must prove itself. Shin et al. (2020), working in the same Seshat record, found that polities first grow in scale and must then improve information processing (writing, records, money) before growing further: the closest published support for Abstraction as externalised memory. Tainter (1988) adds the cost accounting: complexity is problem-solving capacity at rising marginal cost, in information terms a claim about diminishing bits per unit of coordination effort.

**Measured information flow.** Schreiber (2000) introduced transfer entropy; Lizier, Prokopenko and Zomaya (2008) localised it; Watanabe (1960) gave total correlation as the multivariate generalisation of mutual information. Two results discipline everything below. Barnett, Barrett and Seth (2009) proved transfer entropy and Granger causality are equivalent for Gaussian variables: on approximately Gaussian scored series the autoregressive proxy is the same quantity, cheap to estimate. And James, Barnett and Crutchfield (2016) showed transfer entropy is not in general a measure of mechanistic information flow: it quantifies predictive information transfer, and conflating the two is a category error. The audit demonstrates the point inside the laboratory: most raw transfer entropy there is common drive, and only intervention baselines and conditioning separate messages from shared weather. Two further tools join the kit. Kolchinsky and Wolpert (2018) define semantic information as the information about its environment that a system needs in order to persist, measured by scrambling that information and watching viability fall: a candidate analogue for node value, and exactly what the laboratory's silencing experiment measures. Rosas, Mediano, Gastpar and Jensen (2019) give O-information, which separates redundancy-dominated from synergy-dominated systems: the formal shape of the claim that a Sybond is more than its parts. Flack and Krakauer (2006) found macaque power encoded as a consensus in a network of subordination signals, measurable by Shannon entropy: the executive begins life as an agreement that reduces uncertainty, which is why broadcast weight is not information flow.

## 3. The correspondence, revised after audit

Three rows of the first edition's dictionary changed meaning, one was withdrawn, and node value gained a candidate analogue. Every row remains a correspondence to test, not an identity.

| CAMS quantity | Shannon reading | Status |
|---|---|---|
| K − S (functional margin) | The coding condition: reliable processing only while the environment's entropy rate is below capacity; two exits above it (C↑A↓ repetition, or C↓A masking) | Theorem-backed framing; fork testable on the panel |
| Coherence C (within node) | Low entropy of opinion inside one function | Not measured in the toy |
| Cross-node agreement | Total correlation (Watanabe 1960), split into agreement explained by the shared environment and agreement beyond it (TC \| E) | Computed in lab |
| Capacity K | Channel capacity: bits movable per unit time | Illustrative BSC model only |
| Stress S | The environment's entropy rate, how fast the world changes. Not observation noise: a tenfold noise rise costs Helm about 0.02 bits, while sweeping the flip rate costs about 0.6 | Computed in lab |
| Abstraction A | Compression along the bottleneck: healthy keeps bits about the world, masking keeps bits shared across nodes while world information drains. Γ = A/C reads as shared information rising while environmental information falls | Qualitative in lab; fork and Γ testable |
| Bond strength B_ij | Mutual information between node streams | Computed in lab |
| η_loop (Library Attractor) | Withdrawn. The first edition mapped η_loop to directed transfer entropy toward Helm and Lore; the repo defines η_loop as (B_Lore × B_Archive)/S_Hands, a structural operator with no Shannon analogue claimed here. Memory's informational worth is measured directly instead (Section 5) | Correction logged |
| λ₂ | Mixing rate: how fast local news becomes global | Not yet modelled |
| Node value V_i | Candidate analogue: semantic information (Kolchinsky and Wolpert 2018), the viability lost when a node's environmental information is scrambled. The silencing bars measure exactly this | Computed in lab |
| The Sybond (whole) | Synergy, if any: O-information (Rosas et al. 2019) separates redundancy- from synergy-dominated systems | Toy is redundancy-dominated by construction |

Three disciplines attach. First, the CAMS record shows within-node recombinations failing four times when they merely redescribed V (FC3, the σ demotion, the static-responsiveness retraction, the PAS null); the Shannon layer is held to the same bar, and bits must predict something C, K, S and A do not. Second, the James–Barnett–Crutchfield (2016) critique is accepted in advance and now enforced by design: every transfer entropy in the laboratory stands beside its muted-source baseline and its environment-conditioned value. Third, Lore and Archive remain distinct functions throughout: co-vulnerable, not interchangeable.

## 4. The audit: what survived and what fell

The audit ran the laboratory's own code headlessly (maximum difference from the published figures: zero), then tested each headline claim against a control that could break it. This section reports the verdicts; Section 5 reports the rebuilt laboratory.

**What fell.**

*Raw transfer entropy oversells the conversation.* Every node observes the same world, so a good sensor's past belief predicts Helm's next belief even if the two never communicate. The clean test is an intervention: cut the source off from everyone and measure again. With Hands, Craft and Flow cut off, operator-to-Helm transfer entropy still averages 0.082 bits in both societies. Against that baseline, only about 24% of Society A's raw upward figure (0.108 bits) and 48% of Society B's (0.156 bits) is messages. Helm's heavy broadcast adds essentially nothing over its own muted baseline (−0.000 bits in A, +0.001 in B): the first edition's reading, "a loud executive is not an informed one", was directionally right but for the wrong reason. The better reading is that the broadcast sets levels without adding news, and transfer entropy cannot see the level-setting.

*The upward asymmetry is conditional, not raw.* Raw upward transfer entropy exceeds raw downward in both societies (30 of 30 seeds). Conditioned on the environment, the asymmetry inverts in A (upward 0.042 against downward 0.055 bits; the upward figure exceeds the downward in only 1 of 30 seeds) and survives only in B (0.072 against 0.041; 30 of 30). Conditioning is the sharper instrument, and it is what the gate actually buys.

*"Memory is a drag" was an artefact of the design.* The laboratory's Archive had the weakest access to evidence and took most of its social input from Helm's broadcast: an echo of Helm, not a record, in a memoryless two-state environment where the past holds nothing beyond the present. Given a real record (the operators' view one cycle ago, read slightly ahead) and a recurring environment, the record's content is worth 0.218 bits to Helm in Society A and 0.085 in B, against an out-of-phase control that strips the content while keeping the voice. That is six times the largest effect in the original silencing battery.

*Three smaller corrections.* The redundancy ceiling for eight binary streams is 7 bits, not 8 (eight marginals of at most one bit minus a joint entropy of at least one). The first edition's Archive figure quoted A's value next to B's without saying so. And the 50-permutation shuffle null was lenient because it destroyed autocorrelation; the laboratory now uses 199 circular shifts and prints the achievable p-floor (0.005) beside every p.

*The first edition's corpus claims.* The "until now" claim is corrected in Section 1. The figures drawn from the validation programme's records (the FSLS surrogate null registered 2026-08-14, the 92 crisis onsets, PC1 of ΔV at 71.7%, the PAS amendment ruling, the blind ablation) are not yet checkable against the public repo; they are project records with public deposit pending, and they are marked as such wherever they load-bearing below. The audit's criticism of the v1 verifier is also accepted: it checked that strings were present, not that numbers were true. The v2 verifier recomputes the laboratory's quantities headlessly and compares them against the audit's published outputs.

**What survived.**

Opening the gate raises Helm's knowledge of the world (0.170 to 0.248 bits at the default seed; B exceeds A in 30 of 30 seeds). The upward surplus over the muted baseline is positive in every seed in both societies: attenuated, not absent. The decision-fit advantage of the open-gate society holds in 100 of 100 seeds (mean 9.5 percentage points). The coherence dip is the learning: after the shock, coherence and decision fit fall and rise together (r = +0.69 in A, +0.82 in B), because agreement dips while the operators switch first and Helm lags. A society that never dips either never heard the news or suppressed it, which makes the dip a better diagnostic than the level.

**What the audit added.**

*The authority echo.* Raising Helm's influence from 0 to 1 (gate 0.9) adds 0.161 bits of agreement beyond the shared environment and costs Helm 0.015 bits of environmental knowledge (30 of 30 seeds). Silencing Helm's own broadcast raises Helm's knowledge in every seed (0.020 bits in A, 0.021 in B); the laboratory's chart showed this bar and neither the laboratory nor the first edition mentioned it. The mechanism: the broadcast pulls the other nodes towards Helm, so their reports back to Helm carry Helm's own past. Authority contaminates the channels it depends on. This is masking abstraction in miniature: agreement up, information about the world down.

*Stress is rate of change, not noise.* Raising observation noise tenfold (σ 0.2 to 2.0) cuts Helm's knowledge only from 0.286 to 0.271 bits. Raising the environment's flip probability across its range cuts it from 0.704 to 0.035: at the slowest rate Helm holds 89% of what Hands knows, at the fastest 12%. Hands, the direct sensor, loses knowledge slowly; Helm, at the end of the processing chain, loses it fast. This is the K − S threshold inside the toy: when the world's rate of change outruns what the chain can carry, the executive goes blind first. Opening the gate lifts Helm's curve but does not change its slope.

## 5. What the laboratory now demonstrates (v2.2)

The rebuilt laboratory keeps the paired-society design and changes what the numbers mean.

**Flow is reported as surplus.** Every transfer entropy stands beside its muted-source baseline, and the tiles report the surplus and its share of the raw figure. The headline is honest now: opening the gate moves real information, and most of the raw figure is the shared environment both societies watch.

**Conditioning separates the societies.** Environment-conditioned transfer entropy, conditioned total correlation (3.733 bits in A, 3.824 in B, of 5.455 and 5.803 raw) and O-information (+3.897, +4.402) are computed on every run. The toy is redundancy-dominated at every setting, and has to be: weighted averaging copies information but cannot create joint-only bits. A model of emergence needs nonlinear coupling, and the O-information tile now says so.

**The authority echo is a panel, not a footnote.** A sweep of the influence slider charts agreement-beyond-environment against Helm's knowledge, and the two curves move in opposite directions.

**Memory is tested, not asserted.** Archive can now keep a record, the environment can recur (a 60-round cycle with phase slips), and the out-of-phase control separates content from presence. The record's worth decays from 0.218 bits in a clean cycle toward zero as the slips' entropy rate approaches 0.24 bits per round, the memoryless environment's own rate. Memory is worth exactly the environment's predictable structure. And it is worth more than twice as much to the closed-gate society: memory and upward feedback are substitutes, and when the operators cannot reach the executive, the record is the executive's main channel to the world. The Nile flood calendar is the archetype; η_loop measures whether memory holds society together, and the silencing sign measures what memory costs or earns the executive. The first edition's tension, continuity versus conservatism, survives with the sign reversed under honest design.

**The layer match carries its chance rate.** A random three-partition of eight items matches the LAPH layers 6/8 or better 12.9% of the time (16.4% at the LAPH group sizes 2,3,3). The seed-42 match of 6/8 in both societies is suggestive, not decisive; the 30-seed means (A 5.7, B 6.0) sit above the null mean but do not clear a hard bar. Layer emergence stays a candidate, demoted from a demonstration.

None of this validates CAMS. It validates a measurement programme: signatures, emergence checks, interventions and conditioning, each portable to data where the parameters are not imposed.

## 6. Hazards on the path to data

Five hazards are registered, because each has burned this programme or its neighbours before.

*Estimator bias and resolution.* Plug-in entropies are biased downward at short series lengths; the unified panel's irregular sampling makes raw series lengths short and heterogeneous. Mitigation: Miller–Madow or shuffle-corrected estimators, society-normalised statistics, stratification by sampling gap, and the Gaussian shortcut where justified (Barnett et al. 2009), with both computed and concordance reported.

*The common mode.* With PC1 carrying about 72% of shared motion (project records, deposit pending), mutual information and total correlation between node series are guaranteed to be large and uninteresting; the toy reproduces the disease, with two thirds of its redundancy explained by the shared environment. The FSLS surrogate null (project records, deposit pending) exists precisely to destroy cross-node alignment while preserving marginals, autocorrelation and the common mode. No cross-node information claim should be reported except against FSLS, and no flow claim without a muted-style control or conditioning.

*The within-node trap.* New arithmetic on C, K, S, A combinations has failed four times. The informational layer computes over trajectories of the validated component levels, not over new recombinations; any candidate that reduces to a redescription of V is discarded at the gate, as the PAS amendment ruling (project records, deposit pending) requires.

*Truth needs an outside referent.* Entropy cannot tell truth from shared error. Every viability claim (the echo test, the memory test, the dip diagnostic) needs an external outcome series: harvests, prices, mortality, battle outcomes. Shin et al. (2020) show such series exist in Seshat; T-M5's external-validation roster (ILOSTAT, WDI, UNIDO, GRD, UCDP) is the programme's own starting set.

*Interpretation drift.* Transfer entropy without the James–Barnett–Crutchfield caveat, "entropy" without the analogue/operational/validation separation, layer emergence without its chance rate, and any verifier that checks strings rather than numbers: four rhetorical failure modes, all barred from the manuscript register.

## 7. Pre-registration candidates, revised

Eight hypotheses, stated with operational definitions, pass/fail bars and nulls. All use the unified panel (36 societies, about 4,000 society-year transitions), canonical V_i series computed from raw C, K, S, A (bond-drift check first), society-normalised pooling, and the FSLS surrogate null. The starting estimator is the repo's dDIG with two repairs registered in advance: condition on the target's own past so it becomes a true conditional transfer entropy, and coarsen its regime × shock × decade context cells, which are too sparse for annual data.

**INFO-1: Nodes have stable informational signatures.** Per society and node, the four-feature signature over V_i trajectories: entropy, entropy rate, mean transfer entropy in, mean transfer entropy out (quantised ternary symbols; Granger proxy reported alongside). Within-node cross-society signature similarity exceeds between-node similarity. Bar: median within-node Spearman ρ across societies exceeds the between-node median, permutation p < 0.01; blind sets 1, 2, 3, 5, 6, 7 held out for replication.

**INFO-2: The conditional asymmetry, attenuating before crisis.** Revised after audit: the raw upward-versus-downward contrast is common drive and is withdrawn. The registered contrast is environment-conditioned transfer entropy (operator → Helm against Helm → operator), which separates the toy's societies. Hypothesis: the conditional asymmetry favours the upward direction in quiescent windows and attenuates in the five years preceding crisis onsets. Bar: pre-registered directions, permutation p < 0.01 for both contrasts, society-normalised; a muted-style control (third-party node as cut-off) reported alongside. Registered hedge: at annual resolution, lead/lag claims are limited to "among the leading indicators".

**INFO-3: Crisis is a redundancy collapse.** Total correlation across the eight V_i series drops during crisis observations while mean entropy rate rises: desynchronisation (ω↑, λ₂↓) expressed in bits. Bar: AUC ≥ 0.60 against the crisis-flag convention, society-normalised, the effect shrinking but remaining signed under FSLS; a result that vanishes under FSLS is reported as common-mode artefact. The conditioned variant TC | E is reported so that agreement explained by the shared shock is not counted as social glue.

**INFO-4: Layers emerge from information structure.** Average-linkage clustering of the INFO-1 signatures recovers the LAPH assignment at ≥ 6/8 nodes in at least two-thirds of eligible societies, against the surrogate distribution of matches. Bar: observed match rate above the 95th percentile of the FSLS distribution, with the combinatorial chance rate (12.9% for ≥ 6/8 over all three-partitions of eight items) printed beside the result.

**INFO-5: The silencing sign on data.** The panel analogue of silencing: conditional transfer entropy with each node's contribution projected out. Hypothesis: in episodes flagged by the ΔV-based failure-locus convention, removing the locus node's contribution reduces Helm's predictability most; and the sign of the Lore/Archive removal effect distinguishes homeostatic from transformative recoveries. Bar: ranking concordance with the failure-locus distribution at κ ≥ 0.20 (the T-M3d bar), with the mechanical-coupling audit applied since locus and outcome share scoring inputs.

**INFO-6: The coding fork.** In node-years where S overtakes K, classify each episode by exit: repetition (C rises or holds while A falls) or masking (C falls while A holds). Hypothesis: the masking route precedes worse outcomes (deeper V_min, slower recovery) than the repetition route. Bar: pre-registered outcome contrast at p < 0.01, society-normalised, with the fork's classification frozen before outcomes are inspected. This is the audit's sharpest contribution to the programme: a theorem-backed fork where the doctrine already had the vocabulary.

**INFO-7: The echo test.** Does a rise in Helm's informational influence over the other nodes (incoming conditional transfer entropy dominated by Helm's own past) predict a later fall in Helm's fit to an external outcome series? The toy's echo is the mechanism; the panel test needs the external referent (T-M5 roster) and a pre-registered influence measure. Bar: signed lagged association at p < 0.01 against FSLS, society-normalised, with the influence measure frozen before outcomes.

**INFO-8: Memory and feedback as substitutes.** Societies with weak upward channels should lean harder on Archive: the Archive load in Helm's predictability should be higher where upward conditional transfer entropy is low, and higher in recurring environments (flood or monsoon agriculture, long trade cycles) than in volatile ones. Bar: both contrasts signed as registered at p < 0.01, with environment recurrence classified from external records before any CAMS quantity is consulted.

A ninth item is logged as a methodological repair rather than a hypothesis. Paper 4's Germany-to-UK stress-transfer reading in 1914–1918 is the common-drive problem exactly, since both societies absorbed the same war; the repair is a lagged conditional measure with the war as the shared-shock series and a third country as the cut-off control, which is what Paper 4's own figure always required.

**A necessary caveat on the gate.** The feedback gate measures channels, not regime type, and the toy must not be read as democracy versus autocracy. China's reform era ran on experimentation under hierarchy (Heilmann 2008): local pilots whose results feed back to the centre before national rollout, a wide upward gate inside a centralised system, with its own distortions (Wang and Yang 2021: site selection and strategic local effort bias what the centre learns). Pluralist systems narrow upward channels too, through capture or polarised media. Because the gate can be measured in any system, it is a variable for comparing institutions on common ground, not a partisan label.

## 8. What this note is not

It does not claim the eight-node architecture is minimal or optimal: the appendix's own framing makes eight an adjudicable model-order choice, and the blind ablation (project records, deposit pending) retained it as a well-motivated working grammar. It does not claim nodes are organs or agents; they are functional grammar, and the informational signatures are properties of the grammar's trajectories. It does not revise any canonical operator. It does not claim synergy: the toy is redundancy-dominated by construction, and any claim that the Sybond holds joint-only information about the world waits on nonlinear coupling. And it does not promote the toy's numbers to evidence. The toy calibrates intuition and shames vague vocabulary, and after the audit it also demonstrates the virtue that matters most in this programme: a claim that cannot survive its own controls does not leave the laboratory.

## References

- Ashby, W. R. (1956). *An introduction to cybernetics*. Chapman & Hall.
- Barnett, L., Barrett, A. B., & Seth, A. K. (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters*, 103(23), 238701.
- Bialek, W., Nemenman, I., & Tishby, N. (2001). Predictability, complexity, and learning. *Neural Computation*, 13(11), 2409–2463.
- Flack, J. C. (2017). Coarse-graining as a downward causation mechanism. *Philosophical Transactions of the Royal Society A*, 375(2109), 20160338.
- Flack, J. C., & Krakauer, D. C. (2006). Encoding power in communication networks. *American Naturalist*, 168(3), E87–E102.
- Galbraith, J. R. (1974). Organization design: An information processing view. *Interfaces*, 4(3), 28–36.
- Hayek, F. A. (1945). The use of knowledge in society. *American Economic Review*, 35(4), 519–530.
- Heilmann, S. (2008). Experimentation under hierarchy: Policy experiments in the reorganization of China's state sector, 1978–2008. *Harvard CID Working Paper* 172.
- James, R. G., Barnett, N., & Crutchfield, J. P. (2016). Information flows? A critique of transfer entropies. *Physical Review Letters*, 116(23), 238701.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620–630.
- Kolchinsky, A., & Wolpert, D. H. (2018). Semantic information, autonomous agency and non-equilibrium statistical physics. *Interface Focus*, 8(6), 20180041.
- Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2008). Local information transfer as a spatiotemporal filter for complex systems. *Physical Review E*, 77(2), 026110.
- McKern, K. F. (2026). *Entropy flows and collapse signatures* (CAMS Paper 4, corrected final internal draft). KaliBond/wintermute.
- McKern, K. F. (2026). *Appendix on the eight-node optimum in multi-timescale coordination networks*. KaliBond/wintermute.
- North, D. C. (1990). *Institutions, institutional change and economic performance*. Cambridge University Press.
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465–471.
- Rosas, F. E., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). Quantifying high-order interdependencies via multivariate extensions of the mutual information. *Physical Review E*, 100(3), 032305.
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461–464.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423; 27(4), 623–656.
- Shin, J., Price, M. H., Wolpert, D. H., Shimao, H., Tracey, B., & Kohler, T. A. (2020). Scale and information-processing thresholds in Holocene social evolution. *Nature Communications*, 11, 2394.
- Simon, H. A. (1962). The architecture of complexity. *Proceedings of the American Philosophical Society*, 106(6), 467–482.
- Tainter, J. A. (1988). *The collapse of complex societies*. Cambridge University Press.
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *Proceedings of the 37th Allerton Conference on Communication, Control and Computing*, 368–377.
- Turchin, P., Currie, T. E., Whitehouse, H., François, P., Feeney, K., et al. (2018). Quantitative historical analysis uncovers a single dimension of complexity that structures global variation in human social organization. *PNAS*, 115(2), E144–E151.
- van der Leeuw, S., & Dirks, G. (2025). Society building as collaborative selection of information-processing dimensions? *Global Perspectives*, 6(1), 143744.
- van der Leeuw, S., Lane, D., & Read, D. (2009). The long-term evolution of social organization. In D. Lane et al. (Eds.), *Complexity perspectives in innovation and social change*. Springer.
- Wang, S., & Yang, D. Y. (2021). Policy experimentation in China: The political economy of policy learning. *NBER Working Paper* 29402.
- Watanabe, S. (1960). Information theoretical analysis of multivariate correlation. *IBM Journal of Research and Development*, 4(1), 66–82.
- Wiener, N. (1948). *Cybernetics: or control and communication in the animal and the machine*. MIT Press.
