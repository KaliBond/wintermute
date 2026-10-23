From Functional Morphology to Stress-Linked Deterioration

A Preregistered Falsification Sequence in the Complex Adaptive Model of Societies

Kari McKern
Independent Complexity Research · Neural Nations Research Platform
16 October 2026

Research execution and test reporting: Kimi. This manuscript synthesises the sequential protocols and results reported by the research programme. It is not a claim of independent computational replication.

---

Abstract

Background: The Complex Adaptive Model of Societies (CAMS) represents human coordination systems through eight recurrent functions—Helm, Shield, Lore, Stewards, Craft, Hands, Archive and Flow—each assessed along Coherence, Capacity, Stress and Abstraction. Previous work suggested that societies exhibit persistent patterns in the distribution of functional stress. Whether these patterns predict subsequent damage and recovery remained unresolved.

Methods: A sequential programme of preregistered tests, T-M1 through T-M4, examined historical stress episodes in the JUNO corpus. The initial analysis extracted 201 episodes from 48 societies; 158 episodes across 42 societies remained after censoring, and 149 episodes across 33 societies formed the analysis set. Subsequent tests distinguished standing functional weakness from episode-specific deterioration, audited algebraic coupling between the CAMS Stress and Node Value measures, and examined temporal ordering using fixed-effects panel methods. The principal stress-excluded association was subsequently tested on 138 episodes across 48 anonymised blind society series.

Results: Society-specific stress-distribution morphology exhibited modest but statistically detectable persistence (within-society cosine similarity 0.849 versus between-society 0.823; permutation p = 0.004). Morphology did not predict five-year recovery (cross-validated AUC = 0.492), and a society-specific modal failure-locus predictor failed its registered performance threshold. Replacing absolute Node Value with episode-specific change relocated 62% of identified damage loci. Stress uptake initially aligned with calculated deterioration at κ = 0.499, but an algebraic null generated κ = 0.252. After removing Stress from the outcome, the association remained positive (κ = 0.249; permutation p = 0.001). Fixed-effects analysis found that increased Stress preceded subsequent declines in Coherence and Capacity (coefficients approximately −0.094), although reverse-time associations indicated residual measurement dependence. Blind replication yielded pooled κ = 0.267 (p = 0.001), with positive effects across all six blind sets but individual threshold attainment in only three.

Conclusions: The evidence distinguishes three analytically separable processes: persistent functional configuration, episode-specific stress-linked deterioration, and subsequent recovery. Stress uptake is associated with non-Stress functional deterioration, and the relationship replicates within an anonymised holdout corpus. Neither standing morphology nor damage locus has demonstrated useful prediction of aggregate recovery. The remaining decisive requirement is external validation against functional damage measured independently of CAMS scoring.

Keywords: complex adaptive systems; organisational resilience; cybernetics; institutional stress; historical measurement; functional morphology; preregistration; falsification; panel analysis; blind replication.

---

1. Introduction

A society is a historical system of coordinated adaptation.

Its institutions organise collective activity, allocate resources, preserve knowledge, exercise authority and respond to disturbances. The configuration through which these functions are performed changes over time, but the underlying coordination problems recur.

This observation motivates the Complex Adaptive Model of Societies (CAMS).

CAMS proposes that enduring human organisations can be represented through eight recurrent coordination functions. The model was initially conceived as an analytical engine: each function performs a necessary contribution to the continued operation of the whole.

The framework subsequently developed a system of comparative scores intended to describe the operating condition of these functions.

Its central methodological problem is therefore not whether societies resemble organisms metaphorically, but whether a common functional representation can reveal empirical regularities in the way complex human organisations respond to disturbance.

The research programme sits within the intellectual traditions of complex adaptive systems, cybernetics, organisational viability and resilience theory.

Holland (1992) emphasised the emergence of systemic organisation through interacting adaptive agents. Ashby (1956) examined the requirements of regulation under disturbance. Beer's Viable System Model developed a recursive account of organisational regulation, while Holling (1973) distinguished the ability to retain organisation under disturbance from conventional equilibrium stability.

CAMS attempts to turn aspects of these theoretical insights into comparative historical measurements.

Its eight-function architecture is not asserted to be the minimum sufficient representation for every question. Earlier ablation experiments indicated that compressed functional representations can retain substantial crisis-attribution information, while finer differentiation may be useful for describing the morphology of disturbance.

The eight functions are therefore treated as a productive survey basis rather than eight universally separate institutional organs.

1.1 The central problem

Earlier CAMS analyses reported recurring patterns in how societies distribute stress among their functions.

This raised three distinct questions:

1. Do societies exhibit persistent, characteristic patterns of functional stress?
2. Do those patterns predict where damage occurs during a particular crisis?
3. Does knowledge of that morphology help predict whether the society subsequently recovers?

The distinction is essential.

A society might exhibit chronically weak institutions without those institutions suffering the largest deterioration during a new disturbance.

The locus of damage might, in turn, provide little information about the capacity of the system to reorganise and recover.

The T-M research sequence was designed to separate these propositions.

1.2 Research objective

The programme investigated whether CAMS functional stress morphology offers incremental explanatory or predictive information beyond aggregate operating condition and conventional institutional variables.

A further objective emerged through falsification: determining whether apparent relationships between Stress and functional deterioration were genuine empirical associations or mathematical consequences of the scoring system.

The programme therefore combines retrospective historical measurement, preregistered performance thresholds, negative-result retention, algebraic controls, temporal tests and blind replication.

---

2. The CAMS measurement architecture

2.1 Functional decomposition

CAMS represents collective organisation through eight functions.

Function| Primary coordination problem
Helm| Direction, executive coordination and collective decision-making
Shield| Defence, protection, security and coercive capability
Lore| Shared meaning and collective identity
Stewards| Provisioning, administration and resource allocation
Craft| Technical competence, production and innovation
Hands| Labour, execution and operational work
Archive| Institutional memory, record and accumulated knowledge
Flow| Circulation, communication, transport and exchange

The functions may be distributed among institutions, fused within organisations or partially externalised through networks.

Their identities describe recurrent problems of coordination, not necessarily discrete organisational units.

2.2 Primary operating variables

For node (i) in year (t), CAMS records four scored dimensions:

- (C_{it}): Coherence
- (K_{it}): Capacity
- (S_{it}): Stress
- (A_{it}): Abstraction

These are comparative assessments of organisational operating condition, not direct physical measurements.

The canonical Node Value is:

[
V_{it}=C_{it}+K_{it}-S_{it}+0.5A_{it}
]

The mean system value is:

[
\bar V_t=\frac{1}{8}\sum_{i=1}^{8}V_{it}
]

The coefficient applied to Abstraction is part of the current scoring convention and should not be interpreted as an independently calibrated physical constant.

2.3 Relational measures

Canonical CAMS also derives bond measures from the scored operating states, including node properties and Stress.

This imposes an important constraint on interpretation.

The calculated bond matrix is endogenous to the node scores. It is not an independently observed network of material, institutional or communicative connections.

Consequently, correlations involving bond strength cannot automatically be interpreted as independent evidence of causal interaction between functions.

The present study focuses on node-level stress displacement and functional change, while retaining this distinction between calculated structure and observed relationships.

---

3. Data and research provenance

3.1 The JUNO historical corpus

The original T-M1 episode construction used the JUNO unified historical dataset.

The analysis extracted 201 candidate stress episodes from 48 societies. Censoring rules left 158 episodes across 42 societies, and the requirements of leave-one-episode-out estimation left an analysis set of 149 episodes across 33 societies.

Subsequent longitudinal analyses used an updated unified file containing approximately 43,400 node records.

The hardened lag-analysis sample (T-M3e) comprised 40,472 node-year observations after temporal eligibility restrictions, including the requirement that adjacent observed periods be no more than two years apart. The earlier node-fixed-effects specification (T-M3d) used 40,944 node-year observations.

An earlier account referred to a smaller unified dataset containing 32,361 observations. This was a different dataset version and must not be silently reconciled with the later file.

The reproducibility archive should record file checksums, dataset versions, society identifiers, score definitions and temporal-coverage rules.

3.2 Research execution

The tests were executed and reported by Kimi as a sequential research programme.

The programme proceeded through preregistration, analysis, interpretation, explicit falsification decisions and subsequent protocol amendments or extensions.

Failures were retained rather than repaired through post-hoc threshold adjustment.

Two outcome-definition amendments were required during T-M1 after the initially registered recovery-ratio criterion was found to be mathematically unsuitable when baseline mean Node Value approached or crossed zero.

The ratio-based criterion produced a recovery base rate of approximately 0.6%.

It was replaced with an absolute-regain definition, and the amendment was documented.

Because this correction followed examination of outcome behaviour, the amended recovery analysis should be described as preregistered with an outcome-aware correction rather than as a completely untouched confirmatory test.

3.3 Blind replication material

The final replication used 138 episodes drawn from 48 anonymised society series divided into six blind sets.

The scoring and estimator definitions were frozen before the replication analysis.

This provides an important test of transfer beyond the original fitted episode collection. However, anonymisation and analytical holdout status do not independently establish that the underlying historical evidence or scoring process was free of shared model or source bias.

The distinction between an untouched analytical dataset and an independently constructed measurement dataset is retained throughout the interpretation.

---

4. T-M1: Persistence of stress morphology and prediction of recovery

4.1 Hypothesis

T-M1 investigated whether societies possess characteristic patterns of stress distribution and whether those patterns predict subsequent recovery.

Each stress episode was represented by an eight-dimensional, L1-normalised displacement signature describing the allocation of episode stress across CAMS functions.

For each society, a characteristic signature was estimated from its other eligible episodes.

Episode typicality was measured using the cosine similarity between the held-out episode signature and the society's leave-one-episode-out reference signature.

The primary outcome was full recovery of the trough loss in mean Node Value within five years, under the amended absolute-regain definition.

4.2 Competing predictors

The principal comparisons involved:

- Morphology typicality.
- Initial aggregate CAMS condition.
- Episode disturbance amplitude.
- The V-Dem Liberal Democracy Index at episode onset.

The latter is an externally constructed institutional index, used as a competing predictor rather than as proof of any particular causal interpretation of political institutions.

Performance was assessed using cross-validated discrimination, permutation tests and sensitivity analyses.

4.3 Results

Within-society morphology similarity was 0.849, compared with 0.823 between societies.

The permutation test returned:

[
p=0.004
]

The persistence hypothesis therefore passed its registered stability criterion.

However, the effect was modest in absolute magnitude, and the relatively high between-society similarity indicates that much of the functional displacement structure is shared.

Morphology typicality did not predict recovery.

Its cross-validated AUC was:

[
\mathrm{AUC}=0.492
]

This was below the registered success threshold of 0.65 and approximately equivalent to chance discrimination.

The V-Dem comparator achieved AUC 0.547, while the aggregate-state comparator achieved AUC 0.553. Adding morphology to the rival model reduced discrimination from 0.555 to 0.548.

Across recovery thresholds of 0.9–1.1 and observation windows of three to seven years, morphology remained close to chance performance.

Its association with years to recovery was negligible ((\rho=0.06)).

4.4 T-M1 interpretation

The test supported the existence of modest society-specific regularities in stress distribution.

It rejected the stronger proposition that resemblance to a society's historical stress signature usefully predicts aggregate recovery.

Persistent structural morphology and recovery performance must therefore be treated as distinct objects of analysis.

---

5. T-M2: Does societal morphology predict the locus of failure?

5.1 Hypothesis

T-M2 asked whether society-specific stress-distribution patterns predict which CAMS function becomes the failure locus during a crisis.

The original operationalisation identified failure using the lowest absolute Node Value.

A leave-one-episode-out modal-locus classifier was compared with marginal-frequency and permutation baselines.

The registered improvement criterion was 1.5 times the marginal base rate.

5.2 Results

In the initial classification, Hands was the failure locus in 61 of 149 episodes and Flow in 32.

The society-specific modal classifier achieved 51% accuracy, compared with a 41% marginal baseline and a permutation-null mean of 29%.

The permutation result was significant:

[
p=0.002
]

However, the improvement against the distinct marginal-frequency benchmark was only 1.25 times, below the registered 1.5 threshold.

T-M2a therefore failed its preregistered performance criterion.

A secondary test examined whether the function absorbing the most Stress was also the identified failure locus.

Agreement was:

[
\kappa=0.154
]

This fell below the required 0.20 threshold.

Finally, failure-locus identity did not predict recovery:

[
p=0.72
]

5.3 Methodological problem identified

The lowest absolute Node Value is not necessarily the function suffering the greatest episode-specific deterioration.

A function may be chronically weak and remain the lowest-scoring component even when its condition changes little during a crisis.

The original failure-locus definition therefore conflated standing vulnerability with disturbance-induced damage.

This motivated T-M3.

---

6. T-M3: Separating standing weakness from episode-specific damage

6.1 Corrected damage definition

T-M3 replaced the absolute-level failure locus with a change-based locus:

[
L_e^{\Delta V}

\arg\min_i
\left[
V_i(t_{\mathrm{trough}})

V_i(t_{\mathrm{onset}})
\right]
]

The corrected definition identifies the function experiencing the greatest fall in Node Value during the episode.

This is a measurement correction, not a reinterpretation of an unchanged outcome.

6.2 Results

The change-based definition relocated approximately 62% of episode failure loci.

Hands declined from 61 identified failure episodes to 14.

Flow became the most frequent damage locus, with 49 episodes, followed by Helm with 32.

This demonstrated that the earlier prominence of Hands was substantially influenced by chronically low absolute values.

T-M3a found that society-specific modal damage-locus prediction performed at 0.94 times the marginal-frequency benchmark.

Although the permutation test returned p = 0.034, the registered performance criterion was not satisfied.

Society-specific standing morphology did not usefully predict the location of new episode damage.

6.3 Stress–damage alignment

T-M3b compared the node with maximum Stress uptake with the node experiencing maximum calculated Node Value deterioration.

The original analysis reported 56% concordance and Cohen's κ = 0.476.

A trough-aligned recalculation subsequently produced:

[
\kappa=0.499
]

The result exceeded the registered threshold of 0.20.

However, a mathematical dependency was immediately identified.

Because Stress is a component of Node Value, increasing Stress necessarily reduces calculated Node Value, all other components held constant.

The apparent agreement could therefore be partly or entirely mechanical.

An additional preregistered audit was required before the result could be treated as evidence of an independently informative relationship.

T-M3c again found no statistically significant association between damage locus and recovery outcome (p = 0.19).

---

7. T-M3d: Algebraic-coupling audit

7.1 Problem

The canonical formula implies:

[
\Delta V_i

\Delta C_i+\Delta K_i-\Delta S_i+0.5\Delta A_i
]

A classifier based on maximum Stress uptake will therefore share a direct algebraic component with a damage-locus outcome based on minimum (\Delta V).

This is a form of mathematical coupling.

A high concordance between the two measures does not by itself validate a stress-transmission mechanism.

7.2 Mechanical-null construction

T-M3d constructed a null comparison in which the relationship between Stress patterns and other functional changes was disrupted while the contribution of Stress to the Node Value formula was retained.

The mechanical-null estimate was:

[
\kappa_{\mathrm{mechanical}}=0.252
]

Compared with the observed trough-aligned κ of 0.499, this demonstrated that a substantial proportion of the headline agreement could be generated by the scoring equation itself.

This comparison should not be interpreted as an exact additive decomposition of Cohen's κ.

7.3 Stress-excluded outcome

The audit defined a residual functional-change quantity:

[
R_i=\Delta V_i+\Delta S_i
]

Substituting the canonical formula gives:

[
R_i=\Delta C_i+\Delta K_i+0.5\Delta A_i
]

Stress is therefore absent from the outcome by construction.

The test compared the node with maximum Stress uptake against the node with greatest deterioration in (R_i).

7.4 Results

Observed agreement was:

[
\kappa_R=0.249
]

The corresponding permutation-null estimate was approximately 0.022, with:

[
p=0.001
]

The mechanical null (κ = 0.252) and κ_R (0.249) come from different tests: κ_R is compared with its own permutation null of approximately 0.022, not with the mechanical null.

These estimates use the 149-episode analysis set, which is primary. As a sensitivity analysis on the 158-episode censored set, trough-aligned κ was 0.507 and κ_R was 0.265.

The result exceeded the preregistered success threshold of 0.20.

7.5 Interpretation

Stress uptake was associated with deterioration in other scored functional components after removal of the direct Stress term.

This is a statistically non-trivial result within the CAMS measurement system.

It does not, however, demonstrate that Stress causes the deterioration.

Scorers may interpret evidence of institutional damage simultaneously as increasing Stress, decreasing Coherence and decreasing Capacity.

Shared historical evidence and scoring conventions could therefore generate residual correlations without any direct mechanical identity between the equations.

This motivated a temporal test.

---

8. T-M3e: Temporal ordering and shared-shock controls

8.1 Hypothesis

T-M3e investigated whether increases in Stress precede subsequent changes in Coherence and Capacity, rather than merely accompanying them in the same scored historical period.

8.2 Design

The analysis used the updated JUNO panel, restricting eligible observations to those where the following measurement occurred within two years.

The resulting hardened analytical sample contained 40,472 node-year observations.

Models included node fixed effects, controls for prior node condition and society-year fixed effects intended to absorb disturbances shared by all functions within a society-year.

Society-year effects were implemented through a within-transformation rather than explicit expansion of thousands of dummy variables.

Uncertainty estimates were clustered at the society level.

A reverse-time placebo tested whether the corresponding association appeared when temporal ordering was reversed.

8.3 Results

Earlier, less-controlled analyses (the T-M3d node-fixed-effects specification, 40,944 node-year observations) had produced coefficients near −0.138 for both Coherence and Capacity.

After absorbing society-year shocks, the forward coefficients attenuated to approximately:

[
\beta_{\Delta S\rightarrow\Delta C}=-0.094
]

[
\beta_{\Delta S\rightarrow\Delta K}=-0.094
]

The reported tests remained highly statistically significant, with p-values of approximately (10^{-16}).

The attenuation indicated that shared society-year disturbances accounted for a substantial fraction of the original association.

The reverse-time placebo was not zero. It produced coefficients around −0.04 to −0.05.

The forward associations were 1.81 and 2.25 times the corresponding reverse associations for Coherence and Capacity, respectively.

Both exceeded the registered directional-asymmetry threshold of 1.5.

8.4 Interpretation

The results support temporal asymmetry within the scored historical series.

Increases in recorded Stress tend to precede subsequent reductions in recorded Coherence and Capacity, even after important fixed-effect controls.

However, the non-zero reverse-time association indicates that measurement dependence, historical persistence or scoring smear remains.

The asymmetry is consistent with stress-linked functional deterioration, but does not independently identify a causal mechanism.

A stronger causal interpretation would require external observations, additional placebo tests and an identification strategy addressing time-varying confounding.

---

9. T-M4: Blind replication of stress-excluded coupling

9.1 Objective

Following the algebraic audit and temporal hardening, T-M4 tested whether stress-excluded functional damage alignment generalised beyond the original analysis sample.

The estimator and outcome definition were frozen.

9.2 Replication sample

The replication used 138 episodes across 48 anonymised society series organised into six blind sets.

Individual set sizes ranged from approximately 16 to 26 episodes.

The primary outcome remained agreement between maximum episode Stress uptake and maximum deterioration in:

[
R_i=\Delta C_i+\Delta K_i+0.5\Delta A_i
]

9.3 Results

The pooled replication estimate was:

[
\kappa=0.267
]

Observed raw concordance was 0.384.

The permutation test returned:

[
p=0.001
]

The pooled estimate slightly exceeded the original stress-excluded estimate of 0.249.

However, the six individual set estimates ranged from κ = 0.135 to κ = 0.481.

All six were positive, but only three exceeded the registered 0.20 threshold independently.

9.4 Interpretation

T-M4 provides encouraging evidence that the stress-excluded alignment is not confined to the initial episode sample.

The most defensible interpretation is:

The relationship replicated in the pooled blind dataset, remained directionally positive across all six blind sets, and individually exceeded the registered threshold in half of them.

The between-set variation is compatible with a moderate and heterogeneous association.

It also leaves open the possibility that the effect depends on society composition, historical coverage, episode construction or scoring conditions.

The replication does not resolve common-source or shared-scorer dependence because the outcomes remain derived from CAMS-assessed operating variables.

---

10. Summary of the sequential tests

Test| Principal result| Registered verdict
T-M1c| Within-society morphology similarity 0.849 vs 0.823; p = 0.004| Pass
T-M1a| Recovery prediction AUC = 0.492| Fail
T-M2a| Failure-locus typicality 1.25× marginal baseline| Fail
T-M2b| Stress uptake versus absolute failure locus κ = 0.154| Fail
T-M2c| Failure locus versus recovery p = 0.72| Fail
T-M3a| ΔV damage-locus typicality 0.94× marginal baseline| Fail
T-M3b| Stress uptake versus ΔV damage locus κ ≈ 0.50| Pass, subsequently qualified
T-M3c| Damage locus versus recovery p = 0.19| Fail
T-M3d| Stress-excluded damage association κ = 0.249; p = 0.001| Pass
T-M3e| Controlled forward coefficients ≈ −0.094; directional asymmetry above threshold| Pass
T-M4| Blind pooled κ = 0.267; 3/6 individual sets above threshold| Pooled pass, heterogeneous

These results should not be collapsed into a single overall validation score.

Each test addresses a different claim, using partly overlapping measurement foundations.

The failed tests are substantive results. They constrain the theoretical interpretation of the successful ones.

---

11. Discussion

11.1 Three processes rather than one

The findings suggest that CAMS should distinguish at least three components of collective adaptation.

Standing structure describes the historically persistent operating configuration of a society.

The T-M1 and T-M2 results provide modest evidence that societies retain characteristic patterns of relative functional strength and weakness.

However, these patterns are superimposed upon a substantial common functional structure shared across societies.

Episode-specific deterioration describes where functional operating conditions change during a particular disturbance.

Correcting the damage definition from absolute levels to episode-specific changes substantially altered the apparent distribution of failure.

Stress uptake shows a reproducible association with non-Stress deterioration in the scored data.

Adaptive recovery concerns whether the system subsequently regains capacity, reorganises or remains impaired.

Neither standing morphology nor damage-locus identity demonstrated useful predictive power over aggregate recovery.

Aggregate condition and the external institutional comparator performed modestly better.

The findings therefore do not support a single morphology-driven model of crisis outcome.

11.2 The distinction between damage and repair

Damage and repair need not be governed by the same relationships.

A disturbance may reduce the operating condition of particular functions, while recovery depends upon resources, institutions, intervention, historical circumstances and capacities elsewhere in the system.

The observed separation between damage morphology and recovery outcome is consistent with this possibility.

However, the present tests do not identify the mechanism of repair.

Nor do they establish that recovery depends exclusively upon the common statistical mode.

The appropriate conclusion is narrower: the tested morphology variables have not shown incremental predictive value for aggregate recovery.

11.3 Functional differentiation and complex adaptive systems

The results support a refined conception of the eight Paladins.

They are not necessarily eight independent causal mechanisms. Nor must each be institutionally distinguishable at every organisational scale.

They constitute a functional representation through which the distribution of burden and deterioration can be compared historically.

The value of that representation is question-dependent.

A compressed representation may be adequate for aggregate system condition, while a finer decomposition may reveal where disturbances are concentrated.

This distinction parallels a broader principle in complex adaptive systems: aggregate behaviour and local configuration need not carry the same information.

The present evidence suggests a useful descriptive partition between the common mode and residual functional structure.

Whether this partition corresponds to independently measurable causal architecture remains unresolved.

11.4 Why the negative findings matter

The T-M sequence illustrates the importance of distinguishing statistical significance from registered practical performance.

T-M2a produced a significant permutation result but failed its prespecified improvement threshold.

T-M3a also produced a permutation-significant result while performing worse than the marginal-frequency comparator.

These are not contradictory findings.

They arise because different null hypotheses answer different questions.

A pattern can be distinguishable from randomised society labels without improving practical prediction beyond a simple base-rate model.

Likewise, a coefficient can be extremely statistically significant in a large historical panel while leaving major questions about measurement dependence and causal interpretation unresolved.

The programme's methodological strength lies in retaining these distinctions.

11.5 Limits of the evidence

Five limitations remain central.

First, the four primary operating variables are assessments made from historical information. Their accuracy depends upon source coverage, rubric quality, scorer reliability and independence.

Second, the strongest stress–damage results remain internal to those assessments. Removal of direct mathematical coupling does not remove shared interpretive bias.

Third, historical episodes are dependent observations. Common shocks, serial correlation and uneven representation of societies can affect both significance and estimated performance.

Fourth, episode definitions, censoring rules, temporal gaps and amendments affect the eligible sample. Their full implementation must be preserved in executable code.

Fifth, blind analytical replication does not necessarily constitute independent measurement replication. The latter requires new scorers, independently constructed historical evidence or directly observed external outcomes.

These limitations determine the next research stage.

---

12. T-M5: Proposed external validation

12.1 Objective

T-M5 will test whether the CAMS-identified locus of Stress uptake corresponds with functional damage measured independently of CAMS scores.

This is the central outstanding construct-validity test.

12.2 Proposed outcome sources

Candidate external measures include:

CAMS function| Independent observational candidates
Helm| Leadership discontinuity, executive breakdown, documented command disruption
Shield| Security-service breakdown, force degradation, failures of protective capacity
Lore| Legitimacy measures, independently coded institutional-narrative breakdown
Stewards| Fiscal administration, state revenue collection, public-service continuity
Craft| Industrial production, manufacturing output, technical capability loss
Hands| Employment losses, labour disruption, hours worked
Archive| Institutional record loss, professional continuity, knowledge-system disruption
Flow| Trade volumes, transport interruption, exchange and distribution failures

Candidate sources include ILOSTAT, UNIDO, the World Bank, UCDP, V-Dem, historical administrative records and national statistical series.

The correspondence between each external measure and its CAMS function must be justified independently.

Aggregate trade, for example, is not a pure measurement of Flow, and a change in state revenue need not imply administrative failure.

12.3 Design requirements

Before testing, the programme should:

1. Freeze the episode roster, CAMS predictions and source-matching rules.
2. Determine coverage without reference to agreement with CAMS.
3. Establish independent definitions of functional damage.
4. Use blinded external adjudication where direct quantitative measures are unavailable.
5. Report missingness and uncertainty by function and historical period.
6. Compare concordance against prespecified marginal and permutation baselines.
7. Retain negative findings, including failures limited to particular functions.

The primary comparison should distinguish correspondence at the episode trough from genuine advance prediction.

External evidence must not be selected merely because it confirms the CAMS-identified damage locus.

12.4 Falsification

If the stress-excluded association disappears against independently measured damage, the proposed displacement mechanism must be downgraded to an internal measurement regularity.

If it survives, the framework will have demonstrated correspondence between its functional representation and independently observable organisational deterioration.

That would not yet prove a causal law, but it would substantially strengthen construct validity.

---

13. Subsequent research: the recovery mechanism

A separate registered study should investigate why aggregate condition and regime type outperform functional morphology in predicting recovery.

The preliminary association between V-Dem regime measurements and recovery speed suggests that institutional arrangements may contribute information beyond stress-distribution patterns.

One candidate hypothesis is that Stewards and Flow Capacity mediate part of this association.

The proposed test should determine whether their pre-episode values explain incremental variation in recovery speed after controlling for aggregate state, episode amplitude, prior trends and regime type.

Statistical attenuation of a regime coefficient would not, by itself, establish causal mediation.

The study should distinguish predictive contribution, statistical mediation and genuinely identified mechanism.

---

14. Conclusion

The T-M research sequence began with an attractive conjecture: that societies possess characteristic adaptive morphologies and that these morphologies might predict where they fail and whether they recover.

The evidence did not support that conjecture in its original strong form.

Society-specific stress morphology proved modestly persistent, but did not usefully predict aggregate recovery.

The initial failure-locus analysis was compromised by the use of absolute Node Value, which confused chronic weakness with episode-specific damage.

Correcting that definition fundamentally changed the distribution of apparent failure loci.

The resulting strong Stress–damage association was then shown to contain a substantial algebraic component.

After that component was removed, a smaller but statistically meaningful association survived.

The relationship also exhibited temporal asymmetry under fixed-effects controls and replicated in a pooled anonymised holdout sample, although individual-set performance was heterogeneous.

Together, these findings support a disciplined separation between historical functional configuration, episode-specific deterioration and subsequent adaptive outcome.

They do not yet establish a complete causal theory of collective adaptation.

The strongest result is the reproducible association between scored Stress uptake and deterioration in other scored functional conditions.

The weakest remaining link is independent observation of the corresponding institutional damage.

CAMS therefore remains a comparative diagnostic framework with mechanistic ambitions.

Its contribution lies not in claiming that the eight functions explain every aspect of social history, but in offering a common representation through which historically complex patterns can be measured, challenged, decomposed and compared.

The emerging account of national history is neither deterministic nor primarily psychological.

It is a history of coordinated adaptation under disturbance, in which inherited functional structures, episode-specific pressures and contingent recovery processes must be distinguished.

A society's standing configuration describes where it is vulnerable. Stress-linked changes describe where deterioration occurs. Neither alone determines whether it heals.

The next decisive step is to establish whether these internal diagnostic distinctions correspond to independently measurable features of the historical world.

---

Research provenance and reproducibility statement

The T-M1–T-M4 analyses were executed and reported by Kimi under the CAMS research programme. Preregistration, amendments, failed tests, corrected outcomes and blind replication are represented here from the sequential reports provided by the project.

The current manuscript is a methodological synthesis, not an independently rerun computational audit. The core figures were independently rerun from script v2 on 16 October 2026; the originally reported T-M2b value (κ = 0.184) remains unexplained, and the absence of blind set 4 is undocumented.

The complete research deposit should include:

- Original preregistrations and dated amendments.
- Dataset identifiers, versions and cryptographic hashes.
- Episode extraction and censoring rules.
- The T-M1–T-M4 analytical scripts.
- Permutation procedures and random seeds.
- Original and corrected outcome definitions.
- Society-year fixed-effects and lag-model specifications.
- Blind-set keys and permitted unblinding records.
- Full coefficient tables, uncertainty intervals and null distributions.
- All positive, negative and inconclusive test results.

The "morphology_test/" directory is identified as the intended primary analysis archive. Direct verification of its complete contents remains a reproducibility step before journal submission.

References

Ashby, W. R. (1956). An Introduction to Cybernetics. Chapman & Hall.

Beer, S. (1972). Brain of the Firm. Allen Lane.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37–46. https://doi.org/10.1177/001316446002000104

Holland, J. H. (1992). Complex adaptive systems. Daedalus, 121(1), 17–30.

Holling, C. S. (1973). Resilience and stability of ecological systems. Annual Review of Ecology and Systematics, 4, 1–23. https://doi.org/10.1146/annurev.es.04.110173.000245

McKern, K. (2026). Complex Adaptive Model of Societies: Canonical Functional and Scoring Specification. Neural Nations Research Platform. Version-specific archival citation to be completed.

McKern, K. (2026). CAMS T-M1–T-M4: Morphology, Damage and Recovery — Preregistrations, Laboratory Notes and Blind Replication Results. Neural Nations Research Platform. Research archive; persistent identifier to be assigned.

V-Dem Institute. (2026). Varieties of Democracy Dataset, Version 16. University of Gothenburg. https://v-dem.net/data/the-v-dem-dataset/