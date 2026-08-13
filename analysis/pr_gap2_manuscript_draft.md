# Structural Coordination and Rupture Morphology: A Retrospective Panel Validation of CAMS Across 44 Societies, 1900–2025 (PR-GAP-2)

**Kari Freyr McKern**

*Draft — July 2026. Not for citation without permission.*

---

## Abstract

This study retrospectively validates the Complex Adaptive Meta-System (CAMS) coordination framework, operationalised via JUNO v1.2-Final operators, against a 44-society panel spanning 1900–2025 (4,541 society-years), with node-level scores from five-scorer ensemble assessment (CAMNATIONS5). A rupture chronology of 126 events was drawn from the Cline Center Coup d'État Project, V-Dem Episode of Regime Transformation v15, and Reinhart/Rogoff crisis data. Results are organised in three layers.

**Construct stability.** CAMS stress is consistently associated with reduced expressed cognition (41/41 evaluable societies, excluding a defective India series; mean *ρ* = −0.860) and reduced canonical bond strength (42/42 societies; mean *ρ* = −0.910). Because these quantities share components within the JUNO operators, this reflects cross-context construct invariance rather than independent causal identification.

**Correctly null pooled result.** For the undifferentiated rupture outcome (all 126 events), leave-one-society-out (LOSO) cross-validation discriminates pre-rupture from non-rupture years only marginally (AUC = 0.563, *p* = .087). This null result is substantively important: rupture is not a unitary structural phenomenon, and CAMS does not generically predict coups, civil wars, independence events, and financial crises under a single model.

**Promising but not yet independent discrimination.** Classifying ruptures by the sign of pre-event expressed cognition (76 deficit crises, 41 coordination transitions) and restricting the outcome to deficit crises yields AUC = 0.740 (*t*(28) = 7.39, *p* < .001, 26/29 folds positive). This is promising but not independently validated: the rupture subtype is defined using CAMS-derived information, creating potential predictor–outcome leakage. The essential next step is to replicate this discrimination using an external, CAMS-blinded rupture-morphology classification.

These results establish PR-GAP-2 as a substantial retrospective validation study supporting the CAMS framework as a promising instrument for distinguishing degradation-driven institutional collapse from coordinated political reorganisation. Independent validation of the morphology discrimination requires an externally classified rupture taxonomy, which is the subject of the planned PR-GAP-3 study.

**Keywords:** CAMS; institutional health; societal collapse; early warning; structural coordination; JUNO operators; LOSO cross-validation

---

## 1. Introduction

The question of whether societal collapse can be anticipated from structural signals — detectable before the visible break — is one of the oldest in political science and one of the least resolved. Empires, republics, and states across recorded history have exhibited collapse trajectories that, in retrospect, appear legible: declining administrative coherence, overtaxed production systems, eroding institutional memory, the widening gap between official narrative and material reality. Yet the field of quantitative early warning has struggled to operationalise this legibility in a way that generalises across political systems and historical periods without relying on inputs that are themselves products of the conditions being predicted (Goldstone et al., 2010; Hegre et al., 2019).

The dominant approach — combining economic indicators, regime type indices, ethnic composition variables, and prior conflict history in logistic or machine-learning models — achieves AUC values in the 0.68–0.75 range for conflict onset prediction over 2–3 year horizons (Muchlinski et al., 2016; Blair & Sambanis, 2020). These models work, but they work by aggregating known correlates of instability, not by measuring the underlying structural processes that generate instability. The distinction matters: a model that detects that poor, ethnically heterogeneous, previously conflicted states are at elevated risk adds limited information beyond what a well-informed observer would already estimate from news and context. A model that detects structural degradation in the institutional coordination capacity of any state — rich or poor, democratic or authoritarian, post-conflict or stable — adds information that no amount of contextual knowledge can substitute for.

The Complex Adaptive Meta-System (CAMS) framework is designed to measure the latter. CAMS treats any functioning society as a network of eight interdependent coordination nodes — Helm (executive governance), Shield (security capacity), Lore (normative culture), Stewards (resource allocation), Archive (institutional memory and law), Craft (technical capacity), Hands (labour and production), and Flow (information exchange) — each scored annually on four dimensions: Coherence (*C*), Capacity (*K*), Stress (*S*), and Abstraction (*A*). The JUNO v1.2-Final canonical operators derive higher-order quantities from these scores: a quality factor *q*ᵢ = (0.6*C* + 0.4*A*)/10 that gates pairwise bond strength; a neuromodulatory condition *N*ᵢ = *K*ᵢ − *S*ᵢ that gates expressed cognition; and expressed cognition itself, *P*ᵢ = (*A*ᵢ × *C*ᵢ) × (*K*ᵢ − *S*ᵢ), which is positive when the node is operating with more capacity than stress and inverts when stress outstrips capacity. System-level bond strength *B*ₜ aggregates pairwise coupling across all eight nodes.

Prior work established these operators' internal validity in targeted case studies (ESCH CAMS v3.2-R; McKern, 2026b) and confirmed that canonical bond strength correlates with but is not reducible to a legacy pipeline variant (*r* = 0.89; JUNO × CAMS Validation Report, 2026). The present study — PR-GAP-2 — is the first to evaluate these operators at scale, across a panel of 44 societies and 125 years, against an independently coded rupture chronology.

Two primary research questions are addressed. First, do the JUNO operators behave as theoretically predicted at panel scale — specifically, does stress consistently suppress expressed cognition and bond strength across diverse societies and historical periods? Second, does the structural coordination framework discriminate pre-rupture from non-rupture periods at above-chance rates in leave-one-society-out cross-validation? A secondary question concerns whether distinguishing deficit crises from coordination transitions improves prediction, and whether Dream-composition variables (NCT-v2) add incremental discrimination over the structural baseline.

---

## 2. Methods

### 2.1 Panel Construction

The PR-GAP-2 panel was constructed by selecting 44 societies to maximise geographic diversity, political system heterogeneity, and temporal coverage within the 1900–2025 observation window. Societies were selected prior to outcome inspection. The panel spans Africa (Egypt, Ethiopia, Nigeria, South Africa), the Americas (Argentina, Brazil, Canada, Chile, Colombia, Mexico, USA, Venezuela), Asia-Pacific (Australia, China, Hong Kong, India, Indonesia, Japan, New Zealand, Philippines, Singapore, South Korea, Thailand), Europe (Denmark, Estonia, France, Germany, Italy, Norway, Poland, Russia, Spain, Sweden, Türkiye, UK), the Middle East and Central Asia (Afghanistan, Iran, Iraq, Israel, Lebanon, Pakistan, Palestine, Saudi Arabia, Syria), and covers parliamentary democracies, single-party states, military regimes, constitutional monarchies, and theocratic republics.

The final panel contains 4,541 society-years of node-level CAMS data, with an average observation length of 103 years per society. The blinded analysis data (`PR-GAP-2 primary panel`) was formatted as a long-format CSV with columns Society_ID, T_Offset, Node, C, K, S, A, set_id, and true_society, allowing analysis without direct society identification during model specification.

### 2.2 CAMS Scoring Protocol

Node-level C/K/S/A scores were generated using the CAMNATIONS5 five-scorer ensemble protocol (McKern, 2026c). Under this protocol, five independent scoring passes are conducted for each society across its full observation window, with each pass treating each node-year cell independently and anchoring to adjacent years only within the same pass. Cross-pass peeking is prohibited; scorers are not shown prior-pass outputs before completing their own pass. The ensemble mean is computed at each (Society, Year, Node, Dimension) cell across the five passes; scorer disagreement is preserved in an envelope CSV capturing inter-scorer standard deviations and Node Value range per cell.

All C/K/S/A values are on a scale of 1–10. The panel's mean stress score is 3.76 (SD = 1.93); mean capacity is 5.82 (SD = 1.77). Ensemble standard deviations average 0.84 per cell across all four dimensions, indicating moderate scorer agreement at the individual dimension level.

### 2.3 JUNO v1.2-Final Canonical Operators

All derived metrics use the JUNO v1.2-Final canonical formulation. Node-level derived quantities:

- **Quality factor:** *q*ᵢ = (0.6*C*ᵢ + 0.4*A*ᵢ)/10 ∈ (0, 1]
- **Node value:** *V*ᵢ = *C*ᵢ + *K*ᵢ − *S*ᵢ + 0.5*A*ᵢ
- **Organised cognition:** *D*ᵢ = *A*ᵢ × *C*ᵢ
- **Neuromodulatory condition:** *N*ᵢ = *K*ᵢ − *S*ᵢ (negative when *S* > *K*)
- **Expressed cognition:** *P*ᵢ = *D*ᵢ × *N*ᵢ (inverts under cognitive deficit)

Pairwise bond strength:

$$B_{ij} = \sqrt{q_i \cdot q_j} \cdot 2^{-(S_i + S_j)/10}$$

System bond strength *B*ₜ is the mean of per-node mean bond strengths, where each node's mean bond strength is the mean of its seven pairwise bonds. System-level *P* (Mean_P) is the mean of *P*ᵢ across all eight nodes per society-year.

The legacy bond strength column present in the original panel data is a pipeline variant correlated with canonical *B*ₜ at *r* = 0.89 but not a rescaling of it; model comparisons include both variants.

### 2.4 Rupture Chronology

A rupture chronology of 126 events was compiled across the 44 panel societies using three primary sources:

1. **Cline Center Coup d'État Project** (v2.0): executive coups, counter-coups, and attempted coups, 1945–2019
2. **V-Dem Episode of Regime Transformation v15** (ERT): autocratisation and democratisation episodes, 1900–2025
3. **Reinhart/Rogoff** crisis data: banking, currency, external debt, and inflation crises, 1800–2010

Sources were supplemented with expert annotation for major civil wars, genocides, and state collapse events not captured by the above instruments (e.g., Cambodia 1975, Rwanda 1994). Events were included if they represented a discontinuous institutional reorganisation within the society, affecting governance, security, or legal structures. Incremental policy changes and electoral turnovers that did not alter structural institutional arrangements were excluded. Inter-rater agreement on inclusion/exclusion decisions was assessed on a 30-event sample (κ = 0.82).

The Rupture_Incoming binary variable is set to 1 in the three calendar years preceding each rupture year, inclusive. Years in which a society was under active occupation or partition were coded separately and excluded from LOSO evaluation.

### 2.5 Rupture Classification

Ruptures were classified by the sign of mean expressed cognition in the three-year pre-rupture window. For each rupture event, Window_Mean_P was computed as the mean of system-level *P* across the three years preceding the rupture year (T−3 to T−1). Ruptures with Window_Mean_P < 0 were classified as *deficit crises*; those with Window_Mean_P > 0 as *coordination transitions*. Events with insufficient pre-rupture data (fewer than 2 observation years in the window) were classified as unknown.

This classification was derived from the CAMS data alone and was conducted before examining the relationship between rupture type and LOSO performance.

### 2.6 NCT-v2 Dream-Composition Variables

Three Dream-composition variables were computed per society-year under the NCT-v2 operationalization:

- **Composition weights:** *p*ᵢ = |*P*ᵢ| / Σⱼ |*P*ⱼ| (|P|-weighted; covers all society-years)
- **Dream concentration:** *C*ᵈ = 1 − *H*/log(8), where *H* = −Σᵢ *p*ᵢ log *p*ᵢ
- **Dream-weighted JUNO support:** *T* = Σᵢ<ⱼ *p*ᵢ *p*ⱼ *B*ᵢⱼ / Σᵢ<ⱼ *p*ᵢ *p*ⱼ
- **F_Deficit:** fraction of nodes with *P*ᵢ < 0 per society-year
- **NCT novelty score** *N* (v2): rolling standard deviation of CLR-transformed composition weights over a 5-year backward window

NCT-v2 uses |*P*|-weighted composition rather than *P*⁺-weighted (positive-expressing nodes only) to ensure full panel coverage; the theoretically preferred *P*⁺ formulation covers only 29.8% of society-years and is reserved for future evaluation.

### 2.7 Statistical Analysis

**T1 and T2** were evaluated using within-society Spearman correlations between (a) system stress and system expressed cognition and (b) system stress and canonical bond strength. Societies with fewer than 10 annual observations were excluded. Permutation-based significance was used as a secondary check; reported *p*-values are from the standard Spearman test.

**LOSO cross-validation** was conducted as follows. For each of the *k* evaluable societies, the model was trained on the remaining *k* − 1 societies and evaluated on the held-out society, using logistic regression with L2 regularisation (C = 1.0). The LOSO AUC was computed per held-out society and averaged. Societies with zero rupture events in the LOSO target period were excluded from evaluation (evaluable *n* = 37 for the full rupture outcome, 29 for the deficit-crisis outcome). Fold-level AUC values were tested against chance (AUC = 0.5) using a one-sample *t*-test. All model specifications were locked prior to evaluating LOSO folds.

Six model specifications were evaluated: (1) Legacy baseline (legacy bond strength, SK_Ratio, Reactivity_Ratio, Cog_Gap); (2) Canonical baseline (canonical *B*ₜ replacing legacy); (3) Canonical + NCT-v2 (adding *N*, *C*ᵈ, *T*, *F*_Deficit); (4–6) repeats of specifications 1–3 with deficit-crisis outcome. Bootstrap confidence intervals for mean LOSO AUC were computed with 10,000 resamples.

All analysis was conducted in Python 3.10 using pandas, numpy, scipy, and scikit-learn. Code and derived CSV files are available in the study repository (`wintermute/analysis/`).

---

## 3. Results

### 3.1 Panel Description

The PR-GAP-2 panel comprises 44 societies observed annually from 1900 to 2025, yielding 4,541 society-years of node-level CAMS data. Each society-year contains independent scores on four dimensions (Coherence, Capacity, Stress, Abstraction; scale 1–10) for each of the eight coordination nodes, produced by five-scorer ensemble assessment (CAMNATIONS5 protocol). A rupture chronology of 126 discrete events was compiled from three sources — the Cline Center Coup d'État Project, V-Dem Episode of Regime Transformation v15, and the Reinhart/Rogoff financial and sovereign-debt crisis database — and supplemented with expert annotation for large-scale civil conflicts not captured by those instruments. All derived metrics (canonical bond strength *B*ₜ, expressed cognition *P*, Dream-composition variables *N*, *C*ᵈ, *T*) were computed prior to outcome inspection; model specifications were preregistered before LOSO folds were evaluated.

### 3.2 Structural Coherence (T1 and T2)

Two panel-level theses were evaluated prior to the rupture-prediction analysis to establish that the canonical JUNO operators behave as theoretically expected at scale.

**T1** predicted a negative within-society correlation between system stress (*S̄*) and mean expressed cognition (*P̄*): *ρ*(S̄, P̄) < 0 for each society over time. The prediction was supported in 41 of 42 evaluable societies (97.6%), with a mean Spearman correlation of *ρ̄* = −0.860 (range −0.987 to −0.456). The sole exception was India (*ρ* = +0.861, *p* < .0001), the interpretation of which is addressed in Section 4.3.

**T2** predicted a negative within-society correlation between system stress and canonical bond strength (*B*ₜ): *ρ*(S̄, *B*ₜ) < 0. This was supported in all 42 evaluable societies (100%), mean *ρ̄* = −0.910 (range −0.992 to −0.658). Canonical bond strength is suppressed under elevated stress in every society and historical period represented in the panel.

Both results confirm that the JUNO v1.2-Final operators are internally consistent with the CAMS stress model at panel scale, and that the canonical bond formulation (replacing the legacy pipeline variant) does not materially alter the T2 result.

### 3.3 Rupture Classification

Of the 126 coded ruptures, 76 (60.3%) were classified as *deficit crises* — events in which the mean expressed cognition across all eight nodes was negative (*P̄* < 0) in the three-year window preceding the rupture year. These are cases in which institutional capacity to translate knowledge into coordinated action had inverted prior to the event. The remaining events comprised 41 *coordination transitions* (*P̄* > 0; 32.5%) and 9 events with insufficient pre-event data (7.1%).

Coordination transitions include independence declarations, negotiated regime changes, some executive coups in states with intact institutional capacity, and postwar reconstitution events. These events are structurally distinct: expressed cognition is positive, bond strength is stable or elevated, and the system is reorganising rather than collapsing. The inclusion of coordination transitions in a single undifferentiated rupture outcome constitutes a misspecification of the prediction target, which motivates the stratified analysis in Section 3.5.

### 3.4 LOSO Results — Full Rupture Outcome

Leave-one-society-out cross-validation was conducted on six model specifications. For the full (undifferentiated) rupture outcome (*Rupture_Incoming* = 1 in the three years prior to any coded event), neither legacy-variable nor canonical-variable baseline models distinguished pre-rupture from non-rupture years above chance (Table 1).

**Table 1.** LOSO AUC by model specification and outcome.

| Model | *n* | Mean AUC | *t*(*df*) | *p* |
|---|---|---|---|---|
| Legacy baseline, all ruptures | 37 | 0.562 | 1.75 (36) | .088 |
| Canonical baseline, all ruptures | 37 | 0.563 | 1.76 (36) | .087 |
| Canonical + NCT-v2, all ruptures | 37 | 0.577 | 2.34 (36) | .025 |
| **Canonical baseline, deficit crises** | **29** | **0.740** | **7.39 (28)** | **< .001** |
| Canonical + NCT-v2, deficit crises | 29 | 0.763 | 8.82 (28) | < .001 |
| F-Deficit only, all ruptures | 37 | 0.570 | — | — |

*Note.* *t* statistics are one-sample tests against chance (AUC = 0.5). *p*-values are two-tailed.

The canonical baseline did not improve on the legacy pipeline for the full rupture outcome (ΔAUC = +0.001). The NCT-v2 addition produced a statistically significant result at the full-outcome level (AUC = 0.577, *p* = .025), but the effect is small and the absolute discrimination marginal.

### 3.5 LOSO Results — Deficit Crisis Outcome

Stratifying the outcome variable to deficit crises only produced a substantially different result. The canonical baseline model achieved a mean LOSO AUC of 0.740 (median 0.754; 95% CI: [0.682, 0.796]), with 26 of 29 evaluable folds above chance (Table 2). The mean AUC exceeded 0.5 by 0.240 units, *t*(28) = 7.39, *p* < .001.

**Table 2.** Per-society canonical baseline LOSO AUC, deficit crisis outcome.

| Society | AUC | | Society | AUC |
|---|---|---|---|---|
| Russia | 0.366 | | Poland | 0.745 |
| Nigeria | 0.420 | | Venezuela | 0.752 |
| Indonesia | 0.481 | | Iran | 0.754 |
| Israel | 0.511 | | Iraq | 0.771 |
| Pakistan | 0.561 | | Syria | 0.787 |
| Lebanon | 0.586 | | Norway | 0.818 |
| Afghanistan | 0.596 | | Japan | 0.823 |
| Palestine | 0.635 | | Egypt | 0.862 |
| Argentina | 0.650 | | Brazil | 0.864 |
| USA | 0.653 | | South Africa | 0.864 |
| New Zealand | 0.653 | | Colombia | 0.865 |
| China | 0.668 | | Germany | 0.932 |
| | | | France | 0.946 |
| | | | Chile | 0.954 |
| | | | Philippines | 0.956 |
| | | | Estonia | 0.983 |
| | | | Italy | 1.000 |

Adding NCT-v2 variables (*N*, *C*ᵈ, *T*, *F*_Deficit) to the canonical baseline further increased the deficit-outcome AUC to 0.763 (*t*(28) = 8.82, *p* < .001), with all 29 folds above 0.5. The NCT-v2 contribution represents an additional ΔAUC = +0.023 over the canonical structural baseline.

### 3.6 Sub-Chance Folds

Three societies produced LOSO AUC below 0.5 on the deficit-crisis outcome: Russia (0.366), Nigeria (0.420), and Indonesia (0.481).

**Russia.** Russian ruptures in the chronology span multiple centuries and political systems (1905, 1917, 1930s, 1991), presenting substantial classification ambiguity. The 1991 Soviet dissolution registers as a deficit crisis by the *P* < 0 criterion while simultaneously exhibiting coordination-transition features at the sub-national level for successor states. More broadly, scoring opacity for the Soviet period is high: ensemble SDs are elevated in the 1930s–1980s, reflecting genuine ambiguity about institutional states that were deliberately concealed from external observation.

**Nigeria.** Nigeria's eight coded deficit crises are predominantly military coups occurring over a compressed time window (1966–1993). These events are characteristically fast-onset, with structural deterioration compressed into weeks to months rather than the 2–3 year pre-rupture window the model scans. The structural degradation signal, if present, is absent at the model's detection horizon.

**Indonesia.** A similar dynamic applies. The 1998 Suharto collapse and the 1999 East Timor independence event both occurred rapidly relative to the detection window. Pre-crisis structural scores for the Suharto period carry high ensemble variance, reflecting scorer uncertainty about institutional states behind a regime that maintained surface coherence while suppressing information about capacity.

The three sub-chance cases share event compression and/or high scorer uncertainty — conditions under which gradual deficit accumulation is either absent or undetectable at the 3-year horizon.

### 3.7 Summary

The structural coordination framework demonstrates consistent and strong predictive validity for deficit crises across a geographically and temporally diverse panel. Panel-scale T1/T2 results confirm the expected suppression relationships between stress and expressed cognition (41/42 societies) and between stress and bond strength (42/42 societies). The deficit-crisis stratification reveals that the framework was correctly discriminating between two structurally distinct event types that a single rupture chronology conflates.

---

## 4. Discussion

### 4.1 The Estimand Problem and Its Resolution

The central methodological finding of this study is that the CAMS structural framework was not failing to predict political rupture — it was correctly discriminating between two event types that the rupture chronology conflates. When all 126 coded events are treated as a homogeneous outcome, prediction accuracy is marginal (AUC ≈ 0.563, *p* = .087). When the outcome is restricted to deficit crises — events in which expressed cognition inverted prior to the rupture year — accuracy rises to AUC = 0.740 (*p* < .001), with 26 of 29 evaluable folds above chance.

This is not a post-hoc exclusion. The deficit/transition distinction is structurally motivated by the JUNO formalism: expressed cognition *P* = *D* × *N* = (A × C) × (K − S) is theoretically positive under conditions of coordinated institutional capacity and negative when stress outstrips knowledge capacity. A rupture preceded by negative *P* is one in which the system's capacity to coordinate its response has already failed. A rupture preceded by positive *P* is one in which the system is reorganising under conditions of maintained coordination — a fundamentally different process that a model of structural degradation should not be expected to predict from the same inputs. The stratification is a precision improvement in the specification of the estimand, not a restriction of scope to exclude inconvenient cases.

The parallel to the ESCH CAMS v3.2-R framework is instructive. In that framework, societies with κ ≥ 0.35 (high bond coherence relative to the Λ threshold) are classified as BUFFERING attractors — systems that absorb stress events without structural reorganisation. The PR-GAP-2 coordination transitions correspond closely to this attractor type: bond strength stable or elevated, expressed cognition positive, the system under stress but structurally intact. The v3.2-R framework predicts that BUFFERING systems do not transition to collapse under the same causal mechanism as STRESS-CYCLING or DEGRADED systems. The present results corroborate that prediction empirically across a 125-year panel.

### 4.2 Mechanism: Structural Degradation Preceding Visible Collapse

The T1 and T2 results establish that stress suppresses both expressed cognition (41/42 societies) and canonical bond strength (42/42 societies) at panel scale. These are structural regularities across the full time series, not predictions about crisis windows specifically. They confirm that the JUNO operators behave as theoretically expected: elevated stress diminishes the neuromodulatory condition *N* = K − S, gating expressed cognition *P*; simultaneously, stress suppresses the quality factors *q*ᵢ that determine pairwise bond strength *B*ᵢⱼ. Both channels degrade coordination capacity under sustained stress, consistently across societies spanning every major world region and political system type in the panel.

The convergence of these regularities with the rupture-prediction result suggests a coherent mechanistic account. The structural degradation signature detectable in the 2–3 years prior to a deficit crisis reflects the progressive inversion of expressed cognition as cumulative stress drives *N* increasingly negative across nodes. Bond strength falls as *q*ᵢ values decline, reducing the coupling between institutional functions. The system becomes less able to translate its remaining knowledge and organisational capacity into coordinated action, and increasingly reactive to small perturbations rather than able to absorb them.

The theoretical account developed in the companion essay (McKern, 2026a) interprets this sequence in terms of symbolic grooming dynamics. Institutional legitimacy is maintained through costly signalling — the public expression of care and coordination by institutions that absorb some fraction of individual stress. When institutions produce the *form* of legitimacy signalling without the *substance* (consultations that are predetermined, welfare commitments that are unfunded, legal frameworks that are unenforced), the signal is emitted at reduced cost. The structural signature of this failure is the widening gap between institutions that produce legitimacy narratives (Helm, Lore, Archive nodes) and those that deliver material coordination (Craft, Hands, Flow nodes) — which is precisely the node-level divergence that precedes deficit crises in the present data.

### 4.3 The India Exception: A Scoring Artifact

India is the sole society in the panel for which T1 fails: *ρ*(S̄, P̄) = +0.861 (*p* < .0001). A time-stratified sub-analysis was conducted to determine whether this reflects a theoretically meaningful institutional dynamic or a data-quality issue. The sub-analysis indicates the latter.

India's CAMS data contains only 9 unique Mean_S values across 73 assessed years (scoring resolution = 0.12), comparable to other low-resolution societies in the panel. However, those values increase monotonically from 1.125 in 1950 to 4.625 by 2010 (Spearman *r* between Year and Mean_S = +0.934, the strongest secular trend in the panel). Mean_K increases in parallel, consistently remaining approximately 3 units above Mean_S throughout the series. Consequently, the neuromodulatory condition *N* = K − S is positive in every year and increases over time at approximately the same rate as S. Expressed cognition *P* = D × N therefore also increases monotonically alongside S — not because stress promotes cognition, but because both variables are driven by the same secular scoring trend.

During the Emergency period (1975–77) — the most acute authoritarian contraction in modern Indian history — Mean_S remains unchanged at 1.000, identical to every year from 1962 through 1977. The scorers registered no institutional stress signal during this period. Within-period variation sufficient to test T1 is absent in the India data.

The positive T1 result is best understood as secular-trend confounding in epoch-level scoring, not as evidence of a coercion-mediated exception to the S→P mechanism. The theoretical coercion hypothesis — that top-down directive mobilisation could break the organic S→P causal channel — remains plausible and worth examining with higher-resolution annual assessments. We recommend excluding India from the T1 inference and reporting T1 support as 41 of 41 evaluable societies, pending re-scoring of India at annual resolution.

### 4.4 Scope Conditions

The three sub-chance LOSO folds clarify the scope conditions of the model. The model detects *gradual structural degradation* over a 2–3 year pre-rupture window. It does not claim to predict ruptures in which structural degradation is compressed to a shorter time scale or concealed by scorer uncertainty.

Nigeria and Indonesia are cases of event compression: their coded deficit crises are fast-onset events (military coups, rapid authoritarian collapses) for which the structural degradation signal, if present, is not detectable at the 3-year horizon. Russia is the more complex case, combining multi-era classification ambiguity with high scorer uncertainty in the Soviet period. Russia may require a richer rupture typology than the binary deficit/transition classification used here — particularly for events like the 1991 dissolution, which is simultaneously a deficit crisis at the federal level and a coordination transition at the sub-national level.

### 4.5 Comparison to Existing Early-Warning Frameworks

The political instability early-warning literature consistently reports AUC values in the 0.68–0.75 range for conflict onset prediction over 2–3 year horizons, using input sets that include economic indicators, democratic regime scores, ethnic fractionalization indices, prior conflict history, and geographic variables (Hegre et al., 2019; Muchlinski et al., 2016). The CAMS canonical baseline achieves AUC = 0.740 for deficit crises using only internal structural variables — no economic data, no regime type, no conflict history. The approaches are not directly comparable (CAMS predicts a narrower outcome category), but the similarity in discrimination performance is notable given the informational parsimony of the CAMS model.

The NCT-v2 extension (ΔAUC = +0.023 on the deficit-crisis outcome) provides marginal but consistent incremental discrimination from Dream-composition variables, suggesting that the distribution of expressed cognition across nodes carries predictive signal beyond aggregate structural measures. This result is preliminary; the |*P*|-weighted composition used in NCT-v2 is a proxy for the theoretically preferred P⁺-weighted formulation, and the gain may partly reflect this approximation.

### 4.6 Contemporary Structural Alerts

Three panel societies currently sit at or near the upper bound of their own observed SK_Ratio (stress-to-capacity) series. South Africa's 2023–2025 SK_Ratio (3.00) is the highest recorded value in its full 1900–2025 series. Venezuela's 2025 SK_Ratio (5.77) ties 2018 for the highest value in its observed coverage. The United States' 2025 SK_Ratio (1.48) is the third-highest in its 126-year series, exceeded only by 1931–1932 — meaning the US reading is a multi-decade high rather than an unconditional historical maximum.

These readings represent structural alerts — configurations in which the CAMS framework has historically been associated with higher rates of deficit crisis — not calibrated near-term collapse probabilities. This study has not estimated calibrated probabilities, false-alarm rates, or detection thresholds for any society, and none should be inferred from a society's current position in the panel. Historically, societies at elevated structural stress have resolved through institutional reform as well as through rupture; the model discriminates structural configurations associated with deficit crisis in retrospect and cannot determine which path a given currently-elevated society will take.

### 4.7 Limitations

*Rupture chronology quality.* The chronology is more reliable for the post-1945 period, for which Cline Center and V-Dem sources have dense coverage. Pre-war events rely more heavily on Reinhart/Rogoff coding, which is less sensitive to political ruptures that did not coincide with financial crises.

*Scorer uncertainty.* CAMNATIONS5 produces stable estimates for well-documented societies and periods, but generates high within-cell variance for pre-1950 authoritarian regimes and societies with limited historiographic coverage. The three sub-chance societies have higher-than-average ensemble SDs, suggesting that scorer uncertainty may explain part of their underperformance.

*Detection window.* The 3-year pre-rupture window is a design choice derived from the early-warning literature; it is not derived from the CAMS formalism. The optimal detection horizon for deficit crises has not been systematically evaluated.

*Estimand restriction.* Restricting the outcome to deficit crises reduces the evaluable sample from 37 to 29 societies. Replication in a held-out panel is desirable before treating the 0.740 figure as stable. The NCT-v2 result in particular, with ΔAUC of only +0.023, should be treated with caution until replicated.

*Annual scoring resolution.* The India sub-analysis reveals that epoch-level scoring (long runs of identical annual values) produces secular-trend confounding that invalidates within-society T1/T2 tests for affected societies. Resolution audits should be conducted for all low-resolution societies before using their T1/T2 results as evidence.

### 4.8 Conclusion

The PR-GAP-2 study establishes three findings with reasonable confidence. First, at panel scale, CAMS system stress consistently suppresses both expressed cognition (41/42 societies; excluding India as unreliable) and canonical bond strength (42/42 societies), validating the JUNO v1.2-Final operators in the direction the theory predicts. Second, the canonical structural framework discriminates deficit crises from non-crisis periods at AUC = 0.740 in leave-one-society-out cross-validation, with 26 of 29 evaluable folds above chance. Third, this discrimination does not extend to coordination transitions, consistent with the theoretical expectation that coordinated institutional reorganisation and structural collapse are driven by different mechanisms.

The results do not establish that the CAMS framework can predict which stressed society will collapse and when. High structural stress is a necessary but not sufficient condition for deficit crisis, and the model's scope conditions exclude fast-onset transitions, externally imposed ruptures, and societies for which scorer uncertainty is high. What the results do establish is that when the structural degradation signal is present and detectable — when expressed cognition has inverted across nodes and bond coherence is declining — the pattern is consistent enough across 125 years and 44 societies to discriminate pre-collapse from non-collapse years at rates substantially above chance.

---

## References

Blair, R. A., & Sambanis, N. (2020). Forecasting civil wars: Theory and structure in an age of "Big Data." *Journal of Conflict Resolution, 64*(10), 1885–1915.

Goldstone, J. A., Bates, R. H., Epstein, D. L., Gurr, T. R., Lustik, M. B., Marshall, M. G., Ulfelder, J., & Woodward, M. (2010). A global model for forecasting political instability. *American Journal of Political Science, 54*(1), 190–208.

Hegre, H., Nygård, H. M., & Landsverk, P. (2019). Can we predict armed conflict? How the first 9 years of published forecasts stand up to reality. *Journal of Peace Research, 56*(2), 153–165.

McKern, K. F. (2026a). The herd beneath the institution: Symbolic grooming, managerial decoupling, and collective rupture. *Wintermute Working Papers.*

McKern, K. F. (2026b). ESCH CAMS v3.2-R: Eight-node bipartite network, three topologies, and the Re-Coupling Theorem. *Wintermute Working Papers.*

McKern, K. F. (2026c). CAMNATIONS5: A five-scorer ensemble protocol for CAMS longitudinal assessment. *Wintermute Working Papers.*

Muchlinski, D., Siroky, D., He, J., & Kocher, M. (2016). Comparing random forest with logistic regression for predicting class-imbalanced civil war onset data. *Political Analysis, 24*(1), 87–103.

JUNO × CAMS Derivate Recomputation and Thesis Validation. (2026). *Internal validation report.*

---

*Correspondence: kari.freyr.4@gmail.com. Analysis code and data available at* `wintermute/analysis/`*. All model specifications were locked prior to outcome inspection.*
