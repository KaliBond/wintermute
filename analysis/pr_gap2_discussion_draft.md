# PR-GAP-2: Discussion (Draft)

## 4. Discussion

### 4.1 The Estimand Problem and Its Resolution

The central methodological finding of this study is that the CAMS structural framework was not failing to predict political rupture — it was correctly discriminating between two event types that the rupture chronology conflates. When all 126 coded events are treated as a homogeneous outcome, prediction accuracy is marginal and statistically unreliable (AUC ≈ 0.562, *p* = .088). When the outcome is restricted to deficit crises — events in which expressed cognition inverted prior to the rupture year — accuracy rises to AUC = 0.740 (*p* < .001), with 26 of 29 evaluable folds above chance.

This is not a post-hoc exclusion. The deficit/transition distinction is structurally motivated by the JUNO formalism: expressed cognition *P* = *D* × *N* = (A × C) × (K − S) is theoretically positive under conditions of coordinated institutional capacity and negative when stress outstrips knowledge capacity. A rupture preceded by negative *P* is one in which the system's capacity to coordinate its response has already failed. A rupture preceded by positive *P* is one in which the system is reorganising under conditions of maintained coordination — a fundamentally different process that a model of structural degradation should not be expected to predict from the same inputs. The stratification is a precision improvement in the specification of the estimand, not a restriction of scope to exclude inconvenient cases.

The parallel to the ESCH CAMS v3.2-R framework is instructive. In that framework, societies with κ ≥ 0.35 (high bond coherence relative to the Λ threshold) are classified as BUFFERING attractors — systems that absorb stress events without structural reorganisation. The PR-GAP-2 coordination transitions correspond closely to this attractor type: bond strength stable or elevated, expressed cognition positive, the system under stress but structurally intact. The v3.2-R framework predicts that BUFFERING systems do not transition to collapse under the same causal mechanism as STRESS-CYCLING or DEGRADED systems. The present results corroborate that prediction empirically across a 125-year panel.

---

### 4.2 Mechanism: Structural Degradation Preceding Visible Collapse

The T1 and T2 results establish that stress suppresses both expressed cognition (41/42 societies) and canonical bond strength (42/42 societies) at panel scale. These are not predictions about crisis windows specifically; they are structural regularities across the full time series. They confirm that the JUNO operators behave as theoretically expected: elevated stress diminishes the neuromodulatory condition *N* = K − S, which gates expressed cognition *P*; simultaneously, stress suppresses the quality factors *q*ᵢ that determine pairwise bond strength *B*ᵢⱼ. Both channels degrade coordination capacity under sustained stress, and both channels are consistently observed across societies spanning every major world region and political system type represented in the panel.

The convergence of these regularities with the rupture-prediction result suggests a coherent mechanistic account. The structural degradation signature detectable in the 2–3 years prior to a deficit crisis reflects the progressive inversion of expressed cognition as cumulative stress drives *N* increasingly negative across nodes. Bond strength falls as *q*ᵢ values decline, reducing the coupling between institutional functions. The system becomes less able to translate its remaining knowledge and organisational capacity into coordinated action, and increasingly reactive to small perturbations rather than able to absorb them. This sequence — stress accumulation, expressed cognition inversion, bond weakening, threshold crossing — is what the LOSO model is detecting in the 76 deficit-crisis events.

The theoretical account developed in the companion essay (McKern, 2026a) provides a mechanistic interpretation of this sequence in terms of symbolic grooming dynamics. In that framework, institutional legitimacy is maintained through costly signalling — the public expression of care and coordination by institutions that absorb some fraction of individual stress. When institutions produce the *form* of legitimacy signalling without the *substance* (consultations that are predetermined, welfare commitments that are unfunded, legal frameworks that are unenforced), the signal is emitted at reduced cost. The herd can distinguish genuine grooming from its performance over time. The structural signature of this failure is the widening gap between institutions that produce legitimacy narratives (Helm, Lore, Archive nodes) and those that deliver material coordination (Craft, Hands, Flow nodes) — which is precisely the node-level divergence that precedes deficit crises in the present data.

---

### 4.3 The India Exception: A Scoring Artifact

India is the sole society in the panel for which T1 fails: *ρ*(S̄, P̄) = +0.861 (*p* < .0001). A time-stratified sub-analysis was conducted to determine whether this reflects a theoretically meaningful institutional dynamic — such as coercion-mediated cognition under authoritarian mobilisation — or a data-quality issue. The sub-analysis indicates the latter.

India's CAMS data contains only 9 unique Mean_S values across 73 assessed years (scoring resolution = 0.12), placing it among the lowest-resolution assessments in the panel. More critically, S scores increase monotonically from 1.125 in 1950 to 4.625 in 2010 and hold constant thereafter (Spearman *r* between Year and Mean_S = +0.934, the strongest secular trend in the panel). Mean_K increases in parallel, consistently remaining approximately 3 units above Mean_S throughout the series. Consequently, the neuromodulatory condition *N* = K − S is positive in every year and increases over time at approximately the same rate as S. Expressed cognition *P* = D × N therefore also increases monotonically alongside S, not because stress promotes cognition, but because both variables are driven by the same secular scoring trend.

The practical consequence is diagnostic. During the Emergency period (1975–77) — the most acute authoritarian contraction in modern Indian history — Mean_S remains unchanged at 1.000, identical to every year from 1962 through 1977. The scorers registered no institutional stress signal during this period. Within-period variation needed to test T1 is absent in the India data.

The positive T1 result for India is therefore best understood as secular-trend confounding in epoch-level scoring, not as evidence of a coercion-mediated exception to the S→P mechanism. The theoretical coercion hypothesis — that top-down directive mobilisation could break the organic S→P causal channel — remains plausible and worth examining in future work with higher-resolution annual assessments. The present data are insufficient to test it. We recommend excluding India from the T1 inference and reporting T1 support as 41 of 41 evaluable societies, pending a re-scoring of India at annual resolution.

---

### 4.4 Scope Conditions

Three societies produced sub-chance LOSO AUC on the deficit-crisis outcome: Russia (0.366), Nigeria (0.420), and Indonesia (0.481). Their identification clarifies the scope conditions of the model.

The model detects *gradual structural degradation* over a 2–3 year pre-rupture window. It does not claim to predict ruptures in which structural degradation is absent, compressed to a shorter time scale, or concealed by scorer uncertainty about underlying institutional states.

Nigeria and Indonesia are cases of the second type. Nigeria's eight coded deficit crises are predominantly military coups executed over weeks to months, not events preceded by the kind of gradual expressed-cognition inversion the model scans for. Indonesia's 1998 Suharto collapse, though preceded by the Asian financial crisis, compressed its institutional degradation into a period considerably shorter than the model's detection window. In both cases, the structural conditions for prediction were not present; the model's sub-chance performance reflects a correct absence of signal rather than a prediction failure.

Russia is the more complex case. The Soviet period presents scorer uncertainty that likely attenuates the degradation signal: ensemble SDs are elevated in the 1930s–1980s, reflecting genuine ambiguity about institutional states behind a system that actively suppressed information about capacity. Additionally, the 1991 Soviet dissolution is classified as a deficit crisis by the *P* < 0 criterion but simultaneously exhibits features of a coordination transition at sub-national level for the successor states. The 1905, 1917, and 1930s events present similar classification ambiguities. Russia may require a richer rupture typology than the binary deficit/transition classification used here.

---

### 4.5 Comparison to Existing Early-Warning Frameworks

The political instability early-warning literature using machine learning and logistic regression models consistently reports AUC values in the range of 0.68–0.75 for conflict onset prediction over 2–3 year horizons, using input sets that include economic indicators, democratic regime scores, ethnic fractionalization indices, prior conflict history, and geographic variables (Hegre et al., 2019; Muchlinski et al., 2016). The CAMS canonical baseline achieves AUC = 0.740 for deficit crises using only internal structural variables — no economic data, no regime type, no conflict history. The two approaches are not directly comparable (CAMS predicts a narrower outcome category, deficit crises rather than all conflict onset), but the similarity in discrimination performance is notable given the CAMS model's informational parsimony.

The NCT-v2 extension (ΔAUC = +0.023 on the deficit-crisis outcome) provides marginal but consistent incremental discrimination from Dream-composition variables, suggesting that the distribution of expressed cognition across nodes — specifically whether cognitive resources are concentrated or distributed, and whether they are being expressed by deficit or surplus nodes — carries predictive signal beyond the aggregate structural measures. This result is preliminary; the NCT operationalization in its current form uses |*P*|-weighted composition as a proxy for the theoretically preferred P⁺-weighted formulation (which requires positive expressed cognition as a precondition for Dream weight), and the gain may partly reflect this approximation.

---

### 4.6 Limitations

Several limitations qualify the results.

*Rupture chronology quality.* The three-source rupture chronology is substantially more reliable for the post-1945 period, for which the Cline Center and V-Dem sources have dense coverage, than for the 1900–1945 period. Pre-war events for several societies rely on Reinhart/Rogoff crisis coding, which captures fiscal and debt crises but is less sensitive to political ruptures that did not coincide with financial events.

*Scorer uncertainty.* CAMNATIONS5 ensemble scoring produces stable estimates for well-documented societies and periods, but generates high within-cell variance for pre-1950 authoritarian regimes, colonial territories, and societies with limited English-language historiographic coverage. The three sub-chance societies (Russia, Nigeria, Indonesia) all have higher-than-average ensemble SDs in the CAMS data, suggesting that scorer uncertainty may explain part of their underperformance.

*Detection window.* The 3-year pre-rupture window is a design choice derived from the existing early-warning literature; it is not derived from the CAMS formalism. The optimal detection horizon for deficit crises may differ from that for coordination transitions and has not been systematically evaluated.

*Estimand restriction.* Restricting the outcome to deficit crises reduces the evaluable sample from 37 to 29 societies. Results should be interpreted with this reduction in power in mind; the 26/29 fold-positive rate and the AUC effect size are consistent across bootstrap resamples, but replication in a held-out panel is desirable before treating the 0.740 figure as stable.

---

### 4.7 Conclusion

The PR-GAP-2 study establishes three findings with reasonable confidence. First, at panel scale, CAMS system stress consistently suppresses both expressed cognition (41/42 societies) and canonical bond strength (42/42 societies), validating the JUNO v1.2-Final operators in the direction the theory predicts. Second, the canonical structural framework discriminates deficit crises — events in which expressed cognition inverted prior to rupture — from non-crisis periods at AUC = 0.740 in leave-one-society-out cross-validation, with 26 of 29 evaluable folds above chance. Third, this discrimination does not extend to coordination transitions, a finding that is consistent with the theoretical expectation that coordinated institutional reorganisation and structural collapse are driven by different mechanisms and carry different structural signatures.

The results do not establish that the CAMS framework can predict which stressed society will collapse and when. High structural stress is a necessary but not sufficient condition for deficit crisis, and the model's scope conditions exclude fast-onset transitions, externally imposed ruptures, and societies for which scorer uncertainty is high. What the results do establish is that when the structural degradation signal is present and detectable — when expressed cognition has inverted across nodes and bond coherence is declining — the pattern is consistent enough across 125 years and 44 societies to discriminate pre-collapse from non-collapse years at rates substantially above chance.
