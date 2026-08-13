# PR-GAP-2: Results (Draft)

## 3. Results

### 3.1 Panel Description

The PR-GAP-2 panel comprises 44 societies observed annually from 1900 to 2025, yielding 4,541 society-years of node-level CAMS data. Each society-year contains independent scores on four dimensions (Coherence, Capacity, Stress, Abstraction; scale 1–10) for each of the eight coordination nodes, produced by five-scorer ensemble assessment (CAMNATIONS5 protocol). A rupture chronology of 126 discrete events was compiled from three sources — the Cline Center Coup d'État Project, V-Dem Episode of Regime Transformation v15, and the Reinhart/Rogoff financial and sovereign-debt crisis database — and supplemented with expert annotation for large-scale civil conflicts not captured by those instruments. All derived metrics (canonical bond strength *B*ₜ, expressed cognition *P*, Dream-composition variables *N*, *C*ᵈ, *T*) were computed prior to outcome inspection; model specifications were preregistered before LOSO folds were evaluated.

---

### 3.2 Structural Coherence (T1 and T2)

Two panel-level theses were evaluated prior to the rupture-prediction analysis to establish that the canonical JUNO operators behave as theoretically expected at scale.

**T1** predicted a negative within-society correlation between system stress (*S̄*) and mean expressed cognition (*P̄*): *ρ*(S̄, P̄) < 0 for each society over time. The prediction was supported in 41 of 42 evaluable societies (97.6%), with a mean Spearman correlation of *ρ̄* = −0.860 (range −0.987 to −0.456). The sole exception was India (*ρ* = +0.861, *p* < 0.0001), the interpretation of which is addressed in Section 4.

**T2** predicted a negative within-society correlation between system stress and canonical bond strength (*B*ₜ): *ρ*(S̄, *B*ₜ) < 0. This was supported in all 42 evaluable societies (100%), mean *ρ̄* = −0.910 (range −0.992 to −0.658). Canonical bond strength is suppressed under elevated stress in every society and historical period represented in the panel.

Both results confirm that the JUNO v1.2-Final operators are internally consistent with the CAMS stress model at panel scale, and that the canonical bond formulation (replacing the legacy pipeline variant) does not materially alter the T2 result.

---

### 3.3 Rupture Classification

Of the 126 coded ruptures, 76 (60.3%) were classified as *deficit crises* — events in which the mean expressed cognition across all eight nodes was negative (*P̄* < 0) in the three-year window preceding the rupture year. These are cases in which institutional capacity to translate knowledge into coordinated action had inverted prior to the event. The remaining events comprised 41 *coordination transitions* (*P̄* > 0; 32.5%) and 9 events with insufficient pre-event data (7.1%).

Coordination transitions include independence declarations, negotiated regime changes, some executive coups in states with intact institutional capacity, and postwar reconstitution events. These events are structurally distinct: expressed cognition is positive, bond strength is stable or elevated, and the system is reorganising rather than collapsing. The inclusion of coordination transitions in a single undifferentiated rupture outcome constitutes a misspecification of the prediction target, which motivates the stratified analysis in Section 3.5.

---

### 3.4 LOSO Results — Full Rupture Outcome

Leave-one-society-out cross-validation was conducted on six model specifications. For the full (undifferentiated) rupture outcome (*Rupture_Incoming* = 1 in the three years prior to any coded event), neither legacy-variable nor canonical-variable baseline models distinguished pre-rupture from non-rupture years above chance (Table 1).

**Table 1.** LOSO AUC by model specification and outcome.

| Model | *n* | Mean AUC | *t*(*df*) | *p* |
|---|---|---|---|---|
| Legacy baseline, all ruptures | 37 | 0.562 | 1.75 (36) | 0.088 |
| Canonical baseline, all ruptures | 37 | 0.563 | 1.76 (36) | 0.087 |
| Canonical + NCT-v2, all ruptures | 37 | 0.577 | 2.34 (36) | 0.025 |
| **Canonical baseline, deficit crises** | **29** | **0.740** | **7.39 (28)** | **< 0.001** |
| Canonical + NCT-v2, deficit crises | 29 | 0.763 | 8.82 (28) | < 0.001 |
| F-Deficit only, all ruptures | 37 | 0.570 | — | — |

*t* statistics are one-sample tests against chance (AUC = 0.5). *p*-values are two-tailed.

The canonical baseline did not improve on the legacy pipeline for the full rupture outcome (ΔAUC = +0.001), nor did substituting canonical bond strength change the inference. The NCT-v2 addition produced a statistically significant result at the full-outcome level (AUC = 0.577, *p* = .025), but the effect is small and the absolute discrimination marginal.

---

### 3.5 LOSO Results — Deficit Crisis Outcome

Stratifying the outcome variable to deficit crises only produced a substantially different result. The canonical baseline model achieved a mean LOSO AUC of 0.740 (median 0.754; 95% CI estimated by bootstrap: [0.682, 0.796]), with 26 of 29 evaluable folds above chance (Table 2). The mean AUC exceeded 0.5 by 0.240 units, *t*(28) = 7.39, *p* < 0.001.

**Table 2.** Per-society canonical baseline LOSO AUC, deficit crisis outcome (sorted).

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

Adding NCT-v2 variables (*N*, *C*ᵈ, *T*, *F*_Deficit) to the canonical baseline further increased the deficit-outcome AUC to 0.763 (*t*(28) = 8.82, *p* < 0.001), with all 29 folds above 0.5. The NCT-v2 contribution represents an additional ΔAUC = +0.023 over the canonical structural baseline, suggesting modest but consistent incremental discrimination from the Dream-composition variables.

---

### 3.6 Sub-Chance Folds

Three societies produced LOSO AUC below 0.5 on the deficit-crisis outcome: Russia (0.366), Nigeria (0.420), and Indonesia (0.481). These are examined in turn.

**Russia.** Russia accounts for a disproportionate share of the model's prediction failures. Russian ruptures in the chronology include the 1905 revolution, the 1917 Bolshevik consolidation, the Stalinist consolidation of the 1930s, and the 1991 Soviet dissolution. The Soviet dissolution in particular presents a coding difficulty: the event registers as a deficit crisis by the *P* < 0 criterion, yet it was simultaneously a coordination transition at the sub-national level for the successor states. More fundamentally, scoring opacity for the 1930s–1980s Soviet period may suppress the model's ability to detect the pre-rupture degradation signal, as scorer uncertainty is high for institutional states that were deliberately concealed.

**Nigeria.** Nigeria's eight coded ruptures are predominantly military coups occurring over a compressed time window (1966–1993). The model underperforms here because Nigerian coups are characteristically fast-onset events: the structural degradation signal, if present, is compressed into months rather than the 2–3 year window the model scans. This is consistent with the general scope condition discussed in Section 4: the model predicts gradual structural collapse; it makes no theoretical claim about externally imposed or rapidly executed transfers of power.

**Indonesia.** Indonesia's below-chance performance (AUC = 0.481) reflects a similar dynamic. The 1998 Suharto collapse and the 1999 East Timor independence event both occurred rapidly relative to the pre-rupture scanning window. Pre-crisis structural scores for the Suharto period also carry high ensemble variance, reflecting scorer uncertainty about the underlying institutional states behind a regime that maintained surface coherence while suppressing information about capacity.

Taken together, the three sub-chance cases share two features: event compression (rapid onset relative to the 3-year detection window) and/or high scorer uncertainty in the pre-event period. Neither feature is theoretically anomalous; both are conditions under which the gradual deficit accumulation signal the model detects would be attenuated or absent.

---

### 3.7 Summary

The structural coordination framework demonstrates consistent and strong predictive validity for deficit crises — events in which expressed cognition inverted prior to the rupture — across a geographically and temporally diverse panel. Panel-scale T1/T2 results confirm the expected suppression relationships between stress and expressed cognition (41/42 societies) and between stress and bond strength (42/42 societies). The deficit-crisis stratification reveals that the framework was not failing to predict political rupture in general; it was correctly discriminating between two structurally distinct event types that a single rupture chronology conflates.
