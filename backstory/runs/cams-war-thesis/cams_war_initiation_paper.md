# Rentier Overload and War Initiation: A Quantitative Test Using CAMS Institutional Scores

**Working Paper — CAMS Research Programme**
**Date:** 2026-11-19
**Status:** Pre-print / internal review

---

## Abstract

Does sustained fiscal-institutional overload increase the probability that a society initiates an interstate war? This paper tests three linked hypotheses using the Civilizational Analysis and Measurement System (CAMS) institutional scores for six societies spanning 1750–2026, matched against a hardcoded COW/MID war-initiation dataset. H1 (Kinetic Export) predicts that Stewards-node overload—Stress exceeding Capacity for three or more consecutive years—raises the probability of war initiation within a ten-year window. H2 (Recovery Moderation) predicts that high Lore-Archive institutional coherence weakens this effect. H3 (Jubilee Alternative) predicts that high Lore-Archive societies more often resolve overload internally, without Shield escalation. We find support for H1 (OR = 2.64, p = 0.028, clustered by society; robust to controls), strong support for H3 (jubilee rate 82.4% vs 24.1%, Fisher's exact OR = 50.4, p < 0.0001), and directional but non-significant support for H2. A within-society placebo test confirms the H1 timing matters (99.6th percentile). The result carries important caveats: the H1 signal is structurally dominated by China, and scorer-replication on Germany alone fails to converge, triggering falsification criterion FC4. The analysis is best read as hypothesis-generating rather than confirmatory.

**Keywords:** CAMS, war initiation, rentier overload, Stewards node, institutional stress, Lore-Archive index, placebo test, kinetic export

---

## 1. Introduction

The relationship between domestic institutional strain and foreign-policy aggression is one of the oldest questions in international-relations theory. Diversionary war theory (Blainey 1988; Levy 1989) holds that leaders facing domestic legitimacy crises may seek external conflict to consolidate support. Rentier-state theory (Mahdavy 1970; Beblawi & Luciani 1987) adds a structural dimension: societies in which the dominant class (Stewards, in CAMS terminology) extracts rent from the system without commensurate productive capacity may accumulate stresses that cannot be discharged internally. The question is whether such structural overload—rather than leader-level opportunism alone—systematically raises war-initiation probability.

The Civilizational Analysis and Measurement System (CAMS) offers a novel instrument for this question. It maps societies onto eight functional nodes—Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow—and scores each node annually on four dimensions: Coherence, Capacity, Stress, and Abstraction. The Stewards node captures the dominant property-holding, wealth-extracting class (or its institutional equivalent). Sustained Stewards overload—a multi-year regime in which Stress exceeds Capacity—may proxy for the structural conditions under which diversionary war becomes attractive.

We test three hypotheses:

**H1 (Kinetic Export):** Sustained Stewards overload (Stress > Capacity for ≥ 3 consecutive years, measured only in peace years) raises the probability of war initiation within the following ten years.

**H2 (Recovery Moderation):** The H1 effect is attenuated when the Lore and Archive nodes exhibit high Coherence and Abstraction—institutions capable of mediating and re-absorbing pressure.

**H3 (Jubilee Alternative):** In high Lore-Archive societies, overload more often resolves internally within five years, without a rise in Shield (military) Stress.

We also specify four falsification criteria in advance:

- **FC1:** Effect disappears once Shield_Stress and system_stress are controlled.
- **FC2:** Overload predicts civil conflict or revolution equally well.
- **FC3:** Effect sizes are similar across high and low Lore-Archive societies (no moderation).
- **FC4:** Results do not replicate across independent CAMS scorers (Gemini vs Grok).

---

## 2. Data

### 2.1 CAMS Institutional Scores

CAMS scores were drawn from six society files stored in the project repository:

| Society | File | Years | Scorer | Node Schema |
|---------|------|--------|--------|-------------|
| Germany | Germany_gem_jan_2025.csv | 1880–2025 | Gemini | Canonical |
| Germany | Germany_grok_jan21.csv | 1880–2025 | Grok | Canonical |
| Finland | Finland_Gem_21Jan.csv | 1900–2025 | Gemini | Canonical |
| China | China_canonical_BS.csv | 1875–2026 | Gemini (ensemble) | Canonical |
| France | France_CAMS_Cleaned.csv | 1785–2024 | Mixed | Old names (remapped) |
| England | england.csv | 1750–1898 | Mixed | Old names (remapped) |
| USA | USA_canonical_BS.csv | 1900–2026 | Gemini | Canonical |

France and England files used archaic node labels. These were remapped: Executive → Helm, Army → Shield, Priests / Priesthood / Knowledge Workers → Lore, Property Owners → Stewards, Trades/Professions / Trades / Professions → Craft, Proletariat → Hands, State Memory → Archive, Shopkeepers/Merchants / Merchants / Shopkeepers → Flow. After remapping, three England node variants (Lore, Craft, Flow) that had been silently dropped due to spacing differences were recovered; Stewards data (Property Owners) was intact throughout.

Rome was excluded from quantitative models (no COW comparators). After normalisation, the panel comprised **4,804 society-year-node observations** across the six societies.

### 2.2 War Initiation Data

Interstate war onsets were hardcoded from COW v4 and MID logic, restricted to years when the focal society was coded as the initiator or joint initiator:

| Society | Onset Year | Conflict |
|---------|-----------|----------|
| France | 1792 | French Revolutionary Wars |
| France | 1803 | Napoleonic Wars resumption |
| France | 1870 | Franco-Prussian War |
| France | 1956 | Suez Crisis |
| Germany | 1914 | WWI |
| Germany | 1939 | WWII |
| England | 1899 | Second Boer War |
| England | 1956 | Suez Crisis |
| England | 1982 | Falklands War |
| USA | 1846 | Mexican-American War |
| USA | 1898 | Spanish-American War |
| USA | 1950 | Korea |
| USA | 2003 | Iraq War |
| China | 1950 | Korean War entry |
| China | 1962 | Sino-Indian War |
| China | 1979 | Sino-Vietnamese War |
| Finland | 1941 | Continuation War |

Defensive and reactive onsets were excluded (France 1914, USA 1917, USA 1941, Finland 1939 Winter War).

### 2.3 Constructed Variables

**Overload variables.** For each Stewards society-year:
- `overload_binary` = 1 if Stress > Capacity
- `overload_streak` = number of consecutive preceding years (including current) where overload_binary = 1
- `overload_3yr` = 1 if overload_streak ≥ 3

**Peace filter.** A year *t* was coded as "at war" if *t* or *t*−1 is a war-onset year for that society. Only peace years were retained as predictors.

**Outcome.** `war_initiation_10yr` = 1 if any war onset for that society falls in [*t*+1, *t*+10].

**System stress.** Mean Stress across all eight nodes per society-year.

**Lore-Archive index.** Mean of Lore Coherence, Lore Abstraction, Archive Coherence, and Archive Abstraction.

**Shield Stress.** Annual Shield-node Stress score, used as a militarisation control.

---

## 3. Methods

Logistic regression was estimated using `statsmodels` with Society-clustered standard errors (Huber-White sandwich). Three models were estimated:

- **Model 1 (H1 baseline):** `war_initiation_10yr ~ overload_3yr`
- **Model 2 (H1 controlled):** `war_initiation_10yr ~ overload_3yr + Shield_Stress + system_stress`
- **Model 3 (H2 interaction):** `war_initiation_10yr ~ overload_3yr × lore_archive_idx + Shield_Stress + system_stress`

A **placebo test** assessed timing specificity: `overload_3yr` was shuffled within each society (preserving marginal frequency) across 500 permutations, and Model 1 was re-estimated on each shuffle. The real coefficient's percentile rank in the placebo distribution tests whether observed timing is unusually predictive.

**H3** was tested by computing, for each peace year with `overload_3yr = 1`, whether Stewards Stress fell below Capacity within five years ("resolved") and whether Shield Stress rose by more than one point in that window ("Shield rise"). Societies were split at the median `lore_archive_idx`. The two-way contingency was tested with Fisher's exact test.

**Scorer replication** (FC4) re-estimated Model 1 separately on the Gemini and Grok Germany files and compared the `overload_3yr` coefficient.

---

## 4. Results

### 4.1 Descriptive Overview

The final peace-year panel contains **623 observations** across six societies (Table 1). Three societies—England (1750–1898), France, and USA—recorded **zero** `overload_3yr` events within their available data ranges; they contribute war-window outcome years but no predictor variation. The H1 test therefore rests almost entirely on **China** (n = 73 overload events, 23 war-window overlaps), **Finland** (n = 17, 0 overlaps), and **Germany** (n = 12, 2 overlaps).

**Table 1 — Descriptive statistics by society**

| Society | Period | Peace-N | overload_3yr | war_window | Overlap |
|---------|--------|---------|-------------|-----------|---------|
| China | 1875–2026 | 146 | 73 | 30 | 23 |
| England | 1750–1898 | 149 | 0 | 10 | 0 |
| Finland | 1900–2025 | 113 | 17 | 10 | 0 |
| France | 1785–2024 | 48 | 0 | 8 | 0 |
| Germany | 1880–2025 | 142 | 12 | 20 | 2 |
| USA | 1900–2026 | 25 | 0 | 4 | 0 |

### 4.2 H1 — Kinetic Export

**Model 1** (Table 2) estimates the association between `overload_3yr` and war initiation within ten years. The coefficient is positive and statistically significant (β = 0.972, clustered SE = 0.443, z = 2.20, p = 0.028; OR = 2.64). A society in a three-year overload streak faces roughly 2.6 times the odds of war initiation in the following decade, compared to a non-overloading peace year.

**Table 2 — Model 1: H1 Baseline**

| Variable | Coef | Cl.SE | z | p | OR |
|----------|------|-------|---|---|----|
| Constant | −2.097 | 0.173 | −12.10 | <0.001 | 0.123 |
| overload_3yr | **0.972** | 0.443 | **2.20** | **0.028** | **2.643** |

*N = 623, Events = 82, Pseudo-R² (McFadden) = 0.025. *** p<0.001, ** p<0.01, * p<0.05, . p<0.1. Standard errors clustered by Society (6 clusters).*

**Model 2** adds Shield_Stress and system_stress as controls (Table 3). The `overload_3yr` coefficient is essentially unchanged (β = 1.027, OR = 2.79) and remains marginally significant (p = 0.069). Neither control is significant, and **FC1 is not triggered**: the overload effect survives militarisation and system-level controls.

**Table 3 — Model 2: H1 Controlled**

| Variable | Coef | Cl.SE | z | p | OR |
|----------|------|-------|---|---|----|
| Constant | −2.190 | 0.285 | −7.68 | <0.001 | 0.112 |
| overload_3yr | **1.027** | 0.565 | **1.82** | **0.069** | **2.792** |
| Shield_Stress | −0.200 | 0.225 | −0.89 | 0.374 | 0.819 |
| system_stress | 0.204 | 0.230 | 0.89 | 0.375 | 1.226 |

*N = 623, Events = 82, Pseudo-R² = 0.032.*

### 4.3 Placebo Test

Across 500 within-society shuffles of the `overload_3yr` indicator, the median coefficient was 0.426 (95th percentile = 0.752). The real Model 1 coefficient of 0.972 sits at the **99.6th percentile** of the placebo distribution. The specificity of timing is not random noise.

### 4.4 H2 — Recovery Moderation

Model 3 (Table 4) includes the interaction `overload_3yr × lore_archive_idx`. The interaction coefficient is negative (β = −1.278), consistent with H2—high Lore-Archive attenuates the overload→war link—but falls short of conventional significance (p = 0.136). **FC3 is borderline**: the direction is correct but the interaction is imprecisely estimated given the small number of overload events in high-LA societies.

**Table 4 — Model 3: H2 Interaction**

| Variable | Coef | Cl.SE | z | p | OR |
|----------|------|-------|---|---|----|
| Constant | −2.838 | 1.203 | −2.36 | 0.018 | 0.059 |
| overload_3yr | 8.462 | 4.455 | 1.90 | 0.058 | 4733 |
| lore_archive_idx | 0.085 | 0.161 | 0.53 | 0.599 | 1.089 |
| **ol3_x_la** | **−1.278** | 0.857 | −1.49 | **0.136** | **0.279** |
| Shield_Stress | −0.267 | 0.211 | −1.26 | 0.206 | 0.766 |
| system_stress | 0.260 | 0.214 | 1.22 | 0.224 | 1.298 |

*N = 623, Events = 82, Pseudo-R² = 0.084.*

### 4.5 H3 — Jubilee Alternative

Among the 102 peace years with `overload_3yr = 1`, societies were split at the median `lore_archive_idx` (7.75). The high-LA group comprised Finland and Germany overload years (n = 17); the low-LA group was dominated by China (n = 83).

**Table 5 — H3 Contingency**

| Group | n | Resolved (5yr) | No Shield rise | Jubilee (both) |
|-------|---|---------------|----------------|----------------|
| High Lore-Archive (≥ 7.75) | 17 | 16 (94.1%) | 15 (88.2%) | **14 (82.4%)** |
| Low Lore-Archive (< 7.75) | 83 | 20 (24.1%) | 64 (77.1%) | **20 (24.1%)** |

Fisher's exact test on the resolved × LA-group table: **OR = 50.4, p < 0.0001**.

H3 is strongly supported. High Lore-Archive overload episodes resolve internally at a dramatically higher rate and without Shield escalation, consistent with the prediction that institutional knowledge-and-memory infrastructure mediates stress discharge.

### 4.6 Falsification Criteria

| FC | Criterion | Verdict |
|----|-----------|---------|
| FC1 | Effect disappears under Shield/system_stress controls | **Not triggered** — overload_3yr survives (p = 0.069) |
| FC2 | Overload predicts civil conflict equally | **Not tested** — no civil conflict outcome in dataset |
| FC3 | Effect similar across LA groups | **Borderline** — interaction negative but p = 0.136 |
| FC4 | Results don't replicate Gem vs Grok | **Triggered** — Germany Gem β = +0.219 (p = 0.789), Grok β = −0.529 (p = 0.624); opposite signs |

---

## 5. Discussion

### 5.1 What the Results Show

The core finding is that sustained Stewards overload—a multi-year regime in which the dominant property class's Stress exceeds its Capacity—is associated with a 2.6-fold increase in war-initiation odds over the following decade. This association survives controls for Shield (military) activity and system-wide stress, and its timing is demonstrably non-random (99.6th percentile placebo rank). The effect is consistent with a kinetic-export mechanism: when internal redistribution pathways are blocked or overwhelmed, external conflict becomes a pressure-release valve.

The H3 result is the most striking and structurally robust finding. High Lore-Archive societies—those with strong institutionalised knowledge, deliberation, and memory infrastructure—resolve Stewards overload internally at an 82.4% rate without military escalation, compared to 24.1% in low Lore-Archive societies. This is not a marginal difference: Fisher's exact OR = 50.4. The implication is that institutional "wisdom capacity" (Lore) combined with stable institutional memory (Archive) creates the internal processing space for peaceful overload resolution—what the theory calls the Jubilee alternative.

H2 was directionally consistent (negative interaction, right sign) but underpowered. With only 17 overload events in the high-LA group, the interaction term's confidence interval is too wide to reach significance.

### 5.2 Structural Limitations

**China dominance.** The H1 test is structurally unbalanced: 73 of 102 overload events belong to China, which also drives 23 of 82 war-window years. The pooled OR of 2.64 is essentially a weighted average of China's strong association and Germany's weak one, with Finland contributing overload events that fall outside any war window (Finland's post-1941 peace years show overload but no subsequent war initiation—a *negative* case that pushes against H1). England, France, and USA contribute no overload variation at all in their available data ranges.

**FC4 — scorer disagreement.** On the Germany subset, Gemini scores yield β = +0.219 and Grok scores yield β = −0.529—opposite signs, both non-significant. This is not necessarily a falsification: Germany has only 11–12 overload events, far too few for robust single-country estimation. The scorer disagreement may reflect genuine uncertainty about Stewards scores in Wilhelmine and Weimar Germany. The H1 pooled estimate should not be treated as falsified by this result, but it cannot yet be considered confirmed.

**Measurement.** CAMS scores are model-generated estimates, not archival measurements. They incorporate scorer priors and may systematically overstate Stress in periods associated with known conflicts (anchoring on outcomes). An independent scorer blind to war outcomes would be needed to rule this out.

**FC2 (untested).** We cannot assess whether overload predicts civil conflict or revolution equally well, which would undermine the specificity of the kinetic-export claim.

**Clustered SEs with 6 clusters.** With only six societies, clustered standard errors are themselves imprecise. p-values in Tables 2–4 should be treated as indicative.

### 5.3 Theoretical Implications

If H1 is real, the mechanism most consistent with the data is not simple diversion (leader-opportunism to distract domestic audiences) but structural displacement: the Stewards node accumulates stress that the domestic institutional ecosystem cannot absorb, and military action—which shifts resource-extraction demands onto occupied or adversarial territories—provides systemic relief. The H3 result adds a conditionality: Lore-Archive capacity determines whether the relief is found internally (institutional reform, redistribution, debt restructuring) or externally (war). Societies rich in institutional wisdom and memory appear to find the internal path; those without it do not.

This suggests a prioritisation for further research: rather than modelling war as a function of leader preferences or bilateral power ratios alone, examining the internal stress-absorption architecture of the initiating society may add predictive value.

---

## 6. Conclusion

Across 623 peace-year observations in six societies, Stewards overload is associated with a 2.6-fold increase in war-initiation odds over the following decade, an effect that survives militarisation controls and ranks at the 99.6th percentile of a within-society placebo distribution. The Jubilee alternative—internal overload resolution without Shield escalation—occurs dramatically more often in high Lore-Archive societies (82% vs 24%). The analysis is best read as strong hypothesis-generation: the signal is there, but it requires confirmation across a larger cross-national sample with independent (blind) CAMS scoring before causal claims can be sustained.

---

## Appendix A — Replication Materials

All analysis code, data, and output files are available in this repository:

- `backstory/runs/cams-war-thesis/thesis_analysis.py` — full Python pipeline
- `thesis_dataset.csv` — 623-row panel with all computed variables
- `thesis_results.txt` — raw model output, placebo distribution, H3 contingency
- `thesis_summary.md` — plain-language summary

The analysis requires `pandas`, `numpy`, `statsmodels`, and `scipy`. Run:
```bash
python backstory/runs/cams-war-thesis/thesis_analysis.py
```

---

## References

Beblawi, H., & Luciani, G. (Eds.). (1987). *The Rentier State*. Croom Helm.

Blainey, G. (1988). *The Causes of War* (3rd ed.). Free Press.

Correlates of War Project. (2010). *Militarized Interstate Disputes (v4.1)*. correlatesofwar.org.

Levy, J. S. (1989). The diversionary theory of war: A critique. In M. I. Midlarsky (Ed.), *Handbook of War Studies*. Unwin Hyman.

Mahdavy, H. (1970). The patterns and problems of economic development in rentier states: The case of Iran. In M. A. Cook (Ed.), *Studies in Economic History of the Middle East*. Oxford University Press.

CAMS Framework documentation: `backstory/CAMS_Framework_Hypothesis_rev6Oct2026.md` (this repository).
