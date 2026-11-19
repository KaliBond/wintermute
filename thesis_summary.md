# CAMS War Initiation Thesis — Summary of Results

**Date:** 2026-11-19
**Data:** China, England, Finland, France, Germany, USA
**Observations:** 623 society-peace-years
**War initiation events (10yr window):** 82
**Overload_3yr events:** 102

---

## H1 — Kinetic Export Hypothesis

**Claim:** Sustained Stewards overload (Stress > Capacity for 3+ consecutive years)
raises the probability of war initiation within 10 years.

**Result:**

- **Model 1 (baseline):** overload_3yr coefficient = 0.972, OR = 2.643, p = 0.028
- **Model 2 (controlled for Shield_Stress and system_stress):** overload_3yr coefficient = 1.027, OR = 2.792, p = 0.069 (marginal)

p-values use standard errors clustered by society (6 clusters), as in `thesis_results.txt`.

**H1: Supported at conventional significance in Model 1 (p = 0.028); marginal with controls (p = 0.069).** Overload is positively associated with war initiation.

Controlling for Shield_Stress and system_stress, the coefficient grows by 5.7%.
**FC1 not triggered:** overload retains independent predictive value after controls.

### Placebo test

Real coefficient (0.972) ranks at **99.6th percentile** of 500 shuffled runs (median = 0.426, 95th = 0.752).
Placebo test **supports H1**: real timing is unusually predictive.

---

## H2 — Recovery Moderation

**Claim:** High Lore-Archive index weakens the overload→war link.

- **Model 3 interaction term (overload_3yr × lore_archive_idx):** coef = -1.278, p = 0.136

**H2: Directional, not significant.** The negative interaction is in the predicted direction (high Lore-Archive attenuates the overload effect), but it is not significant at conventional levels.

---

## H3 — Jubilee Alternative

**Claim:** In high Lore-Archive societies, overload resolves internally (no Shield rise)
more often than in low Lore-Archive societies.

- High Lore-Archive jubilee rate: 82.4% (n=17)
- Low  Lore-Archive jubilee rate: 24.1% (n=83)

**H3: Supported.** High LA societies substantially more likely to resolve overload without Shield escalation.

---

## Falsification Summary

| Criterion | Assessment |
|-----------|-----------|
| FC1: Effect disappears under Shield/system_stress controls | See Model 1 vs 2 above |
| FC2: Overload predicts civil conflict equally | Not tested — no civil conflict outcome in dataset |
| FC3: Effect similar across LA groups | See Model 3 interaction |
| FC4: Results don't replicate Gem vs Grok | See Replication Check in thesis_results.txt |

---

## Caveats

- **Sample size:** 623 peace-year observations across 6 societies. War onset events are rare (82 outcome-window years), limiting statistical power for rare-event logistic regression.
- **Historical coverage varies** significantly by society, introducing temporal heterogeneity.
- **H1 is structurally dominated by China**, which provides 73 of 102 overload_3yr events. England (1750–1898), France, and USA show zero overload events in this dataset, so the pooled regression is essentially testing China + Germany + Finland.
- **England coverage (1750–1898)** means two of its three coded war onsets (1956, 1982) fall outside the data entirely; only the 1899 Boer War onset is reachable via the 10-year window.
- **War onset coding** is a simplified hardcoded list; disputes about initiator attribution directly affect results.
- **Rome** was excluded from quantitative models (no COW comparators).
- Clustered standard errors by society use very few clusters (6), so reported p-values are approximate.

---

*All details in `thesis_results.txt`. Full panel in `thesis_dataset.csv`.*
