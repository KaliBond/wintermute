---
title: "CAMS/JUNO v1.2 Prospective Predictions 2027–2028: United States, China, Australia"
author: "Kari McKern (ComplexityWorkz) — Neural Nations"
date: "8 November 2026"
---

# CAMS/JUNO v1.2 prospective predictions 2027–2028: United States, China and Australia

**Version 1.0.0 · 8 November 2026 (Australia/Sydney) · Author: Kari McKern, ComplexityWorkz · Licence: CC BY 4.0 (text and data), MIT (code)**

**Archive DOI:** 10.5281/zenodo.24046639. Everything in this document is fixed at deposit. Corrections go in a dated erratum and any new version gets a new version number.

## 1. Status and what this is

This register lodges prospective, falsifiable, probability-weighted predictions about the *scored organisational state* of three societies in 2027 and 2028, under the JUNO v1.2-Final classifier, before the outcomes are scored. It is a thermometer reading projected two years forward, not a crystal ball: it makes no claim about events, causes or the intentions of any government, and it takes no political position.

It holds **32 claims** (12 USA, 11 China, 9 Australia). Of these, 11 are Tier A (stable across datasets), 20 are Tier B (dataset-dependent) and 1 is conditional.

**Relationship to earlier lodgements.** This register does not edit, replace or re-word the frozen 28 August 2026 register (twelve claims P-01 to P-12, locked 10 June 2026) or the 12 July 2026 six-nation memo; both remain in force and are scored separately. It supersedes the working drafts Registers B (USA), C (China) and D (Australia). The 12 July memo was issued under a v1.0-labelled classifier table; this register and the frozen register’s scoring protocol both use JUNO v1.2-Final.

## 2. Method (fixed at lodging)

- **Classifier:** JUNO v1.2-Final (node viability V = C + K − S + 0.5A; bond B_ij = √(q_i q_j)·2^(−(S_i+S_j)/10); sign-preserving cognitive activation s; aggregates rounded to 6 dp before every threshold; precedence Freeze/Collapse → Systemic Crisis → Phantom Type II → Local Node Failure → Stable Adaptive → Strained). “Severe” below means Freeze/Collapse, Systemic Crisis, Phantom Type II or Local Node Failure.
- **Simulation:** from each dataset’s last observed state, a 32-dimensional (8 nodes × Coherence, Capacity, Stress, Abstraction) annual random walk clipped to [1,10]. Noise covariance comes from anchor-to-anchor changes (repeated rows removed, scaled by √gap), shrunk toward the diagonal (30% for 50 or more anchors, otherwise 60%) plus 1e-4·I.
- **Scenarios,** equal weight, 3,000 paths each (9,000 per dataset): PERSIST (no drift); DRIFT (mean annual change of years ≥ 2015); REVERT (0.5 × drift + 0.15 × pull toward the 1980+ mean, or the full-sample mean for series starting after 1980).
- **Stated probability** = primary-dataset (UNIFIED) probability, capped to [5%, 95%] as a model-risk haircut. Exceptions, flagged: US-10 (8%, a judgement blending the simulation range of 1–14% with a historical base rate of about 12% over two years) and US-11 (50%, the historical base rate).
- **Reproducibility:** `code/cams_register_2027_2028.py` regenerates every probability from the extracts in `data/` with seed 2026.

## 3. Scoring rules (fixed at lodging)

1. **When.** Score each claim HIT / MISS / UNSCORABLE after the relevant year closes, using the first complete five-scorer ensemble scoring of that year (eight nodes, C/K/S/A present). Record dataset name and commit SHA.
2. **Primary dataset.** UNIFIED (`juno/JUNO_Unified_Dataset.csv`). If it has no 2027/2028 row, score on whichever datasets exist, say so, and mark the claim “primary dataset unavailable”.
3. **Tier A claims** are scored HIT/MISS on the primary dataset and also reported on every other available dataset. **Tier B claims** are reported on every dataset and receive a single verdict only if all available datasets agree; otherwise they are recorded as dataset-dependent, which is neither a hit nor a miss.
4. **Base state** means the last observed row of the dataset at lodging (2025 for China UNIFIED, 2026 otherwise). Relative claims use the same dataset’s base state.
5. **Brier scores** are reported over all scored claims on the primary dataset, and separately for Tier A. US-11 is scored only if US-10 occurs.
6. **No edits.** Thresholds, wording and probabilities may not change after deposit. Corrections go in an erratum; changed claims need a new version.
7. **Measurement margin.** The formalism cites ±1 NV between-run reliability at year level, wider at node level. Claims whose threshold lies close to the base state (US-4, US-7 and CN-9 within about 1 NV on at least one dataset; AU-5 within 0.01 of B̄) should be read with that margin in mind.

## 4. United States

**Reading.** High stress and a wide gap between capacity and coordination persist. The model has no view on whether the overall score rises or falls, but it expects no return to a “Strained” or better reading by 2028 and expects Shield to stay among the two strongest functions.

**Datasets used.**

- **UNIFIED**: `juno/JUNO_Unified_Dataset.csv`, USA rows, 26 scored slices 1900–2026. **Primary**, because the frozen register names it.
- **MARKER**: USA 1900–2026 ensemble mean, 127 years (55 distinct scored states).
- **MFIN**: `juno/USA_CAMS_mfin_2015-2026_block1.csv`, 12 annual states; V and B recomputed from raw C/K/S/A.
- **M5N**: `juno/USA_CAMS_m5n_2000-2026_block1.csv`, 27 years, evidence-gated ensemble.

**Claims.** Window: calendar years 2027 and 2028. “Both years” means both must hold. Columns after “Stated p” show the simulated probability on each dataset.

| ID | Claim | Stated p | UNIFIED | MARKER | MFIN | M5N | Tier | Falsified if |
|----|----------------------------------------------|-------|--------|--------|--------|--------|----|------------------------|
| US-1 | The USA is classified neither Strained nor Stable Adaptive in 2027 or 2028 | **95%** | 100% | 97% | 97% | 100% | A | Either year classifies Strained or Stable Adaptive |
| US-2 | Shield is among the two highest-V nodes in both years | **95%** | 97% | 99% | 97% | 92% | A | Shield outside the top two in either year |
| US-3 | Lore V < 4.0 in both years | **95%** | 98% | 94% | 61% | 98% | B | Lore V ≥ 4.0 in either year |
| US-4 | Helm V < 4.0 in 2027 (overlaps the 12 July 2026 memo’s Helm arm; not a restatement of it) | **71%** | 71% | 94% | 68% | 100% | B | Helm V ≥ 4.0 in 2027 |
| US-5 | System Bond Strength B̄ < 0.20 in both years | **95%** | 100% | 85% | 54% | 100% | B | B̄ ≥ 0.20 in either year |
| US-6 | Mean stress S̄ ≥ 6.0 in both years | **95%** | 100% | 94% | 81% | 92% | A | S̄ < 6.0 in either year |
| US-7 | V(Shield) − mean[V(Helm), V(Lore)] ≥ 5.0 in both years | **58%** | 58% | 99% | 93% | 77% | B | Gap < 5.0 in either year |
| US-8 | \|V̄(2028) − V̄(2026)\| ≤ 2.0 on the same dataset. No directional call. | **88%** | 88% | 67% | 83% | 80% | A | V̄ moves by more than 2.0 in either direction |
| US-9 | Helm and Lore are not both ≥ 4.0 in the same year | **95%** | 99% | 96% | 79% | 100% | A | Both ≥ 4.0 in one year |
| US-10 | A one-year fall in V̄ of 2.0 or more occurs in 2027 or 2028 (probability is a judgement; see §2) | **8%** | 1% | 14% | 6% | 5% | A | Hits if such a fall occurs; MISS if none |
| US-11 | Conditional on US-10 occurring: Flow is the largest single-node faller that year (probability is the historical base rate, 4 of 8 shocks) | **50%** | n/a | n/a | n/a | n/a | C | Another node falls most. UNSCORABLE if US-10 does not occur |
| US-12 | The USA classifies Systemic Crisis in both 2027 and 2028 | **45%** | 45% | 40% | 14% | 61% | B | Any other label in either year |

**Reference distributions** (system mean V̄, pooled across scenarios; not scored as claims):

| Dataset | Base (year) | Base regime | 2027 median (80% interval) | 2028 median (80% interval) |
|----------|------------|----------------|----------------------|----------------------|
| UNIFIED | 3.55 (2026) | Systemic Crisis | 3.6 (2.5 to 4.8) | 3.7 (2.0 to 5.4) |
| MARKER | 4.11 (2026) | Phantom Type II | 4.1 (2.3 to 5.9) | 4.1 (1.5 to 6.8) |
| MFIN | 6.15 (2026) | Local Node Failure | 6.0 (4.8 to 7.3) | 5.9 (4.1 to 7.8) |
| M5N | 0.92 (2026) | Systemic Crisis | 0.9 (−0.4 to 2.4) | 1.0 (−1.0 to 3.0) |

Regime probabilities, 2027 → 2028:

| Regime | UNIFIED | MARKER | MFIN | M5N |
|---|---|---|---|---|
| Stable Adaptive | 0% → 0% | 0% → 0% | 0% → 0% | 0% → 0% |
| Strained | 0% → 0% | 0% → 2% | 1% → 3% | 0% → 0% |
| Phantom Type II | 7% → 9% | 38% → 32% | 34% → 37% | 1% → 2% |
| Local Node Failure | 35% → 34% | 10% → 10% | 46% → 32% | 2% → 3% |
| Systemic Crisis | 57% → 58% | 52% → 53% | 18% → 29% | 78% → 68% |
| Freeze/Collapse | 0% → 0% | 0% → 2% | 0% → 0% | 19% → 26% |

**Where the datasets disagree.** The four datasets agree on structure and disagree on level. Shield is in the top two on all four and Lore is among the lowest three at the 2026 base on all four. Helm is the single weakest node on two datasets (MARKER, M5N), second weakest on MFIN and fourth lowest on UNIFIED. 2026 mean V̄ ranges from 0.9 (M5N) to 6.15 (MFIN), and the 2026 regime label is Systemic Crisis (UNIFIED, M5N), Phantom Type II (MARKER) or Local Node Failure (MFIN). On M5N the base sits near the Freeze/Collapse boundary, so that label has a 19–26% probability there and 0–2% on the others. The MARKER envelope is floored at 0.3 and is not a measure of scorer spread; the M5N envelope is.

**Context noted, not modelled.** Press reports on 4 November 2026 (for example the ABC News projection) projected Democratic control of both chambers of Congress after the 3 November midterms, with seat counts differing between sources while counting continued. Sources: [ABC News live updates, 4 Nov 2026](https://abc7news.com/live-updates/midterm-elections-2026-live-updates-election-results-key-races-us/20001292/entry/20014313/). The simulation is not conditioned on any of this, and events inside the scoring window are not a reason to reinterpret these claims.

## 5. China

**Reading.** The datasets disagree about China’s level, so the register separates the claims that should hold whichever dataset is used (the weakest functions are the labouring base and resource allocation; Archive stays strong) from those that depend on it (the regime label).

**Datasets used.**

- **UNIFIED**: `juno/JUNO_Unified_Dataset.csv`, China 1800–2025, annual. **Primary.** No 2026 row, so the base state is 2025.
- **CAMS1900**: `cleaned_datasets/China_CAMS_1900_2026_cleaned.csv`, 127 years, base 2026.
- **M5N**: `juno/China_CAMS_m5n_2000-2026_block1.csv`, 27 years, evidence-gated ensemble, base 2026.

**Claims.** Window: calendar years 2027 and 2028. “Both years” means both must hold. Columns after “Stated p” show the simulated probability on each dataset.

| ID | Claim | Stated p | UNIFIED | CAMS1900 | M5N | Tier | Falsified if |
|----|----------------------------------------------|-------|--------|--------|--------|----|------------------------|
| CN-1 | China is never classified as a severe regime (Freeze/Collapse, Systemic Crisis, Phantom Type II, Local Node Failure) in 2027 or 2028 | **79%** | 79% | 63% | 27% | B | A severe label in either year |
| CN-2 | China classifies Stable Adaptive in both years (overlaps the 12 July 2026 memo) | **58%** | 58% | 31% | 1% | B | Any other label in either year |
| CN-3 | System mean V̄ > 10.0 in both years | **95%** | 95% | 69% | 1% | B | V̄ ≤ 10.0 in either year |
| CN-4 | Hands is among the three lowest-V nodes in both years | **84%** | 84% | 90% | 89% | A | Hands outside the bottom three in either year |
| CN-5 | Stewards is among the three lowest-V nodes in both years | **79%** | 79% | 83% | 91% | A | Stewards outside the bottom three in either year |
| CN-6 | Archive is among the three highest-V nodes in both years (tests the Archive leg of frozen claim P-10) | **64%** | 64% | 37% | 41% | B | Archive outside the top three in either year |
| CN-7 | System B̄ > 0.28 in both years | **94%** | 94% | 71% | 8% | B | B̄ ≤ 0.28 in either year |
| CN-8 | \|V̄(2028) − V̄(base)\| ≤ 2.0 on the same dataset. No directional call. | **72%** | 72% | 62% | 87% | B | V̄ moves by more than 2.0 in either direction |
| CN-9 | V_min < 4.0 in 2027 or 2028 (any node below the Local Node Failure threshold) | **5%** | 2% | 8% | 72% | B | V_min ≥ 4.0 in both years |
| CN-10 | The highest-V node is Shield or Craft in both years | **56%** | 56% | 75% | 83% | B | Another node is highest in either year |
| CN-11 | A one-year fall in V̄ of 2.0 or more occurs (2026→27 or 2027→28; actual 2026 row used once it exists) | **16%** | 16% | 23% | 3% | A | Hits if such a fall occurs; MISS if none |

**Reference distributions** (system mean V̄, pooled across scenarios; not scored as claims):

| Dataset | Base (year) | Base regime | 2027 median (80% interval) | 2028 median (80% interval) |
|----------|------------|----------------|----------------------|----------------------|
| UNIFIED | 13.49 (2025) | Stable Adaptive | 13.3 (11.4 to 15.3) | 13.3 (10.9 to 15.6) |
| CAMS1900 | 11.62 (2026) | Stable Adaptive | 11.6 (9.4 to 13.7) | 11.5 (8.6 to 14.4) |
| M5N | 7.99 (2026) | Strained | 8.0 (6.8 to 9.2) | 8.1 (6.4 to 9.8) |

Regime probabilities, 2027 → 2028:

| Regime | UNIFIED | CAMS1900 | M5N |
|---|---|---|---|
| Stable Adaptive | 74% → 63% | 49% → 40% | 1% → 5% |
| Strained | 17% → 19% | 34% → 28% | 43% → 32% |
| Phantom Type II | 9% → 18% | 17% → 32% | 7% → 19% |
| Local Node Failure | 0% → 0% | 0% → 0% | 49% → 43% |
| Systemic Crisis | 0% → 0% | 0% → 0% | 0% → 1% |
| Freeze/Collapse | 0% → 0% | 0% → 0% | 0% → 0% |

**Where the datasets disagree.** The datasets agree on shape and disagree on level. Hands and Stewards sit in the bottom three on every dataset (79–91%) and Archive stays high. UNIFIED reads Stable Adaptive at V̄ 13.5, CAMS1900 reads Stable Adaptive at 11.6, and the newest evidence-gated run (M5N) reads Strained at 8.0. The 5.5 NV gap between UNIFIED and M5N is about 35 standard errors, so it is systematic, not scorer noise, and it widens after 2020 (mean gap 1.3, 2.0, 4.1 NV by decade). It sits in Coherence and Abstraction, not Stress. Whether that reflects thin external evidence on opaque systems, a different scoring protocol, or real organisational difference cannot be settled from these data; the M5N NA rates would help.

**Context noted, not modelled.** The Associated Press reported on 29 September 2026 that second-quarter 2026 growth was 4.3% against a 2026 target of 4.5–5%, and that new measures to support the property sector and economy had been announced. Outlets continue to describe a prolonged property downturn and weak domestic demand. Sources: [AP via WRAL, 29 Sep 2026](https://www.wral.com/news/ap/560c1-china-rolls-out-new-measures-to-bolster-its-property-sector-and-economy/); [Outlook Business, Q2 2026 GDP](https://www.outlookbusiness.com/economy-and-policy/chinas-economy-slows-to-43-why-it-matters-for-the-worldand-for-india). The simulation is not conditioned on any of this, and events inside the scoring window are not a reason to reinterpret these claims.

## 6. Australia

**Reading.** Australia is balanced near a regime boundary. The model leans toward persistence, expects Archive to stay among the strongest functions, and flags that the July 2026 Craft-recovery claim holds or fails depending on the dataset used.

**Datasets used.**

- **UNIFIED**: `juno/JUNO_Unified_Dataset.csv`, Australia 1875–2026, annual. **Primary.**
- **CAM5_1996**: `cleaned_datasets/Australia_cam5_1996_2026_cleaned.csv`, 31 years.
- **M5N**: `juno/Australia_CAMS_m5n_2000-2026_block1.csv`, 27 years, evidence-gated ensemble.

**Claims.** Window: calendar years 2027 and 2028. “Both years” means both must hold. Columns after “Stated p” show the simulated probability on each dataset.

| ID | Claim | Stated p | UNIFIED | CAM5_1996 | M5N | Tier | Falsified if |
|----|----------------------------------------------|-------|--------|--------|--------|----|------------------------|
| AU-1 | Australia is never classified as a severe regime in 2027 or 2028 | **62%** | 62% | 84% | 96% | B | A severe label in either year |
| AU-2 | System V̄ ≥ 7.0 in 2027 and ≥ 7.5 in 2028 (overlaps frozen claim P-09) | **76%** | 76% | 100% | 99% | A | V̄ < 7.0 in 2027 or < 7.5 in 2028 |
| AU-3 | Craft V ≥ 8.08 in 2028 (overlaps the 12 July 2026 memo; Craft base is 6.9 on the primary dataset) | **33%** | 33% | 100% | 91% | B | Craft V < 8.08 in 2028 |
| AU-4 | Archive is among the three highest-V nodes in both years | **88%** | 88% | 61% | 80% | B | Archive outside the top three in either year |
| AU-5 | System B̄ > 0.28 in both years (primary-dataset base is 0.276, at the threshold) | **36%** | 36% | 85% | 81% | B | B̄ ≤ 0.28 in either year |
| AU-6 | V_min ≥ 5.0 in both years | **58%** | 58% | 89% | 91% | B | V_min < 5.0 in either year |
| AU-7 | Australia classifies Stable Adaptive in at least one of the two years | **33%** | 33% | 85% | 61% | B | Strained or a severe label in both years |
| AU-8 | \|V̄(2028) − V̄(2026)\| ≤ 2.0 on the same dataset. No directional call. | **69%** | 69% | 84% | 95% | B | V̄ moves by more than 2.0 in either direction |
| AU-9 | A one-year fall in V̄ of 2.0 or more occurs in 2027 or 2028 | **15%** | 15% | 5% | 1% | A | Hits if such a fall occurs; MISS if none |

**Reference distributions** (system mean V̄, pooled across scenarios; not scored as claims):

| Dataset | Base (year) | Base regime | 2027 median (80% interval) | 2028 median (80% interval) |
|----------|------------|----------------|----------------------|----------------------|
| UNIFIED | 8.90 (2026) | Strained | 9.0 (7.2 to 10.8) | 9.1 (6.5 to 11.6) |
| CAM5_1996 | 11.81 (2026) | Stable Adaptive | 11.8 (10.5 to 13.1) | 11.8 (10.0 to 13.6) |
| M5N | 9.86 (2026) | Strained | 9.9 (9.0 to 10.9) | 10.0 (8.7 to 11.3) |

Regime probabilities, 2027 → 2028:

| Regime | UNIFIED | CAM5_1996 | M5N |
|---|---|---|---|
| Stable Adaptive | 21% → 24% | 80% → 60% | 47% → 47% |
| Strained | 62% → 43% | 17% → 26% | 53% → 49% |
| Phantom Type II | 14% → 29% | 2% → 12% | 0% → 2% |
| Local Node Failure | 2% → 4% | 0% → 2% | 0% → 2% |
| Systemic Crisis | 0% → 0% | 0% → 0% | 0% → 0% |
| Freeze/Collapse | 0% → 0% | 0% → 0% | 0% → 0% |

**Where the datasets disagree.** Australia sits close to a regime boundary on every dataset, so small level differences flip labels (regimes agree in only 26% of overlap years between UNIFIED and M5N). UNIFIED reads Strained at V̄ 8.9, M5N Strained at 9.9 (15% of base-noise draws read Stable Adaptive) and CAM5_1996 Stable Adaptive at 11.8. M5N reads Australia higher than UNIFIED, mainly through lower Stress, the opposite direction to China. Archive in the top three is the one node-level feature all three datasets share. The UNIFIED envelope is largely carried forward (19 distinct blocks in 152 years) and understates Abstraction spread; the finding that its base is Strained holds with that spread floored at M5N levels.

**Context noted, not modelled.** The Reserve Bank of Australia raised the cash rate by 25 basis points to 4.85% on 3 November 2026, its fifth increase of the year, on a 6–3 vote, and projected headline inflation peaking around 4.5% and staying above 3% until late 2027, with GDP growth around 1.6% in the year to December 2026. Sources: [investingLive, RBA 3 Nov 2026](https://investinglive.com/central-banks/more-rba-lifts-cash-rate-to-4-85-in-split-6-3-vote-sees-inflation-above-3-until-late-2027/); [The Adviser, RBA 3 Nov 2026](https://www.theadviser.com.au/borrower/49167-rba-tightens-the-reins-on-cash-rate). The simulation is not conditioned on any of this, and events inside the scoring window are not a reason to reinterpret these claims.

## 7. Changes from the working drafts (Registers B, C, D)

- **US-2 replaces draft C2.** Draft C2 (“Shield is the single highest-V node”) was 85%, 98%, 70% and 33% across UNIFIED, MARKER, MFIN and M5N; the replacement (“among the two highest”) is 92–99% on all four.
- **US-12 is now the joint probability** of Systemic Crisis in both years (45% on UNIFIED) instead of the draft’s per-year 58%.
- **Claims are renumbered** US-, CN- and AU-. US-1, US-3 to US-11 match draft claims C1, C3 to C11; CN-1 to CN-11 match Register C; AU-1 to AU-9 match Register D.
- **All probabilities were regenerated from one script and one seed.** Outside US-2 and US-12, every stated probability is within two points of its draft value.
- **Tiers were added.** Claims are labelled A (probability spread across datasets of 25 points or less) or B (wider), so a result can be read at the right strength.
- **The USA M5N dataset was added** as a fourth sensitivity dataset. Measured scorer noise on the base state was tested for China, Australia and USA M5N and changed no claim probability by more than 1.5 points, so stated probabilities do not include it.

## 8. Limitations

- The scores are model-generated (AI-rater ensembles reading documentary evidence). Ensemble tightness certifies precision, not ground truth, and shared model bias is an unquantified confound.
- The four USA, three China and three Australia datasets disagree by more than the ±1 NV between-run margin the formalism cites. The register does not choose between them. It reports every claim on every available dataset and scores only the primary dataset as a single verdict.
- The simulation has no directional edge: with drift and mean-reversion pooled, medians sit near the base state. The register makes no call on whether V̄ rises or falls.
- Noise covariance is estimated from a small number of anchor-to-anchor changes (12 to 226 scored states) and shrunk toward the diagonal. Intervals are indicative.
- Claims are correlated (many share a node, a threshold or a path). Hit rates and Brier scores are not independent trials.
- Bond strength in JUNO v1.2 is algebraically rank-1, so B̄ and λ₂ carry no pair-specific topology. Claims on B̄ restate stress and coupling quality. Network pictures are illustrative.
- Phantom Type II has no mean-viability condition and fires on s_min < −0.65, so simulation noise raises its share with horizon. The 2028 Phantom share is partly a noise artefact.
- JUNO v1.2 is not externally validated. Scoring this register is part of the prospective test the formalism names, together with the frozen 28 August 2026 register.

## 9. Files in this deposit

- `README.md`: this deposit, how to cite, how to reproduce
- `CAMS_JUNO_Prospective_Register_2027-2028_v1.0.0.md` and `.pdf`: this register
- `predictions-2027-2028.html`: public web page
- `results/FINAL_claims_register.csv`: every claim with stated and per-dataset probabilities, tier and falsifier
- `results/claim_probabilities_by_surface.csv`, `system_forecast_by_surface.csv`, `regime_distribution_by_surface.csv`, `node_forecast_by_surface.csv`: simulation outputs
- `code/cams_register_2027_2028.py`: reproduces all numbers (seed 2026)
- `code/build_documents.py`: builds this document and the web page from the results
- `data/`: the exact input rows and envelopes used
- `SHA256SUMS.txt`: checksums of every file

Upstream repository: <https://github.com/KaliBond/wintermute> (snapshot used for the inputs: commit `b4227c2151b9e58c97838643d521660fdad139d6`, 7 November 2026). The USA MARKER ensemble was supplied separately by the author and is included in `data/`.

## 10. How to cite

McKern, K. (2026). *CAMS/JUNO v1.2 prospective predictions 2027–2028: United States, China and Australia* (Version 1.0.0) [Preregistered prediction register]. Neural Nations / ComplexityWorkz. DOI: 10.5281/zenodo.24046639.
