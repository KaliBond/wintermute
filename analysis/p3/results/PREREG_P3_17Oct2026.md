# P3 frozen specification (written 17 Oct 2026 before any outcome was computed)

## Inputs
- Panel: KaliBond/wintermute `juno/JUNO_Unified_Dataset.csv` at commit 446928c (SHA-256 prefix 49ee7b173abc5856). Columns Society, Year, Node, Coherence, Capacity, Stress, Abstraction only. Canonical nodes Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.
- Primary events: Powell & Thyne Global Coups list, `coups-by-country.csv`, version V2026.08.29, from globalcoups.com (SHA-256 1e8f933b22f4b9a77d0ba5eb59bf0d8158adcaab27746adcf345195be87a9012). External to CAMS.

## Primary event rule (E1)
Event = 1 in society-year t if Powell & Thyne record at least one coup attempt (successful = 2 or failed = 1) in that country-year; 0 otherwise. Coverage: every year from max(1950, the state's first year in the coup list's country-code universe as set out below) to 2025. The list is a complete global register of attempts, so years with no row inside coverage are explicit zeros. Societies with no sovereign state (Hong Kong) are excluded. The coverage start year for post-1950 states is the independence year listed in COVERAGE in build_events.py. Predecessor states map to the successor (USSR→Russia, West Germany→Germany).

## Analysis (unchanged ChatGPT script, analysis/p3_compare.py)
- Horizon 3 is primary: target = any event in (t, t+3].
- Aggregate = 4 dimension means; configuration = 32 node×dimension scores; base-rate control. Leave-one-society-out, standardised L2 logistic regression (C = 0.1).
- Supplementary evaluation (added wrapper, does not change the script): pooled out-of-fold AUC and Brier; society-cluster bootstrap 95% CI for ΔAUC and ΔBrier (configuration − aggregate), 2000 replicates, seed 20261017.
- Success criterion for P3: pooled ΔAUC > 0 with the 95% CI excluding 0 AND ΔBrier < 0.

## Sensitivity (labelled, not primary)
- S1: E1 at horizon 5.
- S2: V-Dem v16 liberal democracy decline: event = 1 in year t if v2x_libdem(t) − v2x_libdem(t−1) ≤ −0.05; coverage = years where both values exist; horizon 3.

## Exploratory P7 bridge metric
Feature `Lambda2_Calc` (algebraic connectivity of the canonical bond matrix, as stored in the canonical dataset; constant within society-year). Compare aggregate + λ2 against aggregate alone under the same LOSO protocol, E1 horizon 3. B is a deterministic function of the node scores, so any gain is a nonlinear summary of the same scores, not independent information. Labelled exploratory.
