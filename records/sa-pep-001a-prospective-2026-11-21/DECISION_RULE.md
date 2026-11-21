# SA-PEP-001a prospective set: decision rule and outcome definitions

Status: DRAFT for Kari's review, 21 November 2026 (AEDT). Not sealed, not deposited. Items marked
**KARI TO CONFIRM** must be settled before the bundle is deposited, because a timestamp only protects
what it contains.

## 1. Predictor (frozen, 001a as restated by Kari on 21 November 2026)

- BS_i(t): node i's bond strength, the mean of its 7 canonical bonds at year t (pipeline_R3.bond_strength).
- dBS_i = BS_i(2026) - BS_i(2024) (predictor window t-2..t, t = 2026).
- **D_Helm = mean over j != Helm of dBS_j, minus dBS_Helm.** D_Helm > 0 means Helm's bond strength fell
  relative to the other seven nodes.
- Reference implementation: decoupling_R3_4.py (in this folder, SHA-256 in MANIFEST.txt; unchanged since
  hashes_R3_4.txt).
- Values as they stand on 21 November 2026: prospective_predictor_D_Helm_2026.csv (32 societies).

## 2. Outcome

- Class: RIGIDITY in the outcome window t+1..t+3 = calendar years 2027, 2028 and 2029.
- Working definition, from the v0.1 prediction text: executive overconfidence, strategic rigidity or
  defensive policy persistence in the outcome window. **KARI TO CONFIRM: the full RIGIDITY codebook entry
  (Appendix A of the prereg draft is still open), inclusions, exclusions and the evidence standard.**
- Coding: binary per society (present / absent), with Unknown coded separately from absent and excluded
  from the analysis (reported). Coded after 31 December 2029 by at least two coders who do not see the
  CAMS scores or D_Helm; disagreements resolved by a third coder; Cohen's kappa reported.
  **KARI TO CONFIRM the coders and the coding date.**

## 3. Decision rule

Confirmatory rule (unchanged from R3.3/R3.4, frozen tau 0.979, R3.5 recheck in calibration_R3_5.md):
- O1 (CAMS-ontology credit) iff pct > 0.979 AND the CAMS-vs-aggregate and CAMS-vs-base-rate Brier
  improvements both have cluster-bootstrap 2.5th percentiles > 0.
- Otherwise O2 (structure, not CAMS-specific) iff the median arbitrary partition beats both the aggregate
  and the base rate on the same bounds.
- Otherwise O3 (no credit).
- Pipeline: pipeline_R3_1 / pipeline_R3_3 features, L2 logistic regression (lambda 1), 5-fold CV grouped by
  society, 1,000 comparison partitions, 1,000 cluster bootstrap resamples.

Scope: 32 cases alone are far below the calibrated need (R3.4: about 293 cases for 50% power at gamma 1.0;
power at n = 160 was 0.13). The confirmatory rule above is therefore applied only when these cases are
pooled with other cases scored before their outcomes were known. **KARI TO CONFIRM.**

Proposed stand-alone descriptive check for the 32 (reported, no CAMS-ontology credit possible):
- Top tercile of D_Helm (prospective_predictor_D_Helm_2026.csv, column tercile; ties broken by
  alphabetical order of society, which is what the stored ranks do) vs the other two terciles.
- HIT if the RIGIDITY rate in the top tercile exceeds the rate in the rest, one-sided Fisher exact
  p < 0.05. MISS otherwise. Stated in advance: power is low.

## 4. Known weaknesses, stated before outcomes

- Cross-scoring reliability of D_Helm is low (r about 0.02 to 0.27 between independent ensemble scorings of
  the same society-years; USA -0.15). See collation_addendum_R3_5_2026-11-21.md.
- Copy-forward: for Colombia and Poland the 2024, 2025 and 2026 matrices are identical (D_Helm = 0 by
  construction); Japan, Norway, UK and Ukraine have 2026 identical to 2025. Flagged in the predictor CSV.
- All 2026 scores were produced by LLM ensembles that know events up to their training and run dates, but
  not 2027 to 2029. They are prospective only from the timestamp date.
- Four societies tie at D_Helm = 0 (Colombia, Norway, Poland, Singapore); the tercile boundary falls among
  them, so tercile membership for these four depends on the tie rule.
