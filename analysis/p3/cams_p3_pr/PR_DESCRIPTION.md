# Test P3: eight-node configuration versus aggregate CAMS scores (LOSO)

## Hypothesis
The full eight-node CAMS configuration offers incremental **out-of-society** predictive information beyond four aggregate dimension means for an **externally defined** crisis/event onset over the next three years.

## Design
- Inputs: frozen annual long-format CAMNATIONS5 ensemble scores (Society, Year, Node, Coherence, Capacity, Stress, Abstraction) and a separately sourced, complete annual binary event register (Society, Year, Event).
- Reject incomplete eight-node observations, ambiguous or duplicate cells, and incomplete future event windows. Never infer that a missing event row is a non-event.
- Baseline: 4 CAMS means. Challenger: 32 node-by-dimension scores. Negative control: training-set base rate.
- Standardised logistic regression with identical fixed regularisation; preprocessing fitted **only on training societies**. Hold out each society completely.
- Report per-society Brier scores, AUC only where both classes are present, unweighted fold means, delta Brier, and out-of-fold prediction CSV. A negative delta Brier favours the full configuration.
- **Do not claim success** on in-sample fit, a positive selected fold, or an AUC with one class. No claim of prospective validation from historically informed scoring.

## Explicit limitations / next iteration
This is an intentionally small first test rather than definitive validation. The design does not yet correct for retrospective scoring leakage, choice of event taxonomy, target recurrence, uncertainty intervals, rare-event calibration, or historical covariate shift. Freeze input file versions, external event taxonomy and analysis specification before examining results. Run a temporal holdout and bootstrap by society in a separate preregistered follow-up. Do not use CAMS-derived `P<0` to define the outcome (a known concern in PR-GAP-2).

## Run
```bash
python -m pip install pandas numpy scikit-learn pytest
python -m pytest -q tests/test_p3_compare.py
python analysis/p3_compare.py --panel FROZEN_CAMS.csv --events INDEPENDENT_EVENTS.csv --horizon 3 --out p3_result.json
```

## Acceptance
- [ ] Freeze data SHA256 and independent event coding provenance.
- [ ] Check missing data and event register coverage; no results if incomplete.
- [ ] Publish all folds including contrary findings and nulls.
- [ ] Retain base-rate baseline and report both aggregate and configuration comparisons.
- [ ] Interpret any performance improvement as *retrospective evidence*, pending independent/prospective replication.

## CI
Please run this test as an **independent workflow**, not as a dependency of the legacy `validate.yml` USA-1861 expectation, which has been failing for unrelated numerical reasons.
