# praxis/ — reference files for CAMS Praxis

Draft material for the CAMS Praxis GPT and the soil-sector pilot. **Nothing here is issued.** Open scientific decisions are listed in Part 0 of the build pack.

Canonical regime classifier: **JUNO v1.2-Final**. Legacy comparator, always named: **v1.0.1**. Labels are never translated between the two, and stored historical outputs and lodged predictions are never rewritten.

| File | What it is |
|---|---|
| `CAMS_PRAXIS_GPT_Build_Pack_v1.1.md` | The build pack: settled rulings, unresolved decisions, GPT instruction set, master instructions, acceptance checks. |
| `CLASSIFIER_SPECIFICATIONS.md` | Exact cascade, thresholds and rounding for both classifiers, with source files and provenance. Section 5 covers activation when Capacity equals Stress. |
| `concordance/CLASSIFIER_CONCORDANCE.md` | Both classifiers applied to all 6,778 in-range society-years: confusion matrix, disagreements by regime and society, anchor cases. Built by `concordance/build_concordance.py`. |
| `controls/CONTROL_VALIDITY.md` | Whether the blind-file controls destroy what they should, keep what they should, and stay hidden. Built by `controls/test_controls.py`. |
| `cams_praxis_calc.py` | Reference calculator. Needs only numpy. |
| `test_vectors.json`, `test_cams_praxis_calc.py` | Five worked cases and the tests. Run `pytest praxis/`. |

## Quick use

```
python praxis/cams_praxis_calc.py --self-test
python praxis/cams_praxis_calc.py scores.csv --spread 2000 --noise-sd 0.5 --seed 1
python praxis/cams_praxis_calc.py soil_scores.csv --unit sector
python praxis/cams_praxis_calc.py scores.csv --activation-mode v2.4-compat
```

Input: a long-format CSV with columns `Society, Year, Node, Coherence, Capacity, Stress, Abstraction`, scores 1 to 10, exactly the eight canonical nodes per unit-year.

- `--unit nation` (default): metrics, the canonical regime, the gap to the nearest thresholds, the legacy comparator, and with `--spread` the share of perturbed draws in each regime.
- `--unit sector`: metrics and functional relationships (bond matrix, strongest and weakest pairs, each node's mean bond). **No regime label**, because the thresholds were fitted on nations. Use this for the soil-sector pilot.

## Rebuilding the reports

```
python praxis/concordance/build_concordance.py
python praxis/controls/test_controls.py
```

Both read `data/v1.0_final_recompute/` and write only inside `praxis/`.

## How the calculator was checked

- **Against the existing engine.** For all 6,778 in-range society-years, V̄, s_V, V_min, B̄ and λ2 match `FG_corpus.csv` (produced by `cams_framework_v2_4.py`) to within 1e-14. In `v2.4-compat` mode the activation and bond functions match the engine's own functions on random inputs.
- **Analytic case.** Eight identical nodes at C=K=S=A=5 give V̄ 7.5, B̄ 0.25 and λ2 = 8 × 0.25 = 2.0.
- **Browser port.** The blind-file game computes the same operators in JavaScript. On 605 score sets its metrics match this calculator to 5e-15 and both classifiers' labels match in every case.

## Limits

- No runnable v1.2-Final implementation existed in the repository before this one, so the cascade is checked against the written specification, not against the code that produced the original blind-corpus results.
- All corpus scores are model-generated. Agreement between classifiers is not accuracy.
- The spread is a sensitivity analysis under an assumed, independent noise level. It is not a confidence interval.
- The game's regime tallies are computed from the Telescope's stored scores, which are not the same files as the primary ensemble corpus used in the concordance.
