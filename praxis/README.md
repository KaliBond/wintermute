# praxis/ — reference files for the CAMS Praxis GPT

Draft material for the departmental CAMS Praxis GPT. Nothing here is issued.

| File | What it is |
|---|---|
| `CAMS_PRAXIS_GPT_Build_Pack_v1.1.md` | Corrected draft of the build pack. Part 0 lists six decisions the framework custodian must make before issue. |
| `cams_praxis_calc.py` | Reference calculator. The GPT runs this for every number it reports. Needs only numpy. |
| `test_vectors.json` | Five worked cases: one analytic, four real society-years from `data/v1.0_final_recompute/series/`. |
| `test_cams_praxis_calc.py` | Tests. Run `pytest praxis/`. |

## Quick use

```
python praxis/cams_praxis_calc.py --self-test
python praxis/cams_praxis_calc.py scores.csv --spread 2000 --noise-sd 0.5 --seed 1
python praxis/cams_praxis_calc.py scores.csv --classifier v1.2-final
```

Input is a long-format CSV with columns `Society, Year, Node, Coherence, Capacity, Stress, Abstraction` and exactly the eight canonical nodes per society-year. Output is JSON: the phase-space metrics, the regime, the regime under the other classifier, the gap to the nearest thresholds, per-node values, and (with `--spread`) the share of perturbed draws landing in each regime.

## How the calculator was checked (14 November 2026)

- **Against the existing engine.** For all 6,778 in-range society-years in `data/v1.0_final_recompute/`, V̄, s_V, V_min, B̄ and λ2 match `FG_corpus.csv` (produced by `cams_framework_v2_4.py`) to within 1e-14.
- **Activation at K = S.** s_min differs from the v2.4 engine in 853 society-years, all where the minimum node has K exactly equal to S (this file gives 0, v2.4 gives +0.1·A·C/100). This changes no regime label under either classifier.
- **Analytic case.** Eight identical nodes at C=K=S=A=5 give V̄ 7.5, B̄ 0.25 and λ2 = 8 × 0.25 = 2.0, as the algebra requires.

## Open custodian decisions

The full table is Part 0 of the build pack. The one that matters most:

**Two regime classifiers are in circulation.** Build pack v1.0 specifies "CAMS v1.0.1". The repository's production specification, and the classifier named in the lodged 2027–2028 register, is JUNO v1.2-Final. They differ in cascade order, thresholds and rounding. On the 6,778 society-years they agree 80.6% of the time. Germany 2024 is Local Node Failure under v1.0.1 and Phantom Type II under v1.2-Final (and "Strained" in the older v2.4 corpus file).

The calculator implements both and requires the caller to name one. It does not choose.

## Limits

- The spread is a sensitivity analysis under an assumed, independent noise level. It is not a confidence interval and is not derived from the between-run drift figures.
- Regime thresholds were fitted on nation-level societies. The calculator will compute metrics for any eight-node unit, but a regime label for a sector or agency is outside what the thresholds support.
- The four real test vectors show the calculator reproduces itself and the v2.4 engine. They say nothing about whether the scores are right.
