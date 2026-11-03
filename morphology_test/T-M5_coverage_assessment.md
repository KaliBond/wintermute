# T-M5 Coverage Assessment (no outcomes computed)

**Date:** 2026-10-16 · **Status:** coverage matrix only — zero external outcome comparisons run
**Inputs:** T-M5_episode_roster.csv (158 frozen episodes, 42 identified unified-set societies; uptake locus, R-based damage locus, per-node trough years). Blind-set episodes excluded pending user's decision on the answer key.

## WDI-only coverage (draft matrix)

Indicators: Hands = unemployment (SL.UEM.TOTL.ZS), Flow = trade % GDP (NE.TRD.GNFS.ZS), Craft = manufacturing VA (NV.IND.MANF.ZS), Stewards = revenue excl. grants (GC.REV.XGRT.GD.ZS). Coverage rule: non-missing baseline (any of onset−2..onset−1) AND ≥1 observation in [onset, end+2].

| Function | All 158 episodes | 1960+ episodes (n=81) |
|---|---|---|
| Hands (unemployment) | 21.5% | 42% |
| Flow (trade) | 43.0% | 84% |
| Craft (manufacturing) | 35.4% | 69% |
| Stewards (revenue) | 21.5% | 42% |
| **All four (Track A eligible)** | **12.7%** | **24.7% (20 episodes, 16 societies)** |

Coverage by decade: effectively zero before 1960, rising to 3.6/4 functions in the 2000s. WDI begins ~1960; this is a source limitation, not a data error.

## Findings for design

1. **WDI alone cannot support Track A.** 20 eligible episodes across 16 societies is thin, and predicted R-loci in the eligible set skew to Flow (9/20) — the selection bias the pre-registration warned against is visible in the coverage matrix itself: the best-documented function is the most-covered function.
2. **Binding constraints are Hands (unemployment) and Stewards (revenue).** ILOSTAT direct (not the WDI mirror) and UNU-WIDER GRD should substantially extend both; GRD covers many countries from ~1980, ILOSTAT modelled estimates from 1991. Historical extensions (Mitchell's International Historical Statistics, national yearbooks) would be required for pre-1960 episodes — high effort, likely better served by Track B adjudication.
3. **Track A is therefore a post-1960 (realistically post-1980) study.** The pre-1960 episode stock (77 episodes) is Track B territory by construction.
4. **Unequal observability confirmed empirically:** Flow 84% vs Hands/Stewards 42% post-1960. Construct-validity implication stands: the eight functions are not equally observable in independent quantitative records.

## Proposed frozen T-M5 gates (for user sign-off before any outcome computation)

- **T-M5-A eligibility:** episodes 1980+, all four WDI+ILOSTAT+GRD series present after source extension. Expected n after extension: 40–60 (estimate; to be confirmed by extended coverage matrix).
- **Primary comparison:** κ between CAMS R-based damage locus and independently measured max-deterioration function among {Hands, Flow, Craft, Stewards}, with per-episode baseline adjustment (z-score vs 5-yr pre-onset trend) and sector-wide shock treatment (drop episodes where all four series deteriorate >1 SD simultaneously — common-shock flag).
- **Nulls:** marginal-frequency baseline and matched permutation (shuffle external-locus labels within society).
- **Bars:** κ ≥ 0.30 against external measurement (higher than the 0.20 internal bar — external validation should clear a tougher fence), permutation p < 0.05, AND no worse than κ − 0.10 degradation vs the internal T-M4 benchmark computed on the same episode subset.
- **Registered scope statement:** four-function result ≠ eight-function validation; Track B (blinded human adjudication, fixed codebook) covers Helm/Shield/Lore/Archive on a stratified episode sample including pre-1960.
- **Anti-selection rule:** eligibility from dates + variable availability only; never from which function CAMS predicts.

## Open decisions for the user

1. Unblind the JUNO answer key for blind-set episodes in T-M5? (T-M4 is already locked; unblinding cannot retroactively compromise it, but precedent matters.)
2. Commission ILOSTAT/GRD extension now, or freeze WDI-only Track A at n≈20 as a pilot?
3. Track B codebook: recruit external coders, or AI-ensemble adjudication with human spot-audit (cheaper, but shares scorer-class risk with CAMS itself)?
