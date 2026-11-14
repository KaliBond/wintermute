# Audit — "SUPPORTED CONCLUSIONS?" document vs JUNO_Blind_motif.csv

*25 September 2026. Data: 12 societies × 61 years × 8 nodes (5,856 rows). Script: `audit_supported_conclusions.py` (reproducible).*

## Data quality gate

Carry-forward audit (protocol flag >30%): 0% for 11 societies; Society_K 19.7%. **The blind panel is clean** — no carry-forward artefact concern. (Contrast: the earlier 6-society JUNO panel used for the Stress × Flow test is heavily smoothed; results from these two files are not interchangeable.)

## Verdict table

| # | Claim | Audit result | Verdict |
|---|-------|--------------|---------|
| 0 | Carry-forward clean | 0–19.7% per society | **Supported** |
| 1 | Repertoire 13–27 states, p<0.0002 vs circular-shift null | Reproduced exactly: 13–27; null p = 0.0000 in all 12 | **Supported** |
| 2 | First PC of K-S ≈ 82.9% mean variance | Mean 81.7% (range 65.6–90.3%) | **Supported** (minor discrepancy) |
| 3 | One-factor sign reconstruction ≈ 89%; ">90% with a few dimensions" | 1-factor 82.4%; 3-factor 84.4% | **Direction right, numbers overstated** |
| 4a | 40/44 split recoveries; 19/40 never realign | My operationalisation: 33 split, 12 never realign | **Direction supported; counts operationalisation-sensitive** |
| 4b | 34/36 = 94.4% recoveries return with different configuration | 36/37 = 97.3% | **Supported** |
| 4c | ~58% of same-M separated pairs differ in configuration | 50.7% (M rounded 2dp, separation ≥5y) | **Approximate; threshold-sensitive** |
| 4d | 53.4% of M>0 observations contain a non-positive node | 51.8% (243/469) | **Supported** |
| 5 | Repertoire span 4.96–20.40; switching 0.283–0.700; ρ(span,switch)≈0.91 | Span 4.96–20.40 exact; switching 0.295–0.705; ρ = 0.91, p = 3.5e-5 | **Supported** (exact on span and ρ) |
| 6 | Node-role concordance Kendall W ≈ 0.33, p ≈ 1e-4 | W = 0.398, permutation p < 5e-5 | **Supported** (W def-dependent, significance robust) |
| 7 | Coalition: Hands, Stewards, Flow, Helm ↔ Archive, Shield, Lore, Craft (relative) | Node mean K-S reproduces this ordering exactly | **Supported as relative tendency** |
| 8 | Shared threshold chain: ~72% of split obs within 1 flip; 14% exact | Split obs: 77.6% within 1 flip vs **74.6% under marginal-matched random chains**; 23.1% exact vs 23.8% null | **Not established — null-matched test shows ~chance** |
| 9 | Node-local persistence improves prediction in all 12; other nodes add nothing | Own-lag AUC gain positive in **12/12** (median +0.083); cross-node lags positive in **0/12** (median −0.081) | **Supported** |
| 10 | No universal lagged pairwise grammar (0.03% RMSE gain) | Light held-out check consistent (cross-node lags overfit); original RMSE pipeline not re-run | **Consistent; not independently reproduced** |

## The two findings that matter most

**1. The threshold-chain claim (#8) does not survive a fair null.** The document's chain (Shield→Lore→Archive→Craft→Helm→Flow→Stewards→Hands) fits 77.6% of mixed-sign observations within one flip — but a random chain fits 74.6%, and the chain implied by raw positivity rates (Archive→Shield→Craft→Lore→Helm→Flow→Stewards→Hands) differs from the claimed one at the top end. Because nodes have different marginal positivity rates, almost any ordering nests almost everything. The "shared approximate threshold hierarchy" should be downgraded to *nodes have stable, society-recurrent positivity rates* — which is claim #6 restated — unless a held-out ordering test with a marginal-matched null can rescue it.

**2. The positive claims are robust.** Repertoire restriction, low dimensionality, role concordance, the revised coalition, aggregate-vs-structural recovery mismatch, repertoire geometry, and node-local persistence all reproduce on my independent implementation, with only minor numeric drift (81.7 vs 82.9; 82.4 vs 89; 50.7 vs 58). The central reframing — *conserved functional response basis + variable global morphology, not a fixed pairwise grammar* — is well supported by this audit.

## Interaction with the earlier Stress × Flow result

The recovery/configuration findings here are computed on a clean, unsmoothed blind panel. The Stress × Flow interaction was computed on the smoothed 6-society panel. Before citing the Stress × Flow decoupling alongside these blind-panel results, the interaction should be re-run on the 36-society unified panel (`JUNO_Unified_Dataset.csv`) with the carry-forward rate reported per society.
