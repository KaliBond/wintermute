Protocol: CAMS Universal Instruction Set v1.4 — Ensemble Mode (3-scorer variant, owner-specified)
Run date: 2026-11-13
Entity: Australian Government
Year(s): 2018Q1–2023Q4, quarterly, eight functions, 192 rows per scorer

Scorer identities: three scorer processes, all Kimi (Moonshot AI) instances — scorer 1: parent session panel (Australia_Govt_2018Q1-2023Q4_cams_v1.4.csv, protocol note of same name); scorer 2: independent session, 15 own web searches (ens2_Australia_Govt_2018Q1-2023Q4_raw.csv, ens2_protocol.md); scorer 3: independent session, 14 own web searches (ens3_Australia_Govt_2018Q1-2023Q4_raw.csv, ens3_protocol.md). Blind extension applied: no scorer saw another's output or reasoning; scorers 2 and 3 were denied read access to all workspace files except the instruction set and reference text. Scorers list field: KimiK2.5|x3.

Recombination rule: no arbitration, no outlier suppression, no cross-scorer reconciliation. ensemble.csv = per-cell median of the non-NA scores across the three raw CSVs (one decimal). envelope.csv = per-cell sample standard deviation (ddof=1, two decimals; NA if N<2), range, min, max, and N, with Protocol, Scorers and Run_date provenance in every row. All three raw CSVs and all protocol notes are preserved alongside these two files.

Spread summary (768 scored cells):
- Range 0 (unanimous): 117 cells; range 1: 430; range 2: 181; range 3: 38; range 4: 2.
- Mean sd per metric: Coherence 0.54, Capacity 0.54, Stress 0.75, Abstraction 0.79.
- Mean sd per node: Hands 0.53, Flow 0.55, Archive 0.60, Shield 0.67, Craft 0.69, Stewards 0.69, Helm 0.73, Lore 0.78.
- Disagreement is concentrated, not uniform: 40 of the 42 largest cells (range ≥3) fall in 2019Q4–2021Q2, the bushfire-to-COVID transition. The contested quantity is Stress calibration (was the load 4, 6 or 7 in 2019Q4? 5, 7 or 8 through 2020?) and Archive/Craft Abstraction (did the inquiry culture of 2019–21 constitute learning — A 3, 4 or 6?). Scorer 1 read 2019Q4 stress as already elevated across nodes; scorers 2 and 3 read some of those quarters as still routine. This spread is reported and left standing; inconsistency is data.
- Zero NA cells across all three scorers (every function-quarter carried both a performance and a strain fact).

Pooling cautions: all three scorers share a model identity — the ensemble decorrelates evidence draws (separate search sessions) but not scorer priors; treat the envelope as a lower bound on true scoring variance. Quarterly adaptation of the annual rule applies to all three raw runs.
