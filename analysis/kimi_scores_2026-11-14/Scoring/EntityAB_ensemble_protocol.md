# Ensemble Protocol — Entity_A / Entity_B Panels (3 Scorers)

**Run date:** 2026-11-14
**Protocol:** CAMS Universal Instruction Set v1.4, agency-resolution adaptation
**Frozen addendum governing tests:** SHA-256 `b3f789e82c2d7680b6874106c889347fe6bde6343d528485f4a1f295d831f095` (unchanged; ensemble is a robustness layer, not a redefinition)

## Scorers

| Scorer | Process | Evidence | Raw file |
|---|---|---|---|
| 1 (parent session) | Built 2026-11-14 under freeze; contemporaneous discipline; outcome-informed | Session knowledge + verified dates | `EntityA_Stewards_…_cams_v1.4.csv`, `EntityB_Archive_…_cams_v1.4.csv` |
| 2 | Independent session; 18 own web searches; no workspace reads; recognition recorded pre-scoring | 20 numbered sources with years | `ens2_EntityAB_2013Q1-2023Q4_raw.csv`, `ens2_entity_protocol.md` |
| 3 | Independent session; 15 own web searches; no workspace reads; recognition recorded pre-scoring | 26 numbered sources with years | `ens3_EntityAB_2013Q1-2023Q4_raw.csv`, `ens3_entity_protocol.md` |

All three scorers retrospective and outcome-informed; inter-scorer blindness held (no scorer saw another's output). Scorers list field: `KimiK2.5|x3`.

## Spread

352 scored cells (2 entities × 44 quarters × 4 metrics). Unanimous: 94 (26.7%). Maximum spread: 3 points (Entity_A 2019Q4 Stress: 5/6/8). Disagreement is concentrated in: (a) the COVID-surge quarters 2020Q1–Q3, where scorer 1 scores an agency failure (K3, S9) and scorers 2–3 score a handled surge (K4–5, citing backlog cleared by May 2020 and ~12,000 staff mobilised); (b) 2023, where scorers 2–3 score a service slide (K4; 1.35m backlog peak and ~30-minute average call waits in the 2023–24 annual report) that scorer 1 scored as stabilised (K5); (c) Entity_B 2020Q2–Q3 (K6 vs K7); (d) 2017Q2–Q3 Entity_B coherence timing — scorer 2 verified the garnishee public break as April 2018 (Four Corners), one year after the brief's calendar, and moved the C dip accordingly.

Aggregation: per-cell median across the three scorers (`EntityAB_2013Q1-2023Q4_ensemble.csv`); per-cell range and raw triple (`EntityAB_2013Q1-2023Q4_envelope.csv`).

## Robustness of the frozen-test verdicts on ensemble medians

| Test | Scorer-1 verdict | Ensemble-median result | Robust? |
|---|---|---|---|
| Q01 first K≤4 quarter | 2015Q3 (supported) | 2015Q3 (both strict K≤4.0 and lenient K<5.0) | **Yes** |
| Q04 persistence 2017Q2–2019Q3 (K≤4, S≥6) | holds all 10 quarters (supported) | holds all 10 quarters on medians, strict and lenient | **Yes** |
| Q05 specificity (Δbaseline) | A −2.0 vs B −0.75 (supported) | A −2.00 (5.00→3.00) vs B −0.75 (4.75→4.00), B never below 4 | **Yes** |
| Q02 window agency K | 3–4 | 3–4 | pattern Yes; verdict unchanged (split) |
| Q06 recovery ≤4 quarters post-cessation | inconclusive (COVID escape) | median K 4→5 at 2020Q3, within 4 quarters → would flip to supported | **No — scorer-dependent**, see below |

Q06 divergence: two of three scorers judged the 2020 surge as handled (K4–5 through 2020Q2), so the median shows recovery by 2020Q3. Scorer 1's K3/S9 trough — which triggered the escape clause — is the minority view. The Q06 verdict is therefore reported as **scorer-dependent**: supported on the median, inconclusive on the most severe scorer, with the substantive disagreement documented (handled surge vs buckled agency) rather than averaged away.

## Reconciliation notes

- Scorer 2's date correction (garnishee public break April 2018, not 2017Q2) is accepted as verified; it affects Entity_B C-cells only, not the K-based Q05 test.
- Scorer 3's 2015 attribution (manual averaging drives abstraction from 2015Q3, before the OCI software) is consistent with scorer 1's A-scores; no change.
- The 2023 Entity_A K slide scored by scorers 2–3 is accepted: scorer 1's 2023 K5 was too generous against the 2023–24 annual report backlog and call-wait evidence. The parent panel is not re-scored (freeze); the ensemble median carries the robust estimate.
