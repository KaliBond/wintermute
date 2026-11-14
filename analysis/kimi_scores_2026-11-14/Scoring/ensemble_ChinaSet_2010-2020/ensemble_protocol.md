# CAMS Ensemble Protocol Note — China Set 2010–2020, five-scorer run

Protocol: CAMS Universal Instruction Set v1.4, Ensemble Mode
Run date: 2026-11-04
Panel: China (national), Evergrande, Huawei, BYD, CATL, Vanke, Longfor, SMIC; 2010–2020; 8 nodes; 704 entity-year-node rows per scorer.

## Scorers

Five independent scorer processes, run in isolation — none saw another's output, reasoning or search results; each extended the blind stance with "as if the other four scorers did not exist". Each ran its own search sessions (12–18 searches per scorer; distinct source mixes per scorer, including RAND, CRS, NBER, USCC, SWP, Caixin, Wharton/HBS/arXiv case studies, Federal Register, company filings via encyclopaedic aggregators, and others — see each scorer's note).

- Scorer identity: Kimi (Moonshot AI) — identical model for all five; envelope `Scorers` field records `Kimi-MoonshotAI|x5`.
- Scorer 2's first launch terminated by a provider content filter before producing output; the scorer was relaunched fresh with a rephrased brief (same rules, same independence). Its outputs are the relaunch's. The failed attempt produced no scores and saw no other scorer's work; it is logged here for provenance.
- Raw outputs preserved: `scorer1_raw.csv` … `scorer5_raw.csv` and `scorer1_note.md` … `scorer5_note.md` (plus scorers' own generator scripts and, for scorer 1, a Score_Status sibling it wrote on its own initiative).

## Recombination rule

- **ensemble.csv** — per Entity, Year, Node: median of the non-NA scores per metric; one decimal permitted; `NA` only when all five scorers returned NA (CATL 2010 only).
- **envelope.csv** — per Entity, Year, Node: population standard deviation, range, min and max per metric over non-NA scores, non-NA count `N`, and full provenance in every row (Protocol `CAMS-Universal-v1.4`, Scorers `Kimi-MoonshotAI|x5`, Run_date `2026-11-04`). `sd` is NA where N < 2. No arbitration, no outlier suppression, no cross-scorer reconciliation — spread is reported and left standing; inconsistency is data.
- Label normalisation: scorer 1 used the entity label `China`; all others used `China (national)`. Keys were aligned by (Entity, Year, Node) after normalising the former to the latter. Scorer 3 and scorer 5 row orders also differed from canonical; alignment was by key, not row order.

## Divergence summary (2,784 scored metric-cells)

- Zero spread (all five identical): 225 cells (8%). Concentrated in: China Craft 2013–2019 (7s with near-total agreement), Longfor Lore (5–6 across the decade), CATL post-2015 Craft (7–8), Evergrande Craft (6–7).
- Widest spread (range 4): Vanke Helm Stress 2016 (scorers 6–10 — read of the Baoneng control war's load); BYD Hands Abstraction 2020.
- Range-3 cells cluster around Vanke 2016 (Shield, Stewards, Flow, Lore, Helm) — the takeover battle's second year drew the most divergent readings, followed by Vanke 2017 (resolution year).
- Systematic scorer stances visible in the spread (not reconciled): scorer 3 used NA liberally beyond the mandated CATL 2010 (whole-node blocks where it found no year-specific facts), which lowers N on affected cells; scorer 2 scored Huawei Helm Stress at 10 for 2020 (only 10 in the panel); scorer 5 scored Evergrande Lore Capacity 8 for 2017–18 (football-brand machinery read at face value, blind to the later collapse, per the no-back-cast rule).
- Median profile 2020 (consensus points): Evergrande Stewards S=9, Huawei Flow S=9, SMIC Shield S=9, China Hands S=9 — the four 2020 rupture rows all hold at the top of the scale across scorers.

## Files

- `scorer1_raw.csv`, `scorer2_raw.csv`, `scorer3_raw.csv`, `scorer4_raw.csv`, `scorer5_raw.csv` — raw integer scores, canonical order, 704 rows each.
- `scorer1_note.md` … `scorer5_note.md` — per-scorer protocol notes (evidence basis, anchors, hostile-witness status, NA justifications).
- `ensemble.csv` — medians (one decimal), 704 rows.
- `envelope.csv` — sd/range/min/max/N + provenance, 704 rows.
- `ensemble_protocol.md` — this note.

Revision log: v1.0, 2026-11-04 — initial ensemble run; no prior versions. Raw rows hold integers only; the CSVs hold current state; this note holds history.
