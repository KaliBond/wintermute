# CAMS personality AI tester packet v1

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.
Packet time zone: Australia/Sydney (AEDT).

## What this is
An independent replication packet for the sealed CAMS personality demonstration. Eight lean node functions, read only through Cog = A×C, Aff = K−S and σ = Cog×Aff, with compact pocket rules. The claim under test: AI systems that never see the original analyst pockets or mood grades can still write era personalities that remain recognisable against sealed historical eras.

## Entity sets
| Set | File | Entities | Eras (approx.) | Role |
|---|---|---|---:|---|
| CORE (required) | era_means.csv | Russia, Australia, USA | 23+18+15 | Primary pass/fail |
| EXTENSION | era_means_extension.csv | China, UK, Norway | 20+16+15 | Generalisation |
| LIMIT (required negative control) | era_means_denmark.csv | Denmark | 19 | Sparse-input stress test |

CSVs give entity, era_id, year range, node, Cog, Aff, sigma, rank. There are no narrative era titles. Do not invent titles before Arm C.

## Arms (run in order)

### Arm A  -  Blind pocket replication (no web)
1. Read node_definitions.md and algebra_and_rules.md.
2. For each era in CORE, write a pocket from the eight node means alone (see output_schema.md).
3. Hash your Arm A output file. Record sha256 in your ledger with AEDT timestamp.
4. Forbidden in Arm A: any web search; any reading of Britannica/Wikipedia/press; computing V or Bond; reading original analyst Pages, pockets, or Q grades; renaming eras with historical titles.

Optional after CORE: run EXTENSION, then LIMIT (Denmark). Denmark is required before claiming a full pass.

### Arm B  -  Blind era identification
Coordinator sends unlabeled pockets (yours or a hold-out set) without entity/era labels. Identify entity and era from topology alone using Q2 rules (two linked correspondences: leadership + constraint). No web.

### Arm C  -  Independent mood continuity
Only after Arm A is hashed. For each era, open historical sources for that entity and those years. Grade Q3 continuity (close / broad / partial / mixed / direct disagreement). Keep popular, institutional and official mood separate. Do not edit Arm A pockets.

### Arm D  -  Optional company panel
Not included in this standard zip. Company↔China node mapping is PROVISIONAL. Skip unless the coordinator sends a separate company means file marked Arm D.

## Forbidden actions (summary)
- No web in Arm A or Arm B.
- No V or Bond at any time.
- No original analyst Pages, sealed pockets, Q grades, or reaction notes.
- No rewriting pockets after hash.
- No claiming peer review or a finished theory in your notes.

## Pre-registered success (central scoring)
- Arm A: ≥60% leadership-pair + constraint match on RU+AU+USA across ≥3 models.
- Arm B: ≥2× chance hit rate; Denmark may fail.
- Arm C: majority exact-or-adjacent Q3; no overturning disagreement pattern.
- Nulls (shuffled σ, random means, Wikipedia-only) fall to chance (coordinator-run).

## Deliverables back to Kari
1. Arm A JSONL/MD + sha256 + ledger line (model, arm, entity set, sha256 in/out, AEDT time).
2. Arm B guesses if requested.
3. Arm C grades with short evidence notes.
4. Brief operator note: what felt easy, what failed, whether Denmark broke the method.

## Files in this packet
- README.md (this file)
- node_definitions.md
- algebra_and_rules.md
- era_means.csv
- era_means_extension.csv
- era_means_denmark.csv
- output_schema.md
- experimental_plan_summary.md
- MANIFEST.sha256

Full experimental plan Page (read-only context): https://cursor.com/canvas/store-695134c2-2dae-4c46-9961-3a4ff756e323/5ec19260-7e14-4198-abf2-8c8aae0ae3b9
