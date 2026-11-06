# Arm C CORE score summary: Claude

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Scored at: 2026-11-07 03:12 AEDT (Australia/Sydney)

## Verdict

**PASS: 48/56 (85.71%) exact-or-adjacent, exceeding the pre-registered majority threshold of 29/56.**
Exact agreement was 27/56 (48.21%). Tester `direct_disagreement` grades: 0. Overturning disagreements: 0. Majority-disagreement pattern: NO.

## Inputs and coverage

- Model: `claude-sonnet-4-6`; Arm C; CORE only.
- Coverage: 56/56 valid, unique rows; 0 missing, 0 duplicates, 0 malformed JSON rows.
- Sealed output: `armC_CORE.jsonl`, sha256 `75a44314ecc219f8b85b6d985d86f1509c5a05bbd817a4c259787ebc225bf4fe`.
- Scored rows: `scored_Claude.jsonl`, sha256 `140346fb0d1f3538a44bb46ea5a709c55f0983eb35cea34e376360639441ea79`.
- Arm A sha256 on every row: `520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7`; internally consistent across all 56 rows.
- Expected packet sha256: `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26`.
- Tester claimed packet sha256: `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26`; match confirmed.
- Tester claimed output sha256: `75a44314ecc219f8b85b6d985d86f1509c5a05bbd817a4c259787ebc225bf4fe`; matches sealed JSONL bytes.
- Tester return timestamp: 2026-11-07 11:30 AEDT.
- HOLD key sha256: `2e378db43727bf09f60bf8a41fdab0e471bfe0bf978f125748704af776d031d2` (126 key rows; 56 CORE rows used).

## Headline metrics

| Set | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CORE | 56 | 27 | 48.21% | 21 | 48 | 85.71% | 8 |

## By entity

| Entity | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Australia | 18 | 8 | 44.44% | 8 | 16 | 88.89% | 2 |
| Russia | 23 | 9 | 39.13% | 9 | 18 | 78.26% | 5 |
| USA | 15 | 10 | 66.67% | 4 | 14 | 93.33% | 1 |

## Returned-grade distribution

| Returned grade | n |
| --- | ---: |
| close | 48 |
| broad | 8 |

## Confusion matrix

Rows are sealed grades; columns are returned grades. Adjacent pairs are one step on close — broad — partial — mixed — direct_disagreement (close/broad, broad/partial, partial/mixed, mixed/direct_disagreement). close/partial is not adjacent.

| Sealed \ Returned | close | broad | partial | mixed | direct_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 26 | 2 | 0 | 0 | 0 |
| broad | 16 | 1 | 0 | 0 | 0 |
| partial | 3 | 3 | 0 | 0 | 0 |
| mixed | 3 | 2 | 0 | 0 | 0 |
| direct_disagreement | 0 | 0 | 0 | 0 | 0 |

## Non-exact pairs

- broad -> close: 16
- mixed -> close: 3
- partial -> broad: 3
- partial -> close: 3
- close -> broad: 2
- mixed -> broad: 2

## Disagreement and pass checks

- Tester `direct_disagreement` against sealed `close` or `broad`: 0.
- Sealed `direct_disagreement` reversed to tester `close` or `broad`: 0.
- Tester `direct_disagreement` share: 0/56; majority-disagreement pattern: NO.
- Majority exact-or-adjacent: YES (48/56; threshold >28).
- Overall pre-registered result: **PASS**.

## Provenance caveats

- The source upload is a pure 56-row JSONL with no embedded ledger line. Sealed bytes are identical to the verified upload (`75a44314ecc219f8b85b6d985d86f1509c5a05bbd817a4c259787ebc225bf4fe`).
- Tester ledger came from the Kari chat paste (tester timestamp 2026-11-07 11:30 AEDT); preserved as TESTER_LEDGER.txt.
- Claimed packet and output hashes both match. No hash mismatch caveats for Claude Arm C.
- EXTENSION and Denmark LIMIT were not run or scored. Arm A/B seals, Gemini Arm C seal bytes, HOLD key contents, and film material were not modified or released.

## Paths

- Return seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/claude_armC_2026-11-07`
- Score pack: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_scores_2026-11-07`
- HOLD key: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/packets/answer_key_HOLD/arm_c_key.csv`
