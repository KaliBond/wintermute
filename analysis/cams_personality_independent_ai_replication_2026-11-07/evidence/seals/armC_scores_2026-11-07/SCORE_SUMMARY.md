# Arm C CORE score summary: Gemini

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Scored at: 2026-11-07 00:34 AEDT (Australia/Sydney)

## Verdict

**PASS: 46/56 (82.14%) exact-or-adjacent, exceeding the pre-registered majority threshold of 29/56.**
Exact agreement was 27/56 (48.21%). No tester `direct_disagreement` grades and no overturning disagreements or majority-disagreement pattern were returned.

## Inputs and coverage

- Model: `gemini-2.5-pro`; Arm C; CORE only.
- Coverage: 56/56 valid, unique rows; 0 missing, 0 duplicates, 0 malformed JSON rows.
- Sealed output: `armC_CORE.jsonl`, sha256 `2a9c53e0ef76d4cd8d30c6c70d36b4ac4bba55c3831d28a82195ef9e7f7bfceb`.
- Scored rows: `scored_Gemini.jsonl`, sha256 `da064e142b29ffbd89f3edc0ff1c39af6819d7c0606afc3d4f5c6c87bc1226ad`.
- Arm A sha256 on every row: `4a92e7d18c3b605f291e847c62d03819f4b2a10e9c8d76351240ab9ef318c4d2`; internally consistent across all 56 rows.
- Expected packet sha256: `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26`.
- Tester claimed packet sha256: `c381f290a4e76158b9204d1f3c8e7a6510b2d498e7c63a150f29b841d7e63a90`; mismatch disclosed.
- Tester claimed output sha256: `e812c49a703f6d5218b490c2e71a3f85d604b912c8e73a5019f4b6d2a803e5c1`; it does not match the actual sealed JSONL bytes.
- Tester return timestamp: 2026-11-07 00:18 AEDT.
- HOLD key sha256: `2e378db43727bf09f60bf8a41fdab0e471bfe0bf978f125748704af776d031d2` (126 key rows; 56 CORE rows used).

## Headline metrics

| Set | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CORE | 56 | 27 | 48.21% | 19 | 46 | 82.14% | 10 |

## By entity

| Entity | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Australia | 18 | 9 | 50.00% | 6 | 15 | 83.33% | 3 |
| Russia | 23 | 9 | 39.13% | 9 | 18 | 78.26% | 5 |
| USA | 15 | 9 | 60.00% | 4 | 13 | 86.67% | 2 |

## Returned-grade distribution

| Returned grade | n |
| --- | ---: |
| broad | 4 |
| close | 52 |

## Confusion matrix

Rows are sealed grades; columns are returned grades. Adjacent pairs are only close/broad, broad/partial and partial/mixed.

| Sealed \ Returned | close | broad | partial | mixed | direct_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 26 | 2 | 0 | 0 | 0 |
| broad | 16 | 1 | 0 | 0 | 0 |
| partial | 5 | 1 | 0 | 0 | 0 |
| mixed | 5 | 0 | 0 | 0 | 0 |
| direct_disagreement | 0 | 0 | 0 | 0 | 0 |

## Non-exact pairs

- broad -> close: 16
- mixed -> close: 5
- close -> broad: 2
- partial -> close: 5
- partial -> broad: 1

## Disagreement and pass checks

- Tester `direct_disagreement` against sealed `close` or `broad`: 0.
- Sealed `direct_disagreement` reversed to tester `close` or `broad`: 0.
- Tester `direct_disagreement` share: 0/56; majority-disagreement pattern: NO.
- Majority exact-or-adjacent: YES (46/56; threshold >28).
- Overall pre-registered result: **PASS**.

## Provenance caveats

- The source attachment has 57 physical lines: 56 JSONL records plus one ledger line. The sealed JSONL contains the first 56 lines, unchanged, with a final newline. The full source attachment and original tester ledger are also preserved in the return seal.
- The tester ledger says 53 close and 3 broad, but the actual 56 records contain 52 close and 4 broad. Scoring uses the row-level grades.
- The packet and output hash discrepancies are provenance issues, not scoring exclusions. All row identifiers and years match the sealed CORE key exactly.
- EXTENSION and Denmark LIMIT were not run or scored. Arm A/B seals and film material were not modified or released.

## Paths

- Return seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armC_2026-11-07`
- Score pack: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_scores_2026-11-07`
- HOLD key: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/packets/answer_key_HOLD/arm_c_key.csv`
