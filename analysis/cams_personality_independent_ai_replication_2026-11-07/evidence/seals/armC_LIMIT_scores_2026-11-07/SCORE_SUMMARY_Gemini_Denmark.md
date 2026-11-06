# Arm C LIMIT (Denmark) score summary: Gemini

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Scored at: 2026-11-07 04:13 AEDT (Australia/Sydney)

## Verdict

### Descriptive (all 19 returned Denmark rows, including malformed XV arm_a)
**PASS: 14/19 (73.68%) exact-or-adjacent** (majority threshold >9.5 i.e. ≥10/19).
Exact 7/19 (36.84%). Overturning disagreements (tester `direct_disagreement` vs sealed `close`/`broad`): 0. Sealed-DD reversals to close/broad: 2. Majority-disagreement pattern: NO.

### Validated subset (18 rows with valid hex64 `arm_a_sha256` matching claimed Arm A sha; XV excluded)
**PASS: 13/18 (72.22%) exact-or-adjacent** (majority threshold >9 i.e. ≥10/18).
Exact 6/18 (33.33%). Overturns: 0. Sealed-DD reversals: 2. Majority-DD: NO.

### EXTENSION
**NOT TESTED — 0/51.** Empty `armC_EXTENSION.jsonl` disclosed (no sealed Arm A EXTENSION pockets). Not scored as pass or fail.

## Inputs and coverage

- Model: `gemini-2.5-pro`; Arm C; LIMIT Denmark only for scoring.
- Denmark coverage: 19/19 observed eras returned; IX (1901–1913) omitted as unobserved (expected).
- Attachment content sha256: `f449a45a7a0950523c4a9d835c9a3ef10c32513a6b573158c9a878232b7439ca` (33677 bytes).
- Attachment filename hash `5ed236…124b` ≠ content sha (disclosed; identical bytes at staged path and attachment path).
- Extracted LIMIT JSONL sha256: `812ff1fdbc29902cddcfafc8f117749ead08505a613725ed8d12e81639ebfba9`.
- Tester claimed LIMIT output sha256: `6c91a4e820f7b35d18c492e0a73f5b19d842c601e9a7b34502f8c1d6e390a4b2` — **mismatch**.
- Claimed Arm A Denmark sha256 on ledger / 18 rows: `9f3c1e7402a8b65d41e9c283750fb4128a6d3091e7c4b52083a91fd6e204c7b5` — status: **UNVERIFIED_NO_FILE**.
- Denmark XV `arm_a_sha256` malformed len=77 value `9f3c1e7402a8b65d41e9c283750fb41e9c283750fb4128a6d3091e7c4b52083a91fd6e204c7b5` (left unrepaired).
- Packet sha256: `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26` (matches claimed sha256_input).
- HOLD key sha256: `2e378db43727bf09f60bf8a41fdab0e471bfe0bf978f125748704af776d031d2` (126 rows; 19 Denmark LIMIT used).
- Tester timestamp: 2026-11-07 03:50 AEDT.
- Adjacency rule (READY / pre-registered): only close↔broad, broad↔partial, partial↔mixed. mixed↔direct_disagreement is **not** adjacent.
- Overturn definition: tester `direct_disagreement` against sealed `close` or `broad`. Denmark is the sparse-input LIMIT negative control — failures retained, not pooled away.

## Headline metrics

| Subset | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | EoA % | Non-adjacent | Overturns | Sealed-DD reversals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Descriptive (all) | 19 | 7 | 36.84% | 7 | 14 | 73.68% | 5 | 0 | 2 |
| Validated (excl. XV) | 18 | 6 | 33.33% | 7 | 13 | 72.22% | 5 | 0 | 2 |

## Returned-grade distribution (actual rows)

| Returned grade | n | Report table claim |
| --- | ---: | --- |
| close | 5 | 5 |
| broad | 6 | 7 (actual 6) |
| partial | 6 | 5 (actual 6; ledger also said 5 partial) |
| mixed | 1 | 1 |
| direct_disagreement | 1 | 1 |

## Confusion matrix (descriptive, all 19)

Rows sealed; columns returned. Adjacent = close/broad, broad/partial, partial/mixed only.

| Sealed \ Returned | close | broad | partial | mixed | direct_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 1 | 0 | 0 | 0 | 0 |
| broad | 1 | 3 | 2 | 0 | 0 |
| partial | 0 | 2 | 2 | 1 | 0 |
| mixed | 2 | 0 | 1 | 0 | 0 |
| direct_disagreement | 1 | 1 | 1 | 0 | 1 |

## Non-exact pairs (descriptive)

- broad -> partial: 2
- mixed -> close: 2
- partial -> broad: 2
- broad -> close: 1
- direct_disagreement -> broad: 1
- direct_disagreement -> close: 1
- direct_disagreement -> partial: 1
- mixed -> partial: 1
- partial -> mixed: 1

## Non-exact eras (descriptive)

- Denmark V (1840-1849): sealed `direct_disagreement` vs returned `partial` [NON-ADJ]
- Denmark VI (1850-1863): sealed `direct_disagreement` vs returned `broad` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark VII (1864-1873): sealed `partial` vs returned `mixed` [adjacent]
- Denmark VIII (1874-1900): sealed `broad` vs returned `partial` [adjacent]
- Denmark X (1914-1919): sealed `direct_disagreement` vs returned `close` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark XI (1920-1932): sealed `broad` vs returned `partial` [adjacent]
- Denmark XII (1933-1939): sealed `partial` vs returned `broad` [adjacent]
- Denmark XIII (1940-1945): sealed `mixed` vs returned `close` [NON-ADJ]
- Denmark XVII (1990-2001): sealed `broad` vs returned `close` [adjacent]
- Denmark XVIII (2002-2012): sealed `partial` vs returned `broad` [adjacent]
- Denmark XIX (2013-2019): sealed `mixed` vs returned `close` [NON-ADJ]
- Denmark XX (2020-2025): sealed `mixed` vs returned `partial` [adjacent]

## Disagreement and pass checks

- Overturning (tester DD vs sealed close/broad): descriptive 0; validated 0.
- Sealed DD reversed to tester close/broad: descriptive 2; validated 2.
- Tester DD share: descriptive 1/19; validated 1/18.
- Majority EoA: descriptive YES; validated YES.
- Sparse negative control: Denmark LIMIT retained as its own set; failures (non-adjacent mismatches and sealed-DD reversals) reported, not erased.

## Provenance audit (report vs sealed artifacts)

1. **LIMIT output hash:** claimed `6c91a4e820f7b35d18c492e0a73f5b19d842c601e9a7b34502f8c1d6e390a4b2` ≠ actual extracted `812ff1fdbc29902cddcfafc8f117749ead08505a613725ed8d12e81639ebfba9`.
2. **Denmark grade counts:** report/ledger understated partial (claimed 5, actual 6) and overstated broad (claimed 7, actual 6). Scoring uses row-level grades.
3. **CORE grade distribution in report:** 52 close / 4 broad — matches sealed CORE JSONL (`2a9c53e0ef76d4cd8d30c6c70d36b4ac4bba55c3831d28a82195ef9e7f7bfceb`).
4. **CORE packet/output hashes (prior seal, not laundered):** expected packet `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26` vs CORE tester claimed packet `c381f290…`; sealed CORE output `2a9c53e0ef76d4cd8d30c6c70d36b4ac4bba55c3831d28a82195ef9e7f7bfceb` vs claimed `e812c49a703f6d5218b490c2e71a3f85d604b912c8e73a5019f4b6d2a803e5c1` — mismatches remain as previously disclosed in `gemini_armC_2026-11-07`.
5. **Arm A Denmark sha `9f3c1e7402a8b65d41e9c283750fb4128a6d3091e7c4b52083a91fd6e204c7b5`:** UNVERIFIED_NO_FILE. Not the same as CORE Arm A sha `4a92e7d1…` on the CORE seal.
6. **XV malformed arm_a:** unrepaired; excluded only from validated subset scoring, still in descriptive.
7. **EXTENSION:** NOT TESTED 0/51; empty-file sha matches claimed `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
8. **Attachment filename vs content hash mismatch** disclosed above.

## Report substantive conclusions (tester's own, not central scores)

- CORE: tester claims strong historical continuity (52 close / 4 broad self-grades); Arm B 100% topological ID claimed.
- Denmark LIMIT functioned as sparse-input negative control: carry-forward identical vectors / ties produced visible degradation — Era III direct_disagreement (1807–1813 bombardment/bankruptcy), Era VII mixed (1864 defeat), several partials on crisis windows.
- Mood-level separation (popular/institutional/official) cited as preventing false positives on coercive or polarised eras.
- EXTENSION omitted because held in Arm A; not generated post-seal.
- Tester explicitly states HOLD agreement is for Reportmister only; no peer-review / finished-theory claim.

## Paths

- Return seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armC_LIMIT_2026-11-07`
- Score pack: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_LIMIT_scores_2026-11-07`
- Prior CORE seal (unchanged): `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armC_2026-11-07`
- HOLD key (unchanged): `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/packets/answer_key_HOLD/arm_c_key.csv`
