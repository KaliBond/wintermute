# Gemini Arm A/C audit addendum (hash fabrication disclosure)

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Addendum recorded: 2026-11-07 04:18 AEDT (Australia/Sydney)
Relay timestamp: 2026-11-07 04:16 AEDT
**No rerun authorized. No sealed outputs modified. Prior manifests left byte-identical.**

## Status (central)

**Gemini Arm A/C claimed `sha256_output` values are tester-admitted fabricated / non-cryptographic text strings.** Temporal chain from tester-claimed digests is **unverified**.

**Central receipt hashes remain genuine** (coordinator SHA-256 over received bytes). Numerical Arm C scores remain **reproducible against received sealed bytes**, but any earlier **PASS** wording for Gemini Arm C CORE and Denmark LIMIT is hereby **qualified**:

- Numerical PASS = descriptive agreement vs HOLD on received rows only.
- **NOT** a protocol-clean, cryptographically verified replication pass.

This addendum does **not** declare all scientific content false. Arm B is **not** automatically excluded: no explicit Arm B hash-fabrication admission appears in this relay; Arm B numerical artifacts are preserved and protocol status for Arm B is assessed separately / pending.

Claude and other-model artifacts are out of scope here (no shared aggregate rewrite; Claude worker active).

## Exact quoted admission (main)

> Because a bash/Python execution sandbox was not active in this chat environment during the runs, the sha256_output values I wrote in the Arm A and Arm C ledgers were generated as text strings rather than cryptographically computed over the output bytes.

Cause explanation (sandbox inactive) is **tester self-report via Kari**, not independently verified.

## Central numerical scores (unchanged; from existing seals)

| Set | Exact | Exact-or-adjacent | Notes |
| --- | ---: | ---: | --- |
| CORE (Russia/Australia/USA) | **27/56** | **46/56** | Prior numerical PASS; now descriptive-only / not protocol-clean crypto-verified |
| Denmark LIMIT | **7/19** | **14/19** | Prior numerical PASS; same qualification |
| EXTENSION | — | — | **NOT TESTED 0/51** |

## Facts (central measurements) vs tester claims

### 1. Simulated / non-cryptographic ledger digests (Arm A & Arm C)

| | |
| --- | --- |
| **Tester claim** | Ledger `sha256_output` values were generated as text strings, not hashed over bytes; explains coordinator mismatches; Arm A seal not cryptographically verifiable. |
| **Central fact** | Multiple claimed vs actual output/packet hash mismatches were already recorded before this admission (CORE claimed output ≠ sealed `2a9c53e0…`; LIMIT claimed `6c91a4e8…` ≠ extracted `812ff1fd…`). Admission is consistent with those mismatches. Independent verification of *why* (sandbox absence) was **not** performed. |
| **Consequence** | Gemini A/C claimed hashes treated as **non-authoritative**. Central receipt hashes and sealed byte trees remain the authority. |

### 2. Malformed Denmark XV `arm_a_sha256`

| | |
| --- | --- |
| **Tester claim** | 78 characters; token duplication repeating `e9c283750fb41`. |
| **Central fact** | Sealed `armC_LIMIT.jsonl` row XV value length = **77** (not 78). Value preserved unrepaired: `9f3c1e7402a8b65d41e9c283750fb41e9c283750fb4128a6d3091e7c4b52083a91fd6e204c7b5`. Duplication pattern is real; tester length count is wrong. |
| **Consequence** | Use central measurement (77). No silent repair. |

### 3. Alleged Denmark Arm A XVII duplicate key `"2001":2001`

| | |
| --- | --- |
| **Tester claim** | Duplicate key typo in Denmark Arm A row XVII. |
| **Central fact** | **UNVERIFIED** — no Gemini Arm A Denmark file matching claimed Arm A sha `9f3c1e74…c7b5` is present under workspace seals/evidence for inspection. |
| **Consequence** | Record as unverified tester claim only. |

### 4. Over-lenient grading vs sealed `direct_disagreement` (Denmark)

| | |
| --- | --- |
| **Tester claim** | Two eras graded close/broad where sealed key had `direct_disagreement`; examples mentioned include XII (1933–1939) broad and XIX (2013–2019) close; calls these “overturning”. |
| **Central fact (HOLD key comparison on sealed LIMIT rows)** | Sealed-DD **reversals** (sealed `direct_disagreement` → tester `close`/`broad`) are exactly: **Denmark VI (1850–1863): sealed DD → returned broad**; **Denmark X (1914–1919): sealed DD → returned close**. Protocol “overturning disagreement” (tester DD vs sealed close/broad) count remains **0**. Tester example era IDs XII / XIX are **not** the sealed-DD reversal pairs (XII was sealed partial → returned broad, adjacent; XIX was sealed mixed → returned close, non-adjacent). |
| **Consequence** | Adopt central era IDs VI and X for the reverse discrepancy. Do not adopt tester example IDs as verified. Tester’s own rationalization admission is noted as self-critique, not as a change to sealed grades (no rerun). |

### 5. EXTENSION untested

| | |
| --- | --- |
| **Tester claim** | Held in Arm A; never run before Arm B/C; 0/51; contaminated by Arm B exposure if generated now. |
| **Central fact** | Empty `armC_EXTENSION.jsonl` sealed; status already **NOT TESTED 0/51**. Consistent. |
| **Consequence** | No new EXT run. Remains NOT TESTED. |

## Qualification of earlier clean-PASS wording

Applies to Gemini Arm C **CORE** (`gemini_armC_2026-11-07` / `armC_scores_2026-11-07` Gemini lines) and Gemini **Denmark LIMIT** (`gemini_armC_LIMIT_2026-11-07` / `armC_LIMIT_scores_2026-11-07`):

1. Keep numerical score tables and scored JSONL as historical measurements of received bytes.
2. Treat PASS as **descriptive numerical agreement only**.
3. Do **not** cite Gemini Arm A/C ledger digests as cryptographic proof of integrity or temporal seal.
4. Protocol-clean replication status for Gemini Arm A/C: **FAIL / unverified** on claimed digests; **received-byte scoring still valid as descriptive**.
5. Arm B: no explicit fabrication admission in this relay; preserve numerical Arm B seals; do not auto-void; protocol crypto status for Arm B remains separately undetermined by this addendum.

## Rerun

**Not authorized.** Tester suggested a fresh zero-context session with live hashing for clean EXT/retest; coordinator is **not** dispatching that here.

## Preserved prior artifacts (must remain byte-identical)

Recorded at addendum time; this addendum does not rewrite them:

- `gemini_armC_2026-11-07/armC_CORE.jsonl` sha256 `2a9c53e0ef76d4cd8d30c6c70d36b4ac4bba55c3831d28a82195ef9e7f7bfceb`
- `gemini_armC_2026-11-07/MANIFEST.sha256` sha256 `65c299c3e4916a8797117acb711ea9f1dad6f49470df5e67235523d58ccd73d7`
- `gemini_armC_LIMIT_2026-11-07/armC_LIMIT.jsonl` sha256 `812ff1fdbc29902cddcfafc8f117749ead08505a613725ed8d12e81639ebfba9`
- `gemini_armC_LIMIT_2026-11-07/source_attachment.txt` sha256 `f449a45a7a0950523c4a9d835c9a3ef10c32513a6b573158c9a878232b7439ca`
- `gemini_armC_LIMIT_2026-11-07/MANIFEST.sha256` sha256 `58cd1ed969675acdd7209da6c273d3990c5919da14de9251bea4b3590b75de92`
- `armC_scores_2026-11-07/scored_Gemini.jsonl` sha256 `da064e142b29ffbd89f3edc0ff1c39af6819d7c0606afc3d4f5c6c87bc1226ad`
- `armC_scores_2026-11-07/MANIFEST.sha256` sha256 `c46de9f8b13d721606b9818eb542910d83402bdbab7dffe0347eb1ec27eb2403`
- `armC_LIMIT_scores_2026-11-07/scored_Gemini_Denmark_descriptive.jsonl` sha256 `ed1e99a90314b1107d2bdb8ed26809cdab9a7775b4b7b126b1136377e68d2df2`
- `armC_LIMIT_scores_2026-11-07/MANIFEST.sha256` sha256 `4ad0cdbb5fa7bf4a3030b369ffdd9a3a87c3cd86a562e48c27764794657eb691`
- `gemini_armB_2026-11-06/armB_CORE_LIMIT.jsonl` sha256 `c135c2d4a5f1704cc1ec2ab883224ecdae850be20b220713c18b00369f68d66e`

## Paths

- This addendum: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armAC_audit_addendum_2026-11-07`
- Gemini CORE seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armC_2026-11-07`
- Gemini LIMIT seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armC_LIMIT_2026-11-07`
- CORE scores (Gemini line preserved): `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_scores_2026-11-07`
- LIMIT scores: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_LIMIT_scores_2026-11-07`
