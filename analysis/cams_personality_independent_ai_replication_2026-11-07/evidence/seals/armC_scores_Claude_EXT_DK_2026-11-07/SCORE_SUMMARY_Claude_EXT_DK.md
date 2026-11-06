# Arm C EXTENSION + LIMIT score summary: Claude

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Scored at: 2026-11-07 04:16 AEDT (Australia/Sydney)

## Verdict

### EXTENSION (descriptive, 51/51)
**PASS: 41/51 (80.39%) exact-or-adjacent** (majority threshold >25.5 i.e. ≥26/51).
Exact 31/51 (60.78%). Overturning disagreements: 0. Sealed-DD reversals: 0. Majority-disagreement pattern: NO.

### LIMIT Denmark (descriptive, 19/19; IX omitted)
**PASS: 10/19 (52.63%) exact-or-adjacent** (majority threshold >9.5 i.e. ≥10/19).
Exact 4/19 (21.05%). Overturns: 0. Sealed-DD reversals: 4. Majority-DD: NO.

### Protocol validation status
**CAVEATED_UNVERIFIED_ARMA** — schema/coverage/years/sources: PASS; Arm A file bytes: UNVERIFIED_NO_FILE. Not an automatic clean protocol pass while Arm A bytes remain unverified on box.

## Inputs and coverage

- Model: `claude-sonnet-4-6`; Arm C; EXTENSION + LIMIT (CORE already sealed separately).
- EXT coverage: 51/51 (China 20, UK 16, Norway 15).
- DK coverage: 19/19; Era IX omitted (expected).
- Sealed EXT: sha256 `da0212f8c715829481787bfec2706cbbc22479cb36b967a5aa24124856f79dfc` (matches claimed).
- Sealed DK: sha256 `ff65e33c5f99b1b8528da72917d01b845db680ba495a4aeccd7ed27cb67f0352` (matches claimed).
- Report copy: sha256 `404a2c10e97cbf66b39f6821f5ae0b24125a4b134ff10f61cfc2acd8a9abb92e`.
- Arm A sha on every row: `520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7`; internally consistent; **UNVERIFIED_NO_FILE**.
- Expected packet sha256: `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26` (matches claimed).
- HOLD key sha256: `2e378db43727bf09f60bf8a41fdab0e471bfe0bf978f125748704af776d031d2`.
- Tester timestamps: EXT 2026-11-07 14:00 AEDT; DK 2026-11-07 15:30 AEDT.
- Seal time: 2026-11-07 04:16 AEDT (earlier than tester grading timestamps; not seal-later-than-grading).
- Adjacency: READY / pre-registered — close↔broad, broad↔partial, partial↔mixed only; mixed↔direct_disagreement **not** adjacent.

## Headline metrics

| Set | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | EoA % | Non-adjacent | Overturns | Sealed-DD reversals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXTENSION | 51 | 31 | 60.78% | 10 | 41 | 80.39% | 10 | 0 | 0 |
| LIMIT | 19 | 4 | 21.05% | 6 | 10 | 52.63% | 9 | 0 | 4 |

## EXTENSION by entity

| Entity | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| China | 20 | 9 | 45.0% | 4 | 13 | 65.0% | 7 |
| Norway | 15 | 11 | 73.33% | 2 | 13 | 86.67% | 2 |
| UK | 16 | 11 | 68.75% | 4 | 15 | 93.75% | 1 |

## LIMIT by entity

| Entity | n | Exact | Exact % | Adjacent only | Exact-or-adjacent | Exact-or-adjacent % | Non-adjacent mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Denmark | 19 | 4 | 21.05% | 6 | 10 | 52.63% | 9 |

## Returned-grade distribution

### EXTENSION
| Returned grade | n | Report claim |
| --- | ---: | ---: |
| close | 46 | 46 |
| broad | 5 | 5 |

### LIMIT
| Returned grade | n | Report claim |
| --- | ---: | ---: |
| close | 12 | 12 |
| broad | 6 | 6 |
| partial | 1 | 1 |

Report grade counts match row-level grades for both sets.

## Confusion matrices (descriptive)

### EXTENSION
| Sealed \ Returned | close | broad | partial | mixed | direct_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 27 | 0 | 0 | 0 | 0 |
| broad | 10 | 4 | 0 | 0 | 0 |
| partial | 1 | 0 | 0 | 0 | 0 |
| mixed | 8 | 1 | 0 | 0 | 0 |
| direct_disagreement | 0 | 0 | 0 | 0 | 0 |

### LIMIT
| Sealed \ Returned | close | broad | partial | mixed | direct_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 1 | 0 | 0 | 0 | 0 |
| broad | 4 | 2 | 0 | 0 | 0 |
| partial | 2 | 2 | 1 | 0 | 0 |
| mixed | 2 | 1 | 0 | 0 | 0 |
| direct_disagreement | 3 | 1 | 0 | 0 | 0 |

## Non-exact pair counts

EXTENSION: {"broad -> close": 10, "mixed -> close": 8, "mixed -> broad": 1, "partial -> close": 1}
LIMIT: {"broad -> close": 4, "direct_disagreement -> close": 3, "mixed -> close": 2, "partial -> broad": 2, "partial -> close": 2, "direct_disagreement -> broad": 1, "mixed -> broad": 1}

## Non-exact eras

### EXTENSION
- China I (1800-1839): sealed `broad` vs returned `close` [adjacent]
- China III (1850-1864): sealed `partial` vs returned `close` [NON-ADJ]
- China IV (1865-1894): sealed `broad` vs returned `close` [adjacent]
- China VII (1912-1916): sealed `broad` vs returned `close` [adjacent]
- China X (1938-1945): sealed `mixed` vs returned `close` [NON-ADJ]
- China XI (1946-1949): sealed `mixed` vs returned `close` [NON-ADJ]
- China XII (1950-1957): sealed `mixed` vs returned `close` [NON-ADJ]
- China XV (1966-1976): sealed `mixed` vs returned `close` [NON-ADJ]
- China XVII (1989-2001): sealed `mixed` vs returned `close` [NON-ADJ]
- China XVIII (2002-2012): sealed `broad` vs returned `close` [adjacent]
- China XIX (2013-2019): sealed `mixed` vs returned `close` [NON-ADJ]
- UK VI (1940-1945): sealed `broad` vs returned `close` [adjacent]
- UK VIII (1952-1963): sealed `broad` vs returned `close` [adjacent]
- UK IX (1964-1970): sealed `broad` vs returned `close` [adjacent]
- UK XII (1991-1997): sealed `broad` vs returned `close` [adjacent]
- UK XIII (1998-2007): sealed `mixed` vs returned `close` [NON-ADJ]
- Norway VIII (1961-1972): sealed `broad` vs returned `close` [adjacent]
- Norway X (1985-1994): sealed `broad` vs returned `close` [adjacent]
- Norway XIV (2020-2021): sealed `mixed` vs returned `close` [NON-ADJ]
- Norway XV (2022-2025): sealed `mixed` vs returned `broad` [NON-ADJ]

### LIMIT
- Denmark I (1752-1783): sealed `broad` vs returned `close` [adjacent]
- Denmark III (1807-1813): sealed `direct_disagreement` vs returned `broad` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark IV (1814-1839): sealed `partial` vs returned `broad` [adjacent]
- Denmark V (1840-1849): sealed `direct_disagreement` vs returned `close` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark VI (1850-1863): sealed `direct_disagreement` vs returned `close` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark X (1914-1919): sealed `direct_disagreement` vs returned `close` [NON-ADJ,SEALED_DD_REVERSAL]
- Denmark XI (1920-1932): sealed `broad` vs returned `close` [adjacent]
- Denmark XII (1933-1939): sealed `partial` vs returned `close` [NON-ADJ]
- Denmark XIII (1940-1945): sealed `mixed` vs returned `close` [NON-ADJ]
- Denmark XIV (1946-1952): sealed `broad` vs returned `close` [adjacent]
- Denmark XVI (1972-1989): sealed `partial` vs returned `close` [NON-ADJ]
- Denmark XVII (1990-2001): sealed `broad` vs returned `close` [adjacent]
- Denmark XVIII (2002-2012): sealed `partial` vs returned `broad` [adjacent]
- Denmark XIX (2013-2019): sealed `mixed` vs returned `broad` [NON-ADJ]
- Denmark XX (2020-2025): sealed `mixed` vs returned `close` [NON-ADJ]

## Optional combined with prior CORE (informational)

| Component | n | Exact | Exact-or-adjacent |
| --- | ---: | ---: | ---: |
| CORE (prior) | 56 | 27 | 48 |
| EXTENSION | 51 | 31 | 41 |
| LIMIT | 19 | 4 | 10 |
| Combined | 126 | 62 (49.21%) | 99 (78.57%) |

Does not replace primary set metrics. CORE used historical adjacency that counted mixed↔DD as adjacent; EXT/DK use READY.

## Provenance caveats

1. Arm A sha `520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7`: **UNVERIFIED_NO_FILE**. wintermute_arm_a_ledger.md claims this final hash for a 126-row Claude Arm A file including EXT+DK, but the Arm A JSONL bytes are not present under seals/uploads for independent hash verification.
2. Claimed EXT/DK output hashes match sealed bytes (unlike Gemini LIMIT hash mismatch).
3. Tester grading timestamps (14:00 / 15:30 AEDT) are later than seal wall-clock 2026-11-07 04:16 AEDT.
4. EXT was graded with the same Arm A sha as CORE; report does not disclose missing EXT pockets — but without Arm A bytes on box, EXT pocket provenance cannot be independently confirmed.
5. Shared `armC_scores_2026-11-07` aggregates and Gemini LIMIT seals were not modified.
6. HOLD key not modified. CORE Claude seal untouched.

## Paths

- Return seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/claude_armC_EXT_DK_2026-11-07`
- Score pack: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armC_scores_Claude_EXT_DK_2026-11-07`
- HOLD key: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/packets/answer_key_HOLD/arm_c_key.csv`
- Prior CORE seal: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/claude_armC_2026-11-07`
