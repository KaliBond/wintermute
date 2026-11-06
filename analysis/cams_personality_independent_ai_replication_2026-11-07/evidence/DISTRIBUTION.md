# Distribution notes - CAMS personality AI tester packets v1

Prepared: 2026-11-06 20:30 AEDT
Prepared by: Reportmister for Kari McKern
Location: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/`

Kari distributes these herself. Reportmister does not email or post them.

## Zips

| Packet | Path | Bytes | sha256 |
|---|---|---:|---|
| Standard AI tester | `CAMS_personality_AI_tester_packet_v1.zip` | 22460 | `da94f1c3f405b4c8f4229178509bf400f5661c694cfb38b909bdc1a70b5d5d1e` |
| Manus special | `CAMS_personality_Manus_special_v1.zip` | 25973 | `1ce384236bd3e8d8ca9819f6bc610a7fc408a9e2b3699548b3acd871569d6809` |

## Who gets what
- **Claude, Gemini, Codex/Kimi, and other independent AI testers:** send the **standard** zip only.
- **Manus:** send the **Manus special** zip (standard materials + film brief for a ~3 minute movie). Tell Manus to finish Arms A-C and hash before opening the film narrative examples inside `MANUS_BRIEF.md`.

## What is inside (standard)
README, node definitions, algebra/rules, CORE/EXTENSION/LIMIT era-means CSVs (no narrative titles), output schema, experimental plan summary, MANIFEST.sha256.

## What is inside (Manus special)
Everything in standard, plus `MANUS_BRIEF.md` and `film_assets_notes.md`.

## What is NOT in either zip
- Answer keys, sealed original pockets, Q grades, Pages, reaction notes.
- Those stubs live only at `packets/answer_key_HOLD/` for Reportmister scoring.

## Source means verified before build
Personality, nations2, usa_marker and china_corps MANIFEST.sha256 checks passed at packet build time.

## Gaps / notes
- Company Arm D means are not in the standard zip (mapping provisional). Film brief may narrate prior sealed company portraits after tester arms.
- Experimental plan Page URL is linked from README; canvas fetch was unavailable at pack time, so the short markdown extract is the in-zip plan summary.
- CORE eras: Russia 23, Australia 18, USA 15. EXTENSION: China 20, UK 16, Norway 15. LIMIT: Denmark 19.

## After returns
Score against `packets/answer_key_HOLD/` paths. Append tester ledger lines (model, arm, entity set, sha256 in/out, AEDT time) to the replication run record.


## Arm B blind packet (v1) — 2026-11-06 20:49 Australia/Sydney

- **File:** `CAMS_ArmB_blind_v1.zip`
- **sha256:** `dceee12e89c1b07b9f63e4ce26836f51fb231089b621afc4f9d8f819ab73aae9`
- **Contents:** README_ArmB.md, pockets_blind.jsonl (126 unlabeled pockets), era_shortlist.csv, output_schema_ArmB.md, MANIFEST.sha256
- **Sets:** CORE Russia/Australia/USA (56); LIMIT Denmark (19); EXTENSION China/UK/Norway (51). No companies.
- **Answer key:** `packets/answer_key_HOLD/arm_b_key.csv` — **not in zip**
- **Recipient use:** Manus / other AI after Arm A hashed. No web. Closed shortlist only.
- **Downloads copy:** `CAMS_ArmB_blind.zip` (and optional `CAMS_ArmB_blind_v1.zip`)

## Arm C independent mood continuity packet (v1) - 2026-11-06 23:59 Australia/Sydney (AEDT)

- **File:** `CAMS_ArmC_mood_v1.zip`
- **Bytes:** 4418
- **sha256:** `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26`
- **Contents:** README_ArmC.md, output_schema_ArmC.md, SHORT_PASTE.md, MANIFEST.sha256. Four files only.
- **Coverage:** CORE first (56 eras), then optional EXTENSION (51), LIMIT Denmark last (19). Denmark IX is explicitly unobserved/ungraded and is omitted.
- **Answer key:** `packets/answer_key_HOLD/arm_c_key.csv` (126 grades), sha256 `2e378db43727bf09f60bf8a41fdab0e471bfe0bf978f125748704af776d031d2`. **Not in zip.**
- **Recipient use:** Claude, Gemini, Kimi, Manus and Grok chats whose own Arm A outputs are hashed and submitted. Arm C permits web research; Arm A and Arm B remain no-web. Testers must not edit Arm A pockets.
- **Not released:** no HOLD key, original analyst pockets, sealed Q3 grades, film or Arm D materials. Kari distributes; no email or post was made.
