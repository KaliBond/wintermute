# File inventory — CAMS public replication source inputs (2026-11-07)

## Classification

| Tag | Meaning |
| --- | --- |
| **GENUINE** | Byte copy of a file issued in the closed 2026 tester packets |
| **NEW** | Written for this public bundle (guidance only; not part of original zips) |
| **WITHHELD** | Exists in the private evidence archive; deliberately omitted |

## Issued archive checksums (authoritative)

| Archive | Bytes | sha256 |
| --- | ---: | --- |
| `CAMS_personality_AI_tester_packet_v1.zip` | 22460 | `da94f1c3f405b4c8f4229178509bf400f5661c694cfb38b909bdc1a70b5d5d1e` |
| `CAMS_ArmB_blind_v1.zip` | 14837 | `dceee12e89c1b07b9f63e4ce26836f51fb231089b621afc4f9d8f819ab73aae9` |
| `CAMS_ArmC_mood_v1.zip` | 4418 | `23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26` |
| `CAMS_Gemini_ArmA_CSVs_v1.zip` | 8192 | `a9d36f5f5273c50359e3f9a45b04490dd9ffda48953836f10734972177781849` |

## Arm A — `arm_a_standard/` (GENUINE = standard packet)

| File | Tag | sha256 |
| --- | --- | --- |
| `MANIFEST.sha256` | GENUINE | `0ff67b31ed6cb5b04ef3e28da9fa0f01abd444b27f3043f34f2d21ad9a8f78b7` |
| `README.md` | GENUINE | `f1c1c635a973e4853a5aaabc2b945695ef66758b6465a7bf81b93d91d7a5a350` |
| `algebra_and_rules.md` | GENUINE | `7c0635e7ad95fb7239c2f2db86a34dcfde2318d9ef4b7286ea43786088c115bd` |
| `era_means.csv` | GENUINE | `887be60d98adc401a6465c2b9da5dd1fc1693a247d79249072f017cbfd8f170c` |
| `era_means_denmark.csv` | GENUINE | `70c9f0be10501cbc545b33d3b00254b4fa8d86a3bfea263cfa7e3147da99bf9a` |
| `era_means_extension.csv` | GENUINE | `c60562213296fc3e8accbcf8534b6cec731ce7bb706f6088be2c1a78fb7407d6` |
| `experimental_plan_summary.md` | GENUINE | `93884b80cafef7084f8b8c3d61049c9b89b51229eb7496ef06425124f8957c72` |
| `node_definitions.md` | GENUINE | `3f8b6aeeef9e26adee8ccb4229c3c9efad6eeabdbcf2764b7d92764bdc3506f0` |
| `output_schema.md` | GENUINE | `052e1c71207e2aa54c6146ef09f6d9c131985cd4c9f6bd7dabf664115ab9826c` |

## Arm B — `arm_b_blind_historical/` (GENUINE issued + NEW spoiler note)

| File | Tag | sha256 |
| --- | --- | --- |
| `COUNTS.txt` | GENUINE | `17c458ea5f2c08d750d7bb975b9b4ccee14735e887110f775419e22209b1fb1f` |
| `MANIFEST.sha256` | GENUINE | `2c89cea0052d2091d3c01d3e0a44d757f3a056e38a25f795e0e70469e085328a` |
| `README_ArmB.md` | GENUINE | `a11ffee93e112573aebd90bcf8ec373a92bdfaa9c1e5eff2f9c94ecd1825f314` |
| `SPOILER_READ_FIRST.md` | NEW | `b865d875e5697ee844298da288103a5b420bec2664e710d766430230886487c1` |
| `era_shortlist.csv` | GENUINE | `7b5828929c799a760dad19b15235cc893d1c18de86f960bc93153821f02da715` |
| `output_schema_ArmB.md` | GENUINE | `8d0abb022364e5ad1b987fe5245f7626599939fa44a6530b77ed10c72d5a9f59` |
| `pockets_blind.jsonl` | GENUINE | `6cd8b8d7ca0d0671a20edcf4c6a89cbb0b5abcedf9e046d011415abc7bcf76a8` |

## Arm C — `arm_c_mood/` (GENUINE)

| File | Tag | sha256 |
| --- | --- | --- |
| `MANIFEST.sha256` | GENUINE | `16909a62fccd488b8ab0297c55ae4e6882e0bb1a974abc76844f25505965d307` |
| `README_ArmC.md` | GENUINE | `917032a536f60edc9a6cc75cb87c3cace3ed473297303bf516ec6fb39be039cc` |
| `SHORT_PASTE.md` | GENUINE | `95e654a3f7e25539b07cb44f2f5963961a6ab300a2b51994d3b3434d820ab8a4` |
| `output_schema_ArmC.md` | GENUINE | `71f091fb72b6056bfbd8d1233b2863133f25a6e2778fb31521db6586c585b5e9` |

## Top-level / operator notes

| File | Tag | Notes |
| --- | --- | --- |
| `README.md` | NEW | |
| `FILE_INVENTORY.md` | NEW | |
| `HOLD_KEYS_WITHHELD.md` | NEW | |
| `CONTAMINATION_AND_SPOILERS.md` | NEW | |
| `operator_notes/experimental_plan_extract.md` | GENUINE | |
| `operator_notes/hold_directory_listing_names_only.txt` | NEW | |

## WITHHELD (not in this zip)

| Item | Reason |
| --- | --- |
| `arm_b_key.csv` | Closed-round Arm B answer key |
| `arm_c_key.csv` | Closed-round sealed Q3 grades |
| Private HOLD README with internal archive paths / grade tallies | Avoid leaking operator scoring detail |
| Sealed analyst pockets / personality Pages | Not tester inputs |
| Closed-round scored JSONL under seals/ | Outputs, not inputs |
| Manus special film brief / Arm D company means | Out of A–C scope; mapping provisional |

## Known gaps / notes

1. PR46 public kit on GitHub already ships Arm C instructions; this zip adds the missing Arm A CSVs/defs/rules and the historical Arm B packet shape.
2. `arm_b_blind_historical/COUNTS.txt` records a zip sha256 (`58ff7c41…`) that does **not** match the canonical issued `CAMS_ArmB_blind_v1.zip` (`dceee12e…`). Prefer the archive hash in `issued_archives/` / this inventory; treat COUNTS as a stale note inside an otherwise genuine folder.
3. Standard packet README links an experimental-plan canvas URL; the short extract in `operator_notes/experimental_plan_extract.md` is the in-archive plan text.
4. Gemini historically also received `CAMS_Gemini_ArmA_CSVs_v1.zip` (CORE+DK CSVs only). Full standard packet supersedes it for new participants.
5. Fresh Arm B requires a **new** blinded pocket set prepared by the operator; this zip does not invent masking beyond the original protocol.
6. No new experiments were run to build this bundle.
