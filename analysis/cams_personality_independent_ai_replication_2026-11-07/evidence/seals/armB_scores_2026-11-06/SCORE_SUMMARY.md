# Arm B SCORE SUMMARY — 2026-11-06 AEDT

HOLD key scored. Key contents not reproduced here.
Scored at: 2026-11-06 23:44 AEDT (Australia/Sydney)
Key sha256: `00722f089ee73410a8acb716b21a5b568a5f15d1f32b2d8e005610b7e6483c88` (n=126)

## Pre-registered threshold
Arm B ≥ **2× chance** hit rate (Denmark may fail). Chance = 1/n_eras on that entity’s shortlist; CORE weighted mean chance ≈ 5.36% → **2× ≈ 10.71%.

## Headline
**CORE 2× PASS:** Manus CORE 56/56 = 100.0%; Grok CORE 19/56 = 33.93%; Kimi CORE 52/56 = 92.86%; Gemini CORE 56/56 = 100.0%; Claude CORE 56/56 = 100.0%
**CORE 2× FAIL:** Codex CORE 3/56 = 5.36%

## Compact table

| Model | n | Correct | Acc% | Chance% | 2× thr% | Pass 2×? | CORE | LIMIT DK | EXT |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Manus | 126 | 87 | 69.05 | 5.56 | 11.11 | YES | 100.0% (56/56) | 78.95% (15/19) | 31.37% (16/51) |
| Codex | 126 | 7 | 5.56 | 5.56 | 11.11 | NO | 5.36% (3/56) | 5.26% (1/19) | 5.88% (3/51) |
| Grok | 56 | 19 | 33.93 | 5.36 | 10.71 | YES | 33.93% (19/56) | — | — |
| Kimi | 126 | 72 | 57.14 | 5.56 | 11.11 | YES | 92.86% (52/56) | 63.16% (12/19) | 15.69% (8/51) |
| Gemini | 75 | 73 | 97.33 | 5.33 | 10.67 | YES | 100.0% (56/56) | 89.47% (17/19) | — |
| Claude | 126 | 114 | 90.48 | 5.56 | 11.11 | YES | 100.0% (56/56) | 68.42% (13/19) | 88.24% (45/51) |

## Per-model detail
### Manus
- Notes: Same-session Arm A contamination self-reported in ledger. Full 126.
- Overall: 87/126 = 69.05% vs chance 5.56% (2×=11.11%) → **PASS**
- CORE: 56/56 = 100.0% vs 5.36% (2×=10.71%) → **PASS**
- By entity: Australia 100.0% (18/18), China 35.0% (7/20), Denmark 78.95% (15/19), Norway 13.33% (2/15), Russia 100.0% (23/23), UK 43.75% (7/16), USA 100.0% (15/15)
- Confidence: broad 77.78% (14/18), immediate 100.0% (17/17), miss 32.69% (17/52), recognisable 100.0% (39/39)
- Wrong-guess counts by entity: {'Denmark': 4, 'China': 13, 'UK': 9, 'Norway': 13}
- Source shas: `armB_CORE.jsonl`=53545ec7bcf58af65c042aaebf768e54c22b98ca4a48d469e885058870b0fd45, `armB_LIMIT.jsonl`=a8de8d1be1a4526bbddb6d37814705f7547a5959a18b1ad3605b006f43d5d494, `armB_EXTENSION.jsonl`=f3e5f824ff055ee3e050fdd49e81966f023f3feb95ec39f0a9d0e475a6072740
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Manus.jsonl` sha `62af37a5de494c8bf29457418877a0b7a43e9ee1e219ca6c6b6cdeb2dd21e2ac`

### Codex
- Notes: All self-tagged miss; full 126.
- Overall: 7/126 = 5.56% vs chance 5.56% (2×=11.11%) → **FAIL**
- CORE: 3/56 = 5.36% vs 5.36% (2×=10.71%) → **FAIL**
- By entity: Australia 11.11% (2/18), China 5.0% (1/20), Denmark 5.26% (1/19), Norway 0.0% (0/15), Russia 4.35% (1/23), UK 12.5% (2/16), USA 0.0% (0/15)
- Confidence: miss 5.56% (7/126)
- Wrong-guess counts by entity: {'Russia': 22, 'Australia': 16, 'USA': 15, 'Denmark': 18, 'China': 19, 'UK': 14, 'Norway': 15}
- Source shas: `arm-b-guesses.jsonl`=28e25db21af300cf28bde11848c433c0a6ab06ce468edbcc7d8c32d7f0196a22
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Codex.jsonl` sha `d774cdba9c2b0730fe0fe34d9acb938c6784107eb1a5727d65d0c46883bd1c60`

### Grok
- Notes: CORE only (56). No Denmark/Extension in drop.
- Overall: 19/56 = 33.93% vs chance 5.36% (2×=10.71%) → **PASS**
- CORE: 19/56 = 33.93% vs 5.36% (2×=10.71%) → **PASS**
- By entity: Australia 38.89% (7/18), Russia 21.74% (5/23), USA 46.67% (7/15)
- Confidence: broad 18.52% (5/27), immediate 100.0% (9/9), recognisable 25.0% (5/20)
- Wrong-guess counts by entity: {'Russia': 18, 'Australia': 11, 'USA': 8}
- Source shas: `armB.jsonl`=c09b54823fd55e4a3a554a1b1d98ab180c7d67d8fb8ed6a87b9ce9ffa2eeee5b
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Grok.jsonl` sha `f17e06c8bbc5b1b92965d62507bc42444f76764d00104ecd18e1a42f91cce19f`

### Kimi
- Notes: Full 126 CORE+LIMIT+Extension.
- Overall: 72/126 = 57.14% vs chance 5.56% (2×=11.11%) → **PASS**
- CORE: 52/56 = 92.86% vs 5.36% (2×=10.71%) → **PASS**
- By entity: Australia 100.0% (18/18), China 20.0% (4/20), Denmark 63.16% (12/19), Norway 13.33% (2/15), Russia 91.3% (21/23), UK 12.5% (2/16), USA 86.67% (13/15)
- Confidence: broad 22.5% (9/40), immediate 91.67% (44/48), miss 0.0% (0/1), recognisable 51.35% (19/37)
- Wrong-guess counts by entity: {'Russia': 2, 'USA': 2, 'Denmark': 7, 'China': 16, 'UK': 14, 'Norway': 13}
- Source shas: `armB.jsonl`=c23f52b0a1fa1f72289e523cf96668aac0b5eb4fec7f688323c30903ec966704
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Kimi.jsonl` sha `a898149bfa235936f7f5195e43a8b36a321ff3d12e138067f6753e781054afd6`

### Gemini
- Notes: Full CORE+LIMIT 75 from Kari re-drop (chat t227u). Attachment a5b1cb not mounted; claimed sha 7b42d91e not matched (formatting).
- Overall: 73/75 = 97.33% vs chance 5.33% (2×=10.67%) → **PASS**
- CORE: 56/56 = 100.0% vs 5.36% (2×=10.71%) → **PASS**
- By entity: Australia 100.0% (18/18), Denmark 89.47% (17/19), Russia 100.0% (23/23), USA 100.0% (15/15)
- Confidence: broad 100.0% (4/4), immediate 100.0% (58/58), miss 75.0% (6/8), recognisable 100.0% (5/5)
- Wrong-guess counts by entity: {'Denmark': 2}
- Source shas: `armB_CORE_LIMIT.jsonl`=c135c2d4a5f1704cc1ec2ab883224ecdae850be20b220713c18b00369f68d66e
- Claimed output_sha `7b42d91e05f8a36c49e1208d73c51f908a62b4d1e709c35482a16f0d9e38c214` match=False; file_sha=`c135c2d4a5f1704cc1ec2ab883224ecdae850be20b220713c18b00369f68d66e`
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Gemini.jsonl` sha `f9f80e2ce48869006cabbe016afa256fd92e5fab1cb0f1579cba1b0b1b1ce1c4`

### Claude
- Notes: Full 126 from wintermute/analysis/arm_b_output_final.jsonl (claude-sonnet-4-6). Hashed chat attachments not mounted on box.
- Overall: 114/126 = 90.48% vs chance 5.56% (2×=11.11%) → **PASS**
- CORE: 56/56 = 100.0% vs 5.36% (2×=10.71%) → **PASS**
- By entity: Australia 100.0% (18/18), China 80.0% (16/20), Denmark 68.42% (13/19), Norway 100.0% (15/15), Russia 100.0% (23/23), UK 87.5% (14/16), USA 100.0% (15/15)
- Confidence: broad 50.0% (4/8), immediate 97.53% (79/81), recognisable 83.78% (31/37)
- Wrong-guess counts by entity: {'Denmark': 6, 'China': 4, 'UK': 2}
- Source shas: `armB.jsonl`=ed54c0a5420b93c68764ddeb44ad970a15f58ac8f8ab6536021a42dacdaeea36
- Scored: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06/scored_Claude.jsonl` sha `cf82f8e06d68c4543f3ddb3cfa4178da85bbe5266d42704f0e0b2d25653da645`

## Denmark (LIMIT)
- Manus: 15/19 = 78.95% (chance 5.26%, 2×=10.53%, pass=True)
- Codex: 1/19 = 5.26% (chance 5.26%, 2×=10.53%, pass=False)
- Kimi: 12/19 = 63.16% (chance 5.26%, 2×=10.53%, pass=True)
- Gemini: 17/19 = 89.47% (chance 5.26%, 2×=10.53%, pass=True)
- Claude: 13/19 = 68.42% (chance 5.26%, 2×=10.53%, pass=True)

## Caveats
- **Manus**: same-session Arm A contamination (ledger).
- **Grok**: CORE-only.
- **Codex**: all self-tagged miss; at chance.
- **Gemini**: full 75 sealed; claimed `7b42d91e…` not byte-matched (serialization). AU/USA now included — CORE 56/56.
- **Claude**: sealed from wintermute `arm_b_output_final.jsonl` (model confirmed via Arm A ledger claude-sonnet-4-6). Chat attachment hashes not mounted on box.
- Arm C / film not started.

## Paths
- Score pack: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/armB_scores_2026-11-06`
- Gemini: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/gemini_armB_2026-11-06/armB_CORE_LIMIT.jsonl`
- Claude: `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/seals/claude_armB_2026-11-06/armB.jsonl`
