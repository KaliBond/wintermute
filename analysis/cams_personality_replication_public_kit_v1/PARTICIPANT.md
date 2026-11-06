# Participant instructions

You are about to try the personality replication blind. The original round is already published. If you read the results first, your run is not blind.

## Do not open these until Arm A and Arm B are hashed and handed over

- `analysis/cams_personality_independent_ai_replication_2026-11-07/CAMS_Personality_Independent_AI_Replication_Report_2026-11-07.md`
- `analysis/cams_personality_independent_ai_replication_2026-11-07/evidence/seals/`
- `analysis/cams_personality_independent_ai_replication_2026-11-07/tables/`
- Any file whose name starts with `scored_`
- The Core Papers discussion of the numeric results, if you have already seen it elsewhere

Arm C allows web pages about the historical period. It still does not allow the sealed grades, the results report, or those scored files. The published 56 core eras are not a secret hold-out. A run on those same eras, after this publication, is a repetition in public, not a fresh blind test. Say which of the two you did.

Use a fresh session for each arm. Do not leave the era-means table open in the session that does Arm B. In the closed round, one Arm B return was flagged because Arm A had already been read in the same session. That return is not a clean blind.

Record the model name and version. A language model is not an independent witness. The historical record of these societies is widely available in training text, and the era means themselves were coded with history in view. Blindness here means you were not shown the titles, the grades, or the key. It does not mean the model has never read a history book.

## What this kit contains

The issued Arm A packet is in [inputs/arm_a_standard/](inputs/arm_a_standard/): node definitions, algebra and rules, and the three era-means tables. Those tables name the country and the years. They do not name the episode. That is the real protocol, not an extra mask added later.

The issued Arm B packet is in [inputs/arm_b_blind_historical/](inputs/arm_b_blind_historical/). Read [SPOILER_READ_FIRST.md](inputs/arm_b_blind_historical/SPOILER_READ_FIRST.md) first. Each pocket already names the country. You choose the era from the shortlist. Because the closed-round report is public, this packet is a historical exercise, not a fresh blind. Say so if you use it. A fresh Arm B is a new set your operator makes after your Arm A hash, with a new key you never see.

Arm C is [inputs/arm_c_mood/](inputs/arm_c_mood/). The old mood key is not included. Writing your own grade is not the same as the old central score.

## Algebra

C is Coherence. K is Capacity. A is Abstraction. S is Stress. The issued file is [algebra_and_rules.md](inputs/arm_a_standard/algebra_and_rules.md).

- Cognition: Cog = A × C
- Affect: Aff = K − S
- Activation: σ = Cog × Aff

Cog is the cognitive quantity. It is not the word “focus.” The rules use focus and activation as separate readings. High focus with negative activation is strain, not health. = (A × C)(K − S)

Positive σ is not happiness. The pocket describes function. It does not use Node Value or bond strength.

## Arm A — no web

Write a portrait from [inputs/arm_a_standard/](inputs/arm_a_standard/). No web and no historical titles. The pocket rules are [algebra_and_rules.md](inputs/arm_a_standard/algebra_and_rules.md). The return shape is [output_schema.md](inputs/arm_a_standard/output_schema.md).

Two sealed ledgers from the closed round are consistent with those rules, and they are not a second rule file:

- Grok's ledger: leaders from the highest σ, described there as the least disabled in a negative regime; constraints from the lowest σ; two functional sentences; no V and no bond.
- Kimi's ledger: era means only; no web; no V or bond; the file was not edited after the hash. On Denmark, identical input tables carried identical pocket text.

If your operator adopts a rule, they must write it down and hash it before you start. Your return, judged by the fields on the sealed Grok and Kimi files, is one JSON object per line with: `arm` = `A`, `model`, `entity`, `era_id`, `year_start`, `year_end`, `leaders`, `constraints`, `pocket`, `notes`, `sha256_input_csv`, `sha256_this_output`, `timestamp_aedt`. There was no separate Arm A schema file in the deposit. Those field names are observed, not a recovered official schema.

Denmark IX (1901–1913) has no observations. Omit it. Do not invent it.

When the file is finished, hash it with the script below. Put that digest in the ledger. If you also put it in `sha256_this_output`, the operator will recompute it. A string you typed is not a hash. In the closed round, one participant later said their Arm A and Arm C digests were text, not hashes of the files. Those runs are not a cryptographic seal.

The clock time must be the time you actually finished, in a named timezone. Do not write a later time than the clock. In the closed round, some tester timestamps sit later than the seal clock. That stayed a caveat.

Hand the file and the ledger to the operator. Do not change the file after the hash.

## Arm B — no web

Only after Arm A is hashed, and only in a fresh session that is not shown the era-means table.

For the historical packet, follow [README_ArmB.md](inputs/arm_b_blind_historical/README_ArmB.md) and [output_schema_ArmB.md](inputs/arm_b_blind_historical/output_schema_ArmB.md). No web. The entity is already on the pocket. Pick one shortlist row. This reuse is contaminated by publication. A fresh blind needs a new packet from your operator.

Hash the finished file before you send it.

## Arm C — web allowed, pockets unchanged

Use the packet in [packet/arm_c_mood/](packet/arm_c_mood/). Read `README_ArmC.md`, `output_schema_ArmC.md`, and `SHORT_PASTE.md`. They are a byte copy of the issued packet.

Compare your own already hashed Arm A pockets with historical sources. Web research is allowed in this arm only. Keep popular feeling, organised institutional mood, and official messaging separate. Grade each era `close`, `broad`, `partial`, `mixed`, or `direct_disagreement`. Do not edit, regenerate, or rehash the Arm A file. If a pocket is missing, omit the era and say so.

The era ids and years are the ones in [inputs/arm_a_standard/](inputs/arm_a_standard/). Do not take them from the results report.

Core first: Russia, Australia, USA, 56 eras. Then stop. Extension is optional: China, UK, Norway, 51 eras. Denmark is last, 19 graded eras, IX omitted.

## Hash and shape

From this folder:

```bash
python3 scripts/sha256_file.py your_return.jsonl --ledger --model "name-and-version" --arm A --entity-set CORE
python3 scripts/check_return_shape.py --arm A your_return.jsonl
```

Use `--arm B` or `--arm C` for the other files. The shape check does not know the right answers. Exit code 0 means the fields are present and the vocabulary matches. It does not mean the portrait is right.

If this environment cannot run Python or `sha256sum`, stop and write that in the ledger. Do not generate a hex string by hand.

## What to report, including the weak parts

Report every era you ran, every era you skipped, and every arm you did not run. An unrun arm is not a failure and not a pass. A wrong Arm B guess and a non-adjacent Arm C grade stay in the file. Denmark is a sparse-input control: identical input rows can force identical pockets. That is a property of thin data, not a requirement that Denmark fail. Do not drop Denmark because it looks bad, and do not claim it must fail.

The pre-registered bars, as written in the plan and the Arm C scoring stub, are for the operator to apply after you have submitted:

- Arm A: at least 60% leadership-pair and constraint match on Russia, Australia, and the USA, across at least three models. The original reference pockets were not published, so this kit cannot score that bar.
- Arm B: at least twice the chance hit rate on the closed shortlist. Chance is one divided by the number of eras on that society's list. Denmark may fail without removing a core result. "May fail" is not "must fail".
- Arm C: a majority of grades exact or adjacent, and no overturning pattern. Adjacent pairs are only close↔broad, broad↔partial, and partial↔mixed. Mixed and direct disagreement are not adjacent. An overturn is your `direct_disagreement` where the sealed grade was close or broad. The operator also inspects the opposite case and keeps it distinct.

Null checks named in the plan (shuffled σ, random means, Wikipedia-only) have no result files in the deposit. Do not invent them.

This experiment does not show that the eight-node portrait causes historical outcomes. It is not peer review.
