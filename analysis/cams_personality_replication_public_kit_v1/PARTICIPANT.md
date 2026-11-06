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

## What you were supposed to receive, and what this kit actually contains

The original standard packet, which is **not** in this repository, contained a readme, node definitions, the algebra and pocket rules, and three era-means tables with no narrative titles: core, extension, and Denmark. The Arm B packet, also **not** here, contained unlabelled pockets and a closed shortlist.

This kit contains the Arm C instructions that were issued, the hash script, and the shape checker. It does not contain the numbers you would need to write the original portraits. If your operator has not given you era means, a node card, and a written pocket rule, stop at Arm A and say the inputs were absent. Do not invent a table of era means, a node manual, or an era answer.

## Algebra

C is Coherence. K is Capacity. A is Abstraction. S is Stress. See [canonical-definitions.html](../../canonical-definitions.html).

- Cog = A × C
- Aff = K − S
- σ = Cog × Aff = (A × C)(K − S)

Positive σ is not happiness. The pocket describes function. It does not use Node Value or bond strength.

## Arm A — no web

Write a portrait from the numeric era means only. No web, no historical titles, no looking up the years.

The original pocket-rules file is missing. Two sealed ledgers record what those participants said they did, and those notes are not a substitute for the missing file:

- Grok's ledger: leaders from the highest σ, described there as the least disabled in a negative regime; constraints from the lowest σ; two functional sentences; no V and no bond.
- Kimi's ledger: era means only; no web; no V or bond; the file was not edited after the hash. On Denmark, identical input tables carried identical pocket text.

If your operator adopts a rule, they must write it down and hash it before you start. Your return, judged by the fields on the sealed Grok and Kimi files, is one JSON object per line with: `arm` = `A`, `model`, `entity`, `era_id`, `year_start`, `year_end`, `leaders`, `constraints`, `pocket`, `notes`, `sha256_input_csv`, `sha256_this_output`, `timestamp_aedt`. There was no separate Arm A schema file in the deposit. Those field names are observed, not a recovered official schema.

Denmark IX (1901–1913) has no observations. Omit it. Do not invent it.

When the file is finished, hash it with the script below. Put that digest in the ledger. If you also put it in `sha256_this_output`, the operator will recompute it. A string you typed is not a hash. In the closed round, one participant later said their Arm A and Arm C digests were text, not hashes of the files. Those runs are not a cryptographic seal.

The clock time must be the time you actually finished, in a named timezone. Do not write a later time than the clock. In the closed round, some tester timestamps sit later than the seal clock. That stayed a caveat.

Hand the file and the ledger to the operator. Do not change the file after the hash.

## Arm B — no web

Only after Arm A is hashed, and only in a fresh session.

You receive unlabelled pockets and a closed list of eras. You do not receive the era means again. No web. Pick one era id from the shortlist for each pocket and say how sure you are.

The original Arm B packet is not in this repository. Its recorded contents were a readme, `pockets_blind.jsonl`, `era_shortlist.csv`, an output schema, and a manifest. Do not reconstruct a shortlist from the published report. If your operator has not given you a blinded file and a shortlist, Arm B cannot start.

Observed fields on the sealed returns, again not a recovered schema file: `blind_id`, `entity`, `chosen_era_id`, `chosen_year_start`, `chosen_year_end`, `confidence`, `brief_reason`. Some files also have `set`. Confidence values that appear on the scored rows are `immediate`, `recognisable`, `broad`, and `miss`, with that spelling. A new operator should state the vocabulary in the packet they issue.

Hash the finished file before you send it.

## Arm C — web allowed, pockets unchanged

Use the packet in [packet/arm_c_mood/](packet/arm_c_mood/). Read `README_ArmC.md`, `output_schema_ArmC.md`, and `SHORT_PASTE.md`. They are a byte copy of the issued packet.

Compare your own already hashed Arm A pockets with historical sources. Web research is allowed in this arm only. Keep popular feeling, organised institutional mood, and official messaging separate. Grade each era `close`, `broad`, `partial`, `mixed`, or `direct_disagreement`. Do not edit, regenerate, or rehash the Arm A file. If a pocket is missing, omit the era and say so.

The era-means tables named in that readme (`era_means.csv`, `era_means_extension.csv`, `era_means_denmark.csv`) are not in this kit. You need the copies your operator already gave you at Arm A, or you cannot check era ids and years. Do not take era ids from the results report.

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
