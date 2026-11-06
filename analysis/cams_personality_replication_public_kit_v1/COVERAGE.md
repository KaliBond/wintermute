# Kit coverage

Checked against the repository at the sealed round (`6e0e558` and the files it contains) and against this folder. Nothing here was inferred into a file that the deposit does not have.

| Item | In this kit | Notes |
|---|---|---|
| Participant route and operator route | yes | [PARTICIPANT.md](PARTICIPANT.md), [OPERATOR.md](OPERATOR.md) |
| Arm C readme, schema, short paste, manifest | yes | Byte copy. Hashes match `evidence/packets/arm_c_mood/` |
| SHA-256 script over file bytes | yes | [scripts/sha256_file.py](scripts/sha256_file.py) |
| Return shape check | yes | Does not score. [scripts/check_return_shape.py](scripts/check_return_shape.py) |
| Canonical letter meanings (C = Coherence, K = Capacity) | cited | [canonical-definitions.html](../../canonical-definitions.html). Not copied, so the definitions page stays the source |
| Experimental claim, sample sizes, pre-registered bars | cited | [experimental_plan.md](../cams_personality_independent_ai_replication_2026-11-07/evidence/experimental_plan.md) and the Arm C packet |
| READY adjacency and overturn rule | cited | [armC_scores_READY/README.md](../cams_personality_independent_ai_replication_2026-11-07/evidence/seals/armC_scores_READY/README.md) |
| Era-means CSVs | no | Named by the Arm C readme. Not in the repository |
| Node-definition file and pocket-rules file | no | Named by the distribution note. Grok and Kimi ledgers are quoted as what those runs said, not as a replacement file |
| Arm B unlabelled pockets, shortlist, and output schema | no | Zip digest is recorded in the distribution note. Files are absent |
| Arm A output-schema file | no | Field names below are observed on sealed returns |
| Hold-out keys | no | Digests only, in the distribution note and the score ledgers. Not reconstructed |
| Original analyst pockets for the Arm A match bar | no | The 60% bar was not scored in the closed deposit |
| Scored JSONL and the results report | not in this kit | They are public elsewhere and are answer-bearing. Participants are told not to open them before the blind arms |
| Null arms (shuffled σ, random means, Wikipedia-only) | no results deposited | Do not invent them |
| Arm D company panel and the film | not part of this kit | No public return in the closed round |
| A new model run packaged as confirmation | no | This kit was not produced by running Arms A–C again |

## Observed return fields

These lists are the keys present on sealed files. They are not a schema document recovered from the missing packets.

Arm A, from the Grok and Kimi JSONL: `arm`, `model`, `entity`, `era_id`, `year_start`, `year_end`, `leaders`, `constraints`, `pocket`, `notes`, `sha256_input_csv`, `sha256_this_output`, `timestamp_aedt`. On those files `sha256_this_output` is an empty string. The file-level ledger hash is the one that matched. A new return should carry a real file hash from `sha256_file.py`.

Arm B, from the sealed guess files: `blind_id`, `entity`, `chosen_era_id`, `chosen_year_start`, `chosen_year_end`, `confidence`, `brief_reason`, and on some files `set`.

Arm C: the issued schema in [packet/arm_c_mood/output_schema_ArmC.md](packet/arm_c_mood/output_schema_ArmC.md).

## Sample cuts named in the plan and the Arm C packet

| Set | Societies | Eras |
|---|---|---:|
| Core | Russia 23, Australia 18, USA 15 | 56 |
| Extension | China 20, UK 16, Norway 15 | 51 |
| Denmark limit | Denmark, IX (1901–1913) omitted | 19 |

Those counts are the denominators. This kit does not include the rows.
