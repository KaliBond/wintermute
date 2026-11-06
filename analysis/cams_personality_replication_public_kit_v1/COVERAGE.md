# Kit coverage

Updated when the issued participant inputs were added. Nothing here was inferred into a missing table.

| Item | Status |
|---|---|
| Arm A node definitions, algebra and rules, output schema, plan summary | Present, genuine. `inputs/arm_a_standard/` |
| Era-means CSVs: core, extension, Denmark | Present. Core sha256 `887be60d…`, Denmark `70c9f0be…`, extension `c6056221…` |
| Standard tester zip | Present. `da94f1c3…`, 22460 bytes |
| Arm B pockets, shortlist, schema, readme | Present as a historical packet. Not a fresh blind |
| Arm B issued zip | Present. `dceee12e…`, 14837 bytes |
| Arm C packet | Present. Byte-identical to the sealed copy. Zip `23fce155…` |
| SHA-256 script and return-shape check | Present. The shape check does not score |
| Participant download | [CAMS_personality_participant_inputs_v1.zip](CAMS_personality_participant_inputs_v1.zip) |
| Closed-round Arm B key and Arm C key | Absent on purpose. Not reconstructed |
| New Arm B hold-out and new mood key | Not in this kit. An operator must create them before a fresh scored test |
| Scored JSONL and the results report | Not in the participant pack. Operator path only, and answer-bearing |
| Company Arm D means and the film brief | Absent. Out of scope |
| A new model run confirming the round | Not done |

## Stale note left as issued

`inputs/arm_b_blind_historical/COUNTS.txt` says the Arm B zip sha256 is `58ff7c4166d0293b2f5ad6cfee33db5d3d46a3b5b9932c3a84594e507c8b6ea6`. The canonical issued archive hashes to `dceee12e89c1b07b9f63e4ce26836f51fb231089b621afc4f9d8f819ab73aae9`. COUNTS was not repaired.

## Actual blinding, not a rewritten one

Arm A tables include country and year range and exclude narrative titles. Arm B pockets include `entity`, `set`, `blind_id`, and `pocket_text`. The era is chosen from `era_shortlist.csv`. The standard packet’s prose sometimes describes Arm B as unlabeled for entity as well as era. The file that was issued does include the entity. This kit does not edit that file to match the other sentence.
