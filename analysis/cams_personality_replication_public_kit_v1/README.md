# Public kit v1: repeat the CAMS personality experiment

7 November 2026. This folder is a new public kit. It is not part of the sealed evidence from the closed round, and it does not change those files.

Two routes:

| Who you are | Start here | What you may open |
|---|---|---|
| Participant, about to do Arms A and B blind | [PARTICIPANT.md](PARTICIPANT.md) | This kit only. Not the results report, not `evidence/seals/`, not the scored tables. |
| Operator, holding the key, or auditor checking the published scores | [OPERATOR.md](OPERATOR.md) | The published report and seals, after the participant's blind files are already hashed. |

The results write-up stays a separate paper: [CAMS personality: independent AI replication](../cams_personality_independent_ai_replication_2026-11-07/CAMS_Personality_Independent_AI_Replication_Report_2026-11-07.md). Read it after the blind arms, not before.

## What this kit can and cannot do

This is not a validated pack, and it is not ready to run all three arms on the original eras. The checks below were run on the files in this folder. No new model arm was run to produce this kit.

You can, with Python 3 and no install:

1. Read the participant rules and the Arm C packet that was actually issued.
2. Hash a file you wrote, with SHA-256 over the bytes (`scripts/sha256_file.py`).
3. Check that a return has the expected fields (`scripts/check_return_shape.py`). That check does not score it.
4. If you are the operator, recompute the published score counts from the sealed scored rows. That is a check of the old files. It is not a new blind test.

You cannot, from this folder alone:

- Rebuild the original Arm A portraits. The era-means tables and the node-definition file are not in the repository.
- Run the original Arm B. The unlabelled pockets and the closed shortlist are not in the repository.
- Score a new return against the original hold-out keys. Those keys were not published, and this kit does not reconstruct them.
- Treat the original 56 core eras as a fresh secret. The report and the scored rows are already public, and they carry the sealed grades and the true era ids.

The full omission list is [COVERAGE.md](COVERAGE.md).

## Algebra used in this experiment

Letters follow the public canonical definitions: [canonical-definitions.html](../../canonical-definitions.html). **C is Coherence. K is Capacity.** C is not complexity.

For each node, cognitive focus and affect are:

- Cog = A × C
- Aff = K − S
- σ = Cog × Aff = (A × C)(K − S)

A is Abstraction. S is Stress. The same product is written on the canonical page as σ = A·C·(K−S). This experiment's pockets use that product. They do not use Node Value and they do not use bond strength. The eight names used on the sealed returns are Helm, Shield, Lore, Stewards, Craft, Hands, Archive, and Flow. The original packet's node-definition file is not in the deposit. Do not write a private node manual and call it the missing file.

## Files

| Path | Role |
|---|---|
| [PARTICIPANT.md](PARTICIPANT.md) | What a blind participant does, and what they must not open |
| [OPERATOR.md](OPERATOR.md) | Hold-out scoring, and the difference between rechecking old files and running a new test |
| [COVERAGE.md](COVERAGE.md) | Present files and missing files |
| [packet/arm_c_mood/](packet/arm_c_mood/) | Byte copy of the issued Arm C packet (four files, no grades) |
| [scripts/sha256_file.py](scripts/sha256_file.py) | Real SHA-256 |
| [scripts/check_return_shape.py](scripts/check_return_shape.py) | Shape check, not a score |
| [scripts/verify_public_kit.py](scripts/verify_public_kit.py) | Checks this kit against the sealed Arm C packet |

Check the kit:

```bash
python3 scripts/verify_public_kit.py
```

A clean check prints `problems=0`.
