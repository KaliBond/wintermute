# Public kit v1: repeat the CAMS personality experiment

7 November 2026. This folder is the public kit. It does not change the sealed evidence from the closed round.

**Start at [START_HERE.md](START_HERE.md).** Do not open the results report, `evidence/seals/`, or any `scored_*` file before you have hashed an Arm A return and decided whether Arm B is a historical exercise or a new hold-out.

| Who you are | Read | What you may open |
|---|---|---|
| Participant | [START_HERE.md](START_HERE.md), then [PARTICIPANT.md](PARTICIPANT.md) | The issued inputs in [inputs/](inputs/) and the [participant zip](CAMS_personality_participant_inputs_v1.zip). Not the results report and not the scored rows. |
| Operator or auditor | [OPERATOR.md](OPERATOR.md) | The published report and seals, on a separate path from the participant pack. The old answer keys are still not in this kit. |

This is not a self-contained new blind validation of Arms A, B, and C. No model was run to build the kit.

## What is now in the kit

Genuine issued files, checked against their manifests:

- Arm A: node definitions, algebra and pocket rules, core, extension, and Denmark era-means CSVs, output schema, and the original standard zip `da94f1c3…`.
- Arm B historical packet: pockets, shortlist, schema, and the original zip `dceee12e…`. This is a spoiled challenge. See [inputs/arm_b_blind_historical/SPOILER_READ_FIRST.md](inputs/arm_b_blind_historical/SPOILER_READ_FIRST.md).
- Arm C instructions, byte-identical to the sealed packet and to [packet/arm_c_mood/](packet/arm_c_mood/), plus the original zip `23fce155…`.

The CSVs name the country and the years. They do not name the episode. That is the protocol that was issued. Arm A and Arm B are no-web. Arm C allows the web only after Arm A is hashed, and it does not allow editing Arm A.

`inputs/arm_b_blind_historical/COUNTS.txt` still records Arm B zip sha256 `58ff7c41…`. That note is stale. The file was not edited. The issued zip in `inputs/issued_archives/` is `dceee12e…`.

## What is still required, and still absent

- The closed-round Arm B and Arm C answer keys. They are not published and not reconstructed. See [inputs/HOLD_KEYS_WITHHELD.md](inputs/HOLD_KEYS_WITHHELD.md).
- A new Arm B blind set, and a new key, if the operator wants a fresh identification test. The historical pockets are not that test.
- A new mood key, written before Arm C and held by the operator, if the operator wants a scored mood validation. A participant’s own grade against sources is descriptive. It is not the old central agreement.
- Company means and the film brief. Not part of A–C.

Recomputing the closed round’s published counts is an operator check of files that already contain the answers. It is not in the participant zip. From the report folder, after the blind work is done: `python3 analysis/cams_personality_independent_ai_replication_2026-11-07/verify_evidence.py`.

## Algebra, as issued

The packet’s own rule file is [inputs/arm_a_standard/algebra_and_rules.md](inputs/arm_a_standard/algebra_and_rules.md). **C is Coherence. K is Capacity.**

- Cognition: Cog = A × C
- Affect: Aff = K − S
- Activation: σ = Cog × Aff

Cog is the cognitive quantity. The packet does not define Cog as focus. Its reading rules speak separately of high focus with negative activation, and they treat that as strain, not health. Do not compute V or bond.

## Check and rebuild

```bash
python3 scripts/verify_public_kit.py
python3 scripts/build_participant_zip.py
```

`verify_public_kit.py` checks the issued hashes, the Arm C bytes against the sealed packet, the absence of key CSVs, and that the participant zip matches a deterministic rebuild. A clean check prints `problems=0`.
