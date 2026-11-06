# Start here

Read this before the other files.

Do not open the replication report, anything under `evidence/seals/`, or any file named `scored_*`, until you have hashed your Arm A return and decided what kind of Arm B you are doing. Those files contain the closed round’s grades and true era ids.

This kit is not a new, self-contained blind validation of Arms A, B, and C.

## What you can do with these files

**Arm A.** The issued tables, node definitions, algebra, and output schema are in `inputs/arm_a_standard/`. The CSVs name the country and the year range. They do not name the historical episode. That is the protocol that was actually issued. No web. Do not add titles. Hash the finished file with `scripts/sha256_file.py` before any later arm. If you cannot run a real hash, stop. Do not type a digest.

**Arm C, after that hash.** Use `inputs/arm_c_mood/` and your own Arm A file. Web research is allowed in this arm only. Do not edit Arm A. You may write your own mood grades. That descriptive grading is not the closed round’s central score. The old mood key is not in this kit. A fresh mood validation needs a new key written down by an operator before you start, and kept off your machine.

**Arm B, historical packet only.** `inputs/arm_b_blind_historical/` is the packet that was issued. Each pocket already carries an entity name. The era is chosen from the shortlist. The closed-round report and scored returns are public, so this packet is not a fresh blind challenge. Reuse it only as a historical exercise, and say that you did. `COUNTS.txt` in that folder records zip sha256 `58ff7c41…`. That is not the issued archive. The archive that matches the ledger is `inputs/issued_archives/CAMS_ArmB_blind_v1.zip`, sha256 `dceee12e…`. Do not edit `COUNTS.txt` to “repair” it.

## What is still not here

The closed-round Arm B and Arm C answer keys are not in this kit and are not reconstructed. A new blind Arm B needs a new pocket set and a new key, both made by an operator after your Arm A file is hashed. Company means and the film brief are not included.

The downloadable copy of the participant inputs, without scored outputs, is [CAMS_personality_participant_inputs_v1.zip](CAMS_personality_participant_inputs_v1.zip). Rebuild it with `python3 scripts/build_participant_zip.py`.
