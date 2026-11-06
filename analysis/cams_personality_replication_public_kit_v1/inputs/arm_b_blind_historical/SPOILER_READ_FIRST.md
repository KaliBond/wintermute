# SPOILER — historical Arm B packet (closed 2026 round)

This folder is the **genuine issued** Arm B blind packet from the closed 2026 replication (`CAMS_ArmB_blind_v1.zip`).

It is **not** a fresh blind challenge.

- The closed-round report and many scored returns are already public.
- Anyone who also runs Arm A on the same era-means tables can reconstruct likely pocket↔era links.
- Country names appear on each pocket row by design of the original protocol (entity known; era window blind).

**For a new blind Arm B:** the operator must mint **new** unlabeled pockets from hashed Arm A outputs (or a hold-out set), with a **new** answer key kept offline. Do not reuse `pockets_blind.jsonl` or the closed-round HOLD key as a fresh secret.

This folder includes protocol files (`README_ArmB.md`, `output_schema_ArmB.md`, `era_shortlist.csv`) so an operator can see the exact shape of a valid Arm B challenge. The answer key that maps `blind_id` → true era is **withheld** from this public bundle.
