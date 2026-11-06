# HOLD keys — withheld from this public bundle

The closed 2026 round used private answer keys for central scoring:

| Key | Role | In this zip? |
| --- | --- | --- |
| Arm B key | Maps each `blind_id` to the true era window | **No** |
| Arm C key | Sealed analyst Q3 mood grades per era | **No** |

They remain with the experiment operator for scoring the **closed** round only.

**Availability (operator-side, not published here):** keys exist in the local evidence archive under the packet tree’s `answer_key_HOLD/` directory, alongside a private README that must not be redistributed with tester materials.

**For a new public replication:**

1. Participants can run **Arm A** (pocket writing) and **Arm C** (independent mood grading) with the files in this zip.
2. For **Arm B**, the operator prepares a **new** blinded pocket set and keeps a **new** HOLD key offline until after participant hashes are recorded.
3. Do not reconstruct or publish the closed-round HOLD keys. Publishing them would not enable a cleaner re-test of the original blind Arm B; that challenge is already spoiled.

This file intentionally omits key contents, path dumps, and sealed grade tallies.
