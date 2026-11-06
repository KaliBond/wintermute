# Arm C scoring stub: READY, no returns scored

Prepared for Kari McKern (Neural Nations / CAMS) by Reportmister.

Status: packet and HOLD key prepared; no Arm C tester returns have been received or scored. Do not infer or invent scores.

## Inputs when a return arrives

- Tester JSONL, final sha256 and ledger line, including the tester's hashed Arm A input sha256.
- Sealed key: `../../packets/answer_key_HOLD/arm_c_key.csv`, 126 graded eras. Never ship it to testers.
- Join by `(entity, era_id)` and validate set and years against the key before scoring. Reject or quarantine malformed, duplicate or missing rows rather than silently dropping them.

## Per-era comparison

- **Exact:** returned `q3_grade` equals `sealed_q3_grade`.
- **Adjacent:** only `close` <-> `broad`, `broad` <-> `partial`, or `partial` <-> `mixed`. The relation is symmetric.
- **Non-adjacent:** every other unequal pair. In particular, `mixed` <-> `direct_disagreement` and `close` <-> `partial` are non-adjacent.
- Keep CORE, EXTENSION and LIMIT separate. Report coverage and ungraded omissions alongside grade comparisons.

## Pre-registered interpretation

- Success requires a majority of scored comparisons to be exact-or-adjacent, with no overturning disagreement pattern. State the denominator and coverage explicitly; pair this with disagreement checks rather than reporting accuracy alone.
- Flag an overturning disagreement instance when the tester returns `direct_disagreement` and the sealed grade was `close` or `broad`.
- Also inspect a majority-disagreement pattern: more than half of valid returned comparisons in the reported set are `direct_disagreement`. Retain the sealed disagreements as expected negatives and inspect reversals in both directions, including sealed `direct_disagreement` cases returned as `close` or `broad`.
- Denmark remains the sparse-input LIMIT control, not a reason to erase failures. Do not pool it away into CORE or EXT.

This stub defines scoring mechanics only; it is not an Arm C result seal.
