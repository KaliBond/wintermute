# Operator notes

You keep the key. The participant does not. This file is for the person who issues packets and scores returns. It is also for an auditor who wants to recompute the published counts.

## Two jobs, and they are not the same

**Computational reproduction of the closed round.** The scored JSONL files under `analysis/cams_personality_independent_ai_replication_2026-11-07/evidence/seals/` already contain `sealed_q3_grade` or `true_era_id`. From the report folder:

```bash
python3 analysis/cams_personality_independent_ai_replication_2026-11-07/verify_evidence.py
```

`problems=0` means the sealed counts match a recomputation under the READY adjacency rule, and the file hashes match. It does not mean a new participant was blind. Anyone who opens those files has seen the answers. Do not describe that check as a fresh replication.

**A new blinded replication.** You must issue inputs the participant has not studied, and you must seal the key before they start. The original core eras are a poor fresh hold-out: the report, the scored rows, and the returned pockets are public. Say so if you reuse them. A fresh hold-out needs new era windows, or a partition the participant is not shown, with the key remaining only on your side.

## What is now public, and what you still hold

The issued Arm A zip (`da94f1c3…`), Arm B zip (`dceee12e…`), and Arm C zip (`23fce155…`) are in [inputs/issued_archives/](inputs/issued_archives/). The loose files beside them match those archives. Do not rebuild the closed-round keys from `sealed_q3_grade` or `true_era_id` in the public scored rows.

| Still absent | Why it still matters |
|---|---|
| Closed-round Arm B key and Arm C key | Not in this kit. Digests were recorded in the score ledgers. Contents stay private |
| A new Arm B pocket set and a new key | Required before anyone can call an identification run a fresh blind |
| A new Arm C key, fixed before the participant starts | Required before a new mood return can be scored as central agreement. The participant’s own grade is descriptive only |
| Original analyst pockets for the 60% Arm A leadership bar | That bar was not scored in the closed deposit, and those pockets are not this kit |

`COUNTS.txt` still names a stale Arm B zip hash, `58ff7c41…`. Leave that file as issued.

## Before the participant starts

1. Give them [inputs/arm_a_standard/](inputs/arm_a_standard/) for Arm A. The node card and rules are the issued files. Hash what you hand over if you need a receipt.
2. If you want a fresh Arm B, mint new pockets after their Arm A hash and keep the new key offline. Do not call `pockets_blind.jsonl` a fresh secret.
3. If you want a scored Arm C, write the mood key first and do not send it. Otherwise accept their grades as descriptive only.
4. Give Arm A the issued CSV. It already has country and years, and no narrative titles. Do not pretend it was geography-blind.
5. Require one fresh session per arm. Arm B must not be shown the era-means table.
6. Tell the participant to hash with [scripts/sha256_file.py](scripts/sha256_file.py) or another real SHA-256 of the file bytes. If they cannot run code, the arm stops. A digest typed from memory is not a seal. That failure mode is already on the record for the closed round.
7. Take their timestamp from a clock you can see, or require the script's `--ledger` line, which prints this computer's clock. Reject a completion time that is later than the receipt and unexplained.

## How to score Arm C

Use the rule in the sealed stub [armC_scores_READY/README.md](../cams_personality_independent_ai_replication_2026-11-07/evidence/seals/armC_scores_READY/README.md), which was written before the returns in that round were scored.

- Exact: returned grade equals sealed grade.
- Adjacent, only: close↔broad, broad↔partial, partial↔mixed.
- Not adjacent: every other pair. Mixed↔direct_disagreement is not adjacent. Close↔partial is not adjacent.
- Overturn: the participant returns `direct_disagreement` and the sealed grade was close or broad.
- Also count the other direction, sealed `direct_disagreement` returned as close or broad, and do not rename it as an overturn.
- Majority means more than half the scored rows in that set. Keep core (56), extension (51), and Denmark (19, IX omitted) separate.
- A malformed `arm_a_sha256` is reported, not repaired. In the closed round one Denmark row was 77 characters. It stayed in the descriptive count and was left out of a validated subset.

There is no Arm C chance model in the sealed round. Do not add one and call it pre-registered. Grades that pile up on close and broad make an exact-or-adjacent majority a coarse screen. Report exact, adjacent, non-adjacent, overturns, and sealed-disagreement reversals separately.

The numeric bar and the protocol bar are different. A majority can be met on rows whose claimed hashes were not computed. Say both.

## How to score Arm B

The sealed score pack defines chance as the mean, over scored rows, of `1 / (number of eras on that entity's shortlist)`. The pre-registered bar is a hit rate at least twice that mean. Core was the decision set. Denmark was allowed to miss without cancelling a core pass. Extension was reported beside them, including weak hit rates. Do the same.

The historical shortlist is [inputs/arm_b_blind_historical/era_shortlist.csv](inputs/arm_b_blind_historical/era_shortlist.csv). Use it for the historical exercise. A fresh test needs the shortlist of the new set you issued.

## Arm A

The plan's bar is at least 60% leadership-pair and constraint match on Russia, Australia, and the USA, across at least three models. The closed deposit never scored that bar. Do not fill in a percentage. A new test needs your sealed reference pockets, compared after the participant's hash is recorded.

## What to publish afterwards

Publish the participant files, your recomputed counts, the key digest (not the key, unless you are deliberately opening it), coverage, and the arms that did not run. Unrun is not a fail. Weak and negative results stay in the table. Denmark's identical input clusters are a reason the control is hard, not a prediction that the control fails.

Do not describe the run as causal evidence, as peer review, or as a clean replication if the hash, the session separation, or the hold-out failed.

The closed round's own limits are in the results report. This kit does not restate them as a new result.
