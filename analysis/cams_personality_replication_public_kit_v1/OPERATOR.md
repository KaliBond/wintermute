# Operator notes

You keep the key. The participant does not. This file is for the person who issues packets and scores returns. It is also for an auditor who wants to recompute the published counts.

## Two jobs, and they are not the same

**Computational reproduction of the closed round.** The scored JSONL files under `analysis/cams_personality_independent_ai_replication_2026-11-07/evidence/seals/` already contain `sealed_q3_grade` or `true_era_id`. From the report folder:

```bash
python3 analysis/cams_personality_independent_ai_replication_2026-11-07/verify_evidence.py
```

`problems=0` means the sealed counts match a recomputation under the READY adjacency rule, and the file hashes match. It does not mean a new participant was blind. Anyone who opens those files has seen the answers. Do not describe that check as a fresh replication.

**A new blinded replication.** You must issue inputs the participant has not studied, and you must seal the key before they start. The original core eras are a poor fresh hold-out: the report, the scored rows, and the returned pockets are public. Say so if you reuse them. A fresh hold-out needs new era windows, or a partition the participant is not shown, with the key remaining only on your side.

## What you still have to supply

These files are named in [the distribution note](../cams_personality_independent_ai_replication_2026-11-07/evidence/DISTRIBUTION.md) and are not in the repository. This kit does not invent them.

| Missing piece | Why it matters |
|---|---|
| Standard tester zip (`da94f1c3…`), including node definitions, pocket rules, and `era_means.csv`, `era_means_extension.csv`, `era_means_denmark.csv` | Arm A cannot be repeated on the original numbers without them |
| Arm B zip (`dceee12e…`): unlabelled pockets, `era_shortlist.csv`, Arm B output schema | Arm B cannot be repeated on the original blind stimuli |
| `packets/answer_key_HOLD/arm_b_key.csv` (digest recorded as `00722f08…3c88`, 126 rows) | Original Arm B scoring |
| `packets/answer_key_HOLD/arm_c_key.csv` (digest recorded as `2e378db4…31d2`, 126 rows) | Original Arm C scoring |
| Original analyst pockets used for the Arm A leadership comparison | The 60% Arm A bar cannot be scored from this kit |

Recorded digests are identifiers of absent files. Do not rebuild a key by collecting `sealed_q3_grade` or `true_era_id` out of the public scored rows and calling it the hold-out key. Those values are already the published answers.

The Arm C packet **is** here, as a byte copy: [packet/arm_c_mood/](packet/arm_c_mood/). Its manifest matches the sealed packet. It contains no grades. It still refers to era-means tables you must provide separately.

## Before the participant starts

1. Write the pocket rule and the node card you are actually issuing. Hash them. The original node-definition file is missing, so you cannot claim they are that file unless you hold it and the hash matches a digest you can show.
2. Seal the answer key in a file the participant's session cannot see. Hash the key and record the digest. Do not send the key.
3. For Arm B, shuffle or otherwise detach pockets from their era titles before the participant session. Hash the blinded file.
4. Give Arm A the numeric table only. No titles.
5. Require one fresh session per arm. Arm B must not be shown the numeric table.
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

You need the shortlist to compute the chance. It is not in this kit.

## Arm A

The plan's bar is at least 60% leadership-pair and constraint match on Russia, Australia, and the USA, across at least three models. The closed deposit never scored that bar. Do not fill in a percentage. A new test needs your sealed reference pockets, compared after the participant's hash is recorded.

## What to publish afterwards

Publish the participant files, your recomputed counts, the key digest (not the key, unless you are deliberately opening it), coverage, and the arms that did not run. Unrun is not a fail. Weak and negative results stay in the table. Denmark's identical input clusters are a reason the control is hard, not a prediction that the control fails.

Do not describe the run as causal evidence, as peer review, or as a clean replication if the hash, the session separation, or the hold-out failed.

The closed round's own limits are in the results report. This kit does not restate them as a new result.
