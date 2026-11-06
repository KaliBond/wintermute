# Validation of this deposit

Checked on this agent after the zip hash matched. No model was called. No HOLD key was opened, because none is in the attachment. No score was changed.

The checker is [`verify_evidence.py`](verify_evidence.py). Its committed transcript is [`validation_output.txt`](validation_output.txt). The machine-readable twin is [`reconciliation_summary.json`](reconciliation_summary.json).

## Result

`problems=0`. Evidence file count 106. Evidence bytes 1379037. Headline Arm B and Arm C counts recomputed from the scored JSONL match the sealed summaries and the closeout.

Re-run:

```bash
python3 verify_evidence.py
```

The script rewrites the tables, the evidence manifest, the summary JSON, and `validation_output.txt`. With these evidence bytes, that rewrite is identical to the committed outputs.

## What was reconciled

1. Every `MANIFEST.sha256` member, with the one self-hash exception below.
2. Arm C exact, READY-adjacent, exact-or-adjacent, overturning, and sealed-`direct_disagreement` reversal counts, from `sealed_q3_grade` and `q3_grade` on the scored rows.
3. Stored row flags (`exact_match`, `adjacent`, `exact_or_adjacent`) against that READY recomputation.
4. A second pass that also treats `mixed` ↔ `direct_disagreement` as adjacent. For these rows it changes no exact-or-adjacent count.
5. Named Denmark rows: Gemini VI, X, XII, XIX, and XV; Claude III, V, VI, and X.
6. Gemini Denmark XV `arm_a_sha256` length 77, left unrepaired, and absent from the validated 18-row file.
7. Denmark IX absent from both scored Denmark files.
8. Sealed Q3 grades agree between Claude and Gemini on all 56 CORE eras and all 19 Denmark eras. That is a consistency check of the scored packs. It is not a reconstruction of the HOLD key, and no key file was written.
9. Claude sealed outputs are byte-identical to the preserved source copies.
10. Gemini’s extension file is empty and hashes to the SHA-256 of the empty string.
11. Arm B correct counts by model, set, and the pre-registered CORE comparison with twice the mean per-row chance stored on the scored rows.
12. Spot hashes named in the Gemini audit addendum, the Claude ledgers, and the Arm A/B file ledgers that have a file present.

## Arm B manifest self-hash

[`evidence/seals/armB_scores_2026-11-06/MANIFEST.sha256`](evidence/seals/armB_scores_2026-11-06/MANIFEST.sha256) records its own digest as `e390b23a60542e74a8c022c26050a902fcee8ba28252b7995602dcbc9b681374`. The file as stored hashes to `e81107966246070772fca7f84b295dd4e145e9ab8c371dbb180db73266f357bb`.

Every other line in that manifest matches its file, including `SCORE_SUMMARY.md`, `aggregate_scores.json`, and all six `scored_*.jsonl` files. No other manifest in the deposit contains a self-line. The self-line was not rewritten. The checker fails if either the recorded self-line or the actual file digest changes, so a later “repair” is visible.

## Numeric results the checker locked

| Pack | Exact | Exact-or-adjacent | Overturns | Sealed DD returned close or broad |
|---|---:|---:|---:|---:|
| Gemini CORE | 27/56 | 46/56 | 0 | 0 |
| Claude CORE | 27/56 | 48/56 | 0 | 0 |
| Claude EXTENSION | 31/51 | 41/51 | 0 | 0 |
| Claude Denmark | 4/19 | 10/19 | 0 | 4 |
| Gemini Denmark, all 19 | 7/19 | 14/19 | 0 | 2 |
| Gemini Denmark, XV excluded | 6/18 | 13/18 | 0 | 2 |

| Arm B model | Correct | CORE | CORE at least twice mean row chance |
|---|---:|---:|---|
| Claude | 114/126 | 56/56 | yes |
| Gemini | 73/75 | 56/56 | yes |
| Manus | 87/126 | 56/56 | yes, contamination flagged in the source ledger |
| Kimi | 72/126 | 52/56 | yes |
| Grok | 19/56 | 19/56 | yes |
| Codex | 7/126 | 3/56 | no |

## What was not validated

- Leadership-pair or constraint agreement for Arm A. No such score pack is in the attachment, so none was created.
- The claimed Claude Arm A bytes `520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7`. The file is absent.
- Gemini’s claimed Arm A and Arm C digests. The audit addendum says they were not cryptographic hashes. The checker uses the coordinator’s receipt hashes.
- Packet-zip digests in [`evidence/DISTRIBUTION.md`](evidence/DISTRIBUTION.md). The zips are not in the attachment. The four loose Arm C packet files do match [`evidence/packets/arm_c_mood/MANIFEST.sha256`](evidence/packets/arm_c_mood/MANIFEST.sha256).
- Any chance baseline for Arm C. None is in the seals, and none was added.
- The canvas page content beyond confirming that the recorded URL opens a closeout page with the same headline fractions. The seals, not the canvas, are the source of truth. The canvas compresses Claude’s provenance to “hashes OK”; the score packs still mark the Arm A file unverified.

## Evidence manifest

[`EVIDENCE_MANIFEST.sha256`](EVIDENCE_MANIFEST.sha256) lists all 106 evidence files. Its own SHA-256, as written by the checker, is `d70dcedbc02d1d623930d3a64002a0702a0ec15e310ad99a2f07744d9bffe4da`.
