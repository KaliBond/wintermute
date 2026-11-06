# Audit history

Append-only. Do not edit an earlier entry. Do not edit, rehash, or replace a file under `evidence/` to make a later entry look cleaner. If a correction is needed, add a new dated entry that points at the earlier one and at the unchanged seal.

## 2026-11-07 — repository deposit of the closed round

Recorded when this folder was added to `KaliBond/wintermute`.

- Source zip SHA-256 `8631465fefa599c3035b4a8c97483394ff8b72d51702de6c5cad74d6442ee7bf`, 306507 bytes, 106 members, 1379037 uncompressed bytes. Local receipt: [RECEIPT.md](RECEIPT.md).
- Seals copied byte-for-byte. No scoring rerun. No model run. No upgrade. No HOLD key added.
- Binding Gemini Arm A/C qualification remains the sealed addendum at [evidence/seals/gemini_armAC_audit_addendum_2026-11-07/AUDIT_DISCLOSURE.md](evidence/seals/gemini_armAC_audit_addendum_2026-11-07/AUDIT_DISCLOSURE.md), recorded there as 2026-11-07 04:18 AEDT. This deposit does not soften that addendum into a clean pass.
- Canonical Arm C adjacency remains the READY rule in [evidence/seals/armC_scores_READY/README.md](evidence/seals/armC_scores_READY/README.md): close↔broad, broad↔partial, partial↔mixed. `mixed`↔`direct_disagreement` is not adjacent.
- Claude CORE pack metadata still says `mixed`↔`direct_disagreement` was adjacent. Recomputation under READY leaves 27/56 exact and 48/56 exact-or-adjacent, because those CORE returns are only close and broad. The metadata was not rewritten.
- Arm B manifest self-hash does not match the manifest bytes. Content hashes in that manifest do match. The file was not repaired. Details: [VALIDATION.md](VALIDATION.md).
- Closeout status remains the sealed summary at [evidence/seals/final_report_2026-11-07/CLOSEOUT_SUMMARY.md](evidence/seals/final_report_2026-11-07/CLOSEOUT_SUMMARY.md): Manus Arm C and the film were not received; Kimi Arm C was not started; Gemini extension was not tested; the user closed the round. Those gaps stay open. They are not entered as failed tests.
- Checker result at deposit: `problems=0` from [verify_evidence.py](verify_evidence.py). Transcript: [validation_output.txt](validation_output.txt).
