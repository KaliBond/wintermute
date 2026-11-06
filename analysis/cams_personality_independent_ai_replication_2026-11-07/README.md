# CAMS personality independent AI replication, 6–7 November 2026

Full report, sealed evidence, and the checks that tie the report’s figures to those seals.

This folder is a documentation deposit. It does not change the telescope, the scorer, or any application. It does not rerun models or rescore against an answer key. The standalone HOLD keys are not in the attachment and are not reconstructed here.

| Read this | For |
|---|---|
| [CAMS_Personality_Independent_AI_Replication_Report_2026-11-07.md](CAMS_Personality_Independent_AI_Replication_Report_2026-11-07.md) | The full report: design, methods, Arms A–C, audit limits, and the calibrated conclusion |
| [RECEIPT.md](RECEIPT.md) | How this copy was received, hashed, and compared with the repository |
| [VALIDATION.md](VALIDATION.md) | What was rechecked, including the one unrepaired manifest self-hash |
| [AUDIT_HISTORY.md](AUDIT_HISTORY.md) | Append-only history. Add a later entry; do not edit an earlier one or a seal |
| [evidence/](evidence/) | The 106 attachment files, byte-for-byte |
| [tables/](tables/) | Score, confusion, non-exact, and coverage tables recomputed from the scored rows |
| [EVIDENCE_MANIFEST.sha256](EVIDENCE_MANIFEST.sha256) | SHA-256 of every evidence file |
| [verify_evidence.py](verify_evidence.py) | Re-runs the hash and row checks. It does not call a model |

The short closeout page is [the existing canvas](https://cursor.com/canvas/store-695134c2-2dae-4c46-9961-3a4ff756e323/6a69bfb0-dc94-4c08-a3db-8962b63c3616). The report in this folder is the record. The canvas is not a substitute for it.

Re-check the deposit from this folder:

```bash
python3 verify_evidence.py
```

A clean check prints `problems=0` and exits 0. It also rewrites `tables/`, `validation_output.txt`, `reconciliation_summary.json`, and `EVIDENCE_MANIFEST.sha256`. Those rewrites match the committed files when the evidence bytes are unchanged.
