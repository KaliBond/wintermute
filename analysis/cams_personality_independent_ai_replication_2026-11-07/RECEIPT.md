# Local receipt

This file records how the evidence reached this repository. It is not a seal created by the original coordinator, and it is not a trusted timestamp.

## Attachment

| Field | Value |
|---|---|
| Filename | `CAMS_AI_replication_final_PR_evidence_2026-11-07_bb74.zip` |
| Path on this agent | `/home/ubuntu/.cursor/projects/workspace/uploads/CAMS_AI_replication_final_PR_evidence_2026-11-07_bb74.zip` |
| Bytes | 306507 |
| File mtime on this VM | 2026-11-06 17:28:09 UTC, which is 2026-11-07 04:28:09 Australia/Sydney (AEDT, UTC+11) |
| SHA-256 requested | `8631465fefa599c3035b4a8c97483394ff8b72d51702de6c5cad74d6442ee7bf` |
| SHA-256 computed before extraction | `8631465fefa599c3035b4a8c97483394ff8b72d51702de6c5cad74d6442ee7bf` |
| Comparison | match |
| Hash command time | 2026-11-06T17:29:21Z |
| Members | 106 files |
| Uncompressed bytes | 1379037 |
| HOLD-key or answer-key filenames inside the zip | none |

The zip mtime is the local filesystem time of the uploaded file. The closeout ledger inside the zip says the round was closed at 2026-11-07 04:25 AEDT, and `PAGE_URL.txt` is stamped 04:26 AEDT in the zip listing. Those are statements in the sealed text, not a claim that this VM’s clock certified them.

## What was copied

The zip was extracted to a temporary directory and copied byte-for-byte into [`evidence/`](evidence/). One sealed JSONL was compared with `cmp` against the temporary extract and matched. `verify_evidence.py` then hashed all 106 files.

No file under `evidence/` was edited. The empty Gemini Arm C extension file remains a zero-byte file. `kimi.zip` remains a zip. Manifests that disagree with themselves were left as received; see [VALIDATION.md](VALIDATION.md).

## Repository comparison

| Field | Value |
|---|---|
| Repository | `KaliBond/wintermute` |
| Base commit before this deposit | `75dc644fd1b01f061f3413cbdffce03f532b644d` |
| Base commit time | 2026-11-06 15:39:30 +1100 |
| Base subject | Fix duplicated clause in best-two scorers note |
| Pre-existing `*armA*`, `*armB*`, `*armC*`, or `*arm_c*` paths | none found |
| Pre-existing copy of Claude Arm A digest `520005f2…` | none found |

The coordinator paths written inside the seals, such as `/workspace/evidence/runs/nations_zeitgeist/personality/ai_replication/`, are paths on the machine that sealed the round. They are not paths in this Git checkout. Nothing already in the checkout was overwritten.

## What this receipt does not establish

- It does not show that a tester computed a hash.
- It does not replace the missing Claude Arm A file whose digest is claimed as `520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7`.
- It does not contain the Arm B or Arm C HOLD keys. Only the key digests recorded by the coordinator are repeated in the report.
- It does not contain the standard tester zip, the Manus special zip, or the Arm B blind zip. [`evidence/DISTRIBUTION.md`](evidence/DISTRIBUTION.md) records their digests. Those zip bytes were not available to recompute.
