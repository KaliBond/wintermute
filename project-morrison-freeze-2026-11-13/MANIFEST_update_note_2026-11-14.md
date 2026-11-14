# Manifest update note — 2026-11-14

Two report files were added to this directory on 2026-11-14, in commit
`a6d69aad` on `KaliBond/wintermute` (branch `main`):

- `CAMS_Robodebt_register_full_report_2026-11-14.md` — full nine-section
  register report covering runs 1–3 (2026-11-13 whole-of-government and Helm
  quarterly panels; 2026-11-14 blinded entity series Q01–Q06; 2026-11-14 Q30
  locked comparison set).
- `robodebt_cams_synthesis_2026-11-14.md` — short synthesis piece for the
  essay file.

Their SHA-256 digests are recorded in `MANIFEST_reports_2026-11-14.sha256`.

**Why a new manifest rather than extending `MANIFEST_v1.sha256`:** the existing
manifests carry OpenTimestamps proofs (`.ots`). Appending lines to a manifest
after timestamping invalidates its proof. Under the project's own freeze
discipline, the timestamped manifests are treated as frozen and were not
modified; the new manifest covers exactly the files added after their
timestamping. `MANIFEST_v1.sha256` therefore no longer lists every file in this
directory — that gap is deliberate and this note is its disclosure.

Hashes were computed over the exact bytes in the repository (verified against
the working copies in `C:\Users\julie\kimi_scores\Scoring`).
