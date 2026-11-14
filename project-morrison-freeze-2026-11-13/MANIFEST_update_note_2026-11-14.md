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

**OpenTimestamps proof.** `MANIFEST_reports_2026-11-14.sha256.ots.b64` is the
OpenTimestamps proof for this manifest, stamped 2026-11-14 with pending
attestations from four default calendars (alice/bob.btc.calendar.opentimestamps.org,
btc.calendar.catallaxy.com, finney.calendar.eternitywall.com). It is stored
base64-encoded because the transport used to write to this repository is
text-only and cannot carry the proof's binary bytes (which begin with a null
byte). Before use, decode it:

    base64 -d MANIFEST_reports_2026-11-14.sha256.ots.b64 > MANIFEST_reports_2026-11-14.sha256.ots
    ots verify MANIFEST_reports_2026-11-14.sha256.ots   # after ots upgrade confirms it

This is a known deviation from the sibling manifests, whose `.ots` files are
raw binary; the deviation is recorded here rather than papered over. The binary
proof also exists locally at
`C:\Users\julie\kimi_scores\Scoring\MANIFEST_reports_2026-11-14.sha256.ots`
(735 bytes, verified to parse and to match the manifest digest).

Hashes were computed over the exact bytes in the repository (verified against
the working copies in `C:\Users\julie\kimi_scores\Scoring`).
