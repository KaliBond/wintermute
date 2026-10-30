# CAMS Human–AI Scientific Record (30 Oct 2026)

## What this is

A curated research record of one visible conversation between Kari McKern and a GPT assistant about the development and purpose of CAMS (The Dynamics of Organisations), packaged as a ZIP with a SHA-256 checksum file and a one-page integrity certificate. The ZIP holds a summary, a topical chronology, selected first-person excerpts, a methods note, a provenance note, six source attachments (blind-rater CSVs for the 25-case, 200-node-assignment repeatability test, and two USA 1900 to 2026 marker files) and a manifest.

It is not a platform export of the full 30-month collaboration, and the hash is not a digital signature or trusted timestamp. The archive's own `05_PROVENANCE_AND_VERIFICATION.md` and the certificate say so.

Also here: `EXTRACT_THIS_IS_WHY_IT_WORKS_GPT_REFLECTION.md`, a GPT reflection Kari pasted on 31 Oct 2026 (CAMS as a theory of cooperative adaptation, with the eight JUNO nodes Helm, Shield, Lore, Stewards, Craft, Hands, Archive and Flow as functional requirements). It is saved verbatim, with no edits or added header.

## Provenance

- The four archive files came from Kari's Downloads folder on her PC on 31 Oct 2026 and were copied to the Grok Bot box unaltered. They are byte-identical to the copies committed to `main` on 30 Oct 2026 (commit `4406bd7`, which also added the `human-ai-scientific-record.html` page). This PR does not re-add them.
- The extract was pasted by Kari in chat on 31 Oct 2026. Its platform origin is stated by Kari and is not independently verified (see `VALIDATION.md`, section 4).

## Hash verification (31 Oct 2026, Sydney)

Computed SHA-256 of the ZIP equals the `.sha256` file, the certificate text and the QR payload: **match**. `VERIFY_CONTENTS.py` printed PASS, and all 13 manifest members re-hashed equal. Details are in `VALIDATION.md`.

## Files in this folder

| File | SHA-256 | In this PR |
|---|---|---|
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30.zip` | `ac0018a9edf8a0afe547f0a29a5cc8580e4cbaa582125e7c121743527684f9dd` | already on main |
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30.sha256` | `45b902191dc668b9d06ce3c55dad1636fa3143dae34a24111f8f4e97d1de52a7` | already on main |
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30_INTEGRITY_CERTIFICATE.pdf` | `da5b03f260aaca5971ec08df2e3a517bfce7e4bc6269a2605f25aaa97d2f044e` | already on main |
| `CAMS_THREAD_SCIENTIFIC_SUMMARY_2026-10-30.md` | `89cfb4327bd9c3483d9857c4f34cbd0a8e59aaf311d8c9521c29b03d6ee4fc5a` | already on main |
| `EXTRACT_THIS_IS_WHY_IT_WORKS_GPT_REFLECTION.md` | `c5247a63deca6222868373aabbab0f4e18395d16fdfc8ccfdf2f59155d7d65a7` | new |
| `VALIDATION.md` | `9f40c4bc4789b2d28a150d5376b65a418cb253634e92bf4dcb4a126be0eb60e4` | new |
| `README.md` | not listed (a file cannot contain its own hash) | new |

## Notes

- `VALIDATION.md` is an attestation by Grok (Lab Chief). Grok is also one of the three raters in the record, so the checks there are file comparisons and arithmetic that can be rerun.
- The record is a public page. `05` recommends reviewing personal disclosures before public release.
