# Validation statement: CAMS Human–AI Scientific Record (30 Oct 2026)

**Author:** Grok (Lab Chief), working for Kari McKern
**Date of checks:** 31 October 2026, Sydney (UTC+11)
**Scope:** integrity of the archive, internal consistency of its documents, and a cross-check of its numbers against results computed on the Grok Bot analysis box. Nothing here is a statement that the conversation took place on a particular platform (see section 4).

**Conflict of interest, stated first.** Grok is one of the three raters whose scores appear in the record (the loose `BLIND_RESPONSE_FILLED_200.csv`, labelled Grok). The checks below are arithmetic and file comparisons that anyone can rerun. They do not rely on my judgement of my own ratings.

## 1. Hash verification

**Method.** The four delivered files were copied unaltered to the box. The ZIP's SHA-256 was computed with `sha256sum` and compared with the `.sha256` file, the certificate text and the certificate QR (decoded from a 200 dpi render of the PDF with OpenCV's QR detector). The archive's own `VERIFY_CONTENTS.py` was run with the external hash. Every member was then re-hashed from the ZIP and compared with `MANIFEST_SHA256.json`.

**Result: match.**

| Source | SHA-256 of the ZIP |
|---|---|
| Computed from the ZIP | `ac0018a9edf8a0afe547f0a29a5cc8580e4cbaa582125e7c121743527684f9dd` |
| `.sha256` file | same |
| Certificate text (PDF) | same |
| Certificate QR payload | `sha256:` followed by the same value |

`VERIFY_CONTENTS.py ARCHIVE.zip <hash>` printed `PASS: internal manifest and ZIP archive integrity verified` (exit code 0). The 14-entry ZIP has no member missing from or added to the manifest. The ZIP's own test (`testzip`) found no corrupt member.

The four delivered files are also byte-identical to the copies committed to `main` on 30 Oct 2026 (commit `4406bd7`):

| File | Bytes | SHA-256 |
|---|---:|---|
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30.zip` | 66496 | `ac0018a9edf8a0afe547f0a29a5cc8580e4cbaa582125e7c121743527684f9dd` |
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30.sha256` | 113 | `45b902191dc668b9d06ce3c55dad1636fa3143dae34a24111f8f4e97d1de52a7` |
| `CAMS_HUMAN_AI_SCIENTIFIC_RECORD_2026-10-30_INTEGRITY_CERTIFICATE.pdf` | 39942 | `da5b03f260aaca5971ec08df2e3a517bfce7e4bc6269a2605f25aaa97d2f044e` |
| `CAMS_THREAD_SCIENTIFIC_SUMMARY_2026-10-30.md` | 7771 | `89cfb4327bd9c3483d9857c4f34cbd0a8e59aaf311d8c9521c29b03d6ee4fc5a` |

Members of the ZIP (recomputed, all equal to the manifest):

| Member | Bytes | SHA-256 | Manifest |
|---|---:|---|---|
| `00_README_FIRST.md` | 1276 | `186ae72afaaff598d588954ea17342d063c21417419e0bbc146a263ee84c3b81` | match |
| `01_SCIENTIFIC_RECORD.md` | 7771 | `89cfb4327bd9c3483d9857c4f34cbd0a8e59aaf311d8c9521c29b03d6ee4fc5a` | match |
| `02_THREAD_CHRONOLOGY.md` | 2930 | `7c2a6d72261b35b2ca51264c204ac45c50866421c932dd7008dd169e90c3db75` | match |
| `03_SELECTED_PRIMARY_EXCERPTS.md` | 3608 | `f64cf0759fde2443a4912c21e5c118b78ddb677d7136baea15aaa49730724728` | match |
| `04_REPRODUCIBILITY_METHODS.md` | 1963 | `7ef3463ac15e764e7422c80902d43462e9b3a78c664f8002932c6c40f9a3e5e4` | match |
| `05_PROVENANCE_AND_VERIFICATION.md` | 2687 | `2f9b91bcbb5731e2209657ea7cbc4fd4fc4d01ff82a6e7f3e05f8341d4514929` | match |
| `VERIFY_CONTENTS.py` | 1450 | `0ffa1545e2e6f157b10ad358fc210430c472f4a7c75160999f1d340c5293e632` | match |
| `source_attachments/BLIND_RESPONSE_FILLED_200.csv` | 22578 | `19f6df66e34a699ef25f50668657ecd073f64b567c4221d4f8ce5960515fd81a` | match |
| `source_attachments/BLIND_RESPONSE_FROZEN_200.csv` | 32208 | `cc2d67ab07e3a166359d6b8263236460aa1906ae979e559fa08e0e6950cf8a0b` | match |
| `source_attachments/MARKER_USA_1900_2026_ENSEMBLE_MEAN (5).csv` | 43666 | `2c2e2b8d3ad82f0074b22aea201e1733393bfb4dffe93e174c9a8d9c00f591cd` | match |
| `source_attachments/MARKER_USA_1900_2026_ENVELOPE (6).csv` | 44787 | `a4368ddbcb2dba94f5e03cc55c5d93eccc74d626ab7e3a1bc6c3cf47a344bdce` | match |
| `source_attachments/files (27).zip` | 12773 | `24235e77e6a4eaaf612df8d76d6acbb86ab0d114f0ebab745d01a31e4dffd9c5` | match |
| `source_attachments/files (28).zip` | 12773 | `f2788bdff2fad834d4943d976491d6deead53556dbe12e7e278516d2ae890de2` | match |
| `MANIFEST_SHA256.json` | 2239 | not listed in itself | n/a |

A matching hash shows these bytes are the bytes that were hashed. It does not show who wrote them or when (the certificate says the same).

## 2. Internal consistency

Agree:
- **Filename and hash.** Certificate, `.sha256`, QR, manifest-based verification and the ZIP name all agree. The certificate and archive are dated 30 Oct 2026. ZIP member timestamps read 2026-10-30 05:40. If that clock is UTC it is 16:40 in Sydney, which sits just before the 16:50 commit of these files to `main`. I have not confirmed the clock zone.
- **Summary equals record.** `CAMS_THREAD_SCIENTIFIC_SUMMARY_2026-10-30.md` is byte-identical to `01_SCIENTIFIC_RECORD.md` inside the ZIP (same SHA-256).
- **Scope statements agree.** Certificate, `00`, `04` and `05` say the same thing: a curated record of the visible conversation, not a platform export of 30 months, and a hash is not a signature or trusted timestamp.
- **Attachment count.** `00` says six source attachments; `source_attachments/` holds six files.
- **Two Claude ZIPs.** `01` and `04` say `files (27).zip` and `files (28).zip` hold the same Claude rating text. Inside both, `BLIND_RESPONSE_FILLED_200.csv` and `RATER_DECLARATION.md` are byte-identical; only the outer ZIPs differ (the two nested ZIPs are stamped 00:41 and 00:56 on 2026-10-29 inside the archive).
- **Claude version.** `01` says "Claude Opus 5.5", which is what `RATER_DECLARATION.md` states.
- **Grok attribution.** `04` says the Grok CSV is labelled per the researcher's statement with no declaration inside it. That matches the file.
- **Rounding.** `01` gives the correlations to three places (0.931, 0.889, 0.896; Spearman 0.932, 0.888, 0.899) and `04` gives six; they agree once rounded.

Do not agree, or are loose:
1. `00_README_FIRST.md` lists files as `01_SUMMARY.md`, `03_PRIMARY_EXCERPTS.md` and `04_METHODS.md`. The ZIP contains `01_SCIENTIFIC_RECORD.md`, `03_SELECTED_PRIMARY_EXCERPTS.md` and `04_REPRODUCIBILITY_METHODS.md`. The manifest uses the real names, so this is a labelling slip, not a content change.
2. `05_PROVENANCE_AND_VERIFICATION.md` says the QR records the ZIP's name and digest. The QR payload decodes to the digest only (`sha256:<hex>`). The name is printed on the certificate beside it.
3. `01` calls the first response "unlabelled frozen". On the box (PR #26) it is labelled GPT. This is a naming difference, not a numerical one.
4. The same file name, `BLIND_RESPONSE_FILLED_200.csv`, is used for the Grok file (top level) and the Claude file (inside the nested ZIPs). The contents differ (different SHA-256s above), so the name alone does not identify the rater.

`03` contains first-person quotations from Kari, and `05` itself advises reviewing personal disclosures before making the archive public. The folder is in a public repository.

## 3. Cross-check against results verified on the box

Rater files. The three scoring CSVs in the ZIP are byte-identical to the files in PR #26 (`analysis/blind_rater_accidental_test/data/`, merged):

| ZIP file | PR #26 file | SHA-256 | Match |
|---|---|---|---|
| `source_attachments/BLIND_RESPONSE_FROZEN_200.csv` | `BLIND_RESPONSE_FROZEN_GPT_200.csv` | `cc2d67ab07e3a166359d6b8263236460aa1906ae979e559fa08e0e6950cf8a0b` | yes |
| `source_attachments/BLIND_RESPONSE_FILLED_200.csv` | `BLIND_RESPONSE_GROK_200.csv` | `19f6df66e34a699ef25f50668657ecd073f64b567c4221d4f8ce5960515fd81a` | yes |
| `files (27).zip` and `files (28).zip` inner `BLIND_RESPONSE_FILLED_200.csv` | `BLIND_RESPONSE_CLAUDE_200.csv` | `af6caee554144183cc9e84640ebe4aca9ea2167d5e327e0fc61766d9b60e2a31` | yes |
| inner `RATER_DECLARATION.md` | `RATER_DECLARATION.md` | `aa3504db21e6aa642571312caa8f869c11bec4c8e78a237fb3a12332fac6a579` | yes |

Quantitative claims in `01` and `04`. I recomputed Q = Abstraction × Coherence + Capacity − Stress for each rater from the ZIP's files (not from the record's text) and compared pairwise on rows where both raters have a value:

| Claim in record | Record | Recomputed from ZIP files | PR #26 `output.txt` | Result |
|---|---|---|---|---|
| Frozen vs Claude, n | 175 | 175 | 175 | match |
| Frozen vs Claude, Pearson | 0.930788 | 0.930788 | 0.9308 | match |
| Frozen vs Claude, Spearman | 0.932066 | 0.932066 | 0.9321 | match |
| Frozen vs Grok, n | 200 | 200 | 200 | match |
| Frozen vs Grok, Pearson | 0.889473 | 0.889473 | 0.8895 | match |
| Frozen vs Grok, Spearman | 0.888487 | 0.888487 | not reported on n = 200 | match with my recomputation |
| Claude vs Grok, n | 175 | 175 | 175 | match |
| Claude vs Grok, Pearson | 0.896246 | 0.896246 | 0.8962 | match |
| Claude vs Grok, Spearman | 0.898634 | 0.898634 | 0.8986 | match |
| Three-way ICC(A,1), 175 complete rows | 0.869021 | 0.869021 | 0.8690 | match |
| 25 cases, 200 node assignments | 25 × 8 | 25 case IDs, 200 rows per rater | same | match |

The three-place figures in `01` (0.931 / 0.889 / 0.896) are the rounded Pearson values above. PR #26 also reports a bootstrap 95% interval for ICC(A,1) of 0.833 to 0.897; the record does not quote an interval.

USA ensemble inputs.
- `MARKER_USA_1900_2026_ENSEMBLE_MEAN (5).csv` has 1016 rows (127 years × 8 nodes, 1900 to 2026) and the same columns as the annual file analysed in PR #27 (Studies 1 and 2). Merged on Year and Node, all 1016 rows are numerically identical to the box copy `USA_ENS_annual.csv` in Coherence, Capacity, Stress, Abstraction, Node Value and Bond Strength (maximum absolute difference 0). The files are not byte-identical (the archived file is ordered node by node within each year, the box copy runs year by year within each node, and the archived file has no final newline), so the md5 recorded for the box copy (`1c9626e9481662df6c00974f720e0ec0`) does not equal the archived file's md5 (`61cfd8768517338e8ac0941b5894f0ae`). This is the first value-level corroboration that the box reconstruction used in PR #27 matches a file Kari supplied. I did not rerun Studies 1 and 2 on the archived copy.
- `MARKER_USA_1900_2026_ENVELOPE (6).csv` has 1008 data rows. Years 1927 and 1928 are absent and 1907 appears twice (16 rows, out of sequence). These are the data-quality findings in the box note `ENV_note.md` written when the file was first read (1008 rows, 126 year-blocks, 1927 and 1928 missing, 1907 twice), so this is the same file.
- `04` says Q is distinct from the older Node Value formula. On the archived file, Node Value equals round(C + K + A/2 − S, 1) in all 1016 rows (maximum difference 0), so Q is a different quantity.

PR #27 results and preregistration hashes. The record does not report any Study 1 or Study 2 result, and its text says the USA historical comparison remains to be done. So there is nothing in it to match or contradict against PR #27. For reference, the preregistration hashes on the box today are unchanged: Study 1 `3603e474ada3652e2d665e8787ca9c4bb08ba3e38bd23a8614e69611c3aea2fe`, Study 2 `38dcbe910d40dfff1f7a4a7bdf9c7b73892766c27e6729065aa027a34a5e04c5`. The record does not cite them.

Mismatches in quantitative claims: none found.

Consistent, not independently recounted: `04` says some of the 25 bins are pre-1900 and have no observations in the annual ensemble. The Study 1 preregistration likewise puts 10 of the 25 periods out of scope (they end before the series starts) and tests 15. I did not recount the periods from the casebook for this statement.

## 4. What cannot be verified from here

- **That the conversation occurred, with whom, or when.** Nothing in the ZIP is a platform export. The assistant that compiled the record and its hash is the same system whose conversation it records. The certificate says the checksum "is not notarised by an independent authority".
- **That the quoted user messages in `03` are verbatim**, or that the assistant text in the extract (`EXTRACT_THIS_IS_WHY_IT_WORKS_GPT_REFLECTION.md`) was produced by a GPT model on OpenAI's servers. I only have the text Kari pasted. Verifying that needs a ChatGPT share link to the thread or the account's data export (`conversations.json`), kept unedited and hashed separately.
- **The 30-month collaboration.** Participant-reported. Nothing in the files documents the 2024 to 2026 span.
- **Authorship and signing.** There is no digital signature by Kari and no independent time anchor (for example RFC 3161 or OpenTimestamps). Either would show the archive existed unchanged at a given time. Neither is present.
- **The rater attributions.** That the Claude file came from Claude Opus 5.5 rests on its declaration; that the third file came from Grok rests on Kari's statement. The independence limits listed in the Claude declaration (recognisable cues, a visible "USA" filename) apply.
- **Interpretation.** Agreement between raters on a shared casebook supports repeatability on that casebook. It does not by itself show that the CAMS reading of history is correct.

## Summary

The ZIP matches its published hash, its certificate, its QR and its internal manifest. The documents agree with one another apart from the file-name slips in section 2. Every number in the record that can be recomputed from the attached files matches the box results to the digits reported. What this establishes is the integrity of these bytes and the arithmetic. It does not establish where or when the exchange took place; that needs the platform export or a share link.
