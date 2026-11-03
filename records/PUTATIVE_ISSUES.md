# PUTATIVE ISSUES: Lab Chief commentary

**Append-only.** Each entry is dated and signed. Earlier entries are never edited or deleted; a later entry qualifies an earlier one ("we don't edit the past, we qualify it"). Entries are commentary by the Lab Chief (the assistant working for Kari McKern). They are **not** findings by Kimi or any other model scorer, and they do not alter, reword or replace any model's results.

**Standing rule (Kari McKern, 2 Nov 2026):** Kimi's results are not rewritten, recomputed over or paraphrased without Kari's approval, case by case. Commentary goes here instead. Recomputations below are labelled *Lab Chief check*: they are checks run beside the record, not replacements for it. The page that presents Kimi's material unedited is [`kimi-response-to-the-book.html`](../kimi-response-to-the-book.html); the Kimi files are in [`kimi-record/`](../kimi-record/).

Status words: **open** (not resolved), **checked** (a Lab Chief check was run, nothing decided), **noted** (a fact for the record).

---

## LC-001 · 2 Nov 2026, 11:53 AEDT · Society and row counts differ between sources · open

- The book (JUNO v3 as described in it): 42 societies, 41,914 rows. Chapter 16 compares 34 societies (24 declined, stress rose in 26; shared pattern 35%, 29%, 47%).
- The Kimi zip (`JUNO_full.csv`) and `juno/JUNO_Unified_Dataset.csv` (identical, SHA-256 `49ee7b17…c4c`): 48 societies, 5,425 society-years, 43,400 rows.
- Kimi's T-M reproducibility log says extraction covers 48 societies and that 42 is the count after censoring.
- Not reconciled. What would resolve it: a written society-by-society bridge between the two panels.

## LC-002 · 2 Nov 2026, 11:53 AEDT · Cambodia "library floor 0.017" not reproduced · open

- The essay in the 2 Nov zip, and `fig2_two_fates.png`, give a reading of 0.017 for Cambodia in 1975.
- Lab Chief check: in the unified dataset Cambodia's mean node value is −0.363 (1973–74), −4.412 (1975), −4.275 (1976–78, identical values each year), +0.925 (1979). Bond strength 0.108 to 0.058; λ2 0.732 to 0.352; both recover in 1979. No 0.017 found among them. The quantity behind 0.017 may be one the Lab Chief did not identify.
- Also noted: identical values across 1976–78 indicate step or carry-forward structure in the annual series.
- What would resolve it: Kimi's (or Kari's) statement of which quantity 0.017 is and where it comes from.

## LC-003 · 2 Nov 2026, 11:53 AEDT · Syria series ends in 2024 · noted

- The essay lists Syria among the societies that declined. In the unified dataset the Syria series ends in 2024, so it is outside a 2000-to-2025 comparison on that file.
- The Lab Chief 2000→2025 check counted 38 societies with readings in both years.

## LC-004 · 2 Nov 2026, 11:53 AEDT · The book's US Helm run cannot be reproduced from the unified dataset · open

- The book reports Helm as the weakest function for 11 years running since 2016, in a 127-year US series.
- The unified dataset holds only 26 US society-years, so that run cannot be recomputed from it. The Lab Chief has not located the series the book used within this repository for this check.
- What would resolve it: pointing to the file and hash of the US series behind the book's figures.

## LC-005 · 2 Nov 2026, 11:53 AEDT · RUN-KIMI-BLIND-1: model version not recorded; A/B isolation unconfirmed · noted

- From the import note (`backstory/results/DT2_adoption_and_Kimi_blind_2026-10-02.md`): the model version was not recorded; prompts A and B came back together, so isolation between them is not confirmed. One pass; DT-1 rubric; seal hashes verified on import.
- Consequence for reading: the run is a reading, not an ensemble (the note's own wording). A repeat with the version recorded is listed in the note's next steps and awaits Kari's go.

## LC-006 · 2 Nov 2026, 11:53 AEDT · T-M3e "32,361 unmatched" count not reconciled · open

- The T-M3e lab note records a count of 32,361 unmatched records. The Lab Chief could not reconcile it with the 40,472 node-years used in the T-M3e panel.
- What would resolve it: Kimi's or Kari's account of what was unmatched against what.

## LC-007 · 2 Nov 2026, 11:53 AEDT · Blind set 4 missing from the repository · open

- T-M4 uses six blind sets. Set 4 is absent from the repository; no reason is documented.
- What would resolve it: the file, or a note of why it was withheld.

## LC-008 · 2 Nov 2026, 11:53 AEDT · Lab Chief checks on the shared dataset · checked

Run by [`records/lab-chief-checks/recompute.py`](lab-chief-checks/recompute.py) on `juno/JUNO_Unified_Dataset.csv`; output in [`recompute-output.txt`](lab-chief-checks/recompute-output.txt); file hashes in [`SHA256SUMS.txt`](lab-chief-checks/SHA256SUMS.txt). These are checks, not replacements for Kimi's or the book's numbers.

| Quantity | Lab Chief check | Kimi essay / figure | Book |
| --- | --- | --- | --- |
| Rows, societies, society-years | 43,400; 48; 5,425 | same | 41,914; 42 |
| Societies with 2000 and 2025 readings | 38 | fig3: 22 of 38 declined | chapter 16: 24 of 34 declined |
| Societies where mean stress rose 2000→2025 | 25 | not stated | 26 (of 34) |
| Shared share of stress movement, eras 1900–49 / 1950–99 / 2000–25 | 34.1% / 30.7% / 41.4% (balanced panels of 22 / 27 / 36 societies; first principal component of z-scored yearly mean stress) | not stated | 35% / 29% / 47% |
| Bond strength vs λ2: pooled r; median within-society r | 0.935; 0.981 | fig4: 0.93; 0.98 | not checked |
| Node responsiveness, leave-self-out (Flow, Helm, Stewards, Craft, Hands, Archive, Shield, Lore) | 1.24, 1.18, 0.93, 0.91, 0.84, 0.82, 0.78, 0.74 | fig4 matches; essay text says Lore "about 0.75" | not checked |
| Share of year-to-year node movement that is one shared direction | 0.688 | essay: about seven tenths | not checked |
| US, 2026 mean node value vs 1900–1999 (20 five-yearly points; mean 10.78, SD 2.64) | 3.55; z = −2.73 | essay figure consistent | not comparable (see LC-004) |

Notes on the table:
- The with-self version of responsiveness gives 0.825 / 1.278 / 1.231 for Lore / Flow / Helm; the leave-self-out version matches Kimi's fig4 exactly.
- The shared-share and 2000→2025 results have the same direction and ordering as the book but smaller magnitude. The cause was not isolated (different society mix and year coverage are candidates only).
- The r between bond strength and λ2 may be partly by construction if both derive from the same pairwise bonds. The Lab Chief has not audited the formulas.
- Charts from the first draft of the page are kept in `lab-chief-checks/` (`kimi-common-pattern-by-era.png`, `kimi-usa-own-century.png`, `kimi-boeing-pass.png`). They are Lab Chief charts, not Kimi's.
- Boeing, computing V = C + K − S + 0.5A from Kimi's scores: mean V by year 2018–2025: 11.44, 2.47, 3.84, 6.44, 8.48, 8.85, 0.47, 6.12; mean stress 4.61, 7.88, 7.46, 5.91, 4.97, 5.05, 8.46, 6.01; Shield −1.0 in both 2019 and 2024. One entity, one pass, no matched control, and the scorer knew the firm's history. This is not the controlled firm test the book describes.

## LC-009 · 2 Nov 2026, 11:53 AEDT · Points where Kimi's material bears on the book's own limits · open

Observations only; none changes any result.
- Kimi's 7 Jun review (reporting the project's falsification battery) says the four metrics collapse to one or two effective dimensions, and that Lore and Archive are the strongest genuine redundancy. Chapter 17 of the book names "two readings are really one" as a way it could be wrong. Not resolved.
- The same review says the bond-strength formula was unresolved at 7 Jun. The Lab Chief has not rechecked whether the current version closes that.
- T-M3 (Kimi, 16 Oct) says the earlier dominance of Hands as failure locus was largely a level artefact. The book's statement that Hands is the weakest function in every era is about level, so is not touched by that finding; naming Hands as where an episode breaks is.
- RUN-KIMI-BLIND-1 H5 ("no ladder") fails under Kimi even after oral evidence is accepted in full. The import note itself says a single pass cannot separate a genuine judgement from the progress-ladder bias the project set out to detect.
- T-M4 blind sets: three of six sit below the preregistered 0.20 bar (set-level range 0.135 to 0.481), so "replicates" holds for the pooled result, not each set.
- The book's Boeing, Theranos and WorldCom tests are described as under way against matched controls; Kimi's Boeing pass of 10 Aug is not that test (see LC-008).

## LC-010 · 2 Nov 2026, 11:53 AEDT · Banned vocabulary inside verbatim Kimi files · noted

- `The_Eight_Jobs.md` (2 Nov zip) uses "healthy" once; `validation_kimi_2026-06-07.txt` uses "health" in several places. Both are shown unedited on the page and flagged there. No other Kimi file shown on the page contains the project's banned words (searched: health, legitim, democratic capacity).
- The book and a July Kimi memo in Drive contain the same words; those are not reproduced on the page.

## LC-011 · 2 Nov 2026, 11:53 AEDT · Correction to the first draft of the Kimi page (PR #35) · noted

- The first draft of the page (commit `1247dd2`) was written in Kimi's voice, reworded Kimi's conclusions and mixed Lab Chief recomputations into the text. That breached the standing rule above. It is replaced in this revision; the earlier text remains in git history only.
- The first draft's figures were also never committed (the repository's `.gitignore` excludes `*.png`), so its images would have been missing. The present revision force-adds the figures it uses.
- The page remains a draft for Kari's approval.

---

*Append new entries below this line. Do not edit entries above it.*

### LC-012 (2 Nov 2026): legacy USA 1861 CI test
The CI step "Run USA 1861 Validation" asserted fixed CAMS v2.1 values (Psi 0.724) and got 0.711, failing every run including main. Possibly explained by data or model changes since v2.1. Resolution: Kari retired the test on 2 Nov 2026 (separate CI PR).
