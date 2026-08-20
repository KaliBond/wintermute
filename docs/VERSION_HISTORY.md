# Version History

This document exists so a reader arriving at any point in this repository's
history — a specific commit, an old paper, a stale README — can see plainly
what changed, when, and what it superseded, rather than assembling that
picture themselves from inconsistent documents. It draws only on dated
repository evidence (commit dates, dates stated inside documents, or file
content that references a specific date). Where evidence runs out, that is
stated explicitly rather than filled in by inference. Several items below
are flagged for the author to confirm or date.

## What "CAMS" Stands For

The acronym has had more than one public expansion. Repository evidence
shows at least six, used at different times by different documents:

| Expansion | Source | Date |
|---|---|---|
| Common Adaptive Model of Society – Catch-All Network | `canonical/CAMS-CAN-MASTER-REFERENCE.md` (title) | last updated 8 Apr 2026 |
| Complex Adaptive Metrics of Society | `CITATION.cff` (prior version), `AUTHOR.md`, `CONTRIBUTORS.md` | dated 2025-08-15 in CITATION.cff; committed 2026-05-29 |
| Complex Adaptive Management Systems – Catch-All Network | `CAMS_INDEX.md` ("CAMS-CAN v3.4") | committed 2025-08-17 |
| Complex Adaptive Model System | `README.md` (title) | committed 2026-05-17 |
| Complex Adaptive Model of Societies | `analysis/operator_portability_synthesis.md` (title) | committed 2026-07-31 |
| Complex Adaptive Meta-System | `analysis/pr_gap2_manuscript_draft.md` | committed 2026-08-13 (most recent found) |

The most recent operator specification (`juno/JUNO_v1.2-Final_Formalism.md`,
2026-07-06) and the pre-registered `CAMS_JUNO_PreRegistration_v15_Program.docx`
(registered 6 July 2026) both use "CAMS" bare, without spelling out the
acronym at all.

**Resolved 2026-08-20.** None of the six expansions above had been declared
canonical anywhere in the repository — each document simply asserted its
own. Per the author: **Complex Adaptive Model State has always been the
intended canonical name.** The variation across documents did not come from
the author changing her mind about the name; it came from different AI
assistants, across different sessions and documents, independently
renaming the acronym on their own initiative while drafting — described by
the author as "a bit of a joke." The six expansions in the table above are
therefore artefacts of that pattern, not a genuine evolution of the
framework's identity, and are left in the table as historical record rather
than treated as a real naming timeline.

`CITATION.cff` and `.zenodo.json` have been updated to *"CAMS (Complex
Adaptive Model State) / JUNO: A Framework for Societal Coordination
Analysis"* accordingly.

## Version and Formulation Timeline

| Date | Version / Milestone | What changed | What it superseded |
|---|---|---|---|
| Jul 2024 | Conceptual origin | Per `neuralnations.org/cams-project-history` (Apr 2026): CAMS originated "around July 2024," initially in collaboration with GPT, motivated by a Socratic problem (distinguishing truth from persuasion) rather than a research plan. | — |
| 27 Sep 2024 | Repository origin | Per `CAMS-CAN-MASTER-REFERENCE.md`: "collaboration with AI as forcing function for logical rigor." Matches this repository's actual git history (first commit 4 Oct 2024). | — |
| 14 Oct 2024 | First public appearance | *Pearls and Irritations*, "Staving off the collapse of Western civilisation: A personal..." — independently verified externally (johnmenadue.com, dated 14 Oct 2024). | — |
| Sep 2024 | v0.1-draft | Eight-node architecture proposed; four canonical metrics defined; Node Value formula introduced; scored on 8 test societies. | — |
| Jul 2025 | v0.5 | Eight-node canonical architecture confirmed; state-space formalization begun. | — |
| Jul 2025 | RD-002 | Per `CAMS-CAN-MASTER-REFERENCE.md`: eight functional nodes adopted as canonical, under the names **Executive, Army, Knowledge, Property, Trades, Labor, Memory, Commerce** — explicitly superseding an earlier **ten-node** variant. **This conflicts with `cams-project-history`** (below) — flagged, not resolved. | 10-node model |
| Aug 2025 | v0.6 / "CAMS-CAN hypothesis" | Per `cams-project-history`: framework extended to corporate organisations (GM, BYD, Tesla), introducing the Network Synchronisation metric and inhibitory-dominance criterion; separate Boeing case study (1990–2025). "CAMS-CAN" used here as the name of the corporate-extension hypothesis specifically — a third, narrower sense of "CAN" alongside the "Catch-All Network" reading elsewhere. | — |
| Sep 2025 | v0.7 | Per `CAMS-CAN-MASTER-REFERENCE.md`: "CAMS-CAN terminology standardized (eliminated inconsistencies with 'Helm/Executive', etc.)" — read by that document as fixing on "Executive." **But `cams-project-history` (Apr 2026) describes Helm/Shield/Lore/Archive/Stewards/Craft/Hands/Flow as already in place "by the time substantive Claude conversations began,"** with no mention of an Executive-named phase at all, and this repository's own `research-diary.html` already uses Helm/Shield/Lore publicly by Feb 2026. **These two of the author's own reconstructions disagree; not resolved here.** Separately, this same month a thermodynamic/entropy formalization was validated against Norway, Singapore, USA, and China data, and the deliberative-vs-reactive governance-mode distinction was formalised. | Mixed Helm/Executive usage (disputed) |
| Nov 2025 | v0.8 / RD-001 | Stress sign convention fixed: stress is always positive (1–10), never negative. `CAMS-CAN-MASTER-REFERENCE.md` describes this as eliminating signed encoding entirely. **`cams-project-history`'s account of the same month reads differently** — it describes the resolution as clarifying that negative values mean functional stress and positive values mean loss of coherence, i.e. retaining a signed convention. The current live dataset (`JUNO_Unified_Dataset.csv`) uses positive-only (1–10) stress, matching the master-reference account, not the history-page one. | Signed/negative stress encoding |
| Dec 2025 | 16 Jan 2025 entry, corrected | **Resolved 2026-08-20.** `research-diary.html` records the original claim in full: a "Thermodynamic Breakthrough" entry (dated 2025-01-16, attributed to "the NNORG team") asserted societies "definitively exhibit quantifiable neural network properties," mapping institutional nodes to neurons and stress signals to neurotransmitters, citing 83% historical-prediction accuracy, "Inhibitory Dominance" (72% of nodes with stress–coherence r < −0.3), and named numerical thresholds (SPE < 1.5, NS < 0.6, API < 0.1). A December 2025 correction banner on the same entry states the hypothesis "has been falsified" and that "the neuroscience analogy has been abandoned in favor of pure thermodynamic analysis," while stating the underlying empirical findings (the 83% figure, the thresholds, cross-cultural consistency) "remain valid." No external description of the falsification test was found. | Neural-network framing of the model |
| Jan 2026 | v0.9 | Thermodynamic formalization (τ, ε, R) completed; critical-slowing/bifurcation thresholds documented; 32+ society dataset completed. Separately, per `cams-project-history`: a "canonical node-mapping correction" fixed **Shield = military/defence (not welfare institutions)** and **Stewards = landowners/asset management (not welfare state)**, changing the interpretation of Sweden's 1990s banking crisis in the existing dataset. | — |
| 8 Apr 2026 | v1.0-RC1 | `CAMS-CAN-MASTER-REFERENCE.md` declared "single source of truth," status release-candidate. Eight-node model, four metrics, and Node Value formula (`V = C + K − S + 0.5A`) confirmed. Bond Strength denominator and Decay Index weights left explicitly unfinalized. | v0.9 |
| ~Apr 2026 | CAMS v3.2-R (referenced in separate project notes) | **Flagged:** this version label appears in prior working notes as roughly contemporaneous with v1.0-RC1 above; the repository does not make clear whether these are the same milestone under two names or genuinely separate branches. Author to confirm. | — |
| ~Jun 2026 | CAMS v2.4 / "CAMS-CAN v1.0-Final" | New bond-strength formula adopted (`B_ij = √(q_i·q_j)·2^(−(S_i+S_j)/10)`, the "JUNO coupling-quality form"), replacing the v3.2 exponential form (retired "per RD-003" — **RD-003 itself is referenced in code comments but no dated decision record for it was found; flagged**). ESCH/cognitive-activation formula corrected: a blanket clamp that had been silently forcing the metric positive (and disabling one of the crisis-detection triggers) was removed. Laplacian connectivity switched from a normalised form (found to be non-discriminating on this corpus) to the raw form. Calibration of regime thresholds dated 2026-06-09 in code comments. | CAMS v3.2 bond formula; clamped ESCH; normalised Laplacian |
| ~Jun 2026 | Node renaming (undated) | All current code and data (`cams_framework_v2_4.py`, `JUNO_Unified_Dataset.csv`, `generate_site_json.py`) use **Helm, Shield, Lore, Archive, Stewards, Craft, Hands, Flow** as canonical — the reverse of the Sep-2025 standardization above, which had fixed on Executive/Army/Knowledge/etc. **Flagged:** no dated decision record for this second rename was found; it occurred sometime between the 8-Apr-2026 master reference (still "Executive" canonical) and the emergence of the JUNO-1.0 code (already "Helm" canonical). Author to confirm date and rationale. | Executive/Army/Knowledge/Property/Trades/Labor/Memory/Commerce naming |
| 6 Jul 2026 | JUNO v1.2-Final | Regime classifier revised: an unreachable branch (the "s-arm," `s_min ≤ −0.80`) deleted after verification that it never fired across 3,002 blind society-years; the "Strained" band's floor lowered from V̄ ≥ 6 to V̄ ≥ 4, closing a coverage gap (six society-years previously fell through to "Unclassified"). New mandatory numerical policy: aggregate metrics rounded to 6 decimal places before threshold comparisons, to prevent floating-point round-trips from silently changing regime labels. | JUNO v1.1 |
| 6 Jul 2026 | `CAMS_JUNO_PreRegistration_v15_Program` registered | The v14 compound variable `crisis_size = base × 1.5^duration` retired, documented as a "construction failure" — mechanically unstable under chronic-crisis regimes (divergence demonstrated at duration = 46 years). Intensity and duration reported separately going forward. | v14 crisis_size construction |
| Aug 2026 | Section 9 scope statement added (this repository, `blind-test/CAMS_JUNO_PreRegistration_v15_Program.docx`) | Framework's scope explicitly narrowed in writing: "It does not predict the future... every dynamical claim must clear a measured noise floor built from the framework's own refuted results." | Earlier "predicts civilisational resilience or collapse" framing (see below) |

## Where Claims Narrowed

This section exists because the discontinuities are more informative than a
smooth story would be.

- **Predictive → diagnostic.** The framework's own citation metadata (dated
  2025-08-15, now superseded by this release) described it as a tool that
  "predicts civilisational resilience or collapse." The most recent scope
  statement committed to this repository states plainly that the framework
  "does not predict the future," that recovery trajectories have "been
  tested and not demonstrated," and that dynamical claims (as opposed to
  morphological/structural ones) require clearing a noise floor derived
  from the framework's own past refutations. This is a real narrowing, not
  a wording change — the earlier framing asserted a predictive capability;
  the current one explicitly disclaims it pending further evidence.
- **A stated hypothesis was falsified.** The neural-network framing of the
  model (see table above) was tested and abandoned in December 2025, with
  the correction left in place on the original diary entry rather than the
  entry being quietly edited or removed.
- **An overstated public claim was caught and retracted before publication,
  not after.** `neuralnations.org/cams-project-history` (Apr 2026) records
  that draft materials had described a "formal broadcast interview on TIO
  Talks," and states plainly that this is not represented in the published
  history because it overstates what actually occurred (a conversation with
  Warwick Powell, not a produced broadcast). The same document also records
  specific overclaiming phrases removed from draft public-positioning
  material for *The Architecture of Civilisation* — "first operational
  predictive framework," "applied statistical mechanics," "high
  retrodictive prediction" — as a deliberate editorial policy, not a
  one-off correction.
- **A load-bearing formula was found unstable and retired.** The v14
  `crisis_size` compound variable was found to diverge under chronic-crisis
  conditions and was retired rather than patched; intensity and duration
  are now reported as separate, more primitive quantities specifically
  because no construction stands between them and the raw scores.
- **A classifier bug suppressed a crisis-detection trigger for an unknown
  period.** The pre-v2.4 ESCH activation formula's blanket clamp forced the
  metric positive, which is recorded as having disabled the
  `sigma_min ≤ −0.85` Local-Node-Failure trigger entirely until the clamp
  was removed. The repository does not date when the clamp was introduced,
  only when it was found and fixed (~June 2026).
- **A regime-classification branch was deleted after being shown
  unreachable**, and a coverage gap in the "Strained" band was closed —
  both changes verified against the same 3,002-society-year blind corpus
  rather than asserted.

## Current Status of Out-of-Sample / Prospective Validation

As of this document, **no completed prospective (true out-of-sample)
validation exists.** The `CAMS_JUNO_PreRegistration_v15_Program` (registered
6 July 2026) is a pre-registration, not a result: it commits in advance to
falsification criteria for tests that had not yet been run at registration
time, including:

- **Tier 1** (kinematic/shock-magnitude test) and **Tier 2** (corpus-contrast
  test) — both explicitly designed to run on data the framework's existing
  pipelines have not yet touched, specifically to test pipeline- and
  corpus-independence claims that "no result yet demonstrates," in the
  pre-registration's own words.
- **Tier 4b**, a forward registry of specific predictions, is scored
  against public record on **1 July 2031** — i.e. genuine prospective
  testing on this item is deliberately deferred five years and has not
  happened.

Prior blind-identification results (e.g. the 24-society blind experiment
referenced in `juno-v1-2.html` and elsewhere) test the framework's
descriptive/classificatory power on historical data already available to
the scorer, which is a different and weaker claim than prospective
prediction on data that did not exist when the model was specified.

## Items Flagged for the Author

Rather than guess, these are listed here for confirmation:

1. ~~Which CAMS acronym expansion (if any) is now canonical~~ — **resolved
   2026-08-20: "Complex Adaptive Model State."** See table above.
2. Whether "CAMS v3.2-R (April 2026)" and "v1.0-RC1 (8 April 2026)" refer to
   the same milestone or two different ones. **Still open** — the newly
   found `cams-project-history` page doesn't mention either label.
3. ~~The date and rationale of the second node-naming change~~ — **partially
   resolved, but now a different and arguably more interesting problem: two
   of the author's own reconstructed histories directly disagree** about
   whether Helm/Shield/Lore/... was the naming from the outset
   (`cams-project-history`, Apr 2026) or whether Executive/Army/Knowledge/...
   was standardized as canonical in Sep 2025 and only reverted later
   (`CAMS-CAN-MASTER-REFERENCE.md`, Apr 2026 — the same month as the other
   document). Both are dated the same month and contradict each other on a
   basic fact. This isn't resolvable from repository evidence; it needs the
   author's memory, not further searching.
4. A dated decision record for RD-003 (bond-formula retirement), referenced
   in code comments (`juno_backcalc.py`) but not found as a dated entry
   anywhere in the repository, including `cams-project-history`. **Still
   open.**
5. ~~What the "neural network hypothesis" actually claimed~~ — **resolved**:
   full original claim recovered from `research-diary.html`'s 2025-01-16
   entry. See table above.
6. The actual version number and release date for this repository's first
   tagged release, to be filled into `CITATION.cff`, `.zenodo.json`, and
   this document once decided (see the unpublished `v1.0.0` release draft).
7. **New, found during research:** `CAMS-CAN-MASTER-REFERENCE.md`'s Nov-2025
   account of the stress-sign-convention fix (RD-001: positive-only,
   1–10) does not match `cams-project-history`'s account of the same month
   (a signed convention retained, with a clarified meaning). The current
   live dataset uses positive-only, matching RD-001 — but the discrepancy
   between the two histories is itself unresolved.
8. **New, found during research:** the "10-node predecessor" that RD-002
   says the eight-node model superseded (Jul 2025) isn't mentioned at all
   in `cams-project-history`, which describes eight nodes as present "by
   the time substantive Claude conversations began." No date or
   description of the ten-node variant has been found anywhere.
