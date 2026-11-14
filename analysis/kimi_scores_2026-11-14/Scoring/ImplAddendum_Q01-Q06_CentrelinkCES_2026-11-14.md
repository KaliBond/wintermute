# Implementation Addendum — Q01–Q06 / Q05 Tests, Entity_A & Entity_B Series

**Status:** FROZEN BEFORE SCORING. SHA-256 of this file recorded in the scoring protocol note *after* scoring; scoring rationale must not reference this hash or post-freeze edits. Any change after the hash invalidates the run.
**Date frozen:** 2026-11-14, before any Entity_A/Entity_B/Helm-extension/Archive-extension score was written.
**Register:** "Robodebt and Helm Failure: Pre-analysis Question Register v1" (Perplexity session bf0c2ebd, compiled 2026-11-13).

---

## 1. Entities and node mapping

| Masked label | True entity | CAMS node scored | Justification |
|---|---|---|---|
| Entity_A | Centrelink, administered by the Department of Human Services (renamed Services Australia 2019–20) — national social-security delivery agency | Stewards (agency resolution) | The agency is the delivery organ of the population-maintenance function. Agency-resolution Stewards is distinct from national Stewards and must not be merged with it. |
| Entity_B | Australian Taxation Office — national revenue collection and fiscal-records agency | Archive (agency resolution) | Custodian of the taxation record; analogous compliance/data-matching role in a different portfolio. Control agency for Q05. |

Existing panels used as-is (built 2026-11-13, unblinded): Australian Government Helm 2018Q1–2023Q4; Australian Government Archive 2018Q1–2023Q4; full 8-node panel 2018Q1–2023Q4. Extensions built in this run: Helm 2013Q1–2017Q4; Archive 2013Q1–2017Q4, same protocol as the 2018–2023 panels.

## 2. Blinding statement (read before scoring)

The scoring analyst is a single agent, already fully informed of the historical outcomes (Royal Commission 2023, NACC 2026, class action 2021). Personal outcome blinding is therefore **unattainable and not claimed**. Three disciplines substitute, each weaker than true blinding and labelled as such:

1. **Masked labels.** Entity_A and Entity_B appear masked in all CSVs so the panels are reusable for genuine third-party blind replication. The unmasking key lives only in the protocol note.
2. **Contemporaneous-source discipline.** Each quarter is scored only on evidence published or occurring on or before that quarter's end. Later findings (court outcomes, royal commissions, audits) are excluded from scoring rationale; they enter only at test time, where the register requires independently dated warnings.
3. **Freeze-before-access.** This addendum (tests, event dates, decision rules) is hashed before the score matrix is written. No test definition may be altered after the scores are seen.

## 3. Frozen event dates (external, independently dated)

| ID | Date | Event | Source basis |
|---|---|---|---|
| E1 | 2015-05-12 | Budget 2015–16 announces "Strengthening the Integrity of Welfare Payments" ($240.4m; net savings $1.7b/5yr); data-matching of welfare vs tax income | Whiteford (2021), AJPA; Clarke & Abbas (2024), AJIS |
| E2 | 2015-07-01 | Interim manual compliance process commences (100,000 discrepancies/yr; income averaging applied manually) | Ombudsman report 02/2017 (April 2017); AJIS |
| E3 | 2016-07-01 / 2016-09-01 | Online Compliance Intervention (OCI) launch, limited then full rollout; ~20,000 interventions/week by late 2016 | Ombudsman 2017; Senate Community Affairs Committee 2020 |
| E4 | 2016-12 | First sustained adverse media; minister publicly announces automated scheme; phone-contacts crisis over 2016–17 summer | Whiteford 2021; Greens/CSMC call-centre figures 2017 |
| W1 | 2017-04-19 | Commonwealth Ombudsman report No. 02/2017 on Centrelink's automated debt raising and recovery system | Ombudsman 2017 |
| W2 | 2017-06-21 | Senate Community Affairs References Committee first report recommends suspension; government does not suspend | Parliament of Australia 2017 |
| W3 | 2017-04 – 2017-09 | Five AAT judgments (Carney) holding income-averaged debts not enforceable on annual income alone | Whiteford 2021 |
| W4 | 2019-02 | First successful court challenge to an income-averaged debt | AJIS timeline |
| W5 | 2019-11-19 | Scheme withdrawal: income-averaged debt raising ceases (announced with the decisive Federal Court challenge) | Services Australia; AJIS |
| W6 | 2020-05 / 2020-07 | Government accepts method was unlawful (May 2020); refund repayments commence (July 2020) | AJIS; Services Australia |
| C1 | 2020-03-23 – 2020-05 | COVID shock: 1.3m JobSeeker claims in 55 days; myGov record load 25 Mar; ATO pays JobKeeper from 30 Mar | Senate Job Security Cttee 2021; APSC State of Service 2019–20; ANAO JobKeeper audit |
| W7 | 2021-06 | Class action settled; Commonwealth settles for $112m incl. costs | Federal Court / session record |
| W8 | 2022-08-18 | Royal Commission into the Robodebt Scheme established | robodebt.royalcommission.gov.au |
| W9 | 2023-07-07 | Royal Commission final report: scheme "a crude and cruel mechanism, neither fair nor legal"; 57 recommendations, 56 accepted in full or in principle | RC report 2023 |

Control agency (Entity_B) frozen dates:
| B1 | 2017-04 | Garnishee-notice treatment of small-business tax debts reported; small-business ombudsman review | ATO 2017 controversy, contemporaneous reporting |
| B2 | 2020-03 – 2020-04 | ATO administers JobKeeper (announced 30 Mar) and early superannuation release; 97% of payments within four business days; ~2.6% net payment gap ($1.8b of $70b original period) | ANAO Administration of JobKeeper (2025 audit of scheme) |

## 4. Frozen tests

**Q01 — Temporal precedence.** Signature: Entity_A Capacity (recipient-facing functioning). Decision rule: E1 (2015-05-12) is the executive change; its operational commencement is E2 (2015-07-01, 2015Q3). Prediction (register): executive signature leads or is concurrent with agency deterioration. Consistent: first Entity_A K ≤ 4 occurs at 2015Q3–Q4 and persists. Weakening: first K ≤ 4 occurs ≥ 2 quarters before 2015Q3, or chronology reversed. Same-period observations leave direction unresolved.

**Q02 — Effective execution with adverse service.** Window: 2016Q3–2019Q3 (OCI operating, E3 to W5). Prediction: Helm C ≥ 5 and Helm K ≥ 5 in ≥ half the window's quarters while Entity_A K ≤ 4 throughout. Weakening: parallel collapse (Helm K ≤ 4 alongside agency K ≤ 4 in most quarters) or agency K ≥ 5. High Helm coherence alone is not corroboration (register).

**Q03 — Corrective feedback.** Signature: Archive K minus Helm K (both national, extended series). Prediction: gap ≥ 2 (oversight custody below executive execution) during the documented failure-to-revise window 2017Q2–2019Q3. Weakening: gap ≤ 1 throughout, or symmetric collapse.

**Q04 — Persistence after warning.** Frozen warnings W1–W4. Prediction: income-averaged debt raising (proxy: Entity_A K ≤ 4 with S ≥ 6) persists beyond W1 without substantive correction until W5. Decision rule: latency = W5 minus W1 = 2019-11 minus 2017-04 = 2.5 years (30 months). Distinguish warning issuance from receipt (register): receipt proxied by government response records (E4–W2 window shows receipt by 2017Q2).

**Q05 — Institutional specificity.** Baseline: each entity's mean K over 2013Q1–2014Q4. Test window: 2016Q3–2019Q3 (OCI operation) plus minimum K anywhere 2015–2019. Prediction: Entity_A K falls ≥ 2 below its baseline while Entity_B K stays within ± 1 of its baseline. Weakening: equivalent Entity_B deterioration (genuine common cause), in which case specificity fails and a national/account-level explanation must be preferred. The whole-of-government aggregates are NOT used as the control (register: an aggregate containing the agency is not independent).

**Q06 — Reversal and repair.** Prediction: Entity_A K recovers ≥ 1 point within 4 quarters of W5 (2019Q4) in recipient-facing functions. Escape clause (register): if a concurrent shock (C1, COVID from 2020Q1) dominates the 4-quarter window, verdict is inconclusive, and improvement must be tested at the first post-shock stabilisation instead (proxy: first quarter after 2021Q4 with no acute surge conditions).

## 5. Multiple testing and reporting

Six primary questions; no p-values (scores are ordinal integers, n=44 with high serial dependence). Verdicts: supported within this design / contradicted / inconclusive / not testable. All weakening conditions reported regardless of direction. Intent, motive, exceptionality: out of scope (register §Intent).

## 6. Amendments

None after the freeze hash. If a defect is discovered, the defect and its consequence are reported; scores are not silently revisited.
