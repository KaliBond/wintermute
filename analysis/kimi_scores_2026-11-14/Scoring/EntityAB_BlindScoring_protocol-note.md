# Protocol Note — Blinded Entity Series (Entity_A / Entity_B) + Helm & Archive Extensions

**Date:** 2026-11-14
**Protocol:** CAMS Universal Instruction Set v1.4
**Governing document:** Implementation Addendum Q01–Q06 (frozen before scoring)
**Addendum SHA-256 (computed before any score was written):** `b3f789e82c2d7680b6874106c889347fe6bde6343d528485f4a1f295d831f095`
**Order of operations (audit trail):** addendum written → SHA-256 recorded → score matrix written (`blind_entity_series_build.py`) → validator run (passed; all step changes ≤ 2) → frozen tests applied → this note written. Test definitions were not altered after scoring.

## Unmasking key

- **Entity_A** = Centrelink, administered by the Department of Human Services (renamed Services Australia from 2019–20). Node scored: **Stewards, agency resolution** (delivery organ of the population-maintenance function).
- **Entity_B** = Australian Taxation Office. Node scored: **Archive, agency resolution** (custodian of the taxation/fiscal record).
- Masked labels are retained in the CSVs so the panels can be handed to a genuinely blind third-party scorer for replication; the key lives only in this note.

## Blinding discipline (stated limits, not claimed strengths)

The scoring analyst was fully aware of the historical outcomes (Royal Commission 2023, NACC 2026, class action 2021). **Personal outcome blinding was unattainable and is not claimed.** Substitute disciplines, per addendum §2:

1. Masked entity labels (above) — reusable for third-party blind replication.
2. Contemporaneous-source discipline — each quarter scored only on evidence available by that quarter's end; later findings (court outcomes, RC, NACC, later ANAO audits) were excluded from scoring rationale and used only as independently dated warnings at test time, as the register requires. As a single analyst this discipline cannot be externally verified; it is disclosed, not certified.
3. Freeze-before-access — the addendum (tests, event dates, decision rules) was hashed before the score matrix existed. One specification defect was discovered at test time (Q03 inequality sign) and is reported in the test report per the addendum's no-silent-revision rule; the scores were not touched.

## Evidence basis (principal, contemporaneous per quarter)

Entity_A scoring anchored on: 2013–14 budget and APS efficiency-dividend pressures on DHS staffing (contemporaneous reporting and budget papers); May 2015 Budget "Strengthening the Integrity of Welfare Payments" (Whiteford 2021; Clarke & Abbas 2024, AJIS); 1 July 2015 interim manual process and two-stage 2015 pilot (Ombudsman report 02/2017); OCI limited rollout July 2016, full September 2016, ~20,000 interventions/week, 216,000 interventions Sep–Dec 2016 (Ombudsman 2017; Senate Community Affairs Committee 2020); December 2016 media surge and ministerial announcement (Whiteford 2021); call-centre failure — 42m busy signals FY2016-17 to April, 55m by October 2017 (Greens media release 25 Oct 2017; CSMC 2023 retrospective on 28m in six months); February 2017 EIC replacement and Senate referral (8 Feb 2017); Ombudsman report 19 April 2017; Senate report 21 June 2017; Carney AAT judgments April–September 2017 (Whiteford 2021); CUPI rollout mid-2018–mid-2019; November 2019 cessation of income-averaged debt raising (Services Australia); COVID surge — 1.3m JobSeeker claims in 55 days, myGov record 2.6m logins 25 March 2020 (Senate Job Security Committee 2021; APSC State of the Service 2019–20); May–July 2020 unlawfulness acceptance and refunds (AJIS timeline); June 2021 settlement (Federal Court); August 2022 Royal Commission establishment; July 2023 RC report.

Entity_B scoring anchored on: ATO budget cuts and restructuring 2014–15; myTax digital expansion 2015–16; April 2017 garnishee-notice treatment of small-business debts (contemporaneous reporting; small-business ombudsman review); May 2017 Plutus tax-fraud scandal; 2018–19 STP expansion; 2020 JobKeeper administration and early superannuation release — 97% of payments within four business days, ~2.6% net gap (ANAO Administration of JobKeeper); post-2021 normalisation.

Helm/Archive extensions (2013Q1–2017Q4) use the same anchors as the 2018–2023 panels (protocol note of 2026-11-13) plus: 2013 leadership change (Rudd) and September 2013 election; 2014 Hockey budget and Senate blockage; February 2015 spill motion; September 2015 Turnbull; July 2016 election; December 2016 OCI defensive handling; 2017 citizenship crisis and High Court judgment (October); marriage postal survey; energy policy conflict. Archive extension: 2014 budget OAIC funding cut; 2015 metadata retention legislation; NAA under-resourcing. Both extensions join the 2018Q1 panels without discontinuity (validator join-checks passed).

## Files

- `EntityA_Stewards_2013Q1-2023Q4_cams_v1.4.csv` (+ `_score-status.csv`) — 44 quarters
- `EntityB_Archive_2013Q1-2023Q4_cams_v1.4.csv` (+ `_score-status.csv`) — 44 quarters
- `Helm_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv` (+ `_score-status.csv`) — 20 quarters
- `Archive_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv` (+ `_score-status.csv`) — 20 quarters
- Generator/validator: `blind_entity_series_build.py` (authoritative score matrix)
- Tests: `Q01-Q06_test_report_2026-11-14.md`
