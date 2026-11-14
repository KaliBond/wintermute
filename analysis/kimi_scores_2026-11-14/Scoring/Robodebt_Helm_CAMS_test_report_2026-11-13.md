# Robodebt and Helm Failure — CAMS Test Report

**Date:** 2026-11-13
**Analyst:** Kimi, for Kari McKern (ComplexityWorkz)
**Task:** Run the pre-analysis question register from the Perplexity session (bf0c2ebd, compiled 2026-11-13) against the CAMS quarterly panels built in this workspace on the same date.
**Entity:** Australian Government, 2018Q1–2023Q4 (24 quarters)

**Update 2026-11-14:** Q01, Q04 and Q05 — not testable or externally-dated in this run — were subsequently tested on a blinded Centrelink/CES entity series (Entity_A), a locked control-agency series (Entity_B, ATO), Helm and Archive extensions 2013–2017, a hash-frozen implementation addendum, and a 3-scorer ensemble. Verdicts below are updated in place and marked. Q30 — withheld in this run — was subsequently tested on a locked seven-segment Australian government comparison set (2008–2023 annual, frozen addendum, 3-scorer ensemble); the "no precedent" claim fails as a general claim. Full detail: `Q01-Q06_test_report_2026-11-14.md`, `Q30_test_report_2026-11-14.md`, `EntityAB_BlindScoring_protocol-note.md`, `EntityAB_ensemble_protocol.md`. The updates change verdicts; they do not re-open any frozen score.

---

## 1. What was run, and under what deviations

The register (Q01–Q40, US-01–08, MIN-01–08, ECON-01–08, HELM-X1–3) was designed for **blinded, entity-level series** — Morrison government, Cabinet, PM, CES/Centrelink — hashed and timestamped before score access. That design could not be executed, for three structural reasons, each of which the register itself anticipates:

1. **No blinding.** All three scorers were retrospective and outcome-informed. Scorers 2 and 3 assembled their own evidence packets from open web sources that include the Robodebt Royal Commission (2023), the Bell Inquiry (2022) and NACC Operation Myrtleford (2026); scorer 1 (the primary panel) was built with the same hindsight. Inter-scorer blindness held — no scorer saw another's output — but **outcome blindness did not exist**. Per the register's own Q33, all claims below are limited to **retrospective discrimination, not historical forecasting or prediction**.
2. **No entity-level series.** Our panels are (a) a whole-of-government 8-node panel and (b) a Helm-only panel for the Federal Executive. There is no Centrelink/CES series, no PM-versus-Cabinet split, no bilateral Australia–US series. Questions requiring those resolutions are classified **not testable**, as the register instructs, not answered by stretching the aggregates.
3. **No implementation freeze.** The register's hash-before-analysis protocol (OSF/OpenTimestamps) was not performed; this analysis is explicitly retrospective. Comparisons, windows and the correction-latency measure were chosen with knowledge of the outcomes and are labelled exploratory where that matters.

**Data used (all in this workspace, all dated 2026-11-13):**

| File | Role |
|---|---|
| `Australia_Govt_2018Q1-2023Q4_cams_v1.4.csv` | Scorer 1 whole-of-government panel (24q × 8 nodes × C/K/S/A) |
| `Australia_FedExec_Helm_2018Q1-2023Q4_cams_v1.4.csv` | Scorer 1 Helm-only panel (identical Helm values to the panel above) |
| `ens2_…_raw.csv`, `ens3_…_raw.csv` + protocol notes | Scorers 2 and 3, independently evidence-searched |
| `Australia_Govt_…_ensemble.csv`, `_envelope.csv` | 3-scorer medians and spread |

**Verdicts are reported on the scorer-1 panel** (the commissioned instrument) **and cross-checked against the 3-scorer ensemble.** Where a load-bearing cell is contested in the ensemble, the verdict is downgraded and the disagreement shown. Contested cells that matter here:

| Cell | Scorer 1 | Scorer 2 | Scorer 3 | Median | Note |
|---|---|---|---|---|---|
| Craft C, 2019Q3–Q4 (robodebt legality) | 4 | 6 / 6 | 5 / 5 | 5 | scorer 1 alone scores legality failure into Craft C |
| Craft K, 2019Q4 | 4 | 5 | 5 | 5 | scorer 1 alone scores K4 |
| Archive A, 2019Q4, 2020Q4–2021 | 6 | 4, 4–3 | 3–4 | 4 | scorer 1 alone scores the RC-era abstraction rise |
| Helm C, 2020Q1 | 6 | 5 | 3 | 5 | wide disagreement; 2020Q2 C6 is unanimous |
| Helm C, 2022Q3 (secret ministries) | 4 | 7 | 5 | 5 | scorer 1 alone scores the drop |
| Stewards S, 2020Q2 | 9 | 8 | 8 | 8 | level, not direction, disputed |
| Archive K, 2020Q2–2022Q2 | 4 | 4–5 | 4 | 4 | **robust** — custody failure is the consensus pattern |

**Uncontested load-bearing cells:** Helm C6 K6 at 2020Q2 (all three scorers); Archive K4 through 2020Q2–2022Q2 (median 4, one dissenting 5 at 2021Q4/2022Q2), recovering to K5 at 2022Q3; Craft K4 at 2021Q2 (median 4); Stewards K6 at 2020Q2 (median 6); Flow S9 at 2022Q2 (median 9).

---

## 2. Verdicts by family

Classification per the register: *supported within this design / contradicted / inconclusive / not testable / exploratory*. "Supported" never means the register's causal claim is proven; it means the panel pattern is the one the question's prediction describes.

### 2.1 Primary family Q01–Q06

| Q | Verdict | Basis |
|---|---|---|
| Q01 Temporal precedence (executive change → Centrelink deterioration) | **Supported with ensemble robustness** (updated 2026-11-14) | Updated by the blinded entity-series run: Entity_A (Centrelink, Stewards agency-resolution) Capacity first falls at 2015Q3 — the quarter the executive policy commenced (1 July 2015), one quarter after its budget announcement. No agency deterioration precedes the executive change. Verdict identical on the 3-scorer ensemble median (strict and lenient thresholds). Same-quarter concurrency noted: the 3-month interval cannot fully resolve cause from common decision. |
| Q02 Effective execution + adverse service | **Supported, with ensemble qualification** | During 2020 — while unlawful robodebt debts were still being raised (scheme ended November 2019) and recovery continued — Helm scores C6 K6 at 2020Q2 (unanimous) while Archive custody sits at K4 (unanimous). Executive coordination coexisting with impaired lawful-service custody is exactly the predicted asymmetry. Qualification: the Craft C4 legality signal is scorer-1-only (ensemble median 5). The 2026-11-14 run adds the agency-side pattern: Entity_A K held 3–4 through every operating quarter on the ensemble median, while Helm execution K stayed ≥5 in 10 of 13 — but Helm coherence sat at 4 through most of the window, so the frozen joint test remains split (see `Q01-Q06_test_report_2026-11-14.md`). |
| Q03 Corrective feedback (Archive→Helm mismatch rises as policy persists) | **Supported on panel (2020–22 phenomenon); not supported as a robodebt-era signature** | Scorer 1: Archive A6 (synthesis of precedent via royal commissions) alongside Archive K4 (custody) 2020Q2–2021 — the system could articulate what had gone wrong for ~2 years while not implementing custody correction. The A-side of that divergence is contested (ensemble median A4). The K side — the correction failure itself — is unanimous. Updated by the 2026-11-14 run (adverse result, reported prominently): the Archive–Helm custody gap stays within ±1 throughout the failure-to-revise window 2017Q2–2019Q3; the gap of −2 appears only from 2020Q2 (Tune-review era). The corrective-feedback failure during robodebt is visible in the agency series itself (Entity_A K3 held four years through Ombudsman, Senate and AAT warnings), not in the national Archive–Helm gap. |
| Q04 Persistence after warning | **Supported with ensemble robustness** (updated 2026-11-14) | Externally dated: Ombudsman investigation of income averaging opened April 2017 (monitoring from 2016); scheme persisted raising debts until November 2019; ~$751m recovered from ~381,000 people before the June 2021 settlement ($112m including costs). Panel: Archive K stays 4 until 2022Q2. Warning-to-cessation ≈ 2.5 years; warning-to-refund ≈ 4 years. Updated: persistence is now scored, not inferred — Entity_A K≤4 with S≥6 in every quarter 2017Q2–2019Q3 on both the parent panel and the 3-scorer ensemble median (strict and lenient thresholds). |
| Q05 Institutional specificity | **Supported with ensemble robustness** (updated 2026-11-14) | Updated by the blinded entity-series run: Entity_A Capacity fell 2.0 points below its 2013–14 baseline (5.0 → 3.0, held four years) while the locked control agency (Entity_B, ATO) stayed within ±1 of its baseline (4.75 → minimum 4.0) across the same window and never scored below 4 in eleven years — including 2020, when it delivered JobKeeper at K6 while Entity_A buckled under the identical shock. Verdict and deltas identical on ensemble medians. Caveat: one control agency cannot separate agency-specific from welfare-portfolio-specific failure. |
| Q06 Reversal and repair | **Scorer-dependent** (updated 2026-11-14) | 2026-11-13 view: inconclusive — post-termination recovery confounded by COVID, vaccine rollout and the 2022 election. 2026-11-14 ensemble: median K recovers 4→5 by 2020Q3, within the frozen 4-quarter window — supported on the median — but this rests on two of three scorers judging the pandemic surge as handled (backlog cleared by May 2020; ~12,000 staff mobilised) against scorer 1's K3/S9 trough. Reported as scorer-dependent, not averaged away. |

### 2.2 Pathway questions Q07–Q14

| Q | Verdict | Basis |
|---|---|---|
| Q07 Location (PM / Cabinet / ministers / administration / frontline) | **Not testable** | Single Helm node cannot resolve levels. |
| Q08 Vertical divergence | **Not testable** | No Cabinet-vs-departmental split in the data. |
| Q09 Upward transmission (not generated / distorted / received-disregarded) | **Not testable** | CAMS records state, not message paths. |
| Q10 Coordination versus correction | **Exploratory (leans supported)** | Same pattern as Q02/Q03: C6K6 Helm 2020Q2 with K4 Archive. One observation, one case; register requires independent replication we do not have. |
| Q11 Administrative abstraction as loss of individual information | **Exploratory** | Our A scores track institutional learning/synthesis, not proxy-driven administration; the register's construct is not what CAMS A measures here. Construct validity not established. |
| Q12 Burden displacement to staff/recipients | **Not testable** | No staff or recipient stress series; register explicitly forbids inferring this from national Hands scores (which measure the national workforce, not Centrelink staff or recipients). |
| Q13 Recognition versus remedy (stage latency) | **Supported (external dates; panel consistent)** | Warning 2017 → cessation 2019 (≈2.5y) → refund/settlement 2021 (≈4y) → royal commission findings 2023 (≈6y). The register's "correction latency" measure, applied with the same rules to the vaccine rollout, gives ≈5 months (see §3). This asymmetry is the strongest quantitative result of the run. |
| Q14 Persistence and repair post-termination | **Inconclusive** | Dataset covers to 2023Q4; recovery is visible (Archive K5 2022Q3) but confounded as per Q06. |

### 2.3 Rival explanations Q15–Q22

| Q | Verdict | Basis |
|---|---|---|
| Q15 Bureaucratic origin | **Inconclusive from scores; external record favours mixed origin** | NACC Operation Myrtleford (March 2026) found serious corrupt conduct by two officials (Withnell, Wilson) and did **not** find corrupt conduct by Morrison, Campbell or three others. The Royal Commission (2023) nonetheless attributed responsibility for continuation to ministers. Scores cannot arbitrate origin; the external record supports distributed administrative distortion plus executive persistence — both register accounts, partially. |
| Q16 Fiscal objective | **Not testable** | No fiscal data in CAMS. ≥$1.763b unlawfully asserted against ~433,000 people is consistent with a savings-driven account; scores are silent. |
| Q17 Technical failure | **Not testable (external record: weak account)** | Income averaging was a legal failure (unlawful from inception per the RC), not a delivery constraint; and technical accounts cannot explain persistence after 2017 warnings. External, not CAMS. |
| Q18 Distributed failure | **Exploratory (consistent with multi-institutional account)** | Failures appear in Craft (legality), Archive (custody) and Helm (stress up) simultaneously in 2019Q4–2020 — spread across functions favours distribution, but these nodes share one government and one evidence base, so "spread" is not independent replication. |
| Q19 Inherited conditions | **Not testable** | Panel opens 2018Q1; scheme began 2015; no baseline exists by design. |
| Q20 General pressure / common shocks | **Inconclusive** | COVID dominates 2020–21 and is a massive confound for every 2020 cell. |
| Q21 Role transition (Turnbull → Morrison) | **Inconclusive** | Helm C3 at 2018Q3 is unanimous (leadership-spill quarter), returning to C6 K6 by 2019Q2–Q3; no resolvable Morrison-specific break at quarterly resolution. |
| Q22 Placebos and false-alarm rate | **Not run** | No event-study or placebo machinery was built; the register's specificity test is unmet. Any apparent precision beyond that is exploratory. |

### 2.4 Intent and targeting Q23–Q30

**Verdict for Q23–Q29: not established — as the register itself requires.** These questions demand evidence beyond system state vectors, and the register forbids inventing a numerical proxy for intent. CAMS scores contribute nothing to intent, and no CAMS combination (including Capacity-minus-Stress or high Abstraction) has validated person-level interpretability.

External record, stated plainly because it bounds what can honestly be claimed: the Royal Commission found the scheme "a crude and cruel mechanism, neither fair nor legal", found that legal advice was suppressed, and made findings about deterrence of public criticism as a media strategy (the Tudge finding, which Tudge disputes). The RC did **not** find a purpose to demoralise recipients, pleasure in suffering, or personal motive on Morrison's part; NACC (2026) found corrupt conduct by two officials, not by the referred ministers. **Kari's "demoralising blow / schadenfreude" prior is therefore supported at most at the level of deterrence-of-criticism media strategy by some actors; it is not supported, and is partly contradicted, at the level of personal motive.** Programme creation and later defence should be considered separately, as Q23 requires — the external record treats them differently.

| Q30 Historical exceptionality | **Tested 2026-11-14 — the "no precedent" claim FAILS as a general claim** | The locked comparison set (7 segments, 21 segment-years, 2008–2023 annual, 8 nodes, 3-scorer ensemble; frozen addendum SHA-256 `2e9a5b3b…c1cfb`) was built and the frozen screen applied. Robodebt is an outlier on automation (C3) unambiguously and intimidation of complainants (C4, contested); precedents exist on unlawfulness (offshore detention, PNG Supreme Court 2016, $70m false-imprisonment settlement), targeting (compulsory income management from 2007; detention) and correction delay (detention > 31 months, remedy still incomplete). Scale (C1) not determinable as frozen — thresholds were robodebt-calibrated (defect disclosed; ordinary reading favours precedent). Verdict: exceptionality confined to mechanism — mass automated administrative unlawfulness — not to scale, unlawfulness, targeting or delay of Australian state harm. Full detail: `Q30_test_report_2026-11-14.md`. |

### 2.5 US family US-01–US-08

**Verdict for the family: not testable from CAMS.** No bilateral series and no cross-administration comparison set exist; the register's own limit applies — similar trajectories cannot establish collusion, which additionally requires evidence of agreement, conduct and improper purpose.

External evidence from the session: continuity across the Trump–Biden transition (Paris posture, AUKUS 16 September 2021, Payne's defence of cooperation on the Mueller-origins investigation) is counterevidence to the specifically partisan-collusion account; China-policy changes (Huawei 5G exclusion August 2018; April 2020 origins call → trade coercion costing ~$20b, iron ore at 82% of exports to China in 2021) track Australian threat assessment and domestic politics as plausibly as US direction. None of this is exoneration; it is simply that the available record supports **alliance continuity plus convergent threat perception**, and CAMS adds nothing to distinguishing the accounts. US-06-type independence tests (policy disagreement with Washington) are not enumerable from our data.

### 2.6 Secret-ministry family MIN-01–MIN-08

| Q | Verdict | Basis |
|---|---|---|
| MIN-01 Continuity planning vs discretionary control | **Not testable from scores** | Requires independent evidence of continuity risks and portfolio powers; external record (Bell Inquiry, 2022) documented the appointments' scope and concealment but motive categories remain institutional explanations, not scoreable states. |
| MIN-02 Secrecy explained | **Not testable from scores** | Contemporaneous reasons are external documents; scores silent. |
| MIN-03 Antecedent CAMS change before the appointments (Mar 2020) | **Supported-weak (suggestive), retrospective only** | Panel shows weakened correction **before** the appointments: Archive C4 at 2019Q2 (median 4) and Craft weakening 2019Q3–Q4 (scorer 1: C4; median 5). A prior weakened-correction pattern is more consistent with an antecedent condition than with a pattern appearing only afterwards — but with 24 intervals, one case, and full hindsight, this is retrospective detection, not advance prediction. Register's caveat applied verbatim. |
| MIN-04 PM trajectory vs Cabinet | **Not testable** | Aggregate Helm cannot separate PM from Cabinet (register says so). |
| MIN-05 Portfolio-anticipation | **Not testable** | Requires independent evidence of anticipated statutory disagreements. |
| MIN-06 Responsibility claim vs control desire | **Not testable** | External documents only; and the powers were barely exercised — one recorded use — which undercuts neither account by itself. |
| MIN-07 Effects of exercised powers | **Not testable (external: minimal exercise)** | Bell Inquiry record: appointments largely unused; consequences scored in Helm 2022Q3 (scorer 1 C4 A4; median C5 — contested). |
| MIN-08 Same signature under transparent emergency arrangements? | **Not testable** | No comparison transparent-emergency episode scored. |

### 2.7 Economic family ECON-01–ECON-08

| Q | Verdict | Basis |
|---|---|---|
| ECON-01 Employer-mediation vs GFC | **Not testable from CAMS** | Needs expenditure-category comparison; external record: JobKeeper $88.8b, ~4m employees, ~1m businesses, announced 30 March 2020, employer-mediated by design; GFC response was direct household payments. Fact established externally; CAMS silent. |
| ECON-02 Final incidence | **Not testable** | Incidence analysis is outside CAMS. |
| ECON-03 Employment preservation vs direct alternatives | **Not testable** | External evaluations (Treasury independent evaluation; ANAO) found stabilisation effective with adaptability and transparency problems. |
| ECON-04 Unequal protection by group | **Not testable from scores** | Design exclusions (casuals, arts, universities) are external facts; scores cannot rank groups. |
| ECON-05 Business error vs welfare debt asymmetry | **Supported (external evidence; CAMS contributes little)** | External: unlawful welfare debts pursued to $751m recovered from ~381,000 people, against a documented reluctance to recover JobKeeper overpayments and ANAO findings on undocumented overpayment discretion. The register's own caveat stands: a business lawfully retaining a payment under the rules is not equivalent to a recipient pursued for an unlawfully established debt — the asymmetry is in enforcement posture toward error, not in legality of retention. CAMS adds only the weak contrast of Craft K4 (vaccine) vs Stewards K6 (JobKeeper) — delivery capacity, not leniency. |
| ECON-06 Costly design retained after evidence changed | **Supported (external dates; consistent with Q04)** | Robodebt retained 2017–2019 after credible warnings; the same latency measured in Q13. |
| ECON-07 Political connections predicting assistance | **Not testable** | Requires programme-level data; scores silent. |
| ECON-08 CAMS separating stabilisation from distributional/accountability failure | **Supported** | This one CAMS does: Stewards delivered K6 under S8–9 stress in 2020 (effective stabilisation visible as maintained capacity under load) while Craft K4 (2021Q2, median 4) and Archive K4 (2020Q2–2022Q2, unanimous) record the accountability failures — different nodes, different metrics, no cancellation. |

### 2.8 Cross-case tests HELM-X1–X3

| Q | Verdict | Basis |
|---|---|---|
| HELM-X1 Selective capacity | **Supported — the strongest CAMS-level result** | Correction latency, same rules both programmes: robodebt ≈ 2.5 years (April 2017 warning → November 2019 cessation), ≈ 4 years to refund (June 2021), ≈ 6 years to authoritative findings (July 2023); vaccine rollout ≈ 5 months (reorganisation mid-2021 → 80% double-vaccinated November 2021, per ANAO). Same state, same executive, ~6–50× faster correction when the constituency was businesses and the vaccinated public rather than welfare recipients. Stewards K6 under S9 (2020Q2) vs Archive K4 through 2022Q2 is the panel-level shadow of the same asymmetry. This is constituency-conditional capacity deployment, not a uniformly incapable state. |
| HELM-X2 Directed coherence | **Supported, with ensemble qualification** | Helm C6 K6 at 2020Q2 (unanimous) coexisted with Archive custody K4 (unanimous) and Craft legality weakness (scorer 1 C4; median 5). Strong executive coordination coexisting with weak legality and unresponsiveness to adverse evidence is the observed pattern. Per the register's prior commitment: high coherence is neither automatically good governance nor automatically coordinated wrongdoing — the scores show coexistence, nothing more. |
| HELM-X3 Common mechanism | **Exploratory** | Concentrated direction plus weakened corrective feedback fits robodebt, the secret appointments and the China-policy concentration (Flow S8–9 2021–22; median 9 at 2022Q2) without changing variable meanings between cases. But the cases share personnel, period and evidence base; no formal test exists; separate case-specific explanations remain viable. |

### 2.9 Validity self-audit Q31–Q40

| Q | Verdict | Basis |
|---|---|---|
| Q31 Provenance | **Disclosed** | Mixed contemporaneous and later findings; full source lists in the three protocol notes; scorer 1 panel built 2026-11-13 with documented event calendar. |
| Q32–Q33 Blinding / leakage | **Failed by design; disclosed** | All scorers outcome-informed; inter-scorer blindness only. Claims limited to retrospective discrimination. |
| Q34 Independent corroboration | **Shared-source consistency only** | Session facts and scores draw on the same public record (RC, ANAO, Bell, NACC); agreement is not external validation. |
| Q35 Comparable scales | **Failed for entity-level questions** | Whole-of-government and Helm nodes cannot substitute for PM/Cabinet/CES series; affected questions marked not testable. |
| Q36 Resolution | **Weak** | 24 quarterly intervals with high serial dependence; effective independent observations far fewer; same-period events unresolvable. |
| Q37 Specification stability | **Exploratory** | Windows and latency measures chosen with hindsight; no frozen alternative set to report against. |
| Q38 Incremental CAMS value | **Mixed, stated honestly** | CAMS adds: latency asymmetry (X1), coherence–legality coexistence (X2), and separation of stabilisation from accountability failure (ECON-08). CAMS adds nothing to: intent, collusion, exceptionality, incidence — and claims to the contrary would be metric invention, which the register forbids. |
| Q39 Symmetric judgement | **Not tested** | One government scored; the Labor-period recovery within the panel (Helm C6 K6 2022Q4–2023Q2) is visible but there is no blinded cross-party replication. |
| Q40 Defeat conditions | **Recorded** | Contrary evidence already on the table: NACC's non-findings against referred ministers; alliance continuity as rival to the partisan account; ensemble disagreement on precisely the cells most favourable to the Helm-failure reading (Craft C, Archive A, Helm C 2020Q1 and 2022Q3) — a pattern that warrants scepticism about scorer-1's sharper verdicts and is reported here rather than suppressed. |

---

## 3. Correction latency — the one quantitative result

Same measurement rule both programmes: elapsed time from a credible adverse signal reaching an authorised decision-maker to substantive remedial action.

| Programme | Credible signal | Substantive remedy | Latency |
|---|---|---|---|
| Robodebt | Ombudsman investigation opened, April 2017 (monitoring from 2016) | Debts stop being raised, November 2019 | ≈ 2.5 years |
| Robodebt (full repair) | April 2017 | Refunds + $112m settlement, June 2021 | ≈ 4 years |
| Robodebt (authoritative findings) | April 2017 | Royal Commission findings, July 2023 | ≈ 6 years |
| Vaccine rollout | Targets missed / ANAO-documented planning failures, early 2021 | Reorganisation mid-2021; 80% double-vaccinated, November 2021 | ≈ 5 months |

Ratio: roughly 6× to 50×, depending on which remedy stage counts. The register's own discriminator applies: this variation is the pattern the selective-capacity hypothesis predicts and which a pure technical-complexity account must explain away — vaccine delivery in a pandemic was not technically simpler than stopping an unlawful debt algorithm. **Data (external dates + unanimous Archive K cells). Inference (the asymmetry reflects constituency-conditional enforcement posture). Speculation (why posture differs — incentives, ideology, contempt — is not determined by anything in this dataset).**

---

## 4. Bottom line

**What the CAMS run establishes (retrospective discrimination only):**

1. A weakened-correction signature exists in the Australian state from at least 2019Q2 — before the secret ministries, before COVID — and persists to 2022Q2 (Archive K4, unanimous across three independent scorers).
2. Executive coherence and legality/accountability failure coexisted through 2020 (Helm C6K6 unanimous at 2020Q2 against Archive K4).
3. Correction latency differed by constituency by one to two orders of magnitude (robodebt vs vaccine).
4. Stabilisation capacity (Stewards K6 under extreme stress) and accountability failure (Craft K4 2021Q2; Archive K4) are separable CAMS outcomes — the state was not uniformly incapable.
5. (2026-11-14, entity-level) The agency signature the register predicted is real and robust: executive policy change (2015Q2/Q3) precedes agency Capacity deterioration; persistence through all dated warnings is scored, not inferred; and the deterioration is concentrated in the robodebt-relevant institution against a locked control agency that never left its baseline — all three results holding on 3-scorer ensemble medians.

**What it does not establish:** executive origination of robodebt versus bureaucratic origin (inconclusive; external record mixed); intent, targeting, schadenfreude or personal motive (not established, per the register's own rule); Trump–Cabinet collusion (not testable; external record consistent with alliance continuity); historical exceptionality (withheld — no comparison set); and any claim at PM-versus-Cabinet resolution.

**What would make this testable later:** ~~a blinded Centrelink/CES entity series 2013–2023~~ **(delivered 2026-11-14 — Entity_A series with Entity_B control, freeze-hashed addendum, 3-scorer ensemble; Q01/Q04/Q05 updated above)**; a locked Australian comparison set across governments (Rudd/Gillard/Abbott/Turnbull/Morrison/Albanese) scored under identical instructions; a second welfare-delivery comparator to separate agency-specific from portfolio-specific failure (the remaining Q05 caveat); a transparent-emergency comparison episode for MIN-08; and an OSF-frozen implementation addendum before any rescoring (procedure demonstrated 2026-11-14). The ensemble disagreement mapped in §1 is itself a finding: the cells most diagnostic of Helm failure are exactly the cells where independent scorers diverge, which is what an honest instrument reports rather than smooths over.

---

## Sources

Session register and factual base: Perplexity session bf0c2ebd-b46d-4a32-9aa3-ae79880403e7 (compiled 2026-11-13), including Federal Court class-action figures (≥$1.763b unlawfully asserted; ~433,000 people; ~$751m recovered from ~381,000; $112m settlement), Treasury JobKeeper evaluation ($88.8b; ~4m employees; 30 Mar 2020–28 Mar 2021), Royal Commission into the Robodebt Scheme (2023), NACC Operation Myrtleford (March 2026), Bell Inquiry (2022), ANAO vaccine-rollout audit (2022).

Panel data: `Australia_Govt_2018Q1-2023Q4_cams_v1.4.csv` (scorer 1, 2026-11-13); `ens2_…_raw.csv`, `ens3_…_raw.csv` with protocol notes; `Australia_Govt_…_ensemble.csv`. Ensemble protocol: `Australia_Govt_2018Q1-2023Q4_ensemble_protocol.md`. Entity-level run (2026-11-14): `EntityA_Stewards_2013Q1-2023Q4_cams_v1.4.csv`, `EntityB_Archive_2013Q1-2023Q4_cams_v1.4.csv`, `Helm_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv`, `Archive_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv`, `EntityAB_2013Q1-2023Q4_ensemble.csv` (+ envelope and both ensemble protocol notes), under `ImplAddendum_Q01-Q06_CentrelinkCES_2026-11-14.md` (SHA-256 `b3f789e8…f095`, frozen pre-scoring). Archived with SHA-256 manifest in `Australia_Govt_2018Q1-2023Q4_CAMS_v1.4_3scorer_2026-11-13.zip`.

External dates used in §3 (stable historical record): Ombudsman investigation April 2017; scheme cessation November 2019; Amato settlement June 2021; RC report July 2023; 80% double-vaccination November 2021.
