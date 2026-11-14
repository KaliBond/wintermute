# Q30 Test Report — Historical Exceptionality, Australian Government Comparison Set

**Date:** 2026-11-14
**Governing document:** `ImplAddendum_Q30_ComparisonSet_2026-11-14.md`, frozen before scoring, SHA-256 `2e9a5b3b688bbdbe53322672b2d928946cd27a6fdb18f307346949ccfbfc1cfb`
**Comparison set:** 7 segments, 21 segment-years, 2008–2023 annual, 8 nodes, 672 cells per scorer; 3-scorer ensemble (parent + two independent sessions, 24 and 18 own searches, 40 and 27 numbered sources; inter-scorer blind; all retrospective/outcome-informed; no identity blinding claimed by design).

---

## 1. Screen verdicts (frozen criteria, frozen event table)

| Criterion | Frozen threshold (robodebt value) | Other segment qualifying? | Verdict |
|---|---|---|---|
| C1 Scale | ≥400,000 persons adversely affected OR ≥$1.5b adverse financial effect | HIP (1.1m households exposed; deficiency rate 29% of inspected; 4 deaths; 224 fires) and DET (~$19–22b programme cost, ~4,000 detainees, deaths incl. Reza Barati 2014, systematic documented harm) under the ordinary-language reading of "scale" | **Not determinable as frozen — specification defect (below)** |
| C2 Unlawfulness | Official finding, operating programme, ≥1,000 persons | **DET** — PNG Supreme Court *Namah v Pato*, 26 Apr 2016: Manus detention unconstitutional and illegal; $70m settlement approved Sep 2017 compensating 1,905 people for false imprisonment; detention continued after the ruling (precedent also on persistence) | **Precedent exists — robodebt NOT an outlier** |
| C3 Automation | Adverse decisions by automated data-matching/proxy, ≥10,000 persons | None. Pre-2015 welfare data-matching (~20k/yr) was manually adjudicated; the 2015 interim process was manual averaging by officers; income management is categorical administration, not data-matching inference | **Robodebt is an outlier on automation** |
| C4 Intimidation | Documented threats connected to programme, directed at rights-exercisers | None in frozen table. RD qualifies via documented prosecution warnings in the compliance process; the RC deterrence-of-criticism finding is disputed (Tudge) | **Outlier (contested)** — per the frozen caveat; the prosecution-warning element alone is documented and undisputed |
| C5 Targeting | Population selected by economic/social category for adverse treatment | **IM** — compulsory income management from 2007: ~20,000 people selected by geography and benefit class, overwhelmingly Indigenous, RDA suspended 2007–2010, discriminatory-effect findings (AHRC; PJC on Human Rights); **DET** — asylum seekers selected by class | **Precedent exists — robodebt NOT an outlier** |
| C6 Correction delay | ≥24 months, credible signal → substantive remedy | **DET** — warnings continuous from 2012; ruling Apr 2016; RPC closed Oct 2017 but detention continued on Manus, Port Moresby and Nauru; substantive remedy incomplete within window and the system persists. HIP by contrast corrected in ~4 months | **Precedent exists — robodebt NOT an outlier** |

### Specification defect disclosed (per amendments policy)

C1's thresholds were calibrated to robodebt's own values (433,000; $1.763b). The register requires decision criteria "justified externally or by the existing codebook, not observed values". Read strictly (citizen-directed financial harm only; realised individual harm only), no other programme reaches $1.5b or 400,000 persons, and the criterion becomes near-tautological — any threshold anchored to the case under study cannot fail to find it an outlier. Read in ordinary language (programme scale and aggregate harm, including state expenditure on harmful programmes), HIP and DET clearly qualify. **Consequence reported, not patched:** C1 is recorded as *not determinable as frozen*; under the ordinary-language reading, precedent exists. The substantive finding is unaffected: robodebt's exceptionality does not rest on scale.

## 2. Q30 verdict

**The "no precedent" claim FAILS as a general claim.** Under the frozen screen, robodebt is an outlier on **one criterion unambiguously — automation (C3)** — and on a second, contested criterion — intimidation of complainants (C4). On unlawfulness (C2), targeting (C5) and correction delay (C6), in-window precedents exist at comparable or greater severity, produced under Labor governments (detention reopening and income management; HIP on the ordinary reading of scale) and continued under Coalition ones. Robodebt's true exceptionality is confined to its **mechanism**: mass automated administrative unlawfulness — the first Australian programme to convert a legal-basis failure into hundreds of thousands of machine-generated adverse decisions. That is a genuine and important first. It is not an exception in the scale, unlawfulness, targeting or correction-delay of Australian state harm.

Per the frozen decision rules: verdicts stated against n = 7 segments; the boundary limitation is recorded (the Pacific Solution operated 2001–2007 outside the window; pre-2007 precedents such as the original Pacific Solution and Stolen Generations-era administration are context, not counted); no absolute "no precedent" language is used.

## 3. Supporting panels (context, not screen evidence)

Ensemble medians at annual resolution (`Australia_Govt_Q30_comparison_2008-2023_ensemble.csv`):

- **Helm median C/K:** Rudd I 2008–09 C7 K7 / C6 K6 (GFC mobilisation scored as the strongest executive functioning in the set); Gillard C5 K5 declining to C4 K4 by 2013; Rudd II C4 K4; Abbott C6 K5 (2013) collapsing to C3 K4 (2014–15); Turnbull C6 K5 declining to C3 K4 (2018); Morrison C4 K4 → C5 K5 → C6 K6 (2020) → C4 K4 (2022); Albanese C6 K6 (2022), C5 K6 (2023). The Morrison-era dip-and-recovery pattern replicates the quarterly panels; the Abbott 2014 and Turnbull 2018 coherence troughs replicate the Helm extension.
- **Spread:** 55/672 cells unanimous (8.2%); 35 cells with spread ≥ 3, concentrated in Stress readings of crisis years (Rudd I 2008–09 GFC; Morrison 2020 COVID; Albanese 2022) and Abstraction readings of Gillard-era Stewards (income-management-era abstraction scored 4 by scorer 1, 7 by scorer 2) — the two axes where the scorers flagged disagreement in advance. Screen verdicts are unaffected: the screen runs on the frozen event table, and panel disagreement is reported as context, per frozen rule 4.

## 4. Deviations accepted

- NEM suspension (June 2022) attributed to Albanese per the handover rule (Morrison's tenure ended 23 May 2022); the framing calendar listed it under Morrison — scorer 2 applied the rule correctly and the parent panel agrees.
- ATO garnishee public break verified (again) as April 2018 (Turnbull), not 2017; affects Turnbull Archive coherence cells only; the event does not meet any frozen criterion (no official unlawfulness finding) either way.

## 5. Files

`ImplAddendum_Q30_ComparisonSet_2026-11-14.md` (frozen) · `build_q30_comparison_set.py` (parent generator + validator) · parent CSV + score-status · `ens2_Q30_comparison_2008-2023_annual_raw.csv` + `ens2_q30_protocol.md` · `ens3_Q30_comparison_2008-2023_annual_raw.csv` + `ens3_q30_protocol.md` · `Australia_Govt_Q30_comparison_2008-2023_ensemble.csv` · `Australia_Govt_Q30_comparison_2008-2023_envelope.csv` · this report.
