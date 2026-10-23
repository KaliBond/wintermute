# CAMS Prospective Register v1 (DRAFT, not lodged)

**Status:** draft for Kari's approval, 20 October 2026 (Australia/Sydney). Not pushed, not hashed, not lodged.
**Window:** calendar years 2027 and 2028.
**Rule in one line:** every entry is graded by an outcome in the world, published by a named public source. No entry is graded by a CAMS score.

CAMS readings motivate each prediction. They do not decide it. Node readings come from `JUNO_Unified_Dataset_v3_2026-10-19.csv` (SHA-256 `54b62badddfd1f7ba17eb67ebaaf604d7e792f335eda685065a0fa096b76b9b4`), with V = C + K − S + 0.5A. No base rate or consensus figure below has been filled in from memory. Where one is missing, it says what to compute and from where, before lodging.

## 1. Lead entries: the three risky predictions

| ID | Society | Plain claim | CAMS reading (JUNO v3) | Outcome and source | Pass / fail threshold | Resolves | Base rate | Consensus contradicted | Risk |
|---|---|---|---|---|---|---|---|---|---|
| R-01 | China | China keeps growing through 2027–28 with no financial crisis | Mean V 13.56 (2023) to 13.49 (2025), steady and the highest of the major powers in the file. Lowest node Hands 10.4; Craft 16.5, Archive 15.5 (2025) | (a) Real GDP growth, National Bureau of Statistics of China, as carried in IMF WEO. (b) Systemic banking crisis flag, IMF Laeven–Valencia systemic banking crises database (or its update) | PASS if real GDP growth ≥ 3.0% in both 2027 and 2028 AND no systemic banking crisis is dated as starting in 2027–28. FAIL otherwise | 30 June 2029 (first full-year official figures for 2028) | Base rate to be computed from IMF WEO historical data (share of two-year windows for upper-middle-income economies with growth ≥ 3.0% both years) before lodging | Widespread forecasts of a property-led stall or "Japanification". Record the IMF WEO October 2026 China projections for 2027 and 2028 here before lodging | Risky (contrarian) |
| R-02 | Russia | Russia holds together and does not slide into economic collapse or regime rupture | Mean V rose 7.48 (2024) to 8.75 (2026). Every node rose; lowest node Flow 7.1 (2026) | (a) Real GDP growth, Rosstat, as carried in IMF WEO. (b) Head of state and form of government, official Kremlin records | PASS if real GDP growth > −2.0% in both 2027 and 2028 AND no unconstitutional transfer of power (coup, forced resignation, state collapse) occurs by 31 Dec 2028. FAIL otherwise | 30 June 2029 | Base rate to be computed from IMF WEO historical data (frequency of a year below −2.0% for economies under sanctions or at war) before lodging | Widespread commentary that sanctions and war spending lead to economic exhaustion or internal crisis by 2027–28. Record the IMF WEO October 2026 Russia projections before lodging | Risky (contrarian) |
| R-03 | USA | The US shows material weakness by 2028: a recession year | Mean V fell 10.23 (2024) to 8.82 (2026). Lowest node Archive 6.9, Lore 7.1, Helm 8.0 (2026); every node fell | (a) Real GDP, annual, US Bureau of Economic Analysis. (b) Recession dates, NBER Business Cycle Dating Committee | PASS if annual real GDP growth < 0.0% in BOTH 2027 AND 2028 (BEA, first annual estimate). FAIL otherwise. Threshold confirmed by Kari, 23 Oct 2026 | 30 June 2029 (BEA) | Base rate to be computed from BEA annual real GDP 1948–2025 and NBER peak dates (share of two-year windows with a negative year or a peak) before lodging | Consensus expects continued positive US growth. Record the IMF WEO October 2026 US projections and the Fed SEP median before lodging. CAMS differs by reading the decline as broad (all eight nodes) rather than cyclical | Risky |

## 2. Other entries (converted from the 12 July 2026 memo)

| ID | Society | Plain claim | CAMS reading (JUNO v3) | Outcome and source | Pass / fail threshold | Resolves | Base rate | Consensus | Risk |
|---|---|---|---|---|---|---|---|---|---|
| R-04 | Germany | Industrial output falls further | Flow 5.2 is the lowest node (2026); mean V 8.59 (2026) | Production in industry index (excl. construction), annual average, Destatis | PASS if the 2028 annual average is below the 2026 annual average. FAIL otherwise | 30 June 2029 | Base rate to be computed from Destatis index history (share of two-year windows with a fall) before lodging | Consensus expects a modest recovery from 2026. Record the Bundesbank or IMF projection before lodging | Risky |
| R-05 | Iran | Inflation stays very high | Flow −2.27 and Hands −1.96 (2026); mean V 3.15, down from 4.97 (2024) | Consumer price inflation, annual average, IMF WEO (Statistical Centre of Iran / Central Bank of Iran) | PASS if annual inflation ≥ 30% in both 2027 and 2028. FAIL otherwise | 30 June 2029 | Base rate to be computed from IMF WEO Iran CPI series before lodging | In line with consensus | Safe |
| R-06 | Australia | No recession | Mean V 8.40 (2024) to 8.90 (2026); Craft 6.9 lowest and flat | Real GDP, annual, Australian Bureau of Statistics (5206.0) | PASS if annual real GDP growth > 0.0% in both 2027 and 2028. FAIL otherwise | 30 June 2029 | Base rate to be computed from ABS annual series 1960–2025 before lodging | In line with consensus | Safe |
| R-07 | USA | The US takes a "next step down" in 2027–28 (source claim: neuralnations.org/crisis-before-trump, "has set the stage for the next step down") | v3 US series: 2008 is the frozen-rule break, mean V shift −2.98 (5-yr post minus 5-yr pre), no reversion within 3 years; 2020 borderline (−2.78). 2025–26 is a uniform −1.75 across all eight nodes, likely scorer drift rather than structure (`/workspace/analysis/slide_robustness/PREREG_usa_breaks.md`, `usa.py`) | Two legs. (a) CAMS-structure leg: v3-method US scores for 2022–2031, rescored on a FROZEN scorer and prompt recorded here before lodging. (b) Public outcome leg: as R-03, annual real GDP (BEA) and recession dates (NBER Business Cycle Dating Committee) | Thresholds proposed, awaiting Kari confirmation. (a) PASS if mean V for 2027–31 minus mean V for 2022–26 is ≤ −2.98 AND no year in 2028–30 returns within the 2022–26 mean ± 1 SD. (b) PASS if annual real GDP growth < 0.0% in BOTH 2027 AND 2028 (BEA, first annual estimate); leg (b) threshold confirmed by Kari, 23 Oct 2026, leg (a) still proposed. R-07 passes only if both legs pass. Leg (b) overlaps R-03; R-07 adds the structural leg. Scorer-drift caveat: leg (a) is UNSCORABLE if the scorer or prompt changes | 30 June 2029 for leg (b); leg (a) needs 2027–31 scores, so not before 2032 |  |  | Risky |

## 3. Mapping of every existing prediction

| Existing prediction | Source | Classification | Becomes |
|---|---|---|---|
| USA P1 (B̄ ≤ 13.0) | `prospective-tests/CAMS_USA_Prospective_Test_2026-2028.md` | Retired as self-scored. Also see the erratum on bond scale | R-03 carries the US claim |
| USA P2 (two of four policy-incoherence indicators) | same | Kept as event-graded, flagged as an easy bar | Unchanged; not counted toward the risky pass rate |
| USA P3 (security share, budget cuts, new authority) | same | Kept as event-graded (OMB budget records, Federal Register) | Unchanged; not counted toward the risky pass rate |
| USA P4, P5, P6, P7 | same | Retired as self-scored (ensemble S, V gaps, V range, B̄) | None |
| Memo 1, USA Helm "breaches V < 4.0" | `CAMS_OOS_Predictions_2027_2028.pdf` | Converted | R-03 |
| Memo 2, Germany Flow/Craft Local Node Failure | same | Converted | R-04 |
| Memo 3, China stays Stable Adaptive | same | Converted | R-01 |
| Memo 4, Iran Flow < 2.0 | same | Converted | R-05 |
| Memo 5, Russia stays Strained | same | Converted | R-02 |
| Memo 6, Australia Craft to 8.08 | same | Converted | R-06 |
| Table 12 P-01, P-02 (USA NV levels and node order) | `juno/OOS_PREDICTIONS_2026-2028.md` | Retired as self-scored | R-03 |
| P-03, P-04, P-11 (Germany Archive/Lore path, Helm-led, coupling range) | same | Retired as self-scored | R-04 |
| P-05 (UK devolution, Hands leads Flow) | same | Retired: (a) names no series; (b) self-scored | None |
| P-06, P-07 (Ukraine Shield branch, mean NV) | same | Retired as self-scored | None |
| P-08 (Russia path by node ordering) | same | Retired as self-scored | R-02 |
| P-09 (Australia mean NV ≥ 7.0 / 7.5) | same | Retired as self-scored | R-06 |
| P-10 (China mean NV ≥ 3.0, Helm > 2.5) | same | Retired as self-scored | R-01 |
| P-12 (cross-corpus coupling speed) | same | Retired as self-scored | None |
| USA–Argentina similarity claim | same, §4 | Not a prediction | None |
| `JUNO_B_EP_Predictions.csv` (197 rows, 1801–2022) | repo root | Not predictions: retrospective episode labels (see erratum) | None |

"Retired" means retired from this register only. The original freezes stay in force as written and can still be scored internally. They do not count as evidence about the world.

Counts: 2 kept, 6 converted (into R-01 to R-06), 17 retired as self-scored (12 from Table 12, 5 from the USA document), plus 1 similarity claim and 1 CSV reclassified as non-predictions.

## 4. Grading rules

1. Scored by outcome only. The named source's first full-year figure decides. A CAMS reading never decides an entry.
2. No revision after lodging. Thresholds, sources, windows and societies are fixed at the hash date.
3. Dated amendments only. A source discontinued or revised is handled by a dated, appended note naming the substitute. The original row is never edited. If no substitute exists, the entry is UNSCORABLE, not passed.
4. Pass rate. R-01 to R-06 are scored PASS, FAIL or UNSCORABLE. Report the pass rate on the three risky lead entries separately from the full six, and report each against its computed base rate. Kept USA P2 and P3 are reported but do not count. If two or more of R-01 to R-03 fail, the register records that CAMS readings did not anticipate the risky outcomes.
5. Before lodging, every "base rate to be computed" and "record the projection" cell must be filled from the named source with an access date. Any cell left empty means that entry is not lodged.

## 5. Lodging

1. Fill section 1 and 2 base rates and consensus figures (Kari).
2. Rename to `CAMS_Prospective_Register_v1.md`, commit, and record `sha256sum` of the file and the commit SHA here.
3. TODO (Kari): lodge the same file on OSF as a timestamped registration and record the OSF DOI here.


## 6. Amendments

- 23 October 2026 (Australia/Sydney): added R-07 (USA "next step down"). Its leg (a) is graded on a CAMS score, an exception to the one-line rule above; it is reported separately and is not counted in the section 4 pass rate unless Kari decides otherwise. Thresholds proposed, awaiting Kari confirmation. Base-rate and consensus cells left empty, as for the other entries.
- 23 October 2026 (Australia/Sydney): R-03 and R-07 leg (b) threshold changed on Kari's clarification ("negative real growth for the next two years was the bet"). Now: BEA annual real GDP growth < 0.0% in both 2027 and 2028; confirmed by Kari. Prior draft wording, kept for the record: "PASS if annual real GDP growth < 0.0% in 2027 or 2028 (BEA, first annual estimate), OR the NBER dates a business-cycle peak between 1 Jan 2027 and 31 Dec 2028", resolving "30 June 2029 for BEA; NBER announcement accepted until 31 Dec 2029". NBER remains listed as a source but no longer grades. All other thresholds remain proposed.
