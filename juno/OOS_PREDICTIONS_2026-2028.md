# Frozen 2026–2028 out-of-sample prediction register

**This file is a dated freeze.** Later work may **score** these claims. It must not **edit** them. Transcription-error corrections go in the erratum appendix; they do not rewrite the numbered table.

| Field | Value |
|---|---|
| Freeze date (this artifact) | 28 August 2026, Australia/Sydney |
| Original lock date | **10 June 2026** (explicit: “locked as of 10 June 2026”) |
| Later public issuance (not a replacement of the twelve) | 12 July 2026 (six nation-level 2027–2028 forecasts; see §5) |
| Claims frozen here | **12** (P-01 … P-12) plus **one similarity claim** (not an OOS prediction) |
| Classifier for subsequent scoring | JUNO v1.2-Final |
| External-validation status | **Not externally validated.** JUNO v1.2 is a closed-form spec. Prospective holdout scoring of this list is the test named in the formalism. Do not describe this freeze, JUNO v1.2, or CAMS as “validated” on the strength of this file. |

Pointers:

- Formalism: [`JUNO_v1.2-Final_Formalism.md`](JUNO_v1.2-Final_Formalism.md) (§6: prospective validation on fresh holdout data remains open; genuine validation of v1.2 requires this holdout).
- Canonical status: [`CANONICAL_STATUS.md`](CANONICAL_STATUS.md) (production classifier = JUNO v1.2-Final; bond matrix is algebraically rank-1, so network visuals are **illustrative**).
- Source paper (HTML): [`../research/cams-validation-2026.html`](../research/cams-validation-2026.html)
- Public URL: <https://neuralnations.org/research/cams-validation-2026.html>
- First-lock commit (10 June 2026, 11:06 +1000): [`bf344e3b6f16aa57436747e8ee7d3f475c1cdec1`](https://github.com/KaliBond/wintermute/commit/bf344e3b6f16aa57436747e8ee7d3f475c1cdec1)
- Paper as of this freeze’s base (`origin/main` `c79bcec`): file last touched by [`d959be47c544182ba6babf37f5ebc5134bd8f442`](https://github.com/KaliBond/wintermute/commit/d959be47c544182ba6babf37f5ebc5134bd8f442) (16 July 2026 Stewards gloss; see Erratum A)

---

## 1. Freeze rule

1. The twelve claims in §3 are the 10 June 2026 lock, quoted from Table 12 of McKern, *Graph-Theoretic Diagnosis of Societal Collapse: Empirical Validation of the Complex Adaptive Model State (CAMS) Framework Across 18 Societies (10–2026 CE)*, Version 1.0, 10 June 2026. The source title uses the word “Validation”; that is the paper’s title, not a status granted by this freeze.
2. This list is frozen. Scoring files, lab notes, and later papers may mark HIT / MISS / UNSCORABLE against it. They may not substitute a quieter wording, a new threshold, a different society, or a shifted window.
3. If a transcription error is found, add a dated erratum in §7. Do not silently rewrite §3.
4. The USA 2024–2026 vs Argentina Dirty War (1975–1983) figure is a **similarity claim** (crisis-fingerprint cosine match in the source paper), **not** one of the twelve OOS predictions. It is frozen separately in §4.
5. The 12 July 2026 six-nation memorandum does **not** supersede Table 12. It is a later, overlapping-but-distinct public issuance. It is recorded in §5 so that neither lock is dropped. The numbered freeze is the twelve, because that is the earliest dated public lock that is explicitly “locked” and that contains twelve claims.
6. Bond strength in JUNO v1.2 is a rank-1 outer product. Claims that invoke “network architecture,” “coupling,” or cascade topology remain frozen as written; network diagrams used when scoring them stay **illustrative** until a successor introduces genuine pair terms ([`CANONICAL_STATUS.md`](CANONICAL_STATUS.md)).
7. Do not import homepage 75–90% figures, “literal thermodynamics,” or other marketing talking points into scoring of this list.

---

## 2. Source lock language (verbatim)

From §6 of the 10 June 2026 paper (`research/cams-validation-2026.html`, Table 12 caption and surrounding text):

> The following predictions are locked as of 10 June 2026 and will be assessed quarterly against CAMS scoring of current data. Each prediction includes: (a) a specific, measurable outcome; (b) the historical analogue from which the prediction is derived; (c) the timeframe; (d) the falsification criterion.

> **Table 12.** Falsifiable Prediction Register — locked 10 June 2026. Status: all predictions are OPEN (unresolved). Assessment cadence: quarterly. Falsification criterion: actual trajectory diverges from predicted by >1.0 Node Value units for >2 consecutive years within the window, OR a categorically different structural outcome (e.g., recovery where consolidation predicted) occurs. Analogue = historical precedent used to derive prediction.

> All predictions below are pre-registered as of 10 June 2026. Quarterly CAMS scoring of each society will assess trajectory conformance. Predictions are considered validated if actual Node Value trajectories track predicted trajectories within ±1.0 annually for ≥75% of measurement windows. Divergences will be documented and used to refine the model.

The source’s own word “validated” in that protocol is quoted here as the paper’s scoring rule. This freeze does not apply that word to JUNO v1.2.

---

## 3. The twelve locked claims

Wording in **Original wording** is from Table 12 at the 10 June 2026 first-lock commit `bf344e3`. One parenthetical gloss on P-02 was later changed (16 July 2026); the locked text is the 10 June wording. See Erratum A.

Observable “NV” in the source is Node Value \(V = C + K - S + 0.5A\) (same operator as JUNO v1.2 \(V_i\)). Where the source does not name a numeric threshold, the threshold cell is left blank.

| # | ID | Claim (compressed) | Target society / window | Observable | Original wording (verbatim) | Numeric threshold(s) as stated | Source |
|---|---|---|---|---|---|---|---|
| 1 | P-01 | USA aggregate NV stays low through 2030 | USA / 2026–2030 | 8-node mean NV (annual; 5-year average) | Aggregate Node Value remains ≤ 5.0 (5-year average across all 8 nodes) through 2030; no single year exceeds NV = 6.0 before 2029 | 5-year mean NV ≤ 5.0 through 2030; no year NV > 6.0 before 2029. Falsify: NV exceeds 6.0 in any year before 2029 | `research/cams-validation-2026.html` Table 12; lock `bf344e3`; <https://neuralnations.org/research/cams-validation-2026.html> |
| 2 | P-02 | USA Stewards & Flow remain weakest through 2028; Helm overtakes Flow by 2030 | USA / 2026–2031 | Node NV: Stewards, Flow, Helm | Stewards (bureaucracy) and Flow (markets/distribution) remain the two lowest-scoring nodes through 2028; Helm recovers faster than Flow (Helm NV exceeds Flow NV by 2030) | Through 2028: Stewards and Flow = two lowest nodes. By 2030: Helm NV > Flow NV. Falsify: Flow recovers before Stewards, OR both recover above NV = 4.0 before 2029 | same (10 June wording; see Erratum A) |
| 3 | P-03 | Germany takes recovery **or** consolidation path by end 2027 | Germany / 2026–2028 | Archive NV, Lore NV (and “declining SD” — SD of what is **not named** in the source) | By end 2027, Germany will have entered one of two measurably distinct paths: (a) Recovery path — Archive NV > 3.5 AND Lore NV > 2.5; (b) Consolidation path — Archive NV < 2.0 AND Lore NV < 2.0 with declining SD | (a) Archive NV > 3.5 and Lore NV > 2.5; (b) Archive NV < 2.0 and Lore NV < 2.0 with declining SD. Falsify: both Archive and Lore remain 2.0–3.5 through 2028 | same |
| 4 | P-04 | If P-03b, Helm rise of ≥2.0 over 3 years precedes Lore rise | Germany / 2026–2034 (conditional on P-03b) | Helm NV, Lore NV, V_Shield vs V_Helm | If consolidation path (P-03b) is entered, a measurable change in Helm node (NV increase ≥ 2.0 over 3 years) will precede a corresponding Lore NV increase, consistent with coercive recoupling flag (V_Shield > V_Helm triggering Helm recovery) | Helm ΔNV ≥ 2.0 over 3 years before Lore increase; flag V_Shield > V_Helm. Falsify: Lore recovers before Helm, OR all nodes recover simultaneously | same |
| 5 | P-05 | UK recovery with devolution pressure; Hands NV leads Flow NV | UK / 2026–2031 | Unspecified “Scotland/Northern Ireland governance metrics”; Hands NV; Flow NV | UK institutional recovery will be accompanied by: (a) measurable devolution pressure (Scotland/Northern Ireland governance metrics deteriorating from London perspective); (b) Hands NV recovery leading Flow NV recovery (labour restructuring preceding market recovery) | (a) no numeric threshold stated for devolution metrics (series unnamed). (b) Hands NV recovers before Flow NV. Falsify: stable constitutional position AND Flow recovers before Hands | same |
| 6 | P-06 | Ukraine Shield NV ≥ 1.0 through 2026 ⇒ resilience path; Shield < 0 in 2026/27 ⇒ Helm/Stewards fragment within 18 months | Ukraine / 2026–2027 | Shield NV, Helm NV, Stewards NV | If Shield node maintains NV ≥ 1.0 through 2026 (state monopoly on coercion preserved), the resilience path (7–12 year recovery) becomes the more probable trajectory. If Shield NV falls below 0 in 2026 or 2027, fragmentation indicators will appear in Helm and Stewards within 18 months | Shield NV ≥ 1.0 through 2026 (resilience branch); Shield NV < 0 in 2026 or 2027 (fragmentation branch, 18-month lag). Falsify: Shield NV < 0 but Helm and Stewards remain stable | same |
| 7 | P-07 | If P-06 resilience: Ukraine mean NV ≤ 4.0 before 2030; 5.0–6.0 by 2034 if EU support holds | Ukraine / 2028–2034 (conditional on P-06 resilience path) | 8-node mean NV | Conditional on P-06 resilience path: Ukraine NV (8-node mean) will not exceed 4.0 before 2030, reflecting sustained post-war reconstruction stress; NV will reach 5.0–6.0 by 2034 if external institutional support (EU integration process) remains active | NV ≯ 4.0 before 2030; NV 5.0–6.0 by 2034 if EU support active. Falsify: NV > 4.0 before 2028, OR NV < 3.0 through 2034 | same |
| 8 | P-08 | Russia 2026–28 path resolved by Craft/Flow vs Helm/Lore ordering | Russia / 2026–2028 | Craft NV, Flow NV, Helm NV, Lore NV | The trajectory ambiguity between economic recovery (USA 1930s analogue) and revolutionary reorganisation (China 1945–1965 analogue) will be resolvable by end 2028 via: Craft and Flow node trajectories. Economic recovery path → Craft NV increases ≥ 1.5 before Flow NV; reorganisation path → Helm and Lore NV increases ≥ 2.0 before Craft or Flow | Recovery: Craft ΔNV ≥ 1.5 before Flow. Reorganisation: Helm and Lore ΔNV ≥ 2.0 before Craft or Flow. Falsify: simultaneous NV increases across all nodes | same |
| 9 | P-09 | Australia 8-node mean NV ≥ 7.0 by 2027 and ≥ 7.5 by 2028 | Australia / 2026–2028 | 8-node mean NV | Australia NV (8-node mean) will return to ≥ 7.0 by 2027 and ≥ 7.5 by 2028, consistent with short-cycle commodity-shock recovery (Australia Depression 1929–1932 recovered within 3–4 years) | NV ≥ 7.0 by 2027; NV ≥ 7.5 by 2028. Falsify: NV remains < 7.0 through 2027 | same |
| 10 | P-10 | China mean NV stays ≥ 3.0 through 2031; Helm NV > 2.5; stress in Stewards & Archive | China / 2026–2031 | 8-node mean NV; Helm NV; Stewards NV; Archive NV | China's managed-stress trajectory will not tip into acute crisis (aggregate NV will not fall below 3.0 in any year through 2031); Stewards and Archive nodes will be the primary stress absorption points; Helm NV will remain above 2.5 throughout | Aggregate NV ≮ 3.0 any year 2026–2031; Helm NV > 2.5 throughout. Falsify: NV < 3.0 any year, OR Helm NV < 1.5 | same |
| 11 | P-11 | Germany coupling-range rises >0.70 on consolidation, falls <0.45 on recovery, by 2028 | Germany / 2026–2028 | Node-pair Spearman-ρ range (max ρ − min ρ); source baseline range = 0.55 | Germany's node coupling volatility (currently range = 0.55) will increase if consolidation path is entered (coupling reorganises toward tighter Helm–Shield, looser Archive–Flow) and decrease if recovery path is entered (coupling stabilises across all nodes). Consolidation path: range increases to >0.70 by 2028. Recovery path: range decreases to <0.45 by 2028 | Consolidation: range > 0.70 by 2028. Recovery: range < 0.45 by 2028. Baseline at lock: 0.55. Falsify: range remains 0.45–0.70 through 2028 | same |
| 12 | P-12 | Tight-coupling societies (mean ρ > 0.80) show faster \|ΔNV\| than loose-coupling (mean ρ < 0.55) in 2026–2028, differential ≥ 1.0 NV/year | Cross-corpus / 2026–2028 | Mean pairwise Spearman ρ; mean absolute year-on-year 8-node-mean NV change | Societies with tight coupling (ρ > 0.80) entering the 2026–2028 stress period will show faster aggregate NV change (both decline and recovery) than societies with loose coupling (ρ < 0.55) entering the same period — predicted differential of ≥ 1.0 NV unit per year in mean absolute NV change | Tight: mean ρ > 0.80. Loose: mean ρ < 0.55. Differential ≥ 1.0 NV unit per year in mean \|ΔNV\|. Falsify: loose faster than tight, OR no significant differential | same |

**Analogues** (verbatim from Table 12; not restated as new claims):

| ID | Analogue (verbatim) | Falsification criterion (verbatim) |
|---|---|---|
| P-01 | USA Great Depression 1929–1939 + Argentina Dirty War 1975–1983 (ρ=0.97) | USA NV exceeds 6.0 in any year before 2029 — would falsify the 8–12 year recovery arc and suggest faster institutional recovery than either analogue |
| P-02 | Argentina Dirty War recovery arc (leadership recovered faster than distribution, 1983–1990) | Flow recovers before Stewards, OR both recover above NV = 4.0 before 2029 |
| P-03 | UK post-WWI recovery (1920–1926) vs Germany interwar consolidation (1925–1933) | Germany remains at intermediate Archive/Lore values (both 2.0–3.5) through 2028 — would indicate path ambiguity persists, falsifying the 2026–2028 decision window prediction |
| P-04 | Germany interwar consolidation 1925–1933; CAMS coercive recoupling flag pattern | Lore recovers before Helm, OR recovery is broad-based (all nodes simultaneously) rather than Helm-led |
| P-05 | UK post-WWI trajectory 1920–1926 — devolution pressures, labour market restructuring, loss of imperial position | UK recovers with stable constitutional position (no measurable devolution pressure increase) AND Flow recovers before Hands — would falsify the post-WWI structural analogy |
| P-06 | Rome Late Empire: Shield collapse (loss of monopoly on coercion) preceded Helm fragmentation by 15–30 years; resilience requires Shield maintenance | Shield NV falls below 0 but Helm and Stewards remain stable — would falsify the cascade prediction and indicate Ukraine has found a novel resilience mechanism |
| P-07 | Post-WWII Western European reconstruction (1945–1955): societies with external Marshall Plan support recovered NV to ~6 within 10 years; without support, recovery took 20+ years | Ukraine NV exceeds 4.0 before 2028 (faster recovery than any post-conflict analogue with comparable starting position) OR remains below 3.0 through 2034 (slower recovery than even the worst-supported post-war analogues) |
| P-08 | USA Great Depression (Craft/Flow-led recovery) vs China Civil War (Helm/Lore-led ideological consolidation) | Both paths show simultaneous NV increases across all nodes — would indicate a novel trajectory distinct from both historical analogues |
| P-09 | Australia Depression 1929–1932 (ρ = 0.86); Canada Depression 1930–1939 (ρ = 0.84) | Australia NV remains below 7.0 through 2027 — would indicate stress is structural (institutional damage), not cyclical (commodity shock), requiring revision of the resilience-outlier classification |
| P-10 | China's Civil War & CCP consolidation (1945–1965): state maintained Helm control throughout, absorbing stress via Stewards and Archive degradation rather than Helm collapse | China NV falls below 3.0 in any year 2026–2031 OR Helm NV falls below 1.5 — would indicate loss of managed-stress capacity and transition to acute crisis mode, requiring reclassification from managed-stress to acute-crisis category |
| P-11 | CAMS coupling volatility finding: Argentina's high-volatility coupling (range 0.71) corresponds to political reorganisation phases; Germany's coupling stability (range 0.55) currently intermediate | Range remains 0.45–0.70 through 2028 — would indicate coupling structure is not a reliable leading indicator of path divergence, falsifying the coupling-as-early-warning prediction |
| P-12 | Coupling architecture predicts propagation speed — tight coupling = faster change in both directions; loose coupling = slower, more modular change | Loose-coupling societies show faster NV change than tight-coupling societies, OR no significant differential emerges — would falsify the coupling-architecture-predicts-propagation-speed claim |

### Items the source leaves unnamed (not filled in here)

- **P-03** “declining SD”: the paper does not say SD of which series (Archive, Lore, the eight-node vector, or something else).
- **P-05(a)** “Scotland/Northern Ireland governance metrics deteriorating from London perspective”: no series, file, or numeric cutoff is named.
- **P-06** “fragmentation indicators” in Helm and Stewards: no extra numeric test beyond the Shield branch.
- **P-12** “no significant differential”: no α / test statistic is named.
- **Similarity vs ρ:** methods §3.4 define the crisis match as **cosine similarity**; Table 5 labels the USA–Argentina figure “Similarity” 0.97; the §4.3.1 heading and P-01 analogue write **ρ = 0.97**. Frozen as written. This freeze does not decide which symbol is correct.

---

## 4. Similarity claim (not an OOS prediction)

The 10 June paper uses crisis-fingerprint matching (8-dimensional mean-NV vectors; cosine similarity; methods §3.4) as a **retrospective homology**, then derives some of the twelve predictions from those analogues. The match itself is **not** one of Table 12’s twelve.

Verbatim, §4.3.1 and Table 5 (`research/cams-validation-2026.html`):

> **4.3.1 USA 2024–2026 — Most Similar to Argentina Dirty War (ρ = 0.97)**

| Rank | Historical Crisis | Period | Similarity | Crisis Type |
|---|---|---|---|---|
| 1 | Argentina Dirty War | 1975–1983 | **0.97** | Military dictatorship, institutional dysfunction |
| 2 | USA Financial Crisis | 2008–2010 | 0.93 | Banking collapse, liquidity crisis |
| 3 | USA Great Depression | 1929–1939 | 0.88 | Economic collapse, institutional stress |

> Both USA 2024–2026 and Argentina 1975–1983 show dominant collapse in Flow (distribution/markets), near-equivalent Stewards degradation (bureaucratic incoherence), and Helm erosion, with Archive and Shield comparatively preserved. Argentina required 15–20 years to recover institutional coherence (1983–2003). Given the USA's higher starting institutional capacity, an estimated **8–12 year recovery arc** (2026–2034) is indicated.

(The 16 July 2026 node-definition commit replaced “bureaucratic incoherence” with “capital/asset-holder incoherence” in that paragraph. The sentence above is the 10 June lock text. See Erratum A.)

> **Critical Finding — USA Stress Matches Dictatorship-Era Argentina** A similarity of 0.97 to Argentina's Dirty War era (1975–1983) indicates the USA's current institutional stress profile is architecturally comparable to a period of military dictatorship, human rights collapse, and bureaucratic incoherence. This is not a prediction of identical outcomes—contextual factors (democratic institutions, rule of law, international position) differ substantially—but it indicates the USA is not in a normal recessionary cycle. Recovery timelines of 8–12 years, not 2–3, are historically indicated.

Methods §3.6 (verbatim constraint, still not an OOS prediction):

> Similarity scores indicate structural homology in institutional stress patterns, not deterministic outcome identity: USA's 0.97 match to Argentina 1975–1983 indicates comparable stress architecture, not that USA will become Argentina.

**Label:** similarity claim / fingerprint match. **Not** Table 12 P-01…P-12. Do not score it as a hit/miss OOS forecast.

---

## 5. 10 June vs 12 July (12 July does not supersede the twelve)

| | 10 June 2026 | 12 July 2026 |
|---|---|---|
| Artifact | McKern graph paper, Table 12 | Memorandum *Graph-Theoretic Civilisational Diagnosis & Out-of-Sample Certification*; site page `predictions.html` |
| Explicit lock? | Yes: “locked as of 10 June 2026”; “Table 12. … locked 10 June 2026” | “Date of Issuance: 12 July 2026”; “The architecture is locked”; six forecasts “officially registered and certified for the 2027–2028 window” |
| Count | **Twelve** claims (P-01…P-12) | **Six** nation-level 2027–2028 trajectories |
| Societies | USA, Germany, UK, Ukraine, Russia, Australia, China, plus cross-corpus | USA, Germany, China, Iran, Russia, Australia |
| Claim type | Node-Value thresholds, node-order, coupling-range, cross-corpus speed | Regime-label trajectories (Local Node Failure, Stable Adaptive, Systemic Crisis, Strained) under the memorandum’s classifier |
| Framework as labelled | CAMS v1.0-Final (paper) | CAMS v1.0-Final · JUNO Ensemble |
| First commit | `bf344e3` 2026-06-10 11:06 +1000 | `a01a69c` 2026-07-12 10:09 +1000 (`predictions.html` + `CAMS_OOS_Predictions_2027_2028.pdf`) |
| Public URL | <https://neuralnations.org/research/cams-validation-2026.html> | <https://neuralnations.org/predictions> |

**Handling:** freeze the 10 June twelve as the canonical numbered list (earliest explicit lock of twelve). Record the 12 July six below so they are not dropped. 12 July does **not** replace P-01…P-12: overlapping societies, different observables, different windows, and a different classifier table (the PDF still lists an LNF s-arm and Stable \(\bar{B} > 0.30\); JUNO v1.2-Final deletes the LNF s-arm and uses Stable \(\bar{B} > 0.28\), Strained floor \(\bar{V} \ge 4\)).

### 5.1 Twelve-July six (verbatim forecasts; not merged into Table 12)

Source: `CAMS_OOS_Predictions_2027_2028.pdf` / `predictions.html`, issuance 12 July 2026, commit `a01a69c56782570865e003d98bf0c5f0372bf536`.

1. **United States (Systemic Contagion Trajectory).** “The derivative of the structural decay curve guarantees that the Helm node will breach the critical \(V_{min} < 4.0\) threshold by mid-2027, triggering an official Local Node Failure. Because the US behaves as a scale-free network, this local failure will cascade, causing secondary coupling degradation in Stewards (execution) and Lore (cultural memory) by 2028.” Basis figure stated: \(V_{Helm} = 4.5\) in 2023.
2. **Germany (Industrial Decoupling Trajectory).** “Germany will trigger a formal Local Node Failure via the Flow/Craft axis by late 2027, resulting in a severe drop in aggregate viability (\(\bar{V}\)).” Basis figures stated: Craft 20.25 in 2023; Flow 5.2 by 2026.
3. **China (Cohesive Transmorphance Trajectory).** “China will remain firmly within the Stable Adaptive regime throughout the 2027–2028 window.” Basis figures stated: Craft 17.1 and Archive 15.5 as of 2025; “internal edge weights (\(\bar{B}\)) consistently average above 30.0” — **quoted as written**; that \(\bar{B}\) scale is incompatible with JUNO v1.2 \(B_{ij} \in [0,1]\) and is not “corrected” here.
4. **Iran (Active Cascading Crisis).** “Iran remains locked in a Systemic Crisis profile. Linear extrapolation indicates that the Flow node will remain pinned below a value of 2.0 through 2028.” Basis figures stated: Stewards 0.2 in 2025; Flow 0.9 in 2026. Iran is **absent** from the 10 June twelve.
5. **Russia (Strained Equilibrium).** “The Flow node shows a consistent downward trajectory, projected to degrade to approximately 5.59 by 2028. … remaining within the Strained envelope without crossing into localized failure (\(V_{min} < 4.0\)).” Basis figure stated: \(\bar{V} = 8.75\) in 2026.
6. **Australia (Resilient Sybond Recovery).** “Australia's Craft node shows a positive trajectory, projected to recover toward 8.08 by 2028. … pull itself safely back toward the upper bounds of the Strained regime.” Basis figure stated: Craft 6.9 in 2026.

### 5.2 Compressed 10 June → 12 July delta (not a silent drop)

| Society | 10 June (frozen twelve) | 12 July (six-nation memo) | Relation |
|---|---|---|---|
| USA | P-01 NV caps; P-02 Stewards/Flow/Helm order | Helm \(V < 4.0\) by mid-2027 → Local Node Failure; Stewards & Lore coupling degradation by 2028 | Overlap on USA stress; **different observables** (mean NV vs Helm \(V_{min}\) / regime label) |
| Germany | P-03 Archive/Lore path by end 2027; P-04 conditional Helm-led; P-11 coupling range | Flow/Craft Local Node Failure by late 2027 | Overlap on Germany 2027; **different nodes** (Archive/Lore vs Flow/Craft) |
| China | P-10 mean NV ≮ 3.0; Helm NV > 2.5 | Remain Stable Adaptive 2027–2028 | Compatible direction, different metric |
| Russia | P-08 path resolved by Craft/Flow vs Helm/Lore | Stay Strained; Flow ≈ 5.59 by 2028; \(V_{min}\) ≮ 4.0 | Overlap on 2026–28; 12 July adds a Flow point forecast the twelve did not state |
| Australia | P-09 mean NV ≥ 7.0 by 2027, ≥ 7.5 by 2028 | Craft → 8.08 by 2028; upper Strained | Overlap on recovery; different scalar (mean NV vs Craft NV / regime) |
| UK | P-05 | *absent* | 12 July drops UK |
| Ukraine | P-06, P-07 | *absent* | 12 July drops Ukraine |
| Cross-corpus | P-12 | *absent* | 12 July drops P-12 |
| Iran | *absent* | Systemic Crisis; Flow < 2.0 through 2028 | 12 July **adds** Iran |

An earlier USA-only seal exists (`prospective-tests/CAMS_USA_Prospective_Test_2026-2028.md`, **Sealed: 28 April 2026**, seven predictions P1–P7 under CAMS v3.0). It is **not** the twelve and is not frozen here.

---

## 6. Scoring protocol (year-close; no peeking restatement)

Score against §3 as written. Do not rephrase a claim to make it easier or harder to hit.

### 6.1 When a year closes

For each claim whose window includes that calendar year (or whose next terminal date has just passed):

1. Take Node Value from the JUNO v1.2 operators (\(V_i = C_i + K_i - S_i + 0.5 A_i\); round aggregates to 6 decimal places before threshold comparison per [`JUNO_v1.2-Final_Formalism.md`](JUNO_v1.2-Final_Formalism.md) §2). Do not switch to a later operator set in order to rescue a claim.
2. Use the then-current `juno/JUNO_Unified_Dataset.csv` (or a dated extract named in the scoring note). Record the dataset commit SHA on the score sheet.
3. Mark **only** one of:

| Mark | Meaning |
|---|---|
| **HIT** | The observation satisfies the claim’s stated pass/threshold condition for that year, or — for window-terminal claims — the whole-window condition is met when the window has closed. |
| **MISS** | The observation meets a stated falsification criterion (Table 12 column, quoted in §3), or a year-level threshold that the claim said would not occur, occurred. |
| **UNSCORABLE** | Required society-year incomplete (missing C/K/S/A or not eight nodes); named series absent (e.g. P-05 devolution metrics); conditional parent failed/unresolved so the child does not apply yet (P-04, P-07); or the year is outside the claim window. State which. |

4. Conditional claims (P-04, P-07): if the parent branch was not entered, mark the child **UNSCORABLE (parent branch not taken)** — not HIT and not MISS.
5. Do not restated the claim, “clarify” a threshold, or drop an analogue at scoring time. If the wording is unusable, mark UNSCORABLE and file an erratum.
6. Source protocol (quoted in §2), applied in addition to the year-close marks: quarterly cadence; trajectory-within-±1.0-NV for ≥75% of windows was the paper’s own “validated” bar; divergence >1.0 NV for >2 consecutive years, or a categorically different structural outcome, was the paper’s falsification bar. Record those diagnostics on the score sheet without rewriting §3.

### 6.2 What this freeze is not

- Not a claim that JUNO v1.2 or CAMS is externally validated.
- Not a licence to treat rank-1 bond visuals as measured topology.
- Not a merge of the 12 July six into the twelve.
- Not a rescoring or re-run of models.

Scoring comes later, in a separate artifact.

---

## 7. Errata (append only)

### Erratum A — 16 July 2026 Stewards gloss (post-lock wording change)

**Commit:** [`d959be47c544182ba6babf37f5ebc5134bd8f442`](https://github.com/KaliBond/wintermute/commit/d959be47c544182ba6babf37f5ebc5134bd8f442) (16 July 2026, +1000) — “fix: correct Stewards node definition, drop Archive bureaucracy framing”.

This commit did **not** add, drop, or retarget any of P-01…P-12. It changed the parenthetical on **P-02** and the USA–Argentina prose:

| Location | 10 June 2026 lock (`bf344e3`) | After 16 July 2026 (`d959be4`, still on `main` at this freeze) |
|---|---|---|
| Table 12 P-02 claim | Stewards **(bureaucracy)** and Flow (markets/distribution) remain the two lowest-scoring nodes through 2028; … | Stewards **(capital/asset-holders)** and Flow (markets/distribution) remain the two lowest-scoring nodes through 2028; … |
| §4.3.1 USA–Argentina paragraph | “Stewards degradation **(bureaucratic incoherence)**” | “Stewards degradation **(capital/asset-holder incoherence)**” |
| Table 1 Stewards row | Functional role: Bureaucratic administration | Functional role: Capital & asset ownership |

**Freeze rule applied:** §3 quotes the **10 June** P-02 wording. The 16 July gloss is recorded here, not silently treated as the original lock. Scoring of P-02 still uses the Stewards **node** (same eight-node label); it does not depend on the parenthetical. If a later reader needs the current public HTML, that HTML follows `d959be4`.

No other Table 12 cell changed between `bf344e3` and `origin/main` at freeze time.

---

## 8. Honesty / status

- Production classifier: **JUNO v1.2-Final** ([`JUNO_v1.2-Final_Formalism.md`](JUNO_v1.2-Final_Formalism.md)).
- Bonds are rank-1; networks are illustrative ([`CANONICAL_STATUS.md`](CANONICAL_STATUS.md)).
- Formalism §6: the 3,002-year blind corpus is operator coherence on model-generated panels, not external validity. “Genuine validation of v1.2 requires fresh holdout data, aligned with the prospective 2026–2028 monitoring programme.”
- This freeze is that monitoring programme’s locked claim list. Until the holdout is scored, do not say the instrument is validated.
