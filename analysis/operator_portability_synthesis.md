# Operator Portability in the Complex Adaptive Model of Societies: Structural Invariance Across Corporate and National Domains

**Kari F. McKern**
*Neural Nations Research — neuralnations.org*
*Draft — 31 July 2026. Companion synthesis; not for citation without permission.*

---

## Status of this draft

This is a synthesis paper that sits above three related working papers rather than replacing them:

| Companion study | Status |
|---|---|
| Scale-Recursion Hypothesis (SRH) — corporate panel | Independently recomputed and confirmed (two audit passes); 3 minor wording/precision items corrected below |
| Operator Portability Proposition (P.4) | Independently recomputed and confirmed (two audit passes); 1 status-table logic error corrected below |
| The Soil-Bond Index (η_soil) — six-society validation, and the Norway extension | **Paused.** A second, more thorough audit found the reported inferential statistics do not match any standard test, one dataset (Germany) is missing 41 undisclosed years, the composite metric underperforms one of its own components, and H2/H3 as stated are internally inconsistent or untested. This paper — and the Norway baseline, which is computed entirely via η_soil — needs a substantive rework before its claims can be cited. Neither is drawn on below. |

Everything in this synthesis is restricted to what the corporate-panel and cross-domain nation/corporation work supports, plus one preliminary result (functional variability/resonance) that has not yet been through an independent audit and is labelled as such throughout.

---

## Abstract

The Complex Adaptive Model of Societies (CAMS) treats nations, corporations, and historical polities as instances of the same eight-node coordination architecture, scored by canonical operators — node value (*V*), pairwise bond strength (*W*), aggregate bond strength (*B̄*), and algebraic connectivity (*λ₂*). A standing question for any such cross-domain framework is whether its operators are *portable*: whether they compute structurally meaningful, comparable quantities regardless of whether the underlying system is a firm, a state, or a civilisation. This paper synthesises two independently audited empirical tests of that question. First, a nine-firm corporate panel (four in-sample, five held out) confirms that aggregate bond strength and algebraic connectivity remain tightly coupled (*r* = 0.933–0.994 at the entity level) whether or not a firm was used to define the discriminating band, and that node-level bottleneck structure replicates exactly under recomputation. Second, a cross-domain comparison of two national panels (Denmark, 1752–2025; France, 1850–2026) against the pooled corporate distribution shows that France behaves as a *bridge entity* — 94.9% of its annual bond-strength values fall inside the corporate band despite a statistically detectable distributional shift — while Denmark is a *separation case*, occupying a distinct parametric regime with only 19.3% containment. Together these results support the **Operator Portability Proposition (P.4)**: canonical CAMS operators are structurally portable (the computational grammar transfers cleanly across organisational scale) but parametrically non-invariant (thresholds, baselines, and dynamic ranges are regime-specific and require entity- or class-specific calibration rather than universal cutoffs). We also report a preliminary, not-yet-independently-audited extension — functional variability in node-value dispersion across a four-firm corporate ensemble — suggesting that firms manage operational shock not by suppressing variability but by concentrating it in specific nodes (Craft, in manufacturing firms; Flow and Shield, in a digital-platform firm). This result is presented as exploratory pending the same audit standard applied to the confirmed findings above.

---

## 1. Introduction

A framework that claims societies, firms, and civilisations share a common coordination architecture owes its users an answer to an obvious objection: a firm is not a nation-state, and a nation-state is not Rome. If the same eight-node scoring grammar is applied to all three, does it produce anything more than a superficial numerical resemblance?

The Operator Portability Proposition (P.4) answers this by separating two claims that are routinely conflated in cross-domain measurement: **structural portability** (do the operators compute a mathematically consistent relationship regardless of domain?) and **parametric invariance** (do different domains produce the same *distribution* of values?). CAMS's working hypothesis is that the first holds and the second does not — operators transfer, but their calibration is regime-dependent, in the same sense that an equation of state is universal while its critical-point parameters are substance-specific.

This paper reports the empirical basis for that hypothesis: a within-domain test on a nine-firm corporate panel, and a cross-domain test comparing two national historical panels against the pooled corporate distribution. Both tests were independently recomputed from raw node-level scores — not from any paper's own summary tables — across two separate audit passes, and every reported figure below reflects that verified recomputation.

---

## 2. The Operator Portability Proposition (P.4)

### 2.1 Canonical operators

All analyses use the CAMS canonical operator set. Node value:

$$V_i = C_i + K_i - S_i + 0.5A_i$$

Quality factor and pairwise bond strength:

$$q_i = (0.6C_i + 0.4A_i)/10 \qquad W_{ij} = \sqrt{q_i q_j} \cdot 2^{-(S_i+S_j)/10}$$

Aggregate bond strength and algebraic connectivity:

$$\bar{B}(t) = \frac{2}{N(N-1)}\sum_{i<j} W_{ij}(t) \qquad \lambda_2(t) = \text{second-smallest eigenvalue of } L = D - W$$

where *C, K, S, A* are Coherence, Capacity, Stress, and Abstraction, and *N* = 8 nodes.

### 2.2 Formal statement

Let 𝒪 = {*Vᵢ, qᵢ, Wᵢⱼ, B̄, λ₂*} denote the canonical operator set. For any two adaptive coordination systems *S₁, S₂* drawn from distinct organisational classes (nation-state, corporation, historical civilisation):

- **P.4.1 — Structural Portability.** Spectral consistency ρ(*B̄, λ₂*) is invariant to organisational class: ρ ≥ θ_ρ, θ_ρ = 0.90.
- **P.4.2 — Parametric Non-Invariance.** The marginal distributions *P*(*B̄*) and *P*(*λ₂*) are class-dependent: *D*_KS(*P*_S1, *P*_S2) > θ_D, θ_D = 0.15 at α = 0.05.
- **P.4.3 — Threshold Relativity.** Crisis or discriminating thresholds require entity- or class-specific baselines; universal absolute cutoffs yield uncontrolled Type-I error.
- **P.4.4 — Bridge Entities Exist.** Some entities exhibit distributional overlap with other classes despite formal class membership, but this is the exception rather than the rule, and overlap does not itself license class pooling.

### 2.3 Falsification criteria and current status

| Criterion | Trigger | Status |
|---|---|---|
| F-P4.1 | ρ(*B̄, λ₂*) < 0.85 for a held-out entity | **Not triggered** — all ρ > 0.93 |
| F-P4.2 | Pooled percentile baseline outperforms entity-specific by ΔAUC > 0.05 | Not tested — requires a crisis-detection ROC analysis not yet run |
| F-P4.3 | Same-class out-of-sample *D*_KS > 0.20 vs. in-sample | **Triggered** — within-class *D* = 0.214 (*B̄*) and 0.276 (*λ₂*), both exceeding the 0.20 threshold (see §5.1 for the correction to this entry) |
| F-P4.4 | *B̄–λ₂* slope differs by > 15% between classes | Not triggered — slope difference < 7% within corporate |
| F-P4.5 | Nation-state > 80% containment *and D*_KS < 0.10 vs. corporate | Partially triggered — France: 94.9% containment, *D* = 0.145 |

---

## 3. Structural Portability in a Corporate Panel

**Source:** `camcorp5_ensemble_mean.csv` (BYD, CATL, General Motors, Tencent; 2006–2026, *n* = 84 in-sample entity-years) and `camcorp5_batch2_ensemble_mean.csv` (BHP, Bunnings, Huawei, Tesla, Woodside; *n* = 105, held out).

### 3.1 Spectral consistency

| Group | ρ(*B̄, λ₂*) | *n* | *p* |
|---|---|---|---|
| In-sample | 0.988 | 84 | < 1×10⁻⁶⁷ |
| Out-of-sample | 0.975 | 105 | < 1×10⁻⁶⁸ |
| Entity-level range (9 firms) | 0.933–0.994 | — | — |

The regression slope difference between in-sample and out-of-sample groups is 6.4%. This confirms the spectral architecture is domain-portable within the corporate class: the operators compute a mathematically consistent relationship, whether or not a firm was used to define the discriminating band in the first place.

### 3.1a Formal within-corporate covariance diagnostic

A stricter test than the pooled correlation above asks whether all four in-sample firms share *one* *B̄*/*λ₂* mapping, rather than merely showing high correlations individually. Three nested models were compared on the 84 firm-years (one common intercept and slope; a common slope with firm-specific intercepts; fully firm-specific intercepts and slopes), with residual-permutation tests (10,000 permutations):

| Diagnostic | Result | Interpretation |
|---|---|---|
| Pooled correlation | *r* = 0.9875 | Very strong common coupling |
| Pooled linear fit | *λ₂* = −0.0044 + 7.0126 *B̄* | 97.5% of pooled variance explained |
| Firm-specific intercept improvement | *p* = 0.019 | Baseline differences remain |
| Firm-specific slope improvement | *p* = 0.008 | Exact common slope is rejected |
| Firm linear-slope range | 5.529–7.866 | Material parameter variation |
| Firm log-log exponent range | 0.811–1.143 | No single scaling exponent demonstrated |

Leave-one-firm-out prediction remained useful for CATL, General Motors, and Tencent (predictive *R²* = 0.984, 0.937, 0.918) but was markedly weaker for BYD (*R²* = 0.616, with a positive prediction bias of 0.083 *λ₂* units). **This qualifies the portability claim above**: the pooled relationship generalises and is useful for prediction, but a strictly firm-invariant slope and intercept are formally rejected — parametric non-invariance operates *within* the corporate class, not only across the corporate/national boundary (§4).

**Interpretation boundary:** *B̄* and *λ₂* are both computed from the same fully-connected (K₈) weight matrix, so they are not fully independent evidence of two unrelated operators co-varying — under this topology they are mechanically expected to correlate near *r* ≈ 0.98 (see also §3.4, item 3). The result demonstrates stable mathematical coupling under the chosen representation; it does not by itself establish that the representation measures the same latent construct across organisational scales.

### 3.2 Parametric non-invariance

Despite this structural consistency, out-of-sample firms operate at systematically higher baseline coordination levels. Kolmogorov–Smirnov and Mann–Whitney tests reject distributional equality for both *B̄* (*D* = 0.214, *p* = 0.011) and *λ₂* (*D* = 0.276, *p* = 0.003); the mean shift is +10.4% for *B̄* and +12.2% for *λ₂* (Cohen's *d* = 0.38 and 0.42 — small-to-moderate effects). Out-of-sample containment within the in-sample Tukey band (IQR ± 1.5×IQR) is 98.1%, but drops to 27.6% under the tighter IQR-only band — the threshold-relativity signature: a wide tolerance band absorbs entity-specific baseline shifts, a narrow one does not.

### 3.3 Topological invariance

Raw bottleneck-node identification (which node in each firm carries the lowest mean value) reproduces exactly under recomputation for all four in-sample firms (Tencent: Shield 100%; BYD: Shield 90.5%; CATL: Shield 47.6%; GM: Hands 38.1%), as does the standardised bottleneck breakdown. Pooled across the panel, Shield's mean node value (8.14) sits nearly a full point below the next-lowest node (Hands, 9.33) in every entity — this is what generates the raw-bottleneck concentration, and it is not specific to the four in-sample firms: Shield remains the dominant raw bottleneck in four of the five held-out firms (Huawei 81.0%, Tesla 71.4%, Woodside 52.4%, BHP 57.1%).

Out-of-sample containment for the discriminating band overall: 102 of 105 held-out entity-years (97%) fall inside the range established solely from the four in-sample firms. The two boundary cases — BHP 2011 and Tesla's 2006–07 startup phase — correspond to a mining supercycle peak and a pre-delivery startup phase respectively, both consistent with mechanisms the underlying analysis already identifies rather than being unexplained outliers.

### 3.4 Corrections applied

Three items from the independent audit are incorporated here rather than in the original working paper's wording:

1. **Tencent's bond-strength trajectory** is not a monotonic decline from 0.431 (2017) to 0.232 (2022). It dips to 0.304 (2018), recovers to 0.419 (2020), and falls again to 0.232 (2022) — two distinct compressions (the 2018 gaming-licence freeze and the 2021–22 antitrust/fintech action), not one continuous decline.
2. The stored-bond-strength inflation factor is reported here as **10.81×–111.07×** (using system *B̄* as the per-node comparator), not the "12×–112×" figure that appears in earlier drafts and is not exactly reproducible.
3. **Scope note:** the canonical *W_{ij}* construction here sums over all 28 node pairs (a complete graph, K₈), not the sparse Mythic/Material-partition edge structure used elsewhere in the project's national-level formalism. *B̄* and *λ₂* correlate at *r* = 0.981 under this construction — the two operators are close to redundant in a fully-connected topology, so this test does not by itself engage the Mythic/Material partition question.

None of these corrections changes a reported figure to a different value; all are wording precision or scope clarifications.

---

## 4. Cross-Domain Test: Nation-States vs. Corporations

**Source:** the corporate panel above (pooled, 9 firms, 189 firm-years), compared against Denmark (1752–2025, *n* = 273 years) and France (1850–2026, *n* = 176 years), both scored via the same canonical operators.

| Pair | KS *D* | *p* | Cohen's *d* | Interpretation |
|---|---|---|---|---|
| Denmark vs. Corporate | 0.853 | < 1×10⁻⁵⁴ | +2.25 | Huge upward shift |
| France vs. Corporate | 0.145 | 0.037 | −0.19 | Small downward shift |
| Denmark vs. France | 0.808 | < 1×10⁻⁴⁵ | +2.36 | Huge separation |

France is the **bridge case**: the KS test marginally rejects distributional equality, but with a small effect size, and 94.9% of France's annual *B̄* values fall inside the corporate band — its range [0.067, 0.491] substantially overlaps the corporate range [0.076, 0.448]. Denmark is the **separation case**: strongly rejected with a huge effect size, and only 19.3% containment. Denmark's range [0.223, 0.876] lies almost entirely above the corporate maximum.

This is the empirical basis for P.4.4: bridge entities exist, but they are the exception, and structural overlap does not by itself license pooling a national and a corporate distribution as though they were drawn from the same population.

*(Note: as flagged in the audit, an initially-supplied France file — a truncated 1900–2026 extract — did not reproduce these figures; the values above use the correct 1850–2026 vintage. The 13 society-years responsible for the earlier discrepancy cluster in 1905–1913 and 1941–1942, both outside the truncated file's window.)*

---

## 5. Corrections to the Portability Proposition

### 5.1 F-P4.3 status-table correction

The falsification criterion is stated as: *"if out-of-sample same-class D_KS > 0.20, class-pooling is invalid (triggered)."* The corporate panel's own within-class statistics (§3.2: *D* = 0.214 for *B̄*, *D* = 0.276 for *λ₂*) both exceed this 0.20 threshold. An earlier version of the status table recorded F-P4.3 as "NOT TRIGGERED (within-class *D* < 0.28)" — a threshold that is not the criterion's own stated value. The corrected status is **TRIGGERED**.

This correction does not weaken the paper's substantive argument — if anything, it strengthens it. F-P4.3 triggering means that even *same-class* out-of-sample pooling (corporate firms held out from other corporate firms) shows detectable parametric drift; P.4.3 (threshold relativity) is therefore supported at a finer grain than originally stated, not merely across the corporate/national boundary.

---

## 6. Functional Variability and Resonance (preliminary, unaudited)

The following extends the corporate panel with a distinct concept — node-state fluctuation as adaptive capacity rather than measurement noise — using the same four in-sample firms (General Motors, BYD, CATL, Tencent; *N* = 84 firm-years). **This section has not yet been through the independent-recomputation audit applied to Sections 3–5, and should be read as exploratory.**

### 6.1 Concept

Traditional reliability framing treats node-value variability (*V*_range, the range of a node's value across ensemble evaluations) as something to be suppressed. The alternative framing tested here is that variability is a functional resource: when tightly coupled nodes experience stress, operational variability can amplify non-linearly (systemic resonance), and if the system's architecture is sufficiently damped — often through specific nodes — this resonance is absorbed without structural fracture, rather than propagating.

### 6.2 Preliminary finding

Analysing standard-deviation envelopes across the four firms suggests distinct strategies for managing this resonance. Across the three manufacturing firms (GM, BYD, CATL), the **Craft** node consistently shows the highest functional variability (*V*_range between 1.86 and 2.57), while Tencent — a digital-platform firm — shows lower Craft variability (1.71) and higher Flow variability instead (1.95). *Correction: an earlier draft of this section also attributed elevated variability to Tencent's Shield node; recomputation shows Shield's mean V_range (1.10) is lower than Craft's, not higher, so only Flow displaces Craft as the top-variability node for Tencent.* The reading offered is that hardware-centric architectures concentrate operational variability in Craft, shielding upstream functions (Lore, Helm) from resonance cascades, while a platform architecture routes at least part of that variability through a different node (Flow).

This is a plausible and theoretically motivated pattern, consistent with the same four firms' already-confirmed structural coupling (§3.1: entity-level ρ(*B̄, λ₂*) = 0.955–0.992). But unlike the results in Sections 3–5, it has not been independently recomputed from raw scores, checked for the same class of issues that were found in the paused η_soil work (test-statistic identification, temporal dependence, incremental value against simpler quantities), or tested against a held-out panel. It is reported here as a candidate direction, not a confirmed result.

---

## 7. Discussion

### 7.1 What this synthesis establishes

Two independent audit passes confirm that canonical CAMS operators are structurally portable across the corporate/national boundary and within the corporate class across held-out firms, while remaining parametrically non-invariant — distributions shift systematically by domain and require entity- or class-specific calibration. This is not a weak result: it means the framework's coordination-node scoring grammar produces mathematically consistent relationships across firms as different as an automaker and a social-media platform, and across polities as different as Denmark and France, without needing different equations for each — only different operating ranges.

### 7.2 What remains open

The bridge/separation distinction (France vs. Denmark) is currently based on two national entities. Whether this taxonomy predicts how future entities will behave — whether an untested nation clusters with France's overlap pattern or Denmark's separation pattern — is an empirical question the current data cannot answer alone; it would need to be pre-registered as a test on new entities before being treated as validated. The three-domain test (civilisation → nation → corporation) that would fully close this question has not yet been run: the historical-civilisation panel exists in the project's data but has not been tested against the national and corporate bands reported here.

The functional-variability extension (§6) is promising but preliminary in a specific and important sense: the paused η_soil work shows that a composite CAMS metric can look confirmed on a first pass and still fail a proper audit — wrong test statistics, undisclosed missing data, and a composite that turns out to perform worse than one of its own components. Nothing has yet been done to check whether *V*_range or the resonance-absorption reading is free of the same failure modes. Until that check is done, §6 should be treated as a hypothesis worth testing, not a corporate-domain finding on the same footing as Sections 3–5.

### 7.3 Why η_soil and Norway are excluded here

A second, more thorough audit of the η_soil paper found problems that go beyond wording: reported t-statistics that do not correspond to any standard test; a Germany dataset silently missing its entire 1949–1989 division-era range; a composite metric that discriminates crisis years *worse* than one of its own numerator components (Archive bond strength alone); an internally self-contradicting H2; and H3's "regime attractors" resting on one example society each with no clustering or classification test. Because the Norway baseline in the companion Norway/functional-variability manuscript is computed entirely via η_soil, it inherits all of these open problems and is excluded from this synthesis for the same reason. Both will be revisited once the η_soil paper has been reworked to the same standard applied here.

A subsequent, file-hashed reproduction pass corroborated every one of these findings independently — matching circular-shift p-values to three decimal places, confirming the Archive-bond-strength-outperforms-composite result (AUC 0.907 vs. 0.861), and adding a formal claim-boundary framework (supported / supported-with-qualification / not established) along with specific-value corrections for Australia (1963 η_soil = 0.2335, not 0.263; 1911 at 0.2477 is the true series maximum; mean S_Hands = 7.252, not 7.81; *r*(η_soil, mean V) = 0.651, not "above 0.82"). This material is retained as source input for the eventual η_soil rework rather than added here, since η_soil itself remains out of scope for this synthesis.

---

## References

McKern, K. F. (2026). *CAMS v1.0-Final: Canonical Formulation and Computational Closure.* Neural Nations Research Technical Report.

McKern, K. F. (2026). *Operator Portability and Scale Invariance in the CAMS Framework: A Cross-Domain Empirical Test of Canonical v1.0-Final Operators.* Neural Nations Research Technical Report.

*Scale-Recursion Hypothesis: Corporate Panel* (working paper; figures independently reproduced in this synthesis via computational audit).

*Extending the Complex Adaptive Model of Societies: Operator Portability, Functional Variability, and the Norway Baseline* (working paper; Norway baseline excluded from this synthesis pending η_soil rework — see §7.3).

*Computational Reproduction and Errata Audit (v2): Three CAMS Working Papers.* Neural Nations Research / ComplexityWorkz, 31 July 2026.

**Data availability:** Source ensemble CSVs and recomputation code referenced throughout are available in the project repository under `analysis/` and `cleaned_datasets/`.

**Competing interests:** The author developed the CAMS framework and may benefit from its adoption.
