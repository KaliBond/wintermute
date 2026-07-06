# JUNO v1.2-Final Formalism

**July 2026 revision, following the 3,002 society-year blind-corpus test of v1.1**
**Status: production specification. Prospective validation on fresh holdout data remains open.**

---

## 1. Core Mathematical Operators (unchanged from v1.1)

**Node Viability (Vᵢ)**

    Vᵢ = C + K − S + 0.5·A

**Cognitive Activation (sᵢ) — sign-preserving clip**

    sᵢ = (A·C / 100) · sign(K − S) · max(|K − S|, 0.1)

**Coupling Quality (qᵢ)**

    qᵢ = (0.6C + 0.4A) / 10

**Bond Strength (Bᵢⱼ)**

    Bᵢⱼ = √(qᵢ·qⱼ) · 2^(−(Sᵢ+Sⱼ)/10),  bounded [0, 1]

**Algebraic Connectivity (λ₂)** — second-smallest eigenvalue of the graph Laplacian of the weighted adjacency matrix W, Wᵢⱼ = Bᵢⱼ (i ≠ j).

---

## 2. Numerical Policy (NEW in v1.2 — mandatory)

All aggregate metrics (V̄, V_min, s_min, B̄) must be **rounded to 6 decimal places before every threshold comparison** in the regime classifier.

Rationale: input scores carry one decimal place, so aggregate values land exactly on classifier thresholds routinely (e.g. V_min = 4.0). IEEE-754 arithmetic can represent such values as 3.9999999999999996, making strict-inequality comparisons path-dependent: the same data yields different regime labels depending on whether values pass through a file round-trip. This was observed on three society-years in the blind corpus. Without a declared rounding policy, regime labels are not reproducible across implementations.

---

## 3. Regime Classifier (v1.2-Final)

**Precedence order** (highest to lowest):
Freeze/Collapse → Systemic Crisis → Phantom Type II → Local Node Failure → Stable Adaptive → Strained

*Note on ordering: this is evaluation order, not a severity ranking. Stable Adaptive is tested before Strained because Strained is the residual band — with the V̄ ≥ 4 floor it now spans everything from mild strain to near-Stable profiles that narrowly miss a Stable sub-condition, so it must be evaluated last.*

| Regime | Conditions (v1.2) | Change from v1.1 |
|---|---|---|
| **Freeze/Collapse** | V̄ < 0 AND V_min < −3 AND B̄ < 0.15 | none |
| **Systemic Crisis** | V̄ < 6 AND V_min < 0 AND B̄ < 0.20 | none |
| **Phantom Type II** | s_min < −0.65 | none |
| **Local Node Failure** | **V_min < 4.0 only** | s-arm (s_min ≤ −0.80) **deleted** — unreachable under precedence: every case satisfying it is captured by Phantom Type II or higher. Verified: zero firings on 3,002 blind society-years. |
| **Stable Adaptive** | V̄ > 10 AND V_min > 5.0 AND B̄ > 0.28 AND s_min > −0.25 | none |
| **Strained** | **4 ≤ V̄ ≤ 10**, OR V̄ > 10 failing any Stable sub-condition | floor lowered from V̄ ≥ 6 to V̄ ≥ 4 |
| **Unclassified** | anything remaining | **preserved as a data-quality tripwire** |

**Completeness note.** The V̄ ≥ 4 floor makes coverage provably complete for well-formed data: any case reaching the Strained rule has survived Local Node Failure, hence V_min ≥ 4, and since V̄ ≥ V_min always, V̄ ≥ 4 is guaranteed. Unclassified therefore fires only on data-integrity failures or future operator changes — its intended role. Verified: 100% coverage on the blind corpus (v1.1: 99.8%, six gap cases with V̄ ∈ [5.35, 5.65], all now Strained; no other reclassifications).

---

## 4. Reference Ranges from Audited Panels

These are **descriptive statistics of specific scored corpora, not instrument bounds**. The only intrinsic bounds are the mathematical ones: Bᵢⱼ ∈ [0, 1]; λ₂ ≥ 0. New panels may legitimately exceed any figure below.

Blind corpus (24 model-scored societies, 3,002 society-years, 24,016 node-years):

| Quantity | Observed range |
|---|---|
| Vᵢ (node-level) | [−7.50, 23.00] |
| sᵢ (node-level) | [−2.50, +8.00] |
| s_min (year-level) | [−2.50, +5.40] |
| Bᵢⱼ | [0.025, 0.758] |
| B̄ (year-level) | [0.045, 0.722] |
| λ₂ | [0.268, 5.354] |

The v1.1 §4 figures (Bᵢⱼ max 0.657; λ₂ ∈ [0.379, 3.668]; sᵢ max +1.68) described the original calibration corpus only and are superseded as validation criteria.

---

## 5. Status of λ₂

λ₂ is a **derived diagnostic, not an independent validator and not a classifier input.**

On the blind corpus λ₂ orders all six regimes monotonically by severity (medians: Stable 2.68 > Strained 1.95 > Phantom II 1.52 > LNF 1.38 > Systemic Crisis 0.83 > Freeze/Collapse 0.51; Kruskal-Wallis H ≈ 2529). However, λ₂ correlates with B̄ at r = 0.972 — near-mechanically, since for a dense 8-node graph λ₂ tracks n·(mean edge weight): λ₂/(8·B̄) = 0.83 ± 0.10 — and B̄ is itself a classifier input. The ordering therefore substantially restates the classifier's own thresholds. After regressing out B̄, residual ordering collapses for several contrasts; per-case separability is strong for Stable vs Strained (AUC 0.95) but weak for Phantom II vs LNF (AUC 0.63).

Legitimate uses: internal-consistency check of the operator algebra; the ~17% deviation of λ₂ from the pure-B̄ prediction, which carries the network-topology information not captured by mean bond strength.

---

## 6. Validation Status

The blind-corpus test demonstrates **operator coherence and classifier coverage on model-generated panels** — compression consistency, not external validity. Ensemble tightness certifies precision, not ground truth; shared model bias remains an unquantified confound. Because the v1.2 threshold adjustments were chosen by inspection of the 3,002-year blind corpus, re-running that corpus is a consistency check only. Genuine validation of v1.2 requires fresh holdout data, aligned with the prospective 2026–2028 monitoring programme. Between-run reliability margins (±1 NV at year level, wider at node level) continue to apply to prospective work.

Corpus sizes cited in this document are limited to directly auditable figures. The verified-corpus size will be stated only when sourced to a specific, citable panel.

---

## 7. Change Log: v1.1 → v1.2

| Component | v1.1 | v1.2-Final |
|---|---|---|
| LNF s-arm | s_min ≤ −0.80 | Deleted (unreachable dead code) |
| Local Node Failure | V_min < 4.0 OR s_min ≤ −0.80 | V_min < 4.0 only |
| Strained floor | V̄ ≥ 6 | V̄ ≥ 4 (provably complete) |
| Unclassified | 0.2% residual on blind corpus | Preserved as tripwire; fires only on data-integrity failure |
| Numerical policy | Unspecified | Round aggregates to 6 dp before threshold comparison (mandatory) |
| §4 bounds | "Validated instrument bounds" | "Reference ranges from audited panels"; removed from validation criteria |
| λ₂ | Supporting indicator; candidate core discriminator | Derived diagnostic only; circularity with B̄ documented |
| Validation framing | External-validation language | Operator coherence on model-generated panels; prospective validation open |

All gains from the v1.1 sign-preserving sᵢ correction are preserved: 31.8% of node-years carry negative activation, Phantom Type II fires (250 society-years under v1.2), and the June 2026 dead-code diagnosis remains resolved.
