# Regime classifier specifications and provenance

Status: reference document for CAMS Praxis, 14 November 2026. Draft pending custodian review.

Two regime classifiers are implemented in `praxis/cams_praxis_calc.py`.

| | Role in Praxis | Function |
|---|---|---|
| **JUNO v1.2-Final** | Canonical. Used for every regime label Praxis reports. | `classify_v1_2_final` |
| **v1.0.1** | Legacy comparator. Reported beside the canonical label, always under its own name. | `classify_v1_0_1` |

They are separate instruments. A v1.0.1 label is never converted into a v1.2-Final label or the reverse. Stored historical outputs and lodged predictions are never re-labelled.

Both classifiers read the same four inputs, computed by the same operators.

---

## 1. Shared operators

Source: `juno/JUNO_v1.2-Final_Formalism.md` section 1. Build Pack v1.0 Part 2 section 2 gives the same formulas.

For each of the eight nodes, with Coherence C, Capacity K, Stress S, Abstraction A scored 1 to 10 (`CAMS_SCORING_PROTOCOL_V2_1.md`):

- Node viability: `V_i = C_i + K_i − S_i + 0.5·A_i`
- Cognitive activation: `s_i = (A_i·C_i / 100) · sign(K_i − S_i) · max(|K_i − S_i|, 0.1)` (see section 5 for K = S)
- Coupling quality: `q_i = (0.6·C_i + 0.4·A_i) / 10`
- Bond strength: `B_ij = √(q_i·q_j) · 2^(−(S_i + S_j)/10)`, for i ≠ j
- `λ2`: second-smallest eigenvalue of the raw Laplacian `L = D − W`, `W_ij = B_ij`

Classifier inputs:

- `V̄` = mean of the eight `V_i`
- `V_min` = smallest `V_i`
- `B̄` = mean of `B_ij` over the 28 unique pairs
- `s_min` = smallest `s_i`

`s_V` (population standard deviation of `V_i`) and `λ2` are reported but are not classifier inputs in either cascade.

---

## 2. JUNO v1.2-Final (canonical)

**Rounding.** Round `V̄`, `V_min`, `s_min` and `B̄` to **6 decimal places** before every comparison.

**Cascade.** Evaluate in this order. The first rule that holds gives the label.

| Order | Regime | Condition |
|---|---|---|
| 1 | Freeze/Collapse | `V̄ < 0` AND `V_min < −3` AND `B̄ < 0.15` |
| 2 | Systemic Crisis | `V̄ < 6` AND `V_min < 0` AND `B̄ < 0.20` |
| 3 | Phantom Type II | `s_min < −0.65` |
| 4 | Local Node Failure | `V_min < 4.0` |
| 5 | Stable Adaptive | `V̄ > 10` AND `V_min > 5.0` AND `B̄ > 0.28` AND `s_min > −0.25` |
| 6 | Strained | `V̄ ≥ 4` |
| 7 | Unclassified | anything left (a data-quality tripwire; unreachable for well-formed data) |

All inequalities are strict except rule 6.

**Provenance**

| Item | Source |
|---|---|
| Specification | `juno/JUNO_v1.2-Final_Formalism.md`, sections 2 and 3. sha256 `10143dd9206797fe…` at repository commit `74ad22f`. |
| Dated | 6 July 2026. `docs/CAMS_LAB_DIARY.md` records it as git `464222f`; `docs/VERSION_HISTORY.md` has the matching entry. |
| How thresholds were set | By inspection of a 3,002 society-year blind corpus (formalism section 6). The same section states that re-running that corpus is a consistency check only, and that validation needs fresh holdout data. |
| Used by | The lodged 2027–2028 prospective register, which names this classifier, the 6 dp rounding and this precedence order (`records/cams-juno-prospective-predictions-2027-2028-v1-0-0-2026-11-08/CAMS_JUNO_Prospective_Register_2027-2028_v1.0.0.md`, line 23; sha256 `4ee1448a62bbcc65…`). |
| Reference implementation before this one | None found in the repository. The formalism is prose and tables. `cams_praxis_calc.py` is, as far as I can find, the first runnable implementation committed here. It should be checked against whatever code produced the 3,002-year blind-corpus results. |

---

## 3. v1.0.1 (legacy comparator)

**Rounding.** Round every metric to **4 decimal places** before comparison.

**Cascade.** Evaluate in this order. The first rule that holds gives the label.

| Order | Regime | Condition |
|---|---|---|
| 1 | Freeze/Collapse | `V̄ < 0` AND `V_min < −3` AND `B̄ < 0.15` |
| 2 | Phantom Type II | `3 ≤ V̄ ≤ 6` AND `V_min < 0` AND `s_min < −0.7` |
| 3 | Systemic Crisis | `V̄ < 6` AND `V_min < 0` AND `B̄ < 0.20` |
| 4 | Local Node Failure | `V_min < 4.5` OR `s_min < −0.75` |
| 5 | Stable Adaptive | `V̄ > 10` AND `B̄ > 0.25` |
| 6 | Strained | otherwise |

**Provenance**

| Item | Source |
|---|---|
| Specification | "Departmental CAMS PRAXIS GPT — Complete Build Pack", dated 14 November 2026, Part 2 section 2. sha256 of the file as supplied: `02990795de612dccbc22b881f9ebdf5422d0913900f98b9e0159d44487fd1483`. The file is not committed to this repository. |
| Dated | The build pack describes it as the "12 June 2026 recalibration". |
| How thresholds were set | Not documented in any file in this repository. The build pack attributes it to a refit on a "verified corpus"; no dataset, script or report here reproduces that refit. |
| Other traces in the repository | None. No file other than those under `praxis/` mentions "v1.0.1". |

Because the provenance is thin, v1.0.1 is kept only as a named comparator.

---

## 4. Where the two differ

| | v1.2-Final | v1.0.1 |
|---|---|---|
| Order of rules 2 and 3 | Systemic Crisis, then Phantom Type II | Phantom Type II, then Systemic Crisis |
| Phantom Type II | `s_min < −0.65` only | `3 ≤ V̄ ≤ 6` AND `V_min < 0` AND `s_min < −0.7` |
| Local Node Failure | `V_min < 4.0` | `V_min < 4.5` OR `s_min < −0.75` |
| Stable Adaptive | four conditions; `B̄ > 0.28` | two conditions; `B̄ > 0.25` |
| Strained | `V̄ ≥ 4` | residual |
| Unclassified | exists | does not exist |
| Rounding | 6 dp | 4 dp |

Freeze/Collapse and the Systemic Crisis condition are identical in both.

The measured effect of these differences on 6,778 society-years is in `praxis/concordance/CLASSIFIER_CONCORDANCE.md`.

---

## 5. Activation when Capacity equals Stress

### The question

The formal definition is `s_i = (A_i·C_i/100) · sign(K_i − S_i) · max(|K_i − S_i|, 0.1)`.

At `K_i = S_i` the existing engine and the formula as written give different values.

| Reading | Value at K = S | Source |
|---|---|---|
| **Formal** | `0` | The formula with the standard signum, where `sign(0) = 0`. |
| **v2.4-compat** | `+0.1 · A_i·C_i/100` | `cams_framework_v2_4.py`, `esch_activation`: "ONLY where K_i == S_i exactly substitute singularity_floor". |

This is not a rare edge case. With integer scores, K = S occurs in 6,140 of 54,224 node-years (11.3%) in the in-range corpus.

The v2.4 engine also differs in a second way: it applies no floor when `0 < |K − S| < 0.1`, where the formal definition would give ±0.1. This case does not occur anywhere in the in-range corpus (0 of 54,224 node-years), so it has no observed effect.

### What the formal definition says

Taken as written, the definition gives `s_i = 0` at `K_i = S_i`. That is the only value consistent with the standard signum function, and it is what `cams_praxis_calc.py` computes by default (`--activation-mode formal`).

The definition is discontinuous at zero: as `K − S` approaches zero from above, `s` approaches `+0.1·A·C/100`; from below, `−0.1·A·C/100`. The stated purpose of the 0.1 floor is that "near-zero E does not erase sign" (`docs/CAMS_IN_HPS_AND_COMPLEX_SYSTEMS.md`). At exactly zero there is no sign to preserve, so a value of zero sits midway between the two one-sided limits. The v2.4 engine instead picks the upper limit, which treats a node in exact balance as slightly positive. Nothing in the formalism supports preferring the positive side.

### Why it cannot change a regime label

Under either reading, a node with K = S has `s_i ≥ 0`.

- If the minimum `s` is at such a node, then `s_min ≥ 0` under both readings.
- If the minimum is at another node, that node's `s` is the same under both readings.

Every `s_min` threshold in both classifiers is negative (−0.25, −0.65, −0.7, −0.75) and each test asks only whether `s_min` is below or above that negative number. A change in `s_min` from `0` to a small positive number, or the reverse, cannot move it across a negative threshold.

Checked on the corpus: `s_min` differs between the two readings in 853 of 6,778 society-years, and the number of regime labels that change is 0 under v1.2-Final and 0 under v1.0.1.

### Position taken here

- Default: `formal`.
- `--activation-mode v2.4-compat` reproduces the existing engine exactly and remains available.
- Stored v2.4 outputs are not recomputed.

### Still open

The reading affects reported `s_i` and `s_min` values even though it does not affect labels. Whether the formal definition should be amended to state its value at K = S explicitly (zero, or a signed convention) is a custodian decision. Until then, any table of `s` values must state the activation mode.

---

## 6. Other cascades in the repository (not implemented in the Praxis calculator)

| Cascade | Where | Note |
|---|---|---|
| CAMS-CAN v1.0-Final | `cams_framework_v2_4.py`, `classify_regime` (sha256 `865642e983e39bfd…`) | Tests Local Node Failure first (`V_min < 4.0 OR s_min ≤ −0.85`). Produced the `regime` column in `data/v1.0_final_recompute/FG_corpus.csv`. Its stored labels match v1.2-Final on 88.3% of the 6,778 rows. |
| CAMS v3.2-R attractors | `test/cams-interpreter/cams-compute.js` | A different operator family (Gaussian kernel bonds, kappa tiers). Not comparable. |
| Telescope basins | `test/telescope_*.html` | Uses S/K ratio and a different bond measure. Not comparable. |

A figure quoted from any stored file must name the cascade that produced it.
