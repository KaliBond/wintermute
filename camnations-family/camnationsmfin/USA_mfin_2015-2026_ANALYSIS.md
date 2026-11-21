# CAMS Ensemble Analysis — United States, 2015–2026 (mfin run)

**Inputs:** `USA_CAMS_mfin_2015-2026_block1.csv` (ensemble means + Node Value + Bond Strength) and `USA_CAMS_mfin_2015-2026_block2.csv` (uncertainty envelope). 5 scoring agents (`n_eff=5`), 28 pairwise bonds (`bond_n=28` = C(8,2)), zero missing cells (`NA_rate=0`).

---

## 1. Structural integrity

- 96 cells per block: 12 years × 8 nodes, no duplicates, no gaps.
- **Seam quality** (block2 footer): two scoring passes merged with `n_overlap_cells=160`, `mean_abs_diff=0.54`, `max_abs_diff=2`. For a 0–10 integer scale this is a clean seam — sub-point average drift between passes.
- **Node Value vs FINAL spec (reconciled, see §7):** the CAMNATIONSMFIN FINAL rule defines Node Value = C_mean + K_mean − S_mean + 0.5·A_mean. **60/96 delivered cells deviate beyond the ±0.175 rounding tolerance**, with structured sign patterns (per-node constant offsets plus a stress-correlated drift). This is aggregation-layer drift from the documented rule — not a formula ambiguity.
- **Frozen-node signature 2015–2019:** Stewards (15.6), Craft (13.0), Hands (9.5), Flow (15.6) are bit-identical across all five pre-shock years, envelopes included. Under the CAMNATIONSMFIN batch design (seam at 2020, de-dup keeping the later batch), seam handling cannot produce this — the scorers themselves held the economic core constant across the pre-shock years. Scorer behaviour, not aggregation artifact.

## 2. Headline result

**System vitality (mean node V) fell 12.95 → 7.42 (−42.7%) over 2015–2026, and Bond Strength fell 9.051 → 5.341 (−41.0%).** The decline is not monotonic — it resolves into four distinct phases:

| Phase | System V | Bond | Character |
|---|---|---|---|
| 2015–2019 Pre-shock erosion | 12.95 → 11.39 (−12.1%) | 9.051 → 7.766 | Decay confined to Helm (11.5→5.7) and Archive; economic core frozen |
| 2019–2020 COVID discontinuity | → 8.35 (−26.7% in one step) | → 5.834 (−24.9%) | All-node synchronous shock; Flow collapses 15.6 → 7.0 |
| 2020–2024 Partial stabilisation | → 10.11 (+21.1%) | → 6.420 | Mean reversion in Helm, Craft, Flow; Lore keeps bleeding |
| 2024–2026 Second-leg decline | → 7.42 (−26.6%) | → 5.341 (−16.8%) | Archive cliff (9.7 → 4.5); no recovery this time |

The 2020–24 rebound never regained the 2019 level — the system stabilised at a lower plateau, then broke downward again. That is the classic step-down staircase, not a V-shaped shock.

## 3. Node-level deltas, 2015 → 2026

| Node | 2015 | 2026 | Δ | % | Reading |
|---|---|---|---|---|---|
| Archive | 12.2 | 4.5 | −7.7 | **−63.1%** | Worst node in the system; cliff is concentrated 2024→2026 |
| Flow | 15.6 | 7.5 | −8.1 | −51.9% | Largest absolute loss; from strongest node to mid-pack |
| Helm | 11.5 | 5.1 | −6.4 | −55.7% | Fell first (2015–17), led the whole structure down |
| Lore | 12.1 | 5.6 | −6.5 | −53.7% | Only node that never rebounds in 2020–24 |
| Stewards | 15.6 | 10.0 | −5.6 | −35.9% | Still second-strongest; slow structural bleed |
| Craft | 13.0 | 7.9 | −5.1 | −39.2% | Tracks the system average |
| Shield | 14.1 | 11.2 | −2.9 | −20.6% | Most resilient node in the system |
| Hands | 9.5 | 7.6 | −1.9 | −20.0% | Weakest in absolute terms throughout — low baseline, low loss |

Ordering of decline matters: **Helm broke first (2015–2017), Archive broke last and hardest (2024–2026)**. The system's memory/continuity layer held through the COVID shock and the first stress cycle, then failed in the second leg.

## 4. Metabolic structure, 2026

Six of eight nodes are now **inverted (Stress > Capacity)** — Helm (S 7.8 / K 5.2), Archive (8.2/4.4), Lore (7.2/4.8), Flow (7.2/5.8), Hands (6.4/5.4), Craft (6.2/5.6). In 2015, *no* node was inverted. Only Shield (+2.0) and Stewards (+1.2) retain a capacity surplus. System-wide means: Coherence 6.25 → 4.68 (series minimum), Stress 3.90 → 6.68 (series maximum). The C–S crossover occurred in 2020, briefly reversed 2021–24, and has reopened wider in 2025–26.

## 5. Uncertainty structure

- Scorer dispersion rises from 0.79 (mean total sd, 2015) to 2.10 (2026), **+164%**; mean V_range doubles from 1.31 to 2.44.
- Disagreement in 2026 concentrates exactly where the decline is fastest: Stewards (sd 2.78), Archive (2.49, V_range 4.0 — the widest cell in the dataset), Helm (2.29). Shield is the only node scorers still agree on (sd 1.16).
- Interpretation: measurement uncertainty is tracking structural transition, not noise — the ensemble is least certain about nodes undergoing regime change. Archive's 2026 envelope (V ∈ [2.5, 6.5]) spans from "failed" to "strained", which is the honest state of knowledge.

## 6. Caveats

- 2015–2019 frozen economic nodes (§1) mean the pre-shock trend is carried entirely by Helm/Lore/Archive; treat the 2015 baseline as a single scoring pass extended across five years.
- Delivered Node Value and Bond Strength deviate from the CAMNATIONSMFIN FINAL aggregation rules (§7); envelope widths remain valid as *dispersion* measures.
- Bond Strength is constant within each year across all nodes by construction (network-level quantity).

## 7. Reconciliation against the CAMNATIONSMFIN FINAL spec

*Added 2026-09-17 after the CAMNATIONSMFIN methodology note. Spec rules tested: (1) Node Value = C_mean + K_mean − S_mean + 0.5·A_mean; (2) block1 V inside per-scorer [V_min, V_max]; (3) Bond Strength = mean over pairs of [0.6·C_iC_j + 0.4·A_iA_j]·exp(−(S_i+S_j)/20). Rounding tolerance ±0.175 (1-dp means). Full cell-level deltas: `reconciliation_nodevalue_violations.csv`, `reconciliation_envelope_violations.csv`, `reconciliation_bond.csv`.*

**Rule 1 — Node Value: 60/96 cells violate spec.** The deltas are structured, not random:

- **Constant per-node offsets:** Hands carries +1.0 in *all 12 years*; Shield carries −1.0 in 2015–2019 then converges to spec; Craft sits +0.2 in the frozen years. This looks like node-level calibration constants applied on top of the linear rule.
- **Stress-correlated positive drift:** from 2020 the delivered V sits increasingly *above* spec in stressed nodes, peaking at Archive 2026 (+3.0), Lore 2024/2025/2026 (+2.0), Helm 2026 (+1.6), Hands 2025 (+1.4). Mean delivered−spec gap by year: ≈0 in 2015–16, growing to +1.27 by 2026. Behaves like a stress-damping/resilience floor not present in the FINAL text.
- 36/96 cells match spec within rounding — including every Flow and Stewards cell in 2015–2020, and all 2020 Shield/Stewards/Craft/Flow cells.

**Rule 2 — Envelope containment: 12/96 cells outside [V_min, V_max].** All breaches are small (≤ +0.4, all above V_max) and concentrated in zero- or low-dispersion cells (Shield 2015–18, Lore 2015, Helm 2016/2019). Consistent with block1 V being computed from unrounded means while the envelope was built from per-scorer integer-derived V's. Minor; does not change any trajectory reading.

**Rule 3 — Bond Strength: fails spec in all 12 years.** Delivered/spec ratio is 0.32–0.44 and *rises with system stress* (0.317 in 2015 → 0.437 in 2026): the delivered series decays more slowly under stress than the spec's exp(−ΣS/20) damping implies. Best-fit form is ≈ mean(base)/4 with much weaker stress damping. The delivered bond layer follows a different normalization than the FINAL rule documents — scale and stress-sensitivity both differ.

**Bottom line:** the *rankings and trajectory shapes* in §§2–5 are robust to these discrepancies (deltas are smooth and systematic), but absolute Node Value and Bond Strength levels are not spec-compliant and should not be mixed with future FINAL-spec runs without recalibration. If the offsets/floor are deliberate, they belong in the skill text; if not, re-aggregating block1 from the raw per-scorer CSVs under the FINAL rules will resolve all three rules at once.

---

*Generated 2026-09-17; reconciliation section added same day. Companion outputs: `usa_mfin_overview.png`, `usa_mfin_envelopes.png`, `usa_mfin_stats.txt`, `analyze_usa_mfin.py`, `reconciliation_*.csv`.*
