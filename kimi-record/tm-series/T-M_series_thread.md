# The T-M Series: Morphology, Damage and Recovery — Complete Thread

**Date:** 2026-10-16 · **Programme:** CAMS/JUNO · **Datasets:** JUNO_Unified_Dataset.csv (canonical-verified, max V drift 3.6e-15), JUNO_Blind_Set_{1,2,3,5,6,7} · **All tests pre-registered before outcome computation; all amendments and fails logged.**

This document threads the full test series in execution order. Individual pre-registrations and lab notes are in the archive; this is the narrative spine.

---

## 0. Origin

The series began with a reframing of CAMS as complex-adaptive-systems science (Holland, Ashby, Beer, Holling): the Paladins as functional architecture, the biosigns as operating conditions, history as coordinated adaptation. From that discussion emerged a testable conjecture: *a society's identity is its historically persistent way of coordinating adaptation under stress* — and its corollary, that displacement morphology (which functions absorb stress) should predict how episodes resolve better than regime type or aggregate state.

## 1. T-M1 — Morphology vs rivals for recovery — FAIL (the productive failure)

**Design:** 201 stress episodes extracted from 48 societies (within-society z_S ≥ 1.0 onsets); 158 / 42 after censoring rules; 149 / 33 after leave-one-episode-out typicality construction. [Episode accounting corrected 2026-10-16: 48 societies at extraction; 42 was the post-censoring count.] Displacement signatures: L1-normalised per-node ΔS. Outcome: full regain of trough V̄ loss within 5 years (base rate 0.73; two amendments logged — the registered ratio criterion was mathematically defective when baseline V̄ ≈ 0).

**Results:** Morphology typicality CV AUC 0.492 — chance. V-Dem liberal democracy 0.547, aggregate state 0.553. Adding morphology to rivals *reduced* the model. Sensitivity grid never left 0.50–0.52. Stability gate passed (within-society signature similarity 0.849 vs between 0.823, p = 0.004).

**Lesson:** morphology is real but does not predict recovery. The identity conjecture's strong form dies here.

## 2. T-M2 — Morphology predicts failure locus? — FAIL on bar, real signal

**Design:** failure locus = argmin V during episode. Society modal-locus (LOEO) prediction vs marginal base rate and permutation null.

**Results:** accuracy 0.510 vs a 0.409 marginal baseline (1.25×; bar was 1.5×) — FAIL as registered, but against a permutation null of 0.287 the result is p = 0.002: society identity carries genuine, sub-threshold information. [Canonical plain-argmin locus; figures corrected 2026-10-16.] Uptake→locus concordance κ = 0.154 (bar 0.20, FAIL; corrected from 0.184, which was not reproducible under script v2). Locus → recovery: nil (p = 0.72). Locus distribution dominated by Hands (61/149, corrected).

## 3. T-M3 — ΔV-only locus — the methodological correction that paid twice

**Design:** locus redefined as largest ΔV drop from baseline (episode-specific damage, not chronic level). Registered as candidate fix in T-M2.

**Results:** 62.4% of episodes relocated (corrected). Hands 61→14, Flow 32→49, Helm 14→32 — T-M2's "Hands basement" was largely a level artefact (within-society Argentina finding, already ΔV-based, unaffected). Society modal accuracy fell to 0.94× marginal (p = 0.034): the society signal lives in *standing* structure, not damage patterns. **T-M3b PASSED strongly: uptake→damage κ = 0.476.** Locus→recovery still nil (p = 0.19; Lore-locus dip 0.40, n=10, exploratory only).

**Lesson:** the correction both invalidated a previous reading and produced the series' first strong positive.

## 4. T-M3d — Mechanical-coupling audit — the circularity confrontation

**Challenge (user):** ΔV = ΔC + ΔK − ΔS + 0.5ΔA, so argmax ΔS vs argmin ΔV is partly arithmetic. If C,K,A constant, concordance = 100% by construction.

**Design:** stress-excluded residual R_i = ΔC_i + ΔK_i + 0.5ΔA_i (identity verified to 1e-9). Trough-aligned deltas. Pure-algebra floor via within-episode stress shuffles. Lag panel: 40,944 node-years, node FE, own-lag controls.

**Results:** total κ 0.499 = algebraic floor 0.252 + **genuine component κ = 0.249 (p = 0.001)** — PASS on the 0.20 bar. Roughly half the original number was the equation talking to itself; half survived. Lag: ΔS_t → ΔC_{t+1}, ΔK_{t+1} both −0.138, p ≈ 1e-16/1e-21 (node-FE panel, 40,944 obs). (Decomposition is heuristic, not a κ identity — user qualification adopted.)

## 5. T-M3e — Hardened lag — PASS with attenuation

**Safeguards applied:** society-year fixed effects (FWL within-transform after dummy expansion killed the kernel — noted for the record), society clustering, reverse-time placebo.

**Results (panel 40,472 obs):** forward −0.094/−0.093 (p ≈ 1e-16/1e-15); reverse placebo −0.052/−0.041. Forward/reverse ratios 1.81× and 2.25× (bar 1.5×) — PASS. ~1/3 of the raw association was shared-shock variance; non-zero placebo confirms partial scoring smear. Manuscript cites the hardened −0.09 figures. Record-count note: panel 40,472 obs from 43,400-row file; user-cited 32,361 unmatched — flagged, unreconciled.

## 6. T-M4 — Blind replication — PASS

**Design:** frozen estimator (exact R definition, trough alignment, κ, permutation null, bars κ ≥ 0.20 & p < 0.05) applied to six untouched anonymised blind sets. 138 episodes, 48 blind societies.

**Results:** pooled κ = 0.267 (in-sample benchmark 0.249), p = 0.001. **No degradation on unseen data.** Registered caveat: set-level heterogeneity 0.135–0.481; all six sets directionally positive, three individually clear the bar. Fixed-effects reading reported alongside pooled.

## 7. T-M5 — Coverage assessment (frozen before outcomes)

**Deliverables:** 158-episode frozen roster (society, onset, per-node trough years, uptake/R/ΔV loci); WDI coverage matrix; zero outcome comparisons run.

**Findings:** all-four-function coverage 12.7% overall, 24.7% post-1960 (20 episodes, 16 societies). Flow observable at 84% post-1960 vs Hands/Stewards 42% — unequal observability is now measured, not asserted. Selection bias visible in the eligible subset (predicted loci skew to Flow, 9/20). Track A becomes a post-1980 study after ILOSTAT/GRD extension (est. 40–60 episodes); 77 pre-1960 episodes are Track B (blinded adjudication) by construction. Proposed frozen bars: external κ ≥ 0.30, degradation ≤ 0.10 vs internal benchmark, common-shock flag, four-function ≠ eight-function scope statement.

**Open decisions:** unblind answer key? ILOSTAT/GRD extension vs WDI-only pilot? Track B human coders vs AI ensemble (recommendation: humans for codebook anchor cases at minimum — AI adjudication shares scorer-class risk with CAMS itself, the exact circularity T-M5 exists to escape).

---

## The calibrated synthesis

Three distinguishable processes, each with its confidence level:

| Process | Status |
|---|---|
| **Standing structure** — characteristic chronic weak nodes | Real, modest (p = 0.002). Identity lives here, not in episodes. |
| **Stress transmission** — uptake → non-stress deterioration | Strongest result. Genuine κ ≈ 0.25, blind-replicated (0.267), correct temporal ordering (hardened, placebo-controlled). |
| **Adaptive recovery** — regain of losses | Null everywhere in morphology. Regime type (V-Dem) carries the modest signal that exists; mechanism test (T-M6: K_Stewards/Flow capacity as carriers) registered as future work. |

The manuscript sentence (hardened form):

> CAMS distinguishes persistent functional vulnerability, the distribution of episode-specific disturbance, and subsequent adaptive outcomes. Registered tests show that stress uptake is associated with deterioration in non-Stress operating components (in-sample κ = 0.249; blind replication κ = 0.267, n = 138, 48 societies, p = 0.001, set-level heterogeneity 0.135–0.481), and that increases in scored Stress precede subsequent reductions in Coherence and Capacity (−0.094/−0.093; forward/reverse placebo 1.8–2.3×). Neither standing morphology nor damage locus predicts aggregate recovery. Structural condition, disturbance and adaptation are distinct, interacting processes.

And the aphorism, with its qualification: *a society's adaptive character describes where it hurts, not whether it heals — and its character appears more clearly in standing vulnerability than in the location of new damage.*

The deeper result: CAMS is beginning to separate the dynamics of damage from the dynamics of repair. Damage has a detectable, blind-replicated relationship with stress uptake. Repair depends on additional capacities and contingencies the morphology does not capture. That distinction — purchased almost entirely with registered negative results — is the series' real contribution.

## Archive contents

| File | Content |
|---|---|
| T-M1_preregistration.md / T-M1_lab_note.md | Morphology vs rivals (recovery) |
| T-M2_preregistration.md / T-M2_lab_note.md | Failure-locus prediction (level-based) |
| T-M3_preregistration.md / T-M3_lab_note.md | ΔV-only locus correction |
| T-M3d_preregistration.md / T-M3d_lab_note.md | Mechanical-coupling audit |
| T-M3e_T-M4_preregistration.md / T-M3e_T-M4_lab_note.md | Hardened lag + blind replication |
| T-M5_episode_roster.csv | 158 frozen episodes with loci and trough years |
| T-M5_coverage_matrix_WDI.csv | Per-episode external data availability |
| T-M5_coverage_assessment.md | Coverage findings + proposed frozen gates |
| episodes.csv | Censored episode-level data (158 rows, pre-typicality) |
| tm_series_analysis.py | Consolidated reproduction script v2, all seeds inline, intermediates saved |
| episodes_censored_158.csv / episodes_analysis_149.csv | Frozen episode sets (censored / analysis) |
| T-M4_blind_set_*_episodes.csv, T-M4_per_set_kappa.csv | Blind replication intermediates |
| REPRODUCIBILITY.md | Seeds, SHA-256 hashes, repo commit, blind-set-4 note, correction log |
