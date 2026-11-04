# ADVERSARY-02 Section 7: Pre-registered predictions for Part 2 (2021-2024)

**STATUS: DRAFT for Kari's sign-off. NOT SIGNED. NOT PRE-REGISTERED. No ledger sign-off line exists.**
Drafted 4 Nov 2026 (AEDT) by Grok Bot on Kari's request (owner message pt70u, "lets make some predictions"). Nothing here is final until Kari edits and signs it. Hashing this file is an integrity record of the draft only.
**Revision 2, 4 Nov 2026 (AEDT), after 20:45:** adds contrast predictions C1 to C3 and a US-decay out-of-scope note (section 4A), records the decisions relayed for section 1, and updates sections 0, 5 and 6. P1 to P12 are unchanged. Revision 1 is kept as `predictions_DRAFT_v1_20-25.md` (sha256 `66a5c17a1e61b582f0027122d57eb80a43545b3c3e675d0c8ae972da494ff11a`). Still UNSIGNED, still no ledger line.

## 0. Honest limits (read first)

1. **Inputs.** Only 2010-2020 results already in `analysis/interim` (recompute_uniform, run2_update, codex_v2_update). No 2021+ data, and no web lookups of any post-2020 event. Every number below is computed by `40_basis_tables.py`, `41_basis_for_predictions.py` and `42_basis_extra.py` in this folder. Outputs: `basis_stdout.txt`, `basis_extra_stdout.txt`.
2. **Outcome knowledge cannot be certified away.** The drafter is a language model whose training knowledge includes events after 2020. I consulted no post-2020 source. I set each threshold from the 2010-2020 tables, and I kept every call that the 2010-2020 trajectory implies even where it felt uncomfortable. I cannot prove the choice of which questions to ask, or the confidence levels, was untouched by that background knowledge. All predictions on Evergrande, Huawei, BYD, CATL, Vanke, Longfor and SMIC are tagged **K (drafter had background awareness of later developments)**. I do not say which direction it pointed. Suggested mitigation: Kari or a reviewer with no such exposure reads the thresholds before sign-off.
3. **The scorers know the outcomes too.** The Claude sets and probably Codex v2 score from models that know 2021+ events. The 2010-2020 trajectories, especially Evergrande's 2019-2020 collapse, carry hindsight risk (flagged since run 1). A prediction that "the 2020 pattern continues" is therefore partly a prediction that the same scorers keep telling the same story.
4. **Forecast origin leak.** Protocol 5.3 sets the forecast origin at 31 Dec 2020, but reconstruction scores for 2020 use evidence up to 30 June 2021. The 2020 jump-off values include up to six months of post-origin information (the Codex v2 run note also says scorers met later snippets).
5. **Most predictions are "no reversal" claims.** All 11 non-coupling predictions would also pass if each entity simply stayed at its 2020 value ("Persistence passes?" in each entry). Passing them will not show CAMS skill beyond persistence. Only P3 would fail under a straight-line extrapolation of 2018-2020 (it is a stabilisation call), and P12 is a negative-result prediction about coupling. Revision 2 adds three contrast predictions (C1 to C3, section 4A) that persistence of the 2020 value fails by construction.
6. **Node mapping is provisional and unapproved** (Board↔Helm, Finance↔Stewards, Markets↔Flow, RD↔Lore, positional Risk→Shield, Operations→Craft, Workforce→Hands, Brand→Archive). Node-level predictions inherit that. Codex v2 covers only 275 of 696 node-years (companies 7-43 of 88 each), so the primary pool's composition varies cell by cell.
7. Confidence levels are my subjective probabilities from the trajectories. They are uncalibrated and were not produced by a model.

## 1. Items needing Kari's decision before sign-off

- **Evidence cutoff for Part 2. (Open. I advised 30 Jun Y+1, NOT yet confirmed by Kari.)** Protocol 5.3 says "strict cutoff 31 December of the scored year" for prospective scores. This request said 30 June Y+1 (the reconstruction rule). Predictions are written to apply to either, but the pass/fail run must use one rule and the draft must state it. A 31 Dec cutoff gives less information per year than the 2010-2020 scores had.
- **Outcome-class bands** (section 2) are my choice, not the protocol's. (Open. I advised keeping them as drafted, NOT yet confirmed by Kari.)
- **Sets in the Part 2 pools.** Same four primary sets (Claude-Opus, Claude-Sonnet, Kimi v2, Codex v2) if they are rerun; a set not rerun is dropped, never replaced, and the change is reported. Excluded sets (Kimi v1, Codex v1) are never pooled.
- **BYD and CATL have no sector comparator** in protocol 5.2, so "most divergent comparator" cannot be answered for them (section 3).
- **Failure Condition 5 rule** ("wrong for the majority of primary companies") is defined here as 3 or more of the 4 outcome predictions P1, P3, P4, P5 failing on the primary pool. **Decided by Kari (relayed 4 Nov 2026): stays as drafted.** The contrast predictions C1 to C3 do not enter FC5.
- **Contrast predictions.** Kari proposed four themes (owner message pt77u): US decay, Chinese strength, Huawei survival, BYD pressure of scaling. Section 4A turns three into predictions (C1 to C3) and records US decay as out of scope. The themes were proposed after the 2010-2020 results were seen and with world knowledge of later events, so they carry outcome-knowledge risk (tag K). Kari to confirm wording and thresholds.

## 2. Definitions (frozen if signed)

- **Scoring rule:** V = C + K - S on set-mean primitives, pooled cell by cell. A node-year cell exists in a set only with n_eff >= 3 complete scorer judgments (uniform recompute rule). V_protocol is reported alongside but not used for pass/fail.
- **Primary pool:** Opus + Sonnet + Kimi v2 + Codex v2, equal weight per set over the sets that hold the cell. **Secondary pool:** the same plus Grok pass 1. Primary pass/fail counts. The secondary result is reported with its own thresholds; it does not change a prediction's verdict, but a disagreement is reported as such.
- **M(entity, year):** mean pooled V over the entity's 8 nodes. It needs at least 6 valid node cells, otherwise the year is UNSCORABLE and counts as neither pass nor fail.
- **Bond (entity-year):** Bond_i = mean over valid partners j of [0.6 C_i C_j + 0.4 A_i A_j] exp(-(S_i+S_j)/20), computed on pooled primitives with partners = valid pooled cells, then averaged over nodes. This pool-level Bond differs from per-set Bond and is not harmonised with it.
- **MAD distance:** mean absolute difference over node x primitive (C,K,S,A) x year, on pooled cells.
- **Pre-shock level:** mean of M over 2016-2018. Primary: Evergrande 6.89, Huawei 10.15, BYD 8.20, CATL 8.99, Vanke 8.49, Longfor 9.66, SMIC 7.49. Secondary: Evergrande 7.45, Huawei 10.14, BYD 8.51, CATL 9.11.
- **Outcome class at 2024** (M in 2024): **Lose viability** = M <= 4.0 (below every non-Evergrande entity-year in 2010-2020; the lowest was 4.68). **Preserve** = M >= pre-shock level - 1.0 (includes strengthening). **Reorganise** = 4.0 < M < pre-shock level - 1.0.
- **Scale of noise:** the standard deviation of annual change in M over 2011-2020 is 0.28 (CATL, Longfor), 0.64 (Huawei), 0.75 (SMIC), 0.84 (Evergrande), 0.96 (BYD), 0.98 (Vanke), 0.57 (China).

## 3. Section 7 template (protocol placeholder), per primary company

The protocol gives no layout beyond the placeholder, so the four required items are followed exactly:

| Primary company | First-response node and direction | Predicted node sequence | Outcome by 2024 | Most divergent comparator and why |
|---|---|---|---|---|
| Evergrande | Stewards (Finance), V down and held low; P2 | Stewards and Shield lowest throughout; Craft and Flow stay above both; P2 | **Lose viability**; P1 | Longfor, with larger distance than Vanke in 2021-24 (stable; P10) |
| Huawei | Shield (external-constraint node) stays lowest; P3 | Shield lowest each year; the rest settle without further large falls; P3 | **Reorganise**; P3 | SMIC, and the two converge, which fits the external-constraint reading (P9) |
| BYD | Stewards or Shield is the lowest node; V does not fall back; P4, P6 | Same Finance/Risk floor; P6 | **Preserve**; P4 | None specified in 5.2 (flag). For information: closest on 2016-2020 profile are SMIC (0.39) and CATL (0.43); farthest Evergrande (0.88) |
| CATL | Shield or Stewards lowest node; P5, P6 | Same floor, rising V, Bond high; P5, P6 | **Preserve**; P5 | None specified in 5.2 (flag). Closest BYD (0.43); farthest Evergrande (1.19) |

## 4. Predictions

Each lists the primary-pool criterion, the secondary-pool criterion, confidence, basis, and whether naive persistence of the 2020 value would pass. Confidence is for the primary criterion. All use V = C+K-S. Tag K applies to all.

**P1. Evergrande loses viability (V).** *Primary:* M(2021)...M(2024) are all <= 5.0 and M(2024) <= 4.0. *Secondary:* all <= 5.5 and M(2024) <= 4.5. *Fail:* any year > 5.0, or M(2024) > 4.0. *Confidence: High, 75%.*
Basis: M fell 6.70 (2018), 4.98 (2019), 2.98 (2020); the 2010-2018 minimum was 6.70. Every node is below its 2018 level. Stewards V went 4.0, 2.3, 0.25 and Shield V 3.7, 2.9, 0.5. A rebound above 5.0 would be a 2.0-point rise, 2.4 times the 0.84 annual noise. Secondary 2020 value 3.96. *Persistence passes? Yes.*

**P2. Evergrande node structure holds (first response Stewards, down).** *Primary criterion, each of 2021-24:* (a) the lowest-V node is Stewards or Shield; (b) Stewards is one of the two lowest; (c) min(V_Craft, V_Flow) > max(V_Shield, V_Stewards); (d) V_Stewards stays below its 2019 value (2.3). Pass if all four hold in at least 3 of the 4 years. *Secondary:* same, with (d) < 3.0. *Confidence: High, 75%.*
Basis: (a) 11 of 11 years, (b) 11 of 11, (c) 11 of 11 in 2010-2020 (secondary pool (b) 10 of 11). Stewards went 6.37 to 0.25 and was the lowest node in 2019-2020. *Persistence passes? Yes.*

**P3. Huawei reorganises, with Shield the weakest node.** *Primary:* M(2024) is above 4.0 and below 9.15 (pre-shock level 10.15 minus 1.0), and M stays below 9.15 in every year 2021-24. *Also reported (not part of pass/fail):* Shield is the lowest-V node in at least 3 of 4 years (basis 11 of 11 years, confidence 85%). *Secondary:* 4.0 < M(2024) < 9.14. *Confidence: Low-medium, 55%.*
Basis: M fell 10.42 (2017), 9.79, 8.68, 7.39; S rose 4.5 to 6.4. Shield V fell 7.1 to 4.7 in 2018 and has sat at 4.8 for three years while other nodes fell later. This is a stabilisation call, not an extrapolation. A straight-line continuation of the 2018-2020 slope (about -1.2 per year) would reach the lose-viability band (M <= 4.0) by about 2023, which would fail P3. *Persistence passes? Yes (7.39). Linear extrapolation fails.*

**P4. BYD preserves.** *Primary:* M(2024) >= 7.2 (pre-shock 8.20 minus 1.0) and M >= 6.5 in every year 2021-24. *Secondary:* M(2024) >= 7.5 and every year >= 6.5. *Confidence: Medium, 65%.*
Basis: M was 8.32, 8.49, 7.99, 8.11, 7.23, 9.24 over 2015-2020 (minimum 7.23). BYD has the largest annual noise among the primaries (0.96), and 42 percent of its pair-node years move by 1 or more. The 2020 value leaves a 2.0 margin above the threshold. *Persistence passes? Yes.*

**P5. CATL preserves, with high Bond.** *Primary:* M(2024) >= 8.0 and M >= 8.3 (its 2016-2020 minimum) in every year 2021-24, **and** Bond >= 23.1 (its 2016-2020 minimum) every year. *Secondary:* M(2024) >= 8.1, every year M >= 8.55, Bond >= 24.2. *Confidence: High, 75%.*
Basis: M has risen every year since 2011 (4.68 to 10.20; annual changes +0.08 to +1.06; noise 0.28, the lowest of any entity). Bond 27.7, 28.5, 30.4 over 2018-2020. *Persistence passes? Yes.*

**P6. Finance and Risk are the floor across the four primaries.** *Primary:* of the 16 company-years (Evergrande, Huawei, BYD, CATL x 2021-24), at least 14 have Stewards or Shield as the lowest-V node (cells with at least 6 valid nodes only). *Secondary:* at least 14 of 16. *Confidence: High, 80%.*
Basis: in 2010-2020 the lowest node was Stewards or Shield in 43 of 43 primary-pool company-years (Evergrande 11, Huawei 11, BYD 11, CATL 10) and 42 of 43 with Grok. This also fails for non-primaries (SMIC 2 of 11, Vanke 0, Longfor 0), so it is specific to the primaries and depends on the unapproved node map. *Persistence passes? Yes (4 of 4 in 2020).*

**P7. Bond separates Evergrande from its comparators.** *Primary:* in every year 2021-24, min(Bond_Vanke, Bond_Longfor) - Bond_Evergrande >= 10.0. *Secondary:* >= 8.0. *Confidence: Medium-high, 70%.*
Basis: that gap was between 2.5 and 9.8 in every year 2010-2018 (never 10 or more) and rose to 13.6 in 2019 and 16.5 in 2020. Bond 2020: Evergrande 11.2, Vanke 27.8, Longfor 29.3. Evergrande's Bond was 17.7 or higher through 2018. Note the earlier caveat that Bond depends on the partner rule and node coverage. *Persistence passes? Yes (16.5).*

**P8. FC3-style ratio stays wide after the event.** Computed on 2021-24 pooled cells: ratio = MAD(Evergrande, X) / MAD(Vanke, Longfor) for X = Vanke and Longfor. *Primary:* both ratios >= 5.0. *Secondary:* both >= 3.5. *Fail:* either ratio below the threshold. *Confidence: Medium, 65%.*
Basis (primary pool, vs Vanke / vs Longfor): 2010-2019 2.49 / 2.24; 2016-2019 3.09 / 3.50; 2018-2020 8.01 / 8.52; 2020 alone 11.47 / 12.19. Secondary: 2.07 / 1.80, 2.17 / 2.43, 5.49 / 5.79, 7.77 / 8.18. The denominator is small (0.17 in 2020; 0.39 over 2010-2019), so a modest rise in the Vanke-Longfor distance (above about 0.4) would sink the ratio. This is a post-event instance, not the pre-registered FC3 (Failure Condition 3 concerns 2010-2020 profiles before the policy event). *Persistence passes? Yes.*

**P9. Huawei and SMIC converge.** *Primary:* |M_Huawei - M_SMIC| <= 1.5 in every year 2021-24. *Secondary:* same. *Confidence: Medium-low, 55%.*
Basis: gap 2.4 to 4.3 over 2010-2018, then 1.03 (2019) and 0.55 (2020); with Grok 0.60 and 0.23. SMIC itself is flat (M 6.84-7.81 since 2016, Helm and Shield dropped in 2020). Convergence fits the external-constraint reading (Failure Condition 4, direction against domestic-alignment H3 for tech). Two-sided: a Huawei collapse well below SMIC would fail it. Only two entities, so no test of H3 itself. *Persistence passes? Yes (0.55).*

**P10. Longfor is the most divergent comparator for Evergrande.** *Primary:* MAD(Evergrande, Longfor) > MAD(Evergrande, Vanke) over the 2021-24 window (all four years pooled). *Secondary:* same. *Confidence: Medium-low, 60%.*
Basis: Longfor-distance exceeds Vanke-distance in 2016-2019 (1.18 vs 1.05) and 2020 (2.07 vs 1.95); with Grok 0.92 vs 0.82 and 1.75 vs 1.67. The margins are only 6-12 percent. Longfor's M exceeded Vanke's in every year 2015-2020 and not before; Longfor is also the most stable entity (annual noise 0.28 vs Vanke 0.98). *Persistence passes? Yes.*

**P11. The comparators Vanke and Longfor preserve.** *Primary:* M(2024) >= 7.49 for Vanke and >= 8.66 for Longfor (each pre-shock level minus 1.0), and each stays above those thresholds in every year 2021-24; pass only if both do. *Secondary:* same thresholds. *Confidence: Medium, 65%.*
Basis: Vanke M 8.93-9.62 over 2017-2020 (with a one-time dip to 7.02 in 2016 and 7.96 in 2015). Longfor M 9.27-10.17 over 2015-2020 (minimum 8.66 in 2012). Vanke's own annual noise is 0.98. The protocol requires comparators whose outcomes may contradict the simple national narrative, and this prediction is the "survivors survive" version. *Persistence passes? Yes.*

**P12. Coupling does not transmit (CAMS result predicted against H3).** Rerun `33_coupling_codex_v2.py` unchanged except the right-censor year moves from 2020 to the last scored year, so the four events censored in Part 1 (Evergrande Three Red Lines 2020, BYD NEV plan 2020, CATL NEV plan 2020, SMIC Entity List Dec 2020) become scorable. *Primary criterion:* on the primary pool, primary rules (|dV| >= 1, national gate G1, strict lag, company-first = fail, V = C+K-S), **no event of the 8 scores 2 or more**. *Fail:* any event scoring 2 or more. *Also reported:* lenient G2/lag-0 and threshold-2 variants; V_protocol basis; secondary pool. *Confidence: Medium-high, 70%.*
Basis: every one of the 8 events scored 0 of 4 in Codex v2 and in both new pools under the primary rules. The highest score under primary rules in any set or earlier pool was 1 (Huawei 2019, a threshold-fragile pass), and the highest under any variant was 2 (BYD 2012, lenient rule, Opus+Sonnet pool). National nodes move by 1 point or more in only 10-30 percent of node-years (primary pool, China: Stewards 0.1, Flow 0.3, Helm 0.3, Lore 0.1). This predicts a result adverse to the experiment's own H3 (and toward Failure Condition 1), pre-registered on purpose. Company coverage for Helm and Lore in Codex v2 is almost empty, so many pair cells may stay NA (UNSCORABLE, not a fail). *Persistence passes? Not applicable.*

## 4A. Contrast predictions (revision 2): persistence of the 2020 value fails each of these

**Why they exist.** P1 to P11 would all pass if every entity stayed at its 2020 value. A contrast prediction needs a change from 2020. Passing one is evidence of something beyond persistence; failing one is not evidence against CAMS (see the scoring notes).

**Provenance and bias (tag K on all).** The themes (US decay, Chinese strength, Huawei survival, BYD pressure of scaling) were proposed by Kari after the 2010-2020 results were visible and with world knowledge of later events (owner message pt77u). I (the drafter) also have background awareness of later developments. Direction is not stated and no post-2020 source was consulted. Selecting these themes is itself a choice that outcome knowledge may have shaped, so a reviewer with no exposure should read them before sign-off.

**Threshold rule (one rule for all three, set from 2010-2020 only).** theta = 2 x the standard deviation of the annual change of the same quantity over 2011-2020, in the same pool. This is the one-sigma size of a four-year move if annual changes were independent noise (random-walk scaling). It is a convention, not a protocol rule, and is Kari's to change. Values come from `43_contrast_basis.py` (output `contrast_basis_stdout.txt`). Under a zero-drift random walk a four-year rise of at least theta has chance of about 16 percent (normal approximation, my arithmetic). In the actual 2010-2020 four-year windows (n = 7, overlapping, so weak evidence) the share at or above theta was 1 of 7 for China M, 2 of 7 for BYD M and 0 of 7 for BYD S.

**C1. Chinese national strength: China M rises clearly above its 2020 level.** *Primary:* M_China(2024) >= 8.16 + 1.13 = **9.30** (2020 value 8.16; theta = 2 x 0.567). *Secondary:* M_China(2024) >= 8.24 + 0.94 = **9.18** (2020 value 8.24; theta = 2 x 0.471). *Fail:* below the threshold in 2024. *Confidence: Low, 30%.*
Basis: China M ran 7.31 to 9.22 over 2010-2020, peaked at 9.22 in 2017 and has fallen each year since (9.22, 8.76, 8.44, 8.16). The threshold sits just above the 2010-2020 maximum. Only 1 of 7 four-year windows rose by theta or more (the largest was +1.26). China S reached its 2010-2020 high in 2020 (5.80). *Persistence (8.16): FAIL. Linear extrapolation of 2018-2020 (slope -0.30 per year, 6.97 in 2024): FAIL.*

**C2. Huawei survival: the slide reverses.** *Primary:* M_Huawei(2024) > M_Huawei(2020) = **7.39** (any rise; strict inequality). *Secondary:* M_Huawei(2024) > **7.26**. *Fail:* M(2024) <= the 2020 value. *Also reported, not part of pass/fail:* a margin version, M(2024) >= 7.39 + 1.28 = **8.67** (theta = 2 x 0.640), because the annual noise is 0.64 and a rise smaller than that is within one year of ordinary movement. *Confidence: Medium-low, 45%.*
Basis: M fell 10.42 (2017), 9.79, 8.68, 7.39, the lowest Huawei value in 2010-2020; S rose to its high of 6.36 in 2020. This is stricter than P3, which passes for any 2024 value between 4.0 and 9.15 and so passes at 7.39 or at 5.0. A pass here requires the fall to stop and turn. The bare inequality is a weak test: with no drift it has about a 50 percent chance, which is why the margin version is reported beside it. *Persistence (7.39, not greater than itself): FAIL. Linear extrapolation (slope -1.20 per year, 2.62 in 2024): FAIL.*

**C3. BYD pressure of scaling: Stress rises while M holds.** *Primary (both must hold in 2024):* (a) mean S_BYD(2024) >= 4.75 + 0.79 = **5.54** (2020 value 4.75; theta = 2 x 0.394) and (b) M_BYD(2024) >= **7.2** (the P4 threshold). *Secondary:* (a) S >= 4.52 + 0.63 = **5.15** (2020 value 4.52; theta = 2 x 0.313) and (b) M >= 7.5. *Fail:* either condition missed. *Also reported, not part of pass/fail:* the same test on the mean S of the two nodes mapped to Operations and Workforce (Craft and Hands): primary >= 4.83 + 0.65 = **5.48** (secondary >= 4.55 + 0.49 = **5.04**). That node-level reading inherits the unapproved positional node map, which is why the pass/fail version uses whole-entity S. *Confidence: Low, 20%.*
Basis: BYD S has stayed between 4.73 and 5.53 in 2010-2020 (5.51 in 2019, 4.75 in 2020), so the threshold equals the historical maximum. M was 7.23 to 9.24 over 2015-2020 and the P4 margin leaves 2.0 above 7.2. No four-year window in 2010-2020 raised BYD S by theta (0 of 7). "Pressure with resilience" means S rises and M does not fall below the P4 line. C3 passes only if P4's M condition also holds in 2024. *Persistence (S 4.75): FAIL. Linear extrapolation of S (slope -0.13 per year, 4.45 in 2024): FAIL.*

**C4. US decay: NOT TESTABLE, out of scope.** The United States is not an entity in this experiment. The scored entities are China (national) and the seven companies Evergrande, Huawei, BYD, CATL, Vanke, Longfor and SMIC. No US primitives exist, so no US prediction can be written, scored or hashed here. A relative claim such as "China strengthens against the US" would need a US scoring run, a design change and Kari's approval. Recorded as an out-of-scope note only; it adds no pass/fail item.

**Reading C1 to C3 together.** Each can pass or fail independently. They are not part of Failure Condition 5. Because they are low-confidence tests built to beat persistence, a majority failing is the expected result by my own confidences (expected passes about 1.0 of 3: 0.30 + 0.45 + 0.20 = 0.95), and should not be read as a CAMS failure. A single pass is informative mainly because persistence cannot produce it.

## 5. Scoring plan on sign-off

- Pass/fail uses the primary pool only. Secondary results are reported alongside and any disagreement is stated.
- Each prediction is reported as PASS, FAIL or UNSCORABLE. Mixed outcomes are partial results (protocol Section 8).
- **Failure Condition 5** is triggered if at least 3 of P1, P3, P4, P5 fail on the primary pool.
- Report the persistence baseline next to each result. For C1 to C3 also report the straight-line 2018-2020 extrapolation.
- C1 to C3 are reported as PASS, FAIL or UNSCORABLE (an entity-year with fewer than 6 valid node cells is UNSCORABLE) and are not counted in FC5. State how many pass against the persistence baseline, which scores 0 of 3 by construction.
- Predictions C2 (bare inequality), P8, P10 and P12 depend on small margins or small denominators. Their results are to be read as weak evidence either way.

## 6. Reproducibility

- Reference pipeline: `analysis/interim/codex_v2_update/` (manifest sha256 `a0df79b3895aa215bd9405d0d1677b2919f3bc144dc231e11702eaa7c38b6650`, ledger 4 Nov 2026 20:13:31 AEDT). Inputs used: `tidy_v3.csv` (sha256 `826f9de0...846a45`), `32_fc3_codex_v2.py` (`2e6ae3e7...a46e7`), `33_coupling_codex_v2.py` (`a424b2f8...8f8`).
- This folder: `40_basis_tables.py`, `41_basis_for_predictions.py`, `42_basis_extra.py`, `43_contrast_basis.py` (revision 2, with `contrast_basis_stdout.txt`), `predictions_DRAFT_v1_20-25.md` (revision 1), pooled cell and entity-year CSVs, stdout files, `MANIFEST.sha256`. The manifest hashes the DRAFT. It is not a sign-off and has no ledger line.
- No 2021+ data, no web lookups, nothing sent or posted.

- 2026-11-04 21:23 AEDT: Kari CONFIRMED Part 2 evidence cutoff = 30 June of Y+1 (t92u).

---
SIGNED by Kari McKern (message t96u) at 2026-11-04 21:28:58 AEDT. Cutoff 30 June Y+1; bands, FC5 and contrast predictions as drafted.
