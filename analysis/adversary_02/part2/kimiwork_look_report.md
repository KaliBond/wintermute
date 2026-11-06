# Kimi Work Part 2 run — look report (2026-11-05)
Source: incoming/t157u attachment.zip sha256 a573d431…be52c. All 16 files match the run's own MANIFEST.sha256. Analysis: ../analysis_v3_kimiwork_2026-11-05/ (50/51 = v2 scripts + KimiWork run + pools e/f; 60_kimiwork_look.py; stdout_51.txt, stdout_60.txt).

## Format
validate_passes.py (BASE path changed from Windows only) prints ALL OK. Central 50_load_validate.py asserts pass too. NA cells: 64/60/60/32/60. All corporate NAs are all-four-or-none; China has none.
NA node-years (69 total): Workforce/Hands 58, SMIC Board/Helm 5, Evergrande Markets/Archive 3, BYD Markets 1, Longfor Board 1, Evergrande RD 1. By year: 2021 21, 2022 8, 2023 17, 2024 23. Manus's 58 NAs are all R&D/Lore, and Gemini/Kimi web are also Lore-only. Kimi Work's gaps fall on a different node.

## Repeatability (pairwise, 4 primitives stacked)
Mean r 0.912 (0.892–0.932), exact 60%, MAD 0.42. Manus 0.918, Codex 0.959, Claude agent 0.955, Grok 0.915, Kimi web p1–p2 0.884. CAVEAT: all passes read one shared evidence pack, which inflates agreement relative to independent gathering.
Carry-forward (all 4 primitives same as prior year): 18–28% of rows. Manus is 28–38%, Codex 27–38%, Grok 75–76%. Kimi Work is the lowest in the set.

## Vs primary pool a (7 runs)
Kimi Work is out of pool, so the only fair comparison is Grok, also out of pool. Rows for in-pool runs are flattered.
5-pass mean: mean by-year Spearman V 0.804, Cog(A×C) 0.845, Aff(K−S) 0.833; cell MAD 0.555; entity-year V MAD 1.35. Grok: 0.851/0.940/0.833, MAD 0.798.
Single passes: V rho 0.73–0.92. Pass 4, the one with the most searches, is closest (0.915).
Bias vs pool: C −0.24, K −0.20, S +0.30, A −0.31, which makes it harsh on the strong firms. V diff: CATL −2.37, Huawei −2.03, SMIC −1.77, BYD −1.49. Evergrande +1.45 and Vanke +0.94 are milder. By node, Helm is −1.52 and Shield −1.18. Largest cells: SMIC Board 2022/23 (−6.0, −5.9), Huawei Workforce 2022 (−5.0), Huawei Risk 2024 (−4.9).

## Vs Kimi web
Kimi web p1 vs KW mean r 0.897 (MAD 0.53); p2 r 0.83 (MAD 0.68). KW mean agrees more closely with Manus (0.927) and Claude agent (0.915) than with either Kimi web pass. Kimi web p1 is closer to the pool (0.936) than to KW. The model family is plausibly the same (K2.8 vs claimed k2d8-preview, unverified), but the harness, evidence and NA behaviour all differ. Treat them as separate runs.

## Section 7 sensitivity (51_score.py logic unchanged; V/Bond recomputed centrally)
Pool e = primary7 + KW. Pool f = primary7 − Kimi web + KW.
- P6: a FAIL (13/16) → e PASS (14/16), f FAIL (13/16). KW alone: PASS.
- P8: a FAIL (ratios 4.35/5.32, limit 5.0) → e PASS (5.03/5.99) and f PASS (5.08/5.94). This is a knife-edge pass, 0.03 over the limit. KW alone: PASS.
- All other items and FC5 (NOT TRIGGERED) are unchanged in e and f.
- KW per-scorer verdicts match the primary-pool verdicts except P6 and P8 (both PASS for KW).

## Grade: C+ (proposed)
Rule compliance is good: the format is clean, NA rules are kept, it declared every deviation and had no contamination exposure. Blanks look honest and well-reasoned (Workforce gate, BYD headcount post-cutoff). Repeatability is middling, and inflated by the shared pack. Carry-forward is the lowest of any run. Independence FAILS the protocol: one shared evidence corpus, China evidence from a single collector, about 12 [memory] items, and searches in passes 4–5 only. Closeness to the pool is the weakest in-family apart from Grok on cells, with a systematic harsh bias on strong firms.
Recommendation: SECONDARY only. Report pool e/f as sensitivity. Do not count it as an 8th independent primary or let it replace Kimi web. Its inclusion alone flips P6 (e) and P8 (e, f).
