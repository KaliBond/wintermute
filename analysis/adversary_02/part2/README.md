# ADVERSARY-02 Part 2 (2021-2024): results files

Base files were copied from verified analysis `analysis_v2_manus_2026-11-04` (plus sealed Section 7). On 2026-11-06 the scoring script/table were refreshed from v3 to add secondary Kimi Work sensitivity; primary results remain unchanged.
Source of truth for the report page `adversary-02-results.html` and the /explore panel.

- `predictions_SIGNED.md` + `section7_SIGNED_MANIFEST.sha256`: sealed predictions, signed 2026-11-04 21:28:58 AEDT. The file's SHA-256 is e15e1198...fb28 (check with `sha256sum -c section7_SIGNED_MANIFEST.sha256`, which will report the other listed files as missing); the manifest file itself hashes to 766a1cbd...be6b (full: 766a1cbd23ccb1dbfe5ca13218c853f3269373fc0b20b31ea2144a47fe0fbe6b); the scoring table misprints it as ...fe6b.
- `scoring_table.md` / `.csv`: verdicts for all 15 items under six pools, per-scorer verdicts, key metrics, sensitivity. Pool (a) remains primary; pools e/f add or hypothetically substitute secondary Kimi Work, respectively.
- `entity_personality_entity_year.csv`: energy (K-S) and focus (A*C) per entity-year. 2021-24 = primary pool; 2010-2020 = Part 1 baseline (method seam at 2020/2021). `adversary02-trajectories.js` embeds these values, rounded to 2 dp.
- `entity_personality_scorer_spread.csv`, `entity_personality_scorer_agreement.csv`: scorer spread and rank agreement.
- `scorer_assessment.md`: scorer grades, including Kimi Work C+ secondary. `best2_scorers_sensitivity.md` and `.csv`: the additive Claude agent + Gemini check (same scorecard as primary) and exact P8 boundary. `scoring_table_v3_with_best2.csv`: combined convenience table with e/f and best-two results. `methods_note_v2.md`: v2 methods; `methods_note_v3_kimiwork.md`: additive e/f sensitivity.
- `51_score.py`: v3 verdict script, reading included `all_passes_long.csv`; P12 also reads the external sealed Part 1 coupling baseline documented in methods v3. `55_entity_personality.py`: map script (path points at original analysis folder). `kimiwork_passes/`: five input grids; full shared evidence is not published. `part2_analysis_MANIFEST.sha256`: hashes of the published analysis inputs, summaries, scripts and figures except itself, derived intermediates and the separately sealed Section 7 manifest. Its legacy entries for `scorer_prompt_PASTE.md` and `prior_record_index.md` refer to the original analysis package, which are not copied into this public folder.
- `figures/`: fig1 personality map, fig3 scorer grades.

V in this experiment is C + K - S (no A term), as sealed. The site's Canonical Definitions page defines V = C + K + A/2 - S.
