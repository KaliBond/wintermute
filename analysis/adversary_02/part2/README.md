# ADVERSARY-02 Part 2 (2021-2024): results files

Copied unchanged from the verified analysis folder `analysis_v2_manus_2026-11-04` (plus the sealed Section 7).
Source of truth for the report page `adversary-02-results.html` and the /explore panel.

- `predictions_SIGNED.md` + `section7_SIGNED_MANIFEST.sha256`: sealed predictions, signed 2026-11-04 21:28:58 AEDT. The file's SHA-256 is e15e1198...fb28 (check with `sha256sum -c section7_SIGNED_MANIFEST.sha256`, which will report the other listed files as missing); the manifest file itself hashes to 766a1cbd...be6b (full: 766a1cbd23ccb1dbfe5ca13218c853f3269373fc0b20b31ea2144a47fe0fbe6b); the scoring table misprints it as ...fe6b.
- `scoring_table.md` / `.csv`: verdicts for all 15 items under four pools, per-scorer verdicts, key metrics, sensitivity.
- `entity_personality_entity_year.csv`: energy (K-S) and focus (A*C) per entity-year. 2021-24 = primary pool; 2010-2020 = Part 1 baseline (method seam at 2020/2021). `adversary02-trajectories.js` embeds these values, rounded to 2 dp.
- `entity_personality_scorer_spread.csv`, `entity_personality_scorer_agreement.csv`: scorer spread and rank agreement.
- `scorer_assessment.md`: scorer grades. `methods_note_v2.md`: methods.
- `51_score.py`, `55_entity_personality.py`: scripts (paths point at the original analysis folder). `part2_analysis_MANIFEST.sha256`: hashes of the analysis folder files.
- `figures/`: fig1 personality map, fig3 scorer grades.

V in this experiment is C + K - S (no A term), as sealed. The site's Canonical Definitions page defines V = C + K + A/2 - S.
