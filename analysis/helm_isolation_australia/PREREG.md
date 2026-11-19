# Helm Isolation, Australia 2008–2025: independent recompute
Frozen 27 Oct 2026 (Sydney), before any computation.

Rule (Organisational Pathologies, 20 Oct 2026): D_t = C_Helm,t − mean(C over the other 7 JUNO v1.2 nodes: Shield, Lore, Stewards, Flow, Craft, Hands, Archive). I_t = |D_t|. Fires iff I_t ≥ 1.86, unrounded (float, compared with tolerance 1e-9 only for equality at exactly 1.86).
Years: 2008–2025 where present. No rescaling, no imputation; a year missing any of the 8 nodes is reported as missing.
Inputs:
- ENS: wintermute cleaned_datasets/Australia_ENS_1875_2026_cleaned.csv
- CAM5: wintermute cleaned_datasets/Australia_cam5_1996_2026_cleaned.csv
- Gemini: Australian_Gem_Jan30_CLEAN.csv not found in repo or zip; used wintermute Australian_Gem_Jan30.csv (root; identical hash to data/cleaned and archive copies). Deviation noted.
- Kimi: zip KSCORESv1_Australia_2015_2025_block0.csv, each Run×Batch a separate series (2020 has B1 and B2).
Outputs: signed_gaps.csv (series, year, D, I, fires). Results reported as computed; no amendment to the rule.
