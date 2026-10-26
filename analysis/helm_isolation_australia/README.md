# Helm Isolation, Australia 2008–2025

Independent recompute of the frozen Helm Isolation rule (|C_Helm − mean C of the other 7 nodes| ≥ 1.86, unrounded; set 20 Oct 2026).

Inputs (SHA-256), repo commit `374bdf7e526fb23880bd88d67d067a384de658d9`:
- `cleaned_datasets/Australia_ENS_1875_2026_cleaned.csv`: `e121e06da563154319eee36173174f472b259ed4b54f8eafe1870ae63df646cc`
- `cleaned_datasets/Australia_cam5_1996_2026_cleaned.csv`: `a72d4a9a0bb834f199a00ab1e85dbab8b216f4ddcf83df0f4ab34e7257dad1ef`
- `Australian_Gem_Jan30.csv` (1,008 rows; the `_CLEAN` 896-row file cited in the draft was not found): `0389328673575f3b78b9b46e03766a5bf4042c66ebb854881a3889d35a9cb983`
- `KSCORESv1_Australia_2015_2025_block0.csv` (from `Kimi_Agent_Skill Build Requirements Discussion.zip`, not in repo): `667889c7127d3688eb44b6b08c96d4934446934c4ee1a4d5ea01730a244ac32c`

Files: `PREREG.md` (frozen before computing), `run.py` → `signed_gaps.csv`, `charts.py` and `dashboard.py` → `charts/`, and the note (`.md`, `.pdf`). Scripts use absolute paths from the analysis box; adjust paths to rerun.
