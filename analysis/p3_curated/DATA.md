# Rebuilding analysis/p3_curated

The panel CSVs are not committed. They are subsets of files already in the repo at 446928c:
- `CURATED_CAMS.csv` (SHA-256 prefix 9320a0e7fb28009a): `select_runs.py` builds it from the ENS files listed in `selected_runs.csv` (`cleaned_datasets/`, `data/nations/`, `analysis/camnations5_eritrea_1993_2026.csv`). For the 17 societies shared with JUNO, it is identical to `juno/JUNO_Unified_Dataset.csv` on all 19,048 shared node-rows.
- `JUNO_RESTRICTED.csv`: written by `p3c_run.py` (JUNO limited to the curated society-years).
- `LAMBDA2_CURATED.csv` and `LAMBDA2_JUNO_RESTRICTED.csv`: written by `build_curated_inputs.py` and `p3c_run.py`.

Order: `inventory.py` → `select_runs.py` → `build_curated_inputs.py` → `p3c_run.py`. The scripts use absolute paths from the analysis box (`/workspace/wm_pr` for the repo, `/workspace/analysis/p3` for the P3 inputs); point them at your checkout and at `analysis/p3/`.
