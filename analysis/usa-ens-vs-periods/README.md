# USA ensemble vs 25 periods, and the external-marker test (29 Oct 2026)

Public page: https://neuralnations.org/usa-ens-vs-periods (source `usa-ens-vs-periods.html` at repo root). Diary entry: https://neuralnations.org/grok.

## Layout
- `study1_ensemble_vs_periods/` Study 1. `PREREG.md` and `FROZEN.txt` (hashes), `run.py` (frozen machinery), `secondary.py`.
  - `hifi_test/` first run, hi-fi annual file (test).
  - `five_yearly/` USA_ENS five-yearly file (H1 strict not supported; H2, H3 not evaluable).
  - `canonical_annual/` the canonical annual USA ensemble run, with `DEVIATIONS.md` (dataset swap log) and `ENV_note.md`.
- `study2_external_markers/` Study 2. `PREREG.md`, `PREREG_draft_v0.md`, `FROZEN.txt`, `run.py`, `run_log.txt`, result tables, `markers/` (derive script and the six marker CSVs).
- `export_panel_data.py` builds `panel_data.json` and `period_profiles_compare.csv` for the page. It rebuilds the rater consensus profiles with the frozen `consensus()` and asserts the per-period r equals the logged values.
- `panel_data.json`, `period_profiles_compare.csv` the data the page loads.

## Paths
Scripts use absolute paths from the analysis box (`/workspace/...`). Edit them to rerun elsewhere.
- Rater files: `analysis/blind_rater_accidental_test/data/` in this repo.
- Hi-fi annual file: `data/nations/USA_v12opt_hifi_block1.csv` in this repo.
- `USA_ENS_annual.csv` (md5 `1c9626e9481662df6c00974f720e0ec0`, 127 years x 8 nodes) is not in this folder. It was reconstructed from a text read of the supplied file, so there is no original checksum. Place it at `/workspace/uploads/usa_ens_annual/USA_ENS_annual.csv` to rerun the canonical run.

## Frozen hashes
- Study 1 PREREG sha256 `3603e474ada3652e2d665e8787ca9c4bb08ba3e38bd23a8614e69611c3aea2fe`
- Study 2 PREREG sha256 `38dcbe910d40dfff1f7a4a7bdf9c7b73892766c27e6729065aa027a34a5e04c5`
- Other frozen hashes are in each `FROZEN.txt`.
