# Separate run — 2026-09 staging (Burma, Italy, Netherlands rescore)

**Status: staging. NOT part of JUNO_Unified_Dataset.csv. Not on the live Explorer/Interpreter/Zeitgeist tools.**

## What this is

Three raw datasets that arrived in the repo without having been run through the
v1.2-Final bond-alignment kernel:

- **Burma** (`cams/Burma_Block1.csv`, 1800–2026) — a society not present in
  `JUNO_Unified_Dataset.csv`'s 48-society panel at all.
- **Italy** and **Netherlands rescores** (`Italy_CAMS_m5n_1850-2026_block1.csv`,
  `Netherlands_CAMS_m5n_1850-2026_block1.csv`) — both societies are already published
  in `JUNO_Unified_Dataset.csv` (Italy 1900–2024, Netherlands 1900–2024), but these
  files carry a wider range (1850–2026) and materially different raw C/K/S/A values
  from what's currently published (e.g. Italy 1900 Helm: published C=4.0/K=4.0/S=5.0/A=5.0
  vs. this file's C=4.8/K=5.0/S=6.2/A=4.2). This is a distinct scoring pass, not a
  duplicate of the published source.

`juno/compute_separate_run_2026-09.py` recomputes the v1.2-Final derived operators
from raw C/K/S/A only, using the exact formulas in
[`verify_bond_alignment.py`](verify_bond_alignment.py) /
[`JUNO_v1.2-Final_Formalism.md`](JUNO_v1.2-Final_Formalism.md):

- `Node_Value_Calc = C + K − S + 0.5·A`
- `Bond_Strength_Calc` (per-node mean of the 7 rank-1 off-diagonal edges)
- `SBD_Calc` (mean of the 28 unique pairwise bonds)
- `Lambda2_Calc` (second-smallest eigenvalue of the graph Laplacian)
- `V_Mean_Calc`, `V_Min_Calc`

Output: [`JUNO_SeparateRun_2026-09_staging.csv`](JUNO_SeparateRun_2026-09_staging.csv)
— 4,128 node-rows, 516 complete society-years (Burma 163, Italy 177, Netherlands 176).
Each row is tagged `Run = separate_run_2026-09_staging` so it can never be silently
confused with a `JUNO_Unified_Dataset.csv` row even if the files are later concatenated.

**Verified:** `python juno/verify_bond_alignment.py` run against this file (with
`CSV_PATH` pointed at it) passes all asserts — `Bond_Strength_Calc`, `SBD_Calc`,
`Lambda2_Calc`, and `Node_Value_Calc` all match independent recomputation to
<5e-7.

## What this is NOT

`Phase_Calc`, `Regime_Label_Calc`, `Decay_Index_Calc`, `ESCH_Calc`,
`CAMS_v1_Regime`, and `Bond_Strength_Calc_legacy` are left as the literal string
`not_computed` (the same convention `JUNO_Unified_Dataset.csv` already uses for
rows an older pipeline stage never touched). Those columns come from a
classifier/pipeline stage not present in this repo — the six-way regime label
in the published file (`Systemic Collapse` … `High Coherence`) does not match
the v1.2-Final formalism doc's precedence-based classifier (`Freeze/Collapse` …
`Strained`), so it was not safe to guess at reproducing it here.

Society-years with a gated/missing node (e.g. Burma pre-1811, before Stewards
existed as a scoreable institution per `cams/burma_score.py`) are dropped
entirely, matching the "incomplete society-years" convention in
`JUNO_Unified_Dataset.csv`.

## Open decision (not made here)

Whether the Italy/Netherlands rows in this file should **supersede** the
currently published Italy/Netherlands rows in `JUNO_Unified_Dataset.csv`, and
whether Burma should be **merged in** as a 49th society, is a methodology call
for Kari — it changes published, citable figures (Zenodo-archived at
`10.5281/zenodo.22029112`). This file exists so that decision can be made by
looking at real recomputed numbers, not raw scores alone.
