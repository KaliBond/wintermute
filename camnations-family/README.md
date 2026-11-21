# CAMS Nations scoring runs

Completed output from the `camnations` family of CAMS scoring pipelines, imported
from [KaliBond/wintermute](https://github.com/KaliBond/wintermute). These are
**not** the same as the main `JUNO_Unified_Dataset.csv` 48-society panel — see
status notes per folder below before citing or merging.

## camnations5/ — five-scorer ensemble
Node Value and Bond Strength computed from the mean of five independent raw
scoring passes. Each nation has a main CSV plus an `_envelope` CSV (per-scorer
standard deviation and V_min/V_max range).

| Nation | Range |
|---|---|
| Democratic Republic of the Congo | 1960–2026 |
| Egypt | 1952–2026 |
| Eritrea | 1993–2026 |
| Ethiopia | 1960–2026 |
| Mauritania | 1960–2026 |
| Nigeria | 1960–2026 |

Status: complete, standalone ensemble output.

## camnations5n/ — NA-aware five-scorer ensemble
| Nation | Range |
|---|---|
| Cuba | 1990–2026 |

`block1` = raw mean scores, `block2` = envelope (scorer SD, V range).

## camnationsm5n_staging_2026-09/ — multi-agent evidence-gated, 2026-09 batch
**Status: staging — NOT merged into `JUNO_Unified_Dataset.csv`.** See
`SeparateRun_2026-09_README.md` in this folder for full detail.

- **Burma** (`Burma_Block1/2.csv`, 1800–2026): a society not yet in the main
  48-society panel at all.
- **Italy** and **Netherlands** (`*_m5n_1850-2026_block*.csv`): both already
  published in the main panel (1900–2024), but these rescores carry a wider
  range and materially different raw C/K/S/A values. Treat as a distinct
  scoring pass, not a duplicate.
- `JUNO_SeparateRun_2026-09_staging.csv`: all three run through the v1.2-Final
  bond-alignment kernel and independently verified (`Run` column tagged
  `separate_run_2026-09_staging`).

Whether Italy/Netherlands here should supersede the published rows, and
whether Burma should be merged in as a 49th society, is still an open
methodology decision — not made by this import.

## camnationsmfin/ — final/seam-diagnostic variant
| Nation | Range |
|---|---|
| USA | 2015–2026 |

**Status: delivered, with a known discrepancy.** Per
`USA_mfin_RECONCILIATION_CAMSBOT_2026-09-17.md`, 60 of 96 delivered cells
deviate from the documented FINAL aggregation rule
(`Node Value = C_mean + K_mean − S_mean + 0.5·A_mean`) beyond the ±0.175
rounding tolerance. Rankings and trajectory shape are judged robust; the
Node Value / Bond Strength figures themselves are not fully reconciled.
See `CAMNATIONSMFIN_methodology.txt` and `USA_mfin_2015-2026_ANALYSIS.md`
for the full writeup.

---
*Imported from wintermute. Source of truth for anything beyond raw files
remains the wintermute repo's `juno/CANONICAL_STATUS.md`.*
