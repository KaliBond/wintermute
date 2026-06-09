# CAMCORP5 Analysis Pipeline

This scaffold turns the CAMCORP5 ensemble mean and envelope CSVs into repeatable analysis artifacts.

## Inputs

- `camcorp5_ensemble_mean.csv`: one row per `Entity`, `Year`, and `Node` with CAMS mean scores.
- `camcorp5_envelope.csv`: matching uncertainty/envelope values for the same grain.

The default config points at:

- `C:/Users/julie/Desktop/HariSeldon/cam5/camcorp5_ensemble_mean.csv`
- `C:/Users/julie/Desktop/HariSeldon/cam5/camcorp5_envelope.csv`

## Run

From `C:/Users/julie/wintermute`:

```powershell
python -m src.camcorp5_pipeline.cli --config configs/camcorp5.json
```

## Outputs

The default config writes to `exports/camcorp5_pipeline/`:

- `camcorp5_node_panel.csv`: validated mean and envelope data merged by `Entity`, `Year`, and `Node`.
- `camcorp5_yearly_summary.csv`: annual system-level aggregates.
- `camcorp5_node_rankings.csv`: node-level panel with annual value and stress ranks.

## Pipeline Stages

1. Load the two source CSVs.
2. Validate required columns and unique `Entity/Year/Node` keys.
3. Coerce numeric score columns.
4. Merge means with envelope values.
5. Add derived metrics: stress/capacity ratio, resilience index, envelope midpoint, and envelope delta.
6. Export node panel, yearly summary, and rankings.

## CAMS-CORP v0.2 Locked Formulation

The pipeline also exports the locked CAMS-CORP v0.2 monitoring stack from 2026-05-29:

- `corp_v02_edews.csv`: Executive Decoupling Early Warning System using locked `mu_raw=0.0058` and `sigma_raw=0.0945`.
- `corp_v02_mu.csv`: entity-level slow-loop memory decay for Archive, Lore, and Stewards.
- `corp_v02_kappa.csv`: within-entity kappa criticality with entity percentile anchors.
- `corp_v02_cf1.csv`: sign-conditional stress-capacity falsification result.
- `corp_v02_hc_divergence.csv`: Helm-Craft divergence and consecutive-year alert state.
- `corp_v02_eta_loop.csv`: regularised library attractor using epsilon `2.0`.
- `corp_v02_crisis_score.csv`: composite crisis score and moderate/severe levels.
- `corp_v02_alarm_protocol.csv`: GREEN/YELLOW/ORANGE/RED/BLACK operational state.
- `corp_v02_discordance.csv`: slow-loop cross-rater concordance proxy from `V_range`.

Locked parameters live in `src/camcorp5_pipeline/constants.py`; change them only as part of a v0.3 revision.
