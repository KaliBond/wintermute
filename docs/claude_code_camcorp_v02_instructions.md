# Claude Code Instruction Set: CAMS-CORP v0.2 Pipeline

Use this instruction set when asking Claude Code to inspect, repair, extend, or verify the CAMS-CORP v0.2 analysis pipeline in this repository.

## Mission

You are operating in `C:\Users\julie\wintermute`. Maintain the CAMCORP5 / CAMS-CORP v0.2 analysis pipeline as a reproducible Python workflow over these source files:

- `C:\Users\julie\Desktop\HariSeldon\cam5\camcorp5_ensemble_mean.csv`
- `C:\Users\julie\Desktop\HariSeldon\cam5\camcorp5_envelope.csv`

The formulation is locked as of `2026-05-29`. Treat locked constants and thresholds as contractual unless explicitly instructed to create a v0.3 revision.

## Primary Files

- `src/camcorp5_pipeline/config.py`: JSON config loader.
- `src/camcorp5_pipeline/io.py`: CSV loading, schema validation, numeric coercion, output writer.
- `src/camcorp5_pipeline/schema.py`: required input schemas and key definitions.
- `src/camcorp5_pipeline/transforms.py`: base node panel, annual summary, and rankings.
- `src/camcorp5_pipeline/constants.py`: locked CAMS-CORP v0.2 parameters. Do not alter for v0.2 work.
- `src/camcorp5_pipeline/corp_v02.py`: locked CAMS-CORP v0.2 metrics.
- `src/camcorp5_pipeline/pipeline.py`: orchestration and output manifest.
- `src/camcorp5_pipeline/cli.py`: command-line entry point.
- `configs/camcorp5.json`: default run config.
- `docs/analysis_pipeline.md`: user-facing pipeline notes.
- `tests/test_camcorp5_pipeline.py`: smoke test; if it fails because of incomplete nodes, update the fixture to include all 8 CAMS nodes.

## Locked Parameters

Do not change these values unless the user explicitly requests a v0.3 formulation:

- EDEWS: `mu_raw=0.0058`, `sigma_raw=0.0945`
- EDEWS thresholds: `WATCH=0.062`, `WARNING=0.693`, `CRITICAL=1.122`, `EXTREME=1.303`
- Memory decay: `MU_P75=0.455`, `MU_P90=0.712`
- Helm-Craft divergence: `D_HC_P90=1.87`, `D_HC_P95=1.90`
- Library attractor: `ETA_EPSILON=2.0`, `P10=19.2`, `P50=97.0`, `P75=166.4`, `P90=219.2`
- Crisis thresholds: `Moderate=0.50`, `Severe=0.60`
- CF-1 growth boundary: `Delta V_mean > 0.3 per year`
- CF-2 discordance threshold: `V_range > 2.0`

## Required Outputs

A successful run writes CSV artifacts to `exports/camcorp5_pipeline/`:

- `camcorp5_node_panel.csv`
- `camcorp5_yearly_summary.csv`
- `camcorp5_node_rankings.csv`
- `corp_v02_edews.csv`
- `corp_v02_mu.csv`
- `corp_v02_kappa.csv`
- `corp_v02_cf1.csv`
- `corp_v02_hc_divergence.csv`
- `corp_v02_eta_loop.csv`
- `corp_v02_crisis_score.csv`
- `corp_v02_alarm_protocol.csv`
- `corp_v02_discordance.csv`

## Commands

Prefer the bundled Codex Python runtime because the global Python may not have `pandas` installed:

```powershell
C:\Users\julie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.camcorp5_pipeline.cli --config configs/camcorp5.json
```

Run tests with standard library `unittest`:

```powershell
C:\Users\julie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_camcorp5_pipeline -v
```

If `pytest` is unavailable, do not install dependencies unless asked. Keep tests runnable with `unittest`.

## Validation Anchors

After running the pipeline, check these anchors against exported CSVs. Small rounding differences are acceptable only at the final decimal place.

`corp_v02_edews.csv`:

- `CATL, 2010`: `edews_v0_2 ~= 6.5928`, level `EXTREME`
- `General Motors, 2008`: `edews_v0_2 ~= -2.6923`, level `EXTREME`
- `General Motors, 2009`: `edews_v0_2 ~= 2.7309`, level `EXTREME`
- `Tencent, 2021`: current executable result is `edews_v0_2 ~= 1.2907`, level `CRITICAL`

`corp_v02_cf1.csv`:

- `General Motors`: mature, pass
- `BYD`: mature, marginal fail
- `CATL`: growth, pass
- `Tencent`: mature, pass

`corp_v02_mu.csv`:

- `BYD`: low decay, `mu_entity ~= 0.1371`
- `CATL`: low decay, `mu_entity ~= 0.1103`
- `General Motors`: high decay, `mu_entity ~= 0.4298`
- `Tencent`: high decay, `mu_entity ~= 0.5301`

`corp_v02_crisis_score.csv`, top expected detections:

- `CATL, 2010`: severe, `~0.7481`
- `General Motors, 2008`: severe, `~0.6309`
- `General Motors, 2009`: severe, `~0.6257`
- `General Motors, 2021`: moderate, `~0.5800`

## Implementation Rules

1. Preserve the input grain: one row per `Entity`, `Year`, `Node`.
2. Validate required columns before computing metrics.
3. Keep the base scaffold outputs backward-compatible.
4. Keep locked constants in `constants.py`; do not scatter numeric thresholds through code.
5. Use vectorized `pandas` operations where practical, but favor clarity for formulas that map directly to the locked text.
6. If a metric requires a node that is absent from a synthetic fixture, fix the fixture rather than weakening the metric.
7. Do not import national CAMS kappa thresholds into corporate calculations.
8. Do not change generated output filenames without updating docs and tests.
9. When adding v0.3 work, create new constants/functions or explicit version switches rather than mutating v0.2 behavior silently.

## Suggested Claude Code Task Prompt

Use this prompt when you want Claude Code to run the pipeline end-to-end:

```text
You are in C:\Users\julie\wintermute. Run and verify the CAMS-CORP v0.2 pipeline using docs/claude_code_camcorp_v02_instructions.md as the governing instruction set. Preserve locked v0.2 constants. Use the bundled Codex Python runtime if global Python lacks pandas. Ensure all expected CSV outputs are written under exports/camcorp5_pipeline, verify the EDEWS, CF-1, mu, and crisis-score anchors, and fix only scaffold/test issues required for reproducibility. Report changed files, commands run, and validation results.
```
