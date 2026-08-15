# Task: Run CAMS Node Dynamics Dashboard

Run the node dynamics analysis script against the JUNO dataset and regenerate the dashboard.

## What this does

Computes per-node response coefficients (β), leave-one-society-out validation, common-mode vs relational variance decomposition, and persistence/change scatter — then generates a 4-panel dashboard PNG.

## Steps

1. **Fix the data path** in `node/cams_node_dynamics_code.py`:
   - Change line: `DATA = Path('/mnt/data/JUNO_Unified_Dataset.csv')`
   - To: `DATA = Path('juno/JUNO_Unified_Dataset.csv')`
   - Run from the repository root (`C:\Users\julie\wintermute`), not from inside `node/`.

2. **Create output directory** if it doesn't exist:
   ```
   node/
   ```

3. **Update output paths** in the script so all CSVs and PNGs write to `node/`:
   - `cams_node_dynamics_metrics.csv` → `node/cams_node_dynamics_metrics.csv`
   - `cams_node_dynamics_LOOS.csv` → `node/cams_node_dynamics_LOOS.csv`
   - Dashboard PNG → `node/cams_node_dynamics_dashboard.png`
   - Individual panel PNGs → `node/` as well

4. **Install dependencies** if needed:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

5. **Run the script** from the repo root:
   ```
   python node/cams_node_dynamics_code.py
   ```

## Expected outputs

- `node/cams_node_dynamics_dashboard.png` — 4-panel figure (β ordering, LOOS scatter, variance decomposition, persistence/change)
- `node/cams_node_dynamics_metrics.csv` — per-node β, persistence, change metrics
- `node/cams_node_dynamics_LOOS.csv` — leave-one-society-out validation results

## Known results (from prior run on 36-society JUNO dataset)

| Node | β |
|---|---|
| Flow | 1.171 |
| Helm | 1.129 |
| Stewards | 1.032 |
| Craft | 0.938 |
| Hands | 0.932 |
| Archive | 0.847 |
| Shield | 0.777 |
| Lore | 0.706 |

Common mode: 69.1% · Relational: 30.9% · LOOS pair accuracy: 74.9% · ρ = 0.643

If the new run produces different numbers, the dataset or script may have changed — check the JUNO_Unified_Dataset.csv row count (expected: 32,361 records across 36 societies).

## Source script

The script is at `node/cams_node_dynamics_code.py` with paths already patched. Run from repo root and it will produce all outputs. The dashboard PNG and both CSVs will appear in `node/`.
