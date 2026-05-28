# data/v2.3 — CAMS v2.3 Canonical Datasets

Datasets in this folder are computed exclusively from raw scores
(Coherence, Capacity, Stress, Abstraction) using `cams_framework_v2_3.py`.

Node Value and Bond Strength are **always derived**, never hand-crafted.

## Formulas

Node Value:
    V_i = C + K + (A / 2) - S

Bond Strength (pairwise):
    Bij = sqrt(max(Vi + 8, 0) * max(Vj + 8, 0)) / 32

Per-node Bond Strength in CSV outputs is the mean of Bij across
all other nodes in the same society-year.

## Relationship to data/cleaned/

`data/cleaned/` contains datasets from earlier development phases.
Those files used the v2.0 weighted formula from `cams_scoring_engine.py`
and are preserved as a historical record. They are **not** comparable
to v2.3 Bond Strength values.

## How to add a new dataset

1. Prepare a CSV with columns:
   Society, Year, Node, Coherence, Capacity, Stress, Abstraction

2. Run:
   python cams_framework_v2_3.py your_raw_scores.csv data/v2.3/your_output.csv

3. Verify the output with cams_v23_tests.py before committing.
