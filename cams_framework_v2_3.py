"""
cams_framework_v2_3.py
CAMS v2.3 canonical computation pipeline.

All Node Value and Bond Strength figures in data/v2.3/ are derived
from raw scores (Coherence, Capacity, Stress, Abstraction) using the
functions in this file. No hand-crafted or externally scored BS values
are permitted in v2.3 datasets.

Node Value formula:
    V_i = C_i + K_i + (A_i / 2) - S_i

Bond Strength formula (pairwise, from cams_spectral.py):
    Bij = sqrt(max(Vi + 8, 0) * max(Vj + 8, 0)) / 32

Per-node Bond Strength in CSV outputs is the mean of Bij
across all other nodes in the same society-year.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

NODES = ["Helm", "Shield", "Lore", "Archive", "Stewards", "Craft", "Hands", "Flow"]

RAW_COLS = ["Coherence", "Capacity", "Stress", "Abstraction"]
DERIVED_COLS = ["Node Value", "Bond Strength"]
OUTPUT_COLS = ["Society", "Year", "Node"] + RAW_COLS + DERIVED_COLS


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def compute_node_value(coherence: float, capacity: float,
                       stress: float, abstraction: float) -> float:
    """
    V_i = C + K + (A / 2) - S
    """
    return coherence + capacity + (abstraction / 2.0) - stress


def compute_pairwise_bond(vi: float, vj: float) -> float:
    """
    CAMS v2.3 canonical pairwise bond strength.
    Bij = sqrt(max(Vi + 8, 0) * max(Vj + 8, 0)) / 32

    The +8 shift keeps the geometric mean meaningful for
    node values as low as -8. Values below -8 produce zero bond strength.
    Division by 32 normalises to approximately [0, 1] for typical scores.
    """
    return math.sqrt(max(vi + 8.0, 0.0) * max(vj + 8.0, 0.0)) / 32.0


def compute_node_bond_strength(node_value: float,
                                all_node_values: list[float]) -> float:
    """
    Per-node Bond Strength = mean pairwise bond with all other nodes
    in the same society-year.

    Args:
        node_value:      V_i for the node being scored
        all_node_values: list of V_j for all OTHER nodes (exclude self)

    Returns:
        Mean of Bij across all pairs, or 0.0 if no other nodes present.
    """
    if not all_node_values:
        return 0.0
    bonds = [compute_pairwise_bond(node_value, vj) for vj in all_node_values]
    return float(np.mean(bonds))


# ---------------------------------------------------------------------------
# DataFrame pipeline
# ---------------------------------------------------------------------------

def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [Society, Year, Node,
    Coherence, Capacity, Stress, Abstraction], add
    'Node Value' and 'Bond Strength' columns.

    Existing Node Value / Bond Strength columns are overwritten.
    All other columns are preserved.

    Args:
        df: raw-scores DataFrame (may contain extra columns)

    Returns:
        DataFrame with Node Value and Bond Strength added/replaced.
    """
    df = df.copy()

    # Validate required columns
    missing = [c for c in ["Society", "Year", "Node"] + RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute Node Value row-wise
    df["Node Value"] = df.apply(
        lambda r: compute_node_value(
            r["Coherence"], r["Capacity"], r["Stress"], r["Abstraction"]
        ),
        axis=1,
    )

    # Compute per-node Bond Strength within each society-year group
    def _group_bs(group: pd.DataFrame) -> pd.Series:
        node_values = group["Node Value"].tolist()
        bs_values = []
        for i, nv in enumerate(node_values):
            others = node_values[:i] + node_values[i + 1:]
            bs_values.append(compute_node_bond_strength(nv, others))
        return pd.Series(bs_values, index=group.index)

    df["Bond Strength"] = (
        df.groupby(["Society", "Year"], group_keys=False)
        .apply(_group_bs, include_groups=False)
    )

    return df


def score_csv(input_path: str,
              output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Read a raw-scores CSV, compute Node Value and Bond Strength,
    write to output_path (or overwrite input if output_path is None),
    and return the result DataFrame.

    The input CSV must have columns:
        Society, Year, Node, Coherence, Capacity, Stress, Abstraction

    Node Value and Bond Strength columns are added or replaced.
    Column order in output follows OUTPUT_COLS, with any extra
    columns appended.
    """
    df = pd.read_csv(input_path)
    df = compute_derived_columns(df)

    # Reorder: canonical columns first, extras after
    extra_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    df = df[OUTPUT_COLS + extra_cols]

    save_path = output_path or input_path
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")
    return df


def batch_score(input_dir: str, output_dir: str) -> None:
    """
    Score all CSV files in input_dir and write results to output_dir.
    Creates output_dir if it does not exist.

    Usage:
        batch_score("data/raw_scores", "data/v2.3")
    """
    import os
    import glob

    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for path in sorted(csv_files):
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        try:
            score_csv(path, out_path)
        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    print(f"\nDone. {len(csv_files)} files processed → {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        batch_score(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        score_csv(sys.argv[1])
    else:
        print("Usage:")
        print("  python cams_framework_v2_3.py <input.csv> [output.csv]")
        print("  python cams_framework_v2_3.py <input_dir/> <output_dir/>")
