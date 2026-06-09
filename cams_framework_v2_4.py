"""
cams_framework_v2_4.py
CAMS v2.4 canonical computation pipeline.  (CAMS-CAN v1.0-Final, 7 Jun 2026)

Supersedes cams_framework_v2_3.py.  DO NOT import v2.3 for new work.
See formulation-decisions log: JUNO_FORMALISM.TXT + FORMULATION_KIMI.TXT.

CHANGES v2.3 → v2.4
--------------------
1. BOND STRENGTH — replaces the Node-Value offset form
       B_ij = sqrt(max(V_i+8,0) * max(V_j+8,0)) / 32      # v2.3 / Book Appendix A
   with the JUNO coupling-quality form (bounded [0,1], stress-decayed):
       q_i  = (0.6*C_i + 0.4*A_i) / 10
       B_ij = T_ij * sqrt(q_i*q_j) * 2**(-(S_i+S_j)/10)
   Inputs to the bond functions are raw C, A, S — NOT Node Values.
   v2.3 is not a scalar rescaling of v2.4 (Spearman 0.982, not 1.0);
   a v2.3 corpus cannot be salvaged by post-hoc rescaling.

2. ESCH ACTIVATION (sigma) — singularity handled ONLY at the exact K==S point.
   The v3.2-R blanket clamp  max(K-S, 0.1)  is REMOVED: it forced sigma
   strictly positive and left the sigma_min ≤ -0.85 Local-Node-Failure trigger
   permanently inert.  Corrected form restores range ~[-2.0, +2.0].

3. LAPLACIAN — algebraic connectivity uses the RAW Laplacian L = D - W
   (lambda2 ~0.27–3.42, discriminating), not the normalised form
   (lambda2 ~1.07–1.14, degenerate on this corpus).

Node Value is UNCHANGED from v2.3 (verified zero-error across all dataset
families).  Public API (score_csv, batch_score, CLI) is unchanged.

CALIBRATION STATUS — sigma_min thresholds (-0.85, -0.7, -0.3) validated
2026-06-09 on 2,377 deduplicated society-years (23 societies, 460–2026).
No numerical changes required: -0.85 sits at corpus p10, historical anchors
are coherent.  See classify_regime docstring for full calibration record.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional

VERSION = "2.4"

# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

NODES = ["Helm", "Shield", "Lore", "Archive", "Stewards", "Craft", "Hands", "Flow"]
N = 8

RAW_COLS = ["Coherence", "Capacity", "Stress", "Abstraction"]
DERIVED_COLS = ["Node Value", "Bond Strength"]
OUTPUT_COLS = ["Society", "Year", "Node"] + RAW_COLS + DERIVED_COLS


# ---------------------------------------------------------------------------
# Operator 1: Node Value (unchanged from v2.3, verified anchor)
# ---------------------------------------------------------------------------

def node_value(C, K, S, A):
    """V_i = C + K - S + 0.5*A   (weights: 1, 1, -1, 0.5)

    Accepts scalars or numpy arrays.  Identical to v2.3 compute_node_value.
    """
    C, K, S, A = map(np.asarray, (C, K, S, A))
    return C + K - S + 0.5 * A


# v2.3-compatible alias so any stray direct call fails loudly with a
# type error rather than silently (the signature is identical here).
def compute_node_value(coherence: float, capacity: float,
                       stress: float, abstraction: float) -> float:
    return float(node_value(coherence, capacity, stress, abstraction))


# ---------------------------------------------------------------------------
# Operator 2: Bond Strength (JUNO bounded coupling form)
# ---------------------------------------------------------------------------

def coupling_quality(C, A):
    """q_i = (0.6*C + 0.4*A) / 10  in [0.1, 1.0] for C, A in [1, 10].

    Clipped at 0 so out-of-range inputs cannot produce a sqrt of a negative.
    Capacity (K) is excluded: it is a node property, not a dyadic one.
    """
    C, A = np.asarray(C, dtype=float), np.asarray(A, dtype=float)
    return np.clip((0.6 * C + 0.4 * A) / 10.0, 0.0, None)


def bond_matrix(C, A, S, T=None):
    """8×8 symmetric Bond Strength matrix, zero diagonal.

        B_ij = T_ij * sqrt(q_i * q_j) * 2 ** (-(S_i + S_j) / 10)

    T defaults to the complete-graph prior (off-diagonal 1, diagonal 0).
    Pass a custom T for the sparse structural prior deferred to v1.1.
    All values are in [0, 1].
    """
    C, A, S = (np.asarray(x, dtype=float) for x in (C, A, S))
    q = coupling_quality(C, A)
    n = len(q)
    if T is None:
        T = np.ones((n, n)) - np.eye(n)
    qi = q.reshape(-1, 1)
    qj = q.reshape(1, -1)
    Si = S.reshape(-1, 1)
    Sj = S.reshape(1, -1)
    B = T * np.sqrt(qi * qj) * np.power(2.0, -(Si + Sj) / 10.0)
    np.fill_diagonal(B, 0.0)
    return B


def node_bond_strength(C, A, S, T=None):
    """Per-node Bond Strength = mean of a node's off-diagonal edges.

    Returns an array of length n.  This is the scalar stored per
    node-year in the output CSVs.
    """
    B = bond_matrix(C, A, S, T)
    n = B.shape[0]
    return B.sum(axis=1) / (n - 1)


# ---------------------------------------------------------------------------
# Operator 3: ESCH cognitive activation (singularity-only handling)
# ---------------------------------------------------------------------------

def esch_activation(C, K, S, A, singularity_floor=0.1):
    """sigma_i = (A_i * C_i / 100) * (K_i - S_i)

    Singularity handling: ONLY where K_i == S_i exactly (activation mode
    indeterminate) substitute `singularity_floor` for the (K-S) term.

    This is deliberately NOT max(K-S, floor): the blanket clamp is the
    v3.2-R bug that pins sigma positive and kills the sigma_min ≤ -0.85
    trigger.

    Set singularity_floor=0.0 to read K==S as sigma=0 ('indeterminate').
    The choice never affects sigma_min: a node at K==S is never the minimum
    when another node carries a genuine K<S deficit.
    """
    C, K, S, A = (np.asarray(x, dtype=float) for x in (C, K, S, A))
    delta = K - S
    delta = np.where(delta == 0.0, singularity_floor, delta)
    return (A * C / 100.0) * delta


# ---------------------------------------------------------------------------
# Spectral: algebraic connectivity on the RAW Laplacian
# ---------------------------------------------------------------------------

def algebraic_connectivity(C, A, S, T=None):
    """lambda_2 (second-smallest eigenvalue) of raw Laplacian L = D - W.

    Range on this corpus: ~0.27–3.42 (discriminating).
    The normalised form degenerates to ~1.07–1.14 and is not used.
    """
    W = bond_matrix(C, A, S, T)
    L = np.diag(W.sum(axis=1)) - W
    ev = np.sort(np.linalg.eigvalsh(L))
    return float(ev[1])


# ---------------------------------------------------------------------------
# Phase-space vector and six-regime classifier
# ---------------------------------------------------------------------------

def phase_state(C, K, S, A, T=None, singularity_floor=0.1):
    """Phi_G = (V_mean, V_std, V_min, B_mean, lambda2, sigma_min).

    V_std uses population std (ddof=0) to match the calibrated corpus.
    's_V' in the spec = V_std here; 's_min' in the spec = sigma_min here.
    """
    V = node_value(C, K, S, A)
    B = node_bond_strength(C, A, S, T)
    sigma = esch_activation(C, K, S, A, singularity_floor)
    return {
        "V_mean":    float(np.mean(V)),
        "V_std":     float(np.std(V)),
        "V_min":     float(np.min(V)),
        "B_mean":    float(np.mean(B)),
        "lambda2":   algebraic_connectivity(C, A, S, T),
        "sigma_min": float(np.min(sigma)),
    }


def classify_regime(phi: dict) -> str:
    """Six-regime classifier (CAMS-CAN v1.0-Final).

    Local Node Failure is tested FIRST: a single failing node triggers it
    even when the aggregate V_mean looks healthy (the Germany-2024 /
    USA-2020 repair).

    THRESHOLD CALIBRATION — validated 2026-06-09 on v2.4 corpus
    (2,377 deduplicated society-years, 23 societies, 460 CE – 2026):

      σ_min ≤ -0.85  (LNF gate)
        Corpus p10 = -0.869; fires at 10.4% of society-years.
        Fires correctly: Russia 1917 (-0.95), Poland 1939 (-1.68),
          Germany 1920 (-0.88), Iran 1979 (-1.01), Rome 450 (-0.96).
        Correctly silent: Germany 1933 (-0.19, Nazi 'recovery'),
          UK 1940 (-0.30, Churchillian resilience), USA 2020 (caught
          instead by V_min < 4.0 gate).
        σ-only contribution (V_min ≥ 4.0, sole trigger): 30 cases —
          NZ economic restructuring 1987–1992, Germany 2021–2026,
          Russia/UK 2022.  These are Phantom precursors: high V_mean
          but a cognitively overloaded node hidden by aggregate health.

      σ_min < -0.70  (Phantom Type II sub-condition)
        Fires only within LNF AND 3 ≤ V_mean < 6 AND V_min < 0.
        4 confirmed corpus cases: Spain 1816–1825 post-Napoleonic.

      σ_min > -0.30  (Stable adaptive requirement)
        Stable adaptive = 38.2% of deduplicated corpus.
        Germany(FRG) 90%, SpaceX 86%, UK 65%, USA 68%, Norway 54%.
        Poland 8%, Iran 7%, Latium Vetus 3% — historically coherent.

    Full regime distribution (2,377 society-years):
      Stable adaptive 38.2% | Strained 27.6% | LNF 22.1%
      Systemic crisis 10.3% | Freeze/Collapse 1.6% | Phantom II 0.2%

    Data note: Argentina has 4× duplicate society-years across canonical
    and USP tiers; dedup before regime-count analysis.
    """
    V    = phi["V_mean"]
    Vmin = phi["V_min"]
    B    = phi["B_mean"]
    smin = phi["sigma_min"]

    # 1. Local Node Failure family — independent of V_mean
    if Vmin < 4.0 or smin <= -0.85:
        if V < 0 and Vmin < -3 and B < 0.15:
            return "Freeze/Collapse"
        if V < 6 and Vmin < 0 and B < 0.20:
            return "Systemic crisis"
        if 3.0 <= V < 6.0 and Vmin < 0 and smin < -0.7:
            return "Phantom Type II"
        return "Local Node Failure"

    # 2. Healthy / strained spectrum
    if V > 10 and Vmin > 5 and B > 0.30 and smin > -0.3:
        return "Stable adaptive"
    if V >= 6:
        return "Strained"
    return "Systemic crisis"


# ---------------------------------------------------------------------------
# DataFrame pipeline
# ---------------------------------------------------------------------------

def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame with columns [Society, Year, Node,
    Coherence, Capacity, Stress, Abstraction], add
    'Node Value' and 'Bond Strength' columns using v2.4 operators.

    Existing Node Value / Bond Strength columns are overwritten.
    All other columns are preserved.

    Bond Strength is computed once per society-year group using the full
    C/A/S arrays — not node-by-node — so the 8×8 bond matrix is built
    once per group (vectorised, not looped).
    """
    df = df.copy()

    missing = [c for c in ["Society", "Year", "Node"] + RAW_COLS
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Node Value: vectorised row-wise (unchanged formula)
    df["Node Value"] = node_value(
        df["Coherence"].values,
        df["Capacity"].values,
        df["Stress"].values,
        df["Abstraction"].values,
    )

    # Bond Strength: one vectorised call per society-year group.
    # Extracts C, A, S arrays; calls node_bond_strength once;
    # maps results back by index.
    def _group_bs(group: pd.DataFrame) -> pd.Series:
        C = group["Coherence"].values.astype(float)
        A = group["Abstraction"].values.astype(float)
        S = group["Stress"].values.astype(float)
        bs = node_bond_strength(C, A, S)   # returns array of len(group)
        return pd.Series(bs, index=group.index)

    df["Bond Strength"] = (
        df.groupby(["Society", "Year"], group_keys=False)
          .apply(_group_bs, include_groups=False)
    )

    return df


def score_csv(input_path: str,
              output_path: Optional[str] = None) -> pd.DataFrame:
    """Read a raw-scores CSV, compute Node Value and Bond Strength (v2.4),
    write to output_path (or overwrite input if output_path is None),
    and return the result DataFrame.

    Input CSV must have columns:
        Society, Year, Node, Coherence, Capacity, Stress, Abstraction

    Node Value and Bond Strength columns are added or replaced.
    Column order follows OUTPUT_COLS; extra columns are appended.
    """
    df = pd.read_csv(input_path)
    df = compute_derived_columns(df)

    extra_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    df = df[OUTPUT_COLS + extra_cols]

    save_path = output_path or input_path
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows → {save_path}")
    return df


def batch_score(input_dir: str, output_dir: str) -> None:
    """Score all CSV files in input_dir and write results to output_dir.
    Creates output_dir if it does not exist.

    Usage:
        batch_score("data/v2.3_input/canonical", "data/v2.3/canonical")
    """
    import os
    import glob

    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    errors = []
    for path in csv_files:
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        try:
            score_csv(path, out_path)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"ERROR {fname}: {e}")

    ok = len(csv_files) - len(errors)
    print(f"\nDone. {ok}/{len(csv_files)} files → {output_dir}/")
    if errors:
        print("Failed files:")
        for f, e in errors:
            print(f"  {f}: {e}")


# ---------------------------------------------------------------------------
# CLI  (identical surface to v2.3)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        # Directory mode: python cams_framework_v2_4.py <input_dir/> <output_dir/>
        batch_score(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        # Single file: python cams_framework_v2_4.py <input.csv> [output.csv]
        score_csv(sys.argv[1])
    else:
        print(f"CAMS Framework v{VERSION}")
        print("Usage:")
        print("  python cams_framework_v2_4.py <input.csv> [output.csv]")
        print("  python cams_framework_v2_4.py <input_dir/> <output_dir/>")
