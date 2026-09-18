#!/usr/bin/env python3
"""Compute JUNO v1.2-Final derived metrics for the Sep 2026 separate-run staging data.

Inputs (raw C/K/S/A + legacy Node Value / Bond Strength, NOT canonical):
  - cams/Burma_Block1.csv                          (Burma not yet in JUNO_Unified_Dataset.csv)
  - juno/Italy_CAMS_m5n_1850-2026_block1.csv        (rescore; differs from published Italy rows)
  - juno/Netherlands_CAMS_m5n_1850-2026_block1.csv  (rescore; differs from published Netherlands rows)

Recomputes the v1.2-Final operators from raw Coherence/Capacity/Stress/Abstraction only,
using the exact formulas in juno/verify_bond_alignment.py and juno/JUNO_v1.2-Final_Formalism.md:

    Node_Value_Calc = C + K - S + 0.5*A
    q_i             = (0.6*C + 0.4*A) / 10
    w_i             = sqrt(q_i) * 2^(-S/10)
    B_ij            = clip(w_i * w_j, 0, 1)     for i != j
    Bond_Strength_Calc (per node) = mean of the 7 off-diagonal edges for that node
    SBD_Calc        = mean of the 28 unique pairwise bonds
    Lambda2_Calc    = second-smallest eigenvalue of the graph Laplacian
    V_Mean_Calc     = mean of the 8 Node_Value_Calc in that society-year
    V_Min_Calc      = min of the 8 Node_Value_Calc in that society-year

Does NOT compute Phase_Calc / Regime_Label_Calc / Decay_Index_Calc / ESCH_Calc /
CAMS_v1_Regime / Bond_Strength_Calc_legacy -- those columns are produced by an
older pipeline stage not present in this repo, and are set to the literal string
"not_computed", the same convention already used for incomplete rows in
JUNO_Unified_Dataset.csv.

Output: juno/JUNO_SeparateRun_2026-09_staging.csv -- NOT merged into
JUNO_Unified_Dataset.csv. Whether/how to merge is a methodology call, not made here.

Usage: python juno/compute_separate_run_2026-09.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_CSV = HERE / "JUNO_SeparateRun_2026-09_staging.csv"

SOURCES = {
    "Burma": ROOT / "cams" / "Burma_Block1.csv",
    "Italy": HERE / "Italy_CAMS_m5n_1850-2026_block1.csv",
    "Netherlands": HERE / "Netherlands_CAMS_m5n_1850-2026_block1.csv",
}

N_NODES = 8
NOT_COMPUTED = "not_computed"


def q_i(C: float, A: float) -> float:
    return (0.6 * C + 0.4 * A) / 10.0


def w_i(C: float, A: float, S: float) -> float:
    q = q_i(C, A)
    if q <= 0.0:
        return 0.0
    return math.sqrt(q) * (2.0 ** (-S / 10.0))


def bond_matrix(C, A, S):
    n = len(C)
    w = [w_i(C[i], A[i], S[i]) for i in range(n)]
    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bij = min(1.0, max(0.0, w[i] * w[j]))
            B[i][j] = bij
    return B


def per_node_means(B):
    n = len(B)
    return [sum(B[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]


def pairwise_mean(B):
    n = len(B)
    acc, k = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += B[i][j]
            k += 1
    return acc / k if k else float("nan")


def lambda2(B):
    n = len(B)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = 0.0
        for j in range(n):
            if i == j:
                continue
            L[i][j] = -B[i][j]
            deg += B[i][j]
        L[i][i] = deg
    evals = np.sort(np.linalg.eigvalsh(np.array(L, dtype=float)))
    return float(evals[1])


def process_society(society: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Coherence": "Coherence", "Capacity": "Capacity",
        "Stress": "Stress", "Abstraction": "Abstraction",
    })
    df["Society"] = society
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

    out_rows = []
    n_skipped = 0
    for (soc, yr), g in df.groupby(["Society", "Year"]):
        score_cols = ["Coherence", "Capacity", "Stress", "Abstraction"]
        if len(g) != N_NODES or g[score_cols].isna().any().any():
            n_skipped += 1
            continue  # incomplete society-year (missing node or gated/blank C/K/S/A), skip
        g = g.reset_index(drop=True)
        C = g["Coherence"].astype(float).tolist()
        K = g["Capacity"].astype(float).tolist()
        S = g["Stress"].astype(float).tolist()
        A = g["Abstraction"].astype(float).tolist()

        node_values = [C[i] + K[i] - S[i] + 0.5 * A[i] for i in range(N_NODES)]
        B = bond_matrix(C, A, S)
        bond_means = per_node_means(B)
        sbd = pairwise_mean(B)
        lam2 = lambda2(B)
        v_mean = sum(node_values) / N_NODES
        v_min = min(node_values)

        for i in range(N_NODES):
            out_rows.append({
                "Society": soc,
                "Year": yr,
                "Node": g.loc[i, "Node"],
                "Coherence": C[i],
                "Capacity": K[i],
                "Stress": S[i],
                "Abstraction": A[i],
                "Node_Value_Calc": round(node_values[i], 6),
                "Bond_Strength_Calc": round(bond_means[i], 6),
                "Bond_Strength_Calc_legacy": NOT_COMPUTED,
                "SBD_Calc": round(sbd, 6),
                "Lambda2_Calc": round(lam2, 6),
                "Decay_Index_Calc": NOT_COMPUTED,
                "Phase_Calc": NOT_COMPUTED,
                "Regime_Label_Calc": NOT_COMPUTED,
                "ESCH_Calc": NOT_COMPUTED,
                "V_Mean_Calc": round(v_mean, 6),
                "V_Min_Calc": round(v_min, 6),
                "CAMS_v1_Regime": NOT_COMPUTED,
                "Run": "separate_run_2026-09_staging",
            })
    if n_skipped:
        print(f"  ({society}: skipped {n_skipped} incomplete/gated society-years)")
    return pd.DataFrame(out_rows)


def main() -> int:
    frames = []
    for society, path in SOURCES.items():
        if not path.exists():
            print(f"SKIP {society}: missing {path}")
            continue
        f = process_society(society, path)
        yr_min = int(f["Year"].min()) if len(f) else None
        yr_max = int(f["Year"].max()) if len(f) else None
        print(f"{society}: {f['Year'].nunique()} society-years, {len(f)} node-rows "
              f"({yr_min}-{yr_max})")
        frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    cols = [
        "Society", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction",
        "Node_Value_Calc", "Bond_Strength_Calc", "Bond_Strength_Calc_legacy",
        "SBD_Calc", "Lambda2_Calc", "Decay_Index_Calc", "Phase_Calc",
        "Regime_Label_Calc", "ESCH_Calc", "V_Mean_Calc", "V_Min_Calc",
        "CAMS_v1_Regime", "Run",
    ]
    out = out[cols]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows, {out.groupby(['Society','Year']).ngroups} society-years)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
