"""
robson_gauge(v1.0) — Canonical implementation of the Robson Gauge (η_soil)

Reference: Kari McKern, "The Robson Gauge (η_soil): A Path-Dependent Measure 
of Civilizational Anchoring", 2 July 2026, Version 3.0.

Formula (unscaled):
    η_soil = (BS_Lore * BS_Archive * C_Hands) / (S_Hands * σ_V + ε)

where ε = 2.0
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple

NODES = ["Helm", "Shield", "Flow", "Hands", "Craft", "Stewards", "Archive", "Lore"]
EPSILON = 2.0


def _node_value(d: Dict[str, float]) -> float:
    """V_i = C + K − S + 0.5A  (CAMS v1.0-Final)"""
    return (d.get("C", 5.0) + d.get("K", 5.0) - d.get("S", 5.0) + 0.5 * d.get("A", 5.0))


def _quality(d: Dict[str, float]) -> float:
    """q_i = (0.6C + 0.4A) / 10   bounded conceptually [0, 1]"""
    return (0.6 * d.get("C", 5.0) + 0.4 * d.get("A", 5.0)) / 10.0


def _bond_weight(di: Dict[str, float], dj: Dict[str, float]) -> float:
    """W_ij = √(q_i · q_j) · 2^(-(S_i + S_j)/10)"""
    qi = _quality(di)
    qj = _quality(dj)
    si = di.get("S", 5.0)
    sj = dj.get("S", 5.0)
    return np.sqrt(qi * qj) * (2.0 ** (-(si + sj) / 10.0))


def _bond_strength(node: str, cams_data: Dict[str, Dict[str, float]]) -> float:
    """BS_i = (1/7) Σ_{j ≠ i} W_ij"""
    others = [n for n in NODES if n != node]
    weights = [_bond_weight(cams_data[node], cams_data[other]) for other in others]
    return float(np.mean(weights))


def robson_gauge(
    cams_data: Dict[str, Dict[str, float]],
    scale: float = 1.0,
    epsilon: float = EPSILON,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the Robson Gauge (η_soil) and full intermediate vector.

    Parameters
    ----------
    cams_data : dict
        Keys must include the eight nodes. Each value is a dict with at least
        C, K, S, A (floats). Missing keys default to 5.0.
    scale : float, optional
        Multiplicative factor applied *after* the core calculation.
        Use 1.0 for published (unscaled) values.
        Use 1000.0 for human-readable reporting (Hermes convention).
    epsilon : float, optional
        Regularisation floor (paper default = 2.0).

    Returns
    -------
    eta_soil : float
        Scaled η_soil (or unscaled if scale=1.0).
    intermediates : dict
        Full diagnostic vector.
    """
    # Ensure all eight nodes exist
    data = {n: cams_data.get(n, {}) for n in NODES}

    # Node values and dispersion
    V = {n: _node_value(data[n]) for n in NODES}
    V_array = np.array(list(V.values()))
    V_mean = float(np.mean(V_array))
    sigma_V = float(np.std(V_array, ddof=0))          # population std (paper)

    # Bond strengths for the slow-loop nodes
    BS_Lore = _bond_strength("Lore", data)
    BS_Archive = _bond_strength("Archive", data)

    # Hands material terms
    C_Hands = float(data["Hands"].get("C", 5.0))
    S_Hands = float(data["Hands"].get("S", 5.0))

    # Core unscaled calculation
    numerator = BS_Lore * BS_Archive * C_Hands
    denominator = S_Hands * max(sigma_V, 1e-12) + epsilon
    eta_unscaled = numerator / denominator

    # Optional scaling
    eta_soil = eta_unscaled * scale

    intermediates = {
        # Primary outputs
        "eta_soil": round(eta_soil, 6),
        "eta_unscaled": round(eta_unscaled, 6),
        "scale": scale,

        # Required intermediate vector
        "BS_Lore": round(BS_Lore, 6),
        "BS_Archive": round(BS_Archive, 6),
        "sigma_V": round(sigma_V, 6),
        "C_Hands": C_Hands,
        "S_Hands": S_Hands,
        "V_mean": round(V_mean, 6),

        # Extra transparency
        "numerator": round(numerator, 6),
        "denominator": round(denominator, 6),
        "epsilon": epsilon,
        "V": {n: round(v, 4) for n, v in V.items()},
    }

    return eta_soil, intermediates


# ----------------------------------------------------------------------
# Minimal self-test against published paper values (approximate)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Rough reconstruction of Singapore 2024 (paper: η ≈ 0.184)
    singapore_2024 = {
        "Helm":     {"C": 8.5, "K": 8.0, "S": 3.0, "A": 8.0},
        "Shield":   {"C": 8.0, "K": 7.5, "S": 3.5, "A": 7.5},
        "Flow":     {"C": 8.5, "K": 8.0, "S": 4.0, "A": 8.0},
        "Hands":    {"C": 8.0, "K": 7.5, "S": 6.0, "A": 7.5},
        "Craft":    {"C": 8.0, "K": 8.0, "S": 4.0, "A": 8.0},
        "Stewards": {"C": 8.5, "K": 8.0, "S": 3.5, "A": 8.5},
        "Archive":  {"C": 9.0, "K": 8.5, "S": 3.0, "A": 9.0},
        "Lore":     {"C": 8.5, "K": 8.0, "S": 3.5, "A": 8.5},
    }

    eta, info = robson_gauge(singapore_2024, scale=1.0)
    print("Singapore-like (unscaled):")
    print(f"  η_soil     = {info['eta_unscaled']:.4f}")
    print(f"  BS_Lore    = {info['BS_Lore']:.4f}")
    print(f"  BS_Archive = {info['BS_Archive']:.4f}")
    print(f"  σ_V        = {info['sigma_V']:.4f}")
    print(f"  C_Hands    = {info['C_Hands']}")
    print(f"  S_Hands    = {info['S_Hands']}")
    print(f"  V_mean     = {info['V_mean']:.4f}")

    # Cross-check: real Singapore ENS 2025 from recompute package
    from pathlib import Path
    import pandas as pd

    series = Path("data/v1.0_final_recompute/series/Singapore_ENS_Bij_v10.csv")
    if series.exists():
        df = pd.read_csv(series)
        y = int(df["Year"].max())
        g = df[df["Year"] == y]
        cams = {}
        for _, row in g.iterrows():
            cams[str(row["Node"])] = {
                "C": float(row["Coherence"]),
                "K": float(row["Capacity"]),
                "S": float(row["Stress"]),
                "A": float(row["Abstraction"]),
            }
        eta2, info2 = robson_gauge(cams, scale=1.0)
        eta3, info3 = robson_gauge(cams, scale=1000.0)
        print(f"\nSingapore ENS {y} (real corpus):")
        print(f"  η_unscaled = {info2['eta_unscaled']:.6f}  (paper ≈ 0.184)")
        print(f"  η×1000     = {info3['eta_soil']:.2f}     (Hermes band scale)")
        print(f"  BS_Lore    = {info2['BS_Lore']:.4f}")
        print(f"  BS_Archive = {info2['BS_Archive']:.4f}")
        print(f"  σ_V        = {info2['sigma_V']:.4f}")
        print(f"  Hands C/S  = {info2['C_Hands']}/{info2['S_Hands']}")
