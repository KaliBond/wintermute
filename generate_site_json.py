#!/usr/bin/env python3
"""
generate_site_json.py — JUNO-1.0 site data export
===================================================
Reads JUNO CSV outputs (from juno_backcalc.py) and produces
data/juno_full.json in the exact schema that neuralnations.org tools
consume via the FULL JavaScript object.

Schema produced (matches existing FULL structure):
  {
    "societies": {
      "Australia": {
        "name": "Australia",
        "latest_year": 2026,
        "B0": 0.3241,          ← 90th-pct of historical BS (JUNO ∈ [0,1])
        "trends": {
          "S": -0.02, "K": 0.01, "C": 0.01, "A": 0.00, "BS": 0.001
        },                      ← linear slopes, last TREND_YEARS years
        "nodes": {              ← latest year, per-node
          "Helm": {"C":5.0,"K":5.2,"S":4.0,"A":5.2,"V":8.5,"BS":0.31},
          ...
        },
        "series": [             ← one entry per year
          {"y":1875,"C":5.2,"K":5.2,"S":3.7,"A":5.1,"V":10.4,"BS":0.299},
          ...
        ]
      }
    }
  }

Note: BS (Bond Strength) is now JUNO-1.0 scale ∈ [0,1].
The Λ coordination index in the site pages uses Λ = B̄ / B̄₀, so the
normalisation is identical — just on a different absolute scale.

Usage:
  # Basic: reads JUNO_all_countries.csv, writes data/juno_full.json
  python generate_site_json.py

  # Custom paths:
  python generate_site_json.py \\
      --input  /path/to/JUNO_all_countries.csv \\
      --nm     /path/to/node_metrics_juno.csv \\
      --sm     /path/to/system_metrics_juno.csv \\
      --out    /path/to/wintermute/data/juno_full.json

  # Inline embed mode — prints JS snippet to paste into an HTML file:
  python generate_site_json.py --embed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NODES = ["Helm", "Shield", "Lore", "Archive", "Stewards", "Craft", "Hands", "Flow"]
# ↑ matches v2.4 canonical node order (Archive at position 3, not 6)

TREND_YEARS = 10        # years of history for linear-trend calculation
B0_PERCENTILE = 90      # same as the site's existing B̄₀ convention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_trend(series: pd.Series) -> float:
    """Slope of a linear fit over the last TREND_YEARS values."""
    s = series.dropna().tail(TREND_YEARS)
    if len(s) < 3:
        return 0.0
    x = np.arange(len(s), dtype=float)
    slope = np.polyfit(x, s.values.astype(float), 1)[0]
    return round(float(slope), 4)


def build_society(society: str, df_raw: pd.DataFrame,
                  df_nm: pd.DataFrame, df_sm: pd.DataFrame) -> dict:
    """
    Build the FULL schema dict for one society.

    df_raw : raw scores filtered to this society
    df_nm  : node_metrics_juno filtered to this society
    df_sm  : system_metrics_juno filtered to this society
    """
    # --- series: one entry per year (system means + system BS) ---
    years = sorted(df_raw["Year"].unique())

    series = []
    for yr in years:
        yr_raw = df_raw[df_raw["Year"] == yr]
        yr_sm  = df_sm[df_sm["Year"] == yr]

        if yr_raw.empty:
            continue

        C_mean = round(float(yr_raw["Coherence"].mean()), 4)
        K_mean = round(float(yr_raw["Capacity"].mean()), 4)
        S_mean = round(float(yr_raw["Stress"].mean()), 4)
        A_mean = round(float(yr_raw["Abstraction"].mean()), 4)
        V_mean = round(C_mean + K_mean + 0.5 * A_mean - S_mean, 4)

        if not yr_sm.empty:
            BS = round(float(yr_sm["System_Bond_Density"].iloc[0]), 4)
        else:
            BS = None

        series.append({"y": int(yr), "C": C_mean, "K": K_mean,
                        "S": S_mean, "A": A_mean, "V": V_mean, "BS": BS})

    if not series:
        return None

    # --- B0: 90th-percentile of historical system bond density ---
    bs_values = [s["BS"] for s in series if s["BS"] is not None]
    B0 = round(float(np.percentile(bs_values, B0_PERCENTILE)), 4) if bs_values else None

    # --- trends: linear slopes over last TREND_YEARS years ---
    series_df = pd.DataFrame(series).set_index("y")
    trends = {col: linear_trend(series_df[col]) for col in ["S", "K", "C", "A", "BS"]}

    # --- latest year: per-node breakdown ---
    latest_year = max(years)
    latest_raw = df_raw[df_raw["Year"] == latest_year].set_index("Node")
    latest_nm  = df_nm[df_nm["Year"] == latest_year].set_index("Node")

    nodes = {}
    for node in NODES:
        if node not in latest_raw.index:
            continue
        row = latest_raw.loc[node]
        C = round(float(row["Coherence"]), 4)
        K = round(float(row["Capacity"]), 4)
        S = round(float(row["Stress"]), 4)
        A = round(float(row["Abstraction"]), 4)
        V = round(C + K + 0.5 * A - S, 4)

        BS_node = None
        if node in latest_nm.index:
            bs_val = latest_nm.loc[node, "Bond_Strength"]
            if isinstance(bs_val, pd.Series):
                bs_val = bs_val.iloc[0]
            BS_node = round(float(bs_val), 4)

        nodes[node] = {"C": C, "K": K, "S": S, "A": A, "V": V, "BS": BS_node}

    return {
        "name": society,
        "latest_year": int(latest_year),
        "B0": B0,
        "trends": trends,
        "nodes": nodes,
        "series": series,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(input_path: str, nm_path: str | None,
             sm_path: str | None, out_path: str,
             embed: bool = False) -> None:

    print(f"[load] {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # Accept either the raw JUNO CSV (with pre-computed columns) or a plain raw-scores CSV.
    # Either way we need: Society, Year, Node, Coherence, Capacity, Stress, Abstraction.
    required = {"Society", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"[error] Missing columns in --input: {missing}")

    df["Year"] = df["Year"].astype(int)

    # --- node_metrics_juno (for per-node BS) ---
    if nm_path and Path(nm_path).exists():
        df_nm = pd.read_csv(nm_path)
        df_nm.columns = df_nm.columns.str.strip()
    elif "Bond_Strength_JUNO" in df.columns:
        # Input is a JUNO CSV — reshape to match node_metrics format
        df_nm = df[["Society","Year","Node","Bond_Strength_JUNO"]].rename(
            columns={"Bond_Strength_JUNO": "Bond_Strength"})
    else:
        # Fall back: recompute bond strength from raw scores
        print("[warn] No node_metrics_juno source found — recomputing bond strength.")
        df_nm = _recompute_node_bs(df)

    # --- system_metrics_juno (for system bond density) ---
    if sm_path and Path(sm_path).exists():
        df_sm = pd.read_csv(sm_path)
        df_sm.columns = df_sm.columns.str.strip()
    elif "System_Bond_Density_JUNO" in df.columns:
        df_sm = (df.groupby(["Society","Year"])
                   .agg(System_Bond_Density=("System_Bond_Density_JUNO","first"))
                   .reset_index())
    else:
        print("[warn] No system_metrics_juno source found — deriving from node metrics.")
        df_sm = _derive_system_sbd(df_nm)

    # --- Build per-society dicts ---
    societies = sorted(df["Society"].unique())
    print(f"[build] {len(societies)} societies: {societies}")

    full = {"societies": {}, "_meta": {
        "formalism": "JUNO-1.0",
        "bond_formula": "B_ij = sqrt(q_i*q_j) * 2^(-(S_i+S_j)/10), q_i=(0.6C+0.4A)/10",
        "bond_range": "[0, 1]",
        "node_value": "V = C + K + 0.5*A - S",
        "B0_percentile": B0_PERCENTILE,
        "generated_by": "generate_site_json.py",
    }}

    for soc in societies:
        rec = build_society(
            soc,
            df_raw = df[df["Society"] == soc].copy(),
            df_nm  = df_nm[df_nm["Society"] == soc].copy(),
            df_sm  = df_sm[df_sm["Society"] == soc].copy(),
        )
        if rec:
            full["societies"][soc] = rec
            print(f"  {soc:<20} {len(rec['series'])} years  "
                  f"latest={rec['latest_year']}  B0={rec['B0']}")

    # --- Write ---
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if embed:
        # Print a JS snippet for direct embedding in an HTML file
        json_str = json.dumps(full, separators=(",", ":"))
        print("\n<!-- paste inside a <script> tag in your HTML page -->")
        print(f"const FULL_JUNO = {json_str};")
        print("<!-- end of JUNO data -->")
    else:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, separators=(",", ":"))
        size_kb = out_p.stat().st_size / 1024
        print(f"\n[done] → {out_p}  ({size_kb:.1f} KB)")
        print("        Load in your site pages with:")
        print("          fetch('/data/juno_full.json').then(r=>r.json()).then(d=>{ FULL_JUNO=d.societies; })")
        print()
        print("        Or add the formalism toggle snippet — see JUNO_TOGGLE_SNIPPET.html")


# ---------------------------------------------------------------------------
# Fallback: recompute BS from raw C/A/S if no pre-computed file available
# ---------------------------------------------------------------------------

def _recompute_node_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute per-node JUNO bond strength from raw scores."""
    from juno_backcalc import juno_bond_matrix, per_node_bond_strength
    rows = []
    for (soc, yr), g in df.groupby(["Society", "Year"]):
        g = g.set_index("Node").reindex(NODES).dropna()
        if len(g) < 8:
            continue
        C = g["Coherence"].values.astype(float)
        A = g["Abstraction"].values.astype(float)
        S = g["Stress"].values.astype(float)
        B = juno_bond_matrix(C, A, S)
        bs = per_node_bond_strength(B)
        for i, node in enumerate(NODES):
            rows.append({"Society": soc, "Year": yr, "Node": node, "Bond_Strength": round(bs[i], 4)})
    return pd.DataFrame(rows)


def _derive_system_sbd(df_nm: pd.DataFrame) -> pd.DataFrame:
    """Derive system bond density as mean of per-node BS."""
    return (df_nm.groupby(["Society", "Year"])
                 .agg(System_Bond_Density=("Bond_Strength", "mean"))
                 .reset_index())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Sensible defaults: look for outputs from juno_backcalc.py in cwd
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate data/juno_full.json for neuralnations.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default=str(here / "JUNO_all_countries.csv"),
        metavar="CSV",
        help="JUNO all-countries CSV (default: JUNO_all_countries.csv alongside this script)",
    )
    parser.add_argument(
        "--nm",
        default=str(here / "node_metrics_juno.csv"),
        metavar="CSV",
        help="node_metrics_juno.csv from juno_backcalc.py (optional — derived from --input if absent)",
    )
    parser.add_argument(
        "--sm",
        default=str(here / "system_metrics_juno.csv"),
        metavar="CSV",
        help="system_metrics_juno.csv from juno_backcalc.py (optional)",
    )
    parser.add_argument(
        "--out", "-o",
        default=str(here / "data" / "juno_full.json"),
        metavar="JSON",
        help="Output path (default: data/juno_full.json alongside this script)",
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Print a JS const snippet for direct HTML embedding instead of writing a file",
    )
    args = parser.parse_args()

    generate(
        input_path=args.input,
        nm_path=args.nm,
        sm_path=args.sm,
        out_path=args.out,
        embed=args.embed,
    )


if __name__ == "__main__":
    main()
