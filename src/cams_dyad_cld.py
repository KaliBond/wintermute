import pandas as pd
import numpy as np
import json

# --- Minimal dyad field extraction (aligned to your revised script) ---
METABOLIC_NODES = ["Hands", "Flow", "Shield"]
MYTH_NODES = ["Lore", "Archive", "Stewards"]

ALPHA = 0.5   # capacity shortfall weight in M
BETA  = 0.5   # coherence weight in Y
GAMMA = 0.3   # stress drag in Y
DELTA = 1.0   # mismatch weight
EPS   = 0.05  # avoid divide-by-zero
ROLL  = 5

def pivot_cams_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert CAMS long format to wide format with error handling."""
    if df_long is None or df_long.empty:
        raise ValueError("Empty dataframe provided")

    df = df_long.copy()
    df.columns = df.columns.str.strip()

    # Validate required columns
    required = ["Year", "Node"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Handle different column name variations
    col_mapping = {
        "Society": "Society",
        "Year": "Year",
        "Node": "Node",
        "Coherence": "Coherence",
        "Capacity": "Capacity",
        "Stress": "Stress",
        "Abstraction": "Abstraction",
        "Bond Strength": "Bond Strength",
        "BondStrength": "Bond Strength",
        "Node Value": "Node Value",
        "NodeValue": "Node Value"
    }

    # Rename columns to standard names
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and old_name != new_name:
            df = df.rename(columns={old_name: new_name})

    # Convert Year to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Node"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Convert metrics to numeric
    for col in ["Coherence","Capacity","Stress","Abstraction","Bond Strength","Node Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build wide format
    out = pd.DataFrame(index=sorted(df["Year"].unique()))

    for node in df["Node"].unique():
        node_data = df[df["Node"] == node].groupby("Year").mean(numeric_only=True)
        for var in ["Coherence","Capacity","Stress","Abstraction"]:
            if var in node_data.columns:
                out[f"{var}_{node}"] = node_data[var]
        if "Bond Strength" in node_data.columns:
            out[f"Bond_{node}"] = node_data["Bond Strength"]

    # Compute average bond strength
    bond_cols = [c for c in out.columns if c.startswith("Bond_")]
    out["B"] = out[bond_cols].mean(axis=1) if bond_cols else np.nan

    return out

def compute_M(wide: pd.DataFrame) -> pd.Series:
    """
    Compute Metabolic Load (M) from stress and capacity shortfall.
    Auto-detects all available nodes instead of hardcoding.
    """
    # Auto-detect nodes from Stress columns
    detected_nodes = []
    for col in wide.columns:
        if col.startswith('Stress_'):
            node = col.replace('Stress_', '')
            detected_nodes.append(node)

    parts = []
    for node in detected_nodes:
        s = wide.get(f"Stress_{node}")
        k = wide.get(f"Capacity_{node}")
        if s is None or k is None:
            continue
        k_base = k.rolling(ROLL, min_periods=1).median()
        dk = (k_base - k).clip(lower=0)
        parts.append(s + ALPHA * dk)
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(index=wide.index, dtype=float)

def compute_Y(wide: pd.DataFrame) -> pd.Series:
    """
    Compute Mythic Integration (Y) from coherence, abstraction, and stress.
    Auto-detects all available nodes instead of hardcoding.
    """
    # Auto-detect nodes from Coherence columns
    detected_nodes = []
    for col in wide.columns:
        if col.startswith('Coherence_'):
            node = col.replace('Coherence_', '')
            detected_nodes.append(node)

    parts = []
    for node in detected_nodes:
        c = wide.get(f"Coherence_{node}")
        a = wide.get(f"Abstraction_{node}")
        s = wide.get(f"Stress_{node}")
        if c is None or a is None or s is None:
            continue
        parts.append(BETA*c + (1-BETA)*a - GAMMA*s)
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(index=wide.index, dtype=float)

def normalise_B(B: pd.Series) -> pd.Series:
    if B.isna().all():
        return pd.Series(index=B.index, data=np.nan)
    mn, mx = B.min(), B.max()
    return (B - mn) / (mx - mn) if mx > mn else pd.Series(index=B.index, data=0.5)

def compute_Omega(wide: pd.DataFrame, nodes: list = None) -> pd.Series:
    """Compute stress volatility (Omega) across nodes."""
    if nodes is None:
        # Auto-detect nodes from column names
        nodes = []
        for col in wide.columns:
            if col.startswith('Stress_'):
                nodes.append(col.replace('Stress_', ''))

    diffs = []
    for node in nodes:
        s = wide.get(f"Stress_{node}")
        if s is not None:
            diffs.append(s.diff())
    return pd.concat(diffs, axis=1).std(axis=1) if diffs else pd.Series(index=wide.index, dtype=float)

def build_cld_json(topic: str) -> dict:
    # You'll likely refine node texts/loops later; this is a starter scaffold.
    return {
        "metadata": {"topic": topic},
        "nodes": [],
        "edges": [],
        "loops": [],
        "archetypes": []
    }

def main(cams_csv_path: str, out_json_path: str):
    df_long = pd.read_csv(cams_csv_path)
    wide = pivot_cams_long_to_wide(df_long)

    M = compute_M(wide)
    Y = compute_Y(wide)
    D = M - DELTA*Y
    Bn = normalise_B(wide["B"])
    R = D / (EPS + Bn)
    Om = compute_Omega(wide)

    # A tiny "teaching" summary you can show students/readers:
    summary = {
        "years": [int(wide.index.min()), int(wide.index.max())],
        "mean_M": float(M.mean()),
        "mean_Y": float(Y.mean()),
        "mean_D": float(D.mean()),
        "mean_Bn": float(Bn.mean()),
        "max_R": float(R.max()),
        "mean_Omega": float(Om.mean()),
    }

    model = build_cld_json("CAMS Dyad: M–Y mismatch, coupling, risk")
    model["metadata"]["computed_summary"] = summary

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Example:
    # main("data/cleaned/china_dec_gem_1970_2025.csv", "cld_scaffold.json")
    pass
