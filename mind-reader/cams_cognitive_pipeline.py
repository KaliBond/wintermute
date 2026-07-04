#!/usr/bin/env python3
"""
CAMS Cognitive Pipeline + Mind Reader
Lightweight batch processor for activation vectors (σ) and bond/cognitive synchronisability
across the full Neural Nations 43-society dataset collection.

Examines societies as complex adaptive systems: computes per-node activation
σ_i = (A_i · C_i) · (K_i - S_i), derives system-level synchronisability proxies,
classifies attractor states, and generates interpretable "mind reader" portraits
of how each society is managing (or mismanaging) its distributed 8-node cognition.

Designed to run on a local clone of https://github.com/KaliBond/wintermute
(data/cleaned/ or data/nations/ folders containing the standardized CSVs).

Usage:
    python cams_cognitive_pipeline.py --demo                 # run on synthetic data, print mind-read
    python cams_cognitive_pipeline.py --input data/cleaned --output reports/cams_cog
    python cams_cognitive_pipeline.py --society Australia --recent 20   # focused mind-read

Dependencies (lightweight):
    pandas, numpy  (pip install pandas numpy)

Author: Grok xAI + Kari McKern CAMS collaboration, July 2026
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CANONICAL 8-NODE MODEL (CAMS v3.2-R + JUNO)
# =============================================================================
CANONICAL_NODES: List[str] = [
    "Helm", "Shield", "Flow", "Hands", "Craft", "Archive", "Lore", "Stewards"
]
NODE_ORDER: List[str] = CANONICAL_NODES[:]  # fixed order for consistent 8x8 matrices

# Flexible alias mapping for real datasets (extend as needed)
NODE_ALIASES: Dict[str, str] = {
    "helm": "Helm", "leadership": "Helm", "executive": "Helm", "sovereign": "Helm",
    "shield": "Shield", "security": "Shield", "defence": "Shield", "defense": "Shield", "guardian": "Shield",
    "flow": "Flow", "exchange": "Flow", "circulation": "Flow", "trade": "Flow", "messenger": "Flow",
    "hands": "Hands", "labour": "Hands", "labor": "Hands", "worker": "Hands", "farmer": "Hands", "body": "Hands",
    "craft": "Craft", "skills": "Craft", "artisan": "Craft", "engineer": "Craft", "smith": "Craft",
    "archive": "Archive", "memory": "Archive", "knowledge": "Archive", "scribe": "Archive", "library": "Archive",
    "lore": "Lore", "meaning": "Lore", "priest": "Lore", "bard": "Lore", "myth": "Lore", "normative": "Lore",
    "stewards": "Stewards", "stewardship": "Stewards", "resources": "Stewards", "treasurer": "Stewards",
    "gardener": "Stewards", "husbandry": "Stewards",
}

def normalize_node(node: str) -> Optional[str]:
    """Map raw node label to canonical CAMS node."""
    if pd.isna(node):
        return None
    key = str(node).strip().lower()
    return NODE_ALIASES.get(key, node if node in CANONICAL_NODES else None)

# =============================================================================
# ACTIVATION INDEX (core cognitive throughput metric)
# =============================================================================
def compute_activation(row: pd.Series) -> float:
    """
    σ_i = (A_i · C_i) · (K_i - S_i)
    Effective cognitive activation of the node: coherent abstraction × net capacity headroom.
    Positive and high → node is strongly contributing to distributed societal cognition.
    Low/negative → dysregulated, offline, or actively damping collective intelligence.
    """
    try:
        a = float(row.get("Abstraction", row.get("A", 0)))
        c = float(row.get("Coherence", row.get("C", 0)))
        k = float(row.get("Capacity", row.get("K", 0)))
        s = float(row.get("Stress", row.get("S", 0)))
        return (a * c) * (k - s)
    except (ValueError, TypeError):
        return np.nan

# =============================================================================
# SYNCHRONISABILITY & SYSTEM METRICS (lightweight proxies)
# =============================================================================
def compute_juno_bond_matrix(node_metrics: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Reconstruct full 8×8 pairwise bond matrix using JUNO-1.0 formula.
    B_ij = sqrt(q_i * q_j) * 2^(-(S_i + S_j)/10)
    where q_i = (0.6 * C_i + 0.4 * A_i) / 10
    Higher B_ij = stronger effective coupling between nodes i and j under current stress.
    """
    n = len(NODE_ORDER)
    B = np.zeros((n, n))
    q = {}
    for idx, node in enumerate(NODE_ORDER):
        m = node_metrics.get(node, {"C": 5.0, "A": 5.0, "S": 5.0})
        c = float(m.get("C", 5.0))
        a = float(m.get("A", 5.0))
        s = float(m.get("S", 5.0))
        q[node] = (0.6 * c + 0.4 * a) / 10.0
        # store S for later
        node_metrics[node] = node_metrics.get(node, {})
        node_metrics[node]["S"] = s   # ensure present

    for i, ni in enumerate(NODE_ORDER):
        for j, nj in enumerate(NODE_ORDER):
            if i == j:
                continue
            si = node_metrics[ni].get("S", 5.0)
            sj = node_metrics[nj].get("S", 5.0)
            B[i, j] = np.sqrt(q[ni] * q[nj]) * (2 ** (-(si + sj) / 10.0))
    return B


def compute_spectral_gap(B: np.ndarray) -> float:
    """
    Algebraic connectivity (lambda2) of the weighted graph Laplacian.
    L = D - B (unnormalized). lambda2 is the second-smallest eigenvalue.
    Higher lambda2 = stronger global synchronisability / harder to partition the 8-node cognitive graph.
    This is the rigorous CAS measure of network headroom and phase-transition resilience.
    """
    if B.shape[0] != 8:
        return np.nan
    D = np.diag(np.sum(B, axis=1))
    L = D - B
    try:
        eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))  # symmetric, real eigenvalues
        # lambda2 is the smallest strictly positive eigenvalue (skip near-zero)
        for ev in eigvals:
            if ev > 1e-6:
                return float(ev)
        return 0.0
    except Exception:
        return np.nan


def compute_synchronisability(sigmas: np.ndarray, bs_values: Optional[np.ndarray] = None,
                              node_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """
    Lightweight proxies + full JUNO bond-matrix spectral diagnostics.
    When node_metrics (C, A, S per node) is provided, computes the complete 8×8 JUNO B matrix
    and its Laplacian algebraic connectivity (lambda2) for rigorous synchronisability.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    valid = ~np.isnan(sigmas)
    if valid.sum() < 3:
        return {"mean_sigma": np.nan, "sigma_balance": np.nan, "active_nodes": 0, "mean_BS": np.nan,
                "juno_lambda2": np.nan, "mean_bond": np.nan}

    mean_sig = np.nanmean(sigmas)
    std_sig = np.nanstd(sigmas)
    cv = std_sig / (abs(mean_sig) + 1e-6)
    sigma_balance = max(0.0, 1.0 - min(cv, 2.0) / 2.0)

    active = int(np.sum(sigmas[valid] > 3.0))

    mean_bs = float(np.nanmean(bs_values)) if bs_values is not None and len(bs_values) > 0 else np.nan

    # === NEW: Full JUNO spectral gap when node_metrics available ===
    juno_lambda2 = np.nan
    mean_bond = np.nan
    if node_metrics is not None:
        try:
            B = compute_juno_bond_matrix(node_metrics)
            juno_lambda2 = round(compute_spectral_gap(B), 4)
            mean_bond = round(float(np.mean(B[B > 0])), 4) if np.any(B > 0) else np.nan
        except Exception:
            juno_lambda2 = np.nan
            mean_bond = np.nan

    return {
        "mean_sigma": round(mean_sig, 3),
        "sigma_std": round(std_sig, 3),
        "sigma_balance": round(sigma_balance, 3),
        "active_nodes": active,
        "mean_BS": round(mean_bs, 3) if not np.isnan(mean_bs) else np.nan,
        "juno_lambda2": juno_lambda2,
        "mean_bond": mean_bond,
    }

# =============================================================================
# ATTRACTOR STATE CLASSIFIER (heuristic, extensible)
# =============================================================================
def classify_attractor_state(
    sigma_dict: Dict[str, float],
    mean_bs: float = np.nan,
    recent_trend: Optional[float] = None
) -> str:
    """
    Classify the 8-node organism's dominant attractor state from activation pattern + bond proxy.
    Grounded in CAMS v3.2-R attractor taxonomy.
    Extend rules with more longitudinal data or full bond-matrix eigenvalues.
    """
    vals = np.array(list(sigma_dict.values()))
    mean_sig = np.nanmean(vals)
    helm = sigma_dict.get("Helm", np.nan)
    shield = sigma_dict.get("Shield", np.nan)
    hands = sigma_dict.get("Hands", np.nan)
    archive = sigma_dict.get("Archive", np.nan)
    lore = sigma_dict.get("Lore", np.nan)
    flow = sigma_dict.get("Flow", np.nan)

    # Fracture signatures
    if (not np.isnan(helm) and helm < 2.5) and (not np.isnan(shield) and shield > 5.5) and mean_sig < 3.5:
        return "Fracture — Executive Decoupling (Helm offline, Shield hypertrophy)"
    if (not np.isnan(helm) and helm < 3.0) and mean_sig < 2.5:
        return "Fracture — Strategic Collapse"

    # Thermodynamic Freeze / Library suppression
    library_sig = (archive if not np.isnan(archive) else 0) + (lore if not np.isnan(lore) else 0)
    if library_sig < 5.0 and (not np.isnan(hands) and hands > 6.0):
        return "Thermodynamic Freeze — Library Attractor suppressed by metabolic stress (Hands)"
    if mean_sig < 2.0 and (np.isnan(mean_bs) or mean_bs < 0.35):
        return "Thermodynamic Freeze — Chronic low-value survival (external entropy injection required)"

    # Strong re-synchronisation / buffering
    active_nodes = sum(v > 4 for v in vals if not np.isnan(v))
    if mean_sig > 5.5 and (np.isnan(mean_bs) or mean_bs > 0.55) and active_nodes >= 5:
        if recent_trend is not None and recent_trend > 0.05:
            return "Re-synchronisation (recovering graph, positive momentum)"
        return "Buffering — Shock absorbed with strong node coupling"

    # Oscillation / limit-cycle
    if 2.5 < mean_sig < 5.0 and (np.isnan(mean_bs) or 0.35 < mean_bs < 0.55):
        return "Oscillation — Repeating limit-cycle coordination without stable settlement"

    return "Buffering / Transitional (monitor trend and bond matrix for phase shift)"

# =============================================================================
# MIND READER — interpretive portrait of societal cognitive management
# =============================================================================
def mind_read_cognition(
    society: str,
    recent_window: pd.DataFrame,
    attractor: str,
    sync_metrics: Dict[str, float]
) -> Tuple[str, Dict]:
    """
    Generate a structured, CAS-framed narrative of how the society is currently
    managing (or failing to manage) its distributed 8-node cognition.
    Uses activation vector, synchronisability proxies, and attractor classification.
    """
    if recent_window.empty:
        return "Insufficient recent data for cognitive portrait.", {}

    # Aggregate recent statistics (assume recent_window has 'sigma_{Node}' columns or long-form)
    node_cols = [f"sigma_{n}" for n in CANONICAL_NODES]
    available = [c for c in node_cols if c in recent_window.columns]
    if not available:
        # try long-form fallback
        if "Node_norm" in recent_window.columns and "sigma" in recent_window.columns:
            pivot = recent_window.pivot_table(index="Year", columns="Node_norm", values="sigma", aggfunc="mean")
            recent_stats = pivot.tail(10).mean()
        else:
            return "Data format not recognised for mind-read.", {}
    else:
        recent_stats = recent_window[available].tail(10).mean()

    # Key node highlights
    helm_sig = recent_stats.get("sigma_Helm", np.nan)
    shield_sig = recent_stats.get("sigma_Shield", np.nan)
    hands_sig = recent_stats.get("sigma_Hands", np.nan)
    flow_sig = recent_stats.get("sigma_Flow", np.nan)
    archive_sig = recent_stats.get("sigma_Archive", np.nan)
    lore_sig = recent_stats.get("sigma_Lore", np.nan)
    craft_sig = recent_stats.get("sigma_Craft", np.nan)

    library_sig = (archive_sig if not np.isnan(archive_sig) else 0) + (lore_sig if not np.isnan(lore_sig) else 0)
    fast_loop_avg = np.nanmean([helm_sig, shield_sig, flow_sig, hands_sig, craft_sig])

    # Trend proxy (simple)
    if len(recent_window) >= 5:
        years = recent_window["Year"].tail(10).values
        mean_sig_series = recent_window[[c for c in available if "sigma_" in c]].mean(axis=1).tail(10).values
        slope = np.polyfit(years - years[0], mean_sig_series, 1)[0] if len(years) > 1 else 0.0
    else:
        slope = 0.0

    # Build narrative
    lines = []
    lines.append(f"**{society} — Distributed Cognitive Portrait** (recent window, attractor: {attractor})")
    lines.append("")

    # Overall posture
    if sync_metrics.get("mean_sigma", 0) > 5.0 and sync_metrics.get("sigma_balance", 0) > 0.65:
        posture = "robust distributed cognition with good node coupling"
    elif sync_metrics.get("sigma_balance", 0) < 0.4:
        posture = "fragmented cognitive activation — high variance across nodes"
    else:
        posture = "moderate but uneven cognitive throughput"
    lines.append(f"**Overall posture**: {posture}. Mean activation σ = {sync_metrics.get('mean_sigma', np.nan):.2f}, "
                 f"balance = {sync_metrics.get('sigma_balance', np.nan):.2f}, active nodes ≈ {sync_metrics.get('active_nodes', 0)}/8.")

    # Helm / strategic cognition
    if not np.isnan(helm_sig):
        if helm_sig < 3.0:
            lines.append(f"**Helm** (strategic direction) is cognitively offline (σ ≈ {helm_sig:.1f}). Executive abstraction has collapsed; the society is steering reactively or by inertia.")
        elif helm_sig > 6.5 and shield_sig > 5.5 and fast_loop_avg < 4.0:
            lines.append(f"**Helm + Shield** over-activated (σ_Helm ≈ {helm_sig:.1f}, σ_Shield ≈ {shield_sig:.1f}) while metabolic nodes lag. Classic executive–praetorian decoupling pattern.")
        else:
            lines.append(f"**Helm** σ ≈ {helm_sig:.1f} — strategic node is contributing but requires monitoring for coherence with material base.")

    # Library Attractor (slow-loop memory + meaning)
    if library_sig < 5.5:
        lines.append(f"**Library Attractor (Archive + Lore)** is weak (combined σ ≈ {library_sig:.1f}). Long-horizon memory and normative ordering are eroding — fast-loop nodes will increasingly lack stable reference frames.")
    elif library_sig > 7.0 and hands_sig < 3.5:
        lines.append(f"**Library Attractor strong** but Hands/Flow depleted. Risk of abstract elite cognition decoupled from embodied reality.")

    # Metabolic / embodiment nodes
    if not np.isnan(hands_sig) and hands_sig < 2.5:
        lines.append(f"**Hands** (metabolic throughput) severely suppressed (σ ≈ {hands_sig:.1f}). High stress is throttling the society's ability to embody its own plans and abstractions.")
    if not np.isnan(flow_sig) and flow_sig < 3.0:
        lines.append(f"**Flow** (circulation & translation) is constricted. Information, goods, and people are not moving efficiently — a classic precursor to broader desynchronisation.")

    # Trajectory & management style
    if slope > 0.08:
        traj = "positive momentum — graph is re-synchronising"
    elif slope < -0.08:
        traj = "negative trajectory — accelerating desynchronisation"
    else:
        traj = "stable but fragile"
    lines.append(f"**Trajectory**: {traj} (Δmean_σ ≈ {slope:+.3f} per year).")

    lines.append(f"**Observed management style**: The society is currently managing its distributed cognition through "
                 f"{'compensatory Shield inflation at the expense of Flow/Hands' if (not np.isnan(shield_sig) and shield_sig > 5.5 and flow_sig < 3.5) else 'uneven node activation with Library neglect' if library_sig < 5.5 else 'broad but shallow engagement across nodes'}. "
                 f"This pattern increases long-term entropy accumulation and reduces adaptive headroom.")

    lines.append("")
    lines.append("*Interpretation grounded in CAMS v3.2-R: crisis = cognitive desynchronisation across the 8-node graph.*")

    narrative = "\n".join(lines)

    metrics = {
        "mean_sigma": sync_metrics.get("mean_sigma"),
        "sigma_balance": sync_metrics.get("sigma_balance"),
        "active_nodes": sync_metrics.get("active_nodes"),
        "helm_sigma": round(helm_sig, 2) if not np.isnan(helm_sig) else None,
        "library_sigma": round(library_sig, 2),
        "hands_sigma": round(hands_sig, 2) if not np.isnan(hands_sig) else None,
        "mean_sigma_slope": round(slope, 4),
        "attractor_state": attractor,
    }
    return narrative, metrics

# =============================================================================
# BATCH PIPELINE
# =============================================================================
def run_full_pipeline(
    input_dir: str = "data/cleaned",
    output_dir: str = "reports/cams_cognitive",
    min_years: int = 5,
    recent_window_years: int = 15,
) -> None:
    """
    Batch process all CSVs in input_dir.
    Produces:
      - activations_long.csv (every node-year + σ)
      - activations_wide.csv (per society-year with full σ vector + sync metrics)
      - society_reports/*.md (mind-reader portraits + stats for each society)
      - summary.json (cross-society overview)
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    reports_dir = out_path / "society_reports"
    reports_dir.mkdir(exist_ok=True)

    all_long_rows: List[Dict] = []
    all_wide_rows: List[Dict] = []
    society_data: Dict[str, pd.DataFrame] = {}  # for later mind-read

    csv_files = list(in_path.glob("*.csv")) + list(in_path.glob("*.CSV"))
    if not csv_files:
        print(f"No CSVs found in {input_dir}. Run with --demo for synthetic example.")
        return

    print(f"Processing {len(csv_files)} dataset files from {input_dir}...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  Skipping {csv_file.name}: {e}")
            continue

        # Column normalisation (flexible for real data)
        col_rename = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ["coherence (c)", "coherence_c", "c"]: col_rename[col] = "Coherence"
            elif cl in ["capacity (k)", "capacity_k", "k"]: col_rename[col] = "Capacity"
            elif cl in ["stress (s)", "stress_s", "s"]: col_rename[col] = "Stress"
            elif cl in ["abstraction (a)", "abstraction_a", "a"]: col_rename[col] = "Abstraction"
            elif cl in ["bond strength", "bond_strength", "bs", "bondstrength"]: col_rename[col] = "Bond_Strength"
            elif cl in ["node", "institutional node", "cams node"]: col_rename[col] = "Node"
            elif cl in ["year", "date"]: col_rename[col] = "Year"
            elif cl in ["society", "country", "polity", "entity"]: col_rename[col] = "Society"
        if col_rename:
            df = df.rename(columns=col_rename)

        if "Node" not in df.columns or "Year" not in df.columns:
            continue

        df["Node_norm"] = df["Node"].apply(normalize_node)
        df = df[df["Node_norm"].isin(CANONICAL_NODES)].copy()
        if df.empty:
            continue

        df["sigma"] = df.apply(compute_activation, axis=1)

        # Ensure Society column
        if "Society" not in df.columns:
            df["Society"] = csv_file.stem.split("_")[0].replace("CAMS", "").replace("_Cleaned", "").strip()

        # Group by Society + Year
        grouped = df.groupby(["Society", "Year"], dropna=False)

        for (society, year), g in grouped:
            if len(g) < 4:  # need reasonable node coverage
                continue

            sig_dict: Dict[str, float] = {}
            for _, row in g.iterrows():
                node = row["Node_norm"]
                if node:
                    sig_dict[node] = row["sigma"]

            # Fill missing canonical nodes with NaN
            for n in CANONICAL_NODES:
                if n not in sig_dict:
                    sig_dict[n] = np.nan

            bs_vals = g["Bond_Strength"].values if "Bond_Strength" in g.columns else None
            sync = compute_synchronisability(np.array(list(sig_dict.values())), bs_vals)

            # Simple trend placeholder (will be refined per-society later)
            attractor = classify_attractor_state(sig_dict, sync.get("mean_BS", np.nan))

            long_row = {
                "Society": society,
                "Year": int(year) if not pd.isna(year) else None,
                **{f"sigma_{n}": round(v, 3) if not np.isnan(v) else None for n, v in sig_dict.items()},
                "mean_sigma": sync["mean_sigma"],
                "sigma_std": sync["sigma_std"],
                "sigma_balance": sync["sigma_balance"],
                "active_nodes": sync["active_nodes"],
                "mean_BS": sync["mean_BS"],
                "attractor_state": attractor,
                "source_file": csv_file.name,
            }
            all_long_rows.append(long_row)

            wide_row = long_row.copy()
            all_wide_rows.append(wide_row)

            # Accumulate for per-society mind-read
            if society not in society_data:
                society_data[society] = []
            society_data[society].append(long_row)

    if not all_long_rows:
        print("No valid node-year records processed.")
        return

    # Save long format
    df_long = pd.DataFrame(all_long_rows)
    df_long = df_long.sort_values(["Society", "Year"])
    df_long.to_csv(out_path / "activations_long.csv", index=False)
    print(f"  Saved activations_long.csv ({len(df_long)} rows)")

    # Wide format (already per society-year)
    df_wide = pd.DataFrame(all_wide_rows)
    df_wide = df_wide.sort_values(["Society", "Year"])
    df_wide.to_csv(out_path / "activations_wide.csv", index=False)
    print(f"  Saved activations_wide.csv ({len(df_wide)} rows)")

    # Per-society reports + mind reader
    summary = {"generated": datetime.now().isoformat(), "societies": {}, "global_stats": {}}

    for society, rows in society_data.items():
        if len(rows) < min_years:
            continue
        df_soc = pd.DataFrame(rows).sort_values("Year")
        recent = df_soc.tail(recent_window_years)

        # Recompute sync on recent window
        sig_cols = [f"sigma_{n}" for n in CANONICAL_NODES]
        recent_sigmas = recent[sig_cols].values.flatten()
        recent_sync = compute_synchronisability(recent_sigmas, recent["mean_BS"].values if "mean_BS" in recent.columns else None)

        # Latest attractor
        latest_attractor = df_soc["attractor_state"].iloc[-1] if not df_soc.empty else "Unknown"

        narrative, metrics = mind_read_cognition(society, recent, latest_attractor, recent_sync)

        # Save report
        report_path = reports_dir / f"{society.replace(' ', '_')}_cognitive_portrait.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(narrative)
            f.write("\n\n## Quantitative Snapshot (recent window)\n")
            f.write(json.dumps(metrics, indent=2))
            f.write("\n\n*Generated by CAMS Cognitive Pipeline — examine societies as complex adaptive systems.*")

        summary["societies"][society] = {
            "records": len(df_soc),
            "latest_attractor": latest_attractor,
            "recent_mean_sigma": metrics.get("mean_sigma"),
            "recent_sigma_balance": metrics.get("sigma_balance"),
            "report": str(report_path),
        }

    # Global stats
    if not df_wide.empty:
        summary["global_stats"] = {
            "total_societies_processed": df_wide["Society"].nunique(),
            "total_node_year_records": len(df_long),
            "mean_sigma_all": round(df_wide["mean_sigma"].mean(), 3),
            "societies_with_high_balance": int((df_wide.groupby("Society")["sigma_balance"].mean() > 0.6).sum()),
        }

    with open(out_path / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPipeline complete. Outputs in {output_dir}/")
    print(f"  - activations_long.csv & activations_wide.csv")
    print(f"  - society_reports/ ({len(summary['societies'])} portraits generated)")
    print(f"  - pipeline_summary.json")

# =============================================================================
# DEMO / SYNTHETIC DATA (for testing without full repo)
# =============================================================================
def generate_synthetic_demo() -> pd.DataFrame:
    """Create a plausible 25-year synthetic society for pipeline + mind-read demo."""
    np.random.seed(42)
    years = list(range(2000, 2026))
    nodes = CANONICAL_NODES
    rows = []
    base = {"Helm": 5.5, "Shield": 4.8, "Flow": 5.2, "Hands": 4.5, "Craft": 5.8, "Archive": 6.2, "Lore": 5.9, "Stewards": 5.1}
    for y in years:
        drift = (y - 2015) * 0.08  # gradual desynchronisation after 2015
        for node in nodes:
            c = max(1, min(10, base[node] + np.random.normal(0, 0.8) - (drift if node in ["Hands", "Flow"] else 0)))
            k = max(1, min(10, base[node] + np.random.normal(0, 0.6)))
            s = max(1, min(10, 4.5 + np.random.normal(0, 1.2) + (drift * 0.7 if node in ["Hands", "Flow", "Craft"] else 0)))
            a = max(1, min(10, base[node] + np.random.normal(0, 1.0) + (0.3 if node in ["Lore", "Archive", "Helm"] else 0)))
            rows.append({
                "Society": "Synthetic_Demo_Land",
                "Year": y,
                "Node": node,
                "Coherence": round(c, 1),
                "Capacity": round(k, 1),
                "Stress": round(s, 1),
                "Abstraction": round(a, 1),
            })
    return pd.DataFrame(rows)

def run_demo():
    print("=== CAMS Cognitive Pipeline DEMO (Synthetic_Demo_Land 2000–2025) ===\n")
    demo_df = generate_synthetic_demo()
    demo_df["Node_norm"] = demo_df["Node"]
    demo_df["sigma"] = demo_df.apply(compute_activation, axis=1)

    # Simulate wide processing for last 12 years
    recent = demo_df[demo_df["Year"] >= 2014].copy()
    sig_pivot = recent.pivot_table(index="Year", columns="Node_norm", values="sigma", aggfunc="mean")
    sig_pivot = sig_pivot.reindex(columns=CANONICAL_NODES)

    print("Sample recent σ matrix (last 5 years):")
    print(sig_pivot.tail(5).round(2).to_string())
    print()

    # Compute sync on latest year
    latest_year = sig_pivot.index.max()
    latest_sigmas = sig_pivot.loc[latest_year].values
    sync = compute_synchronisability(latest_sigmas)
    print(f"Latest year synchronisability proxies: {sync}")

    # Attractor
    sig_dict = dict(zip(CANONICAL_NODES, latest_sigmas))
    attractor = classify_attractor_state(sig_dict, sync.get("mean_BS", 0.5))
    print(f"Classified attractor state: {attractor}\n")

    # Mind read
    narrative, metrics = mind_read_cognition("Synthetic_Demo_Land", recent, attractor, sync)
    print(narrative)
    print("\nMetrics JSON:")
    print(json.dumps(metrics, indent=2))

    # Also run full pipeline on demo data (saves to artifacts/demo_reports)
    demo_dir = Path("/home/workdir/artifacts/demo_cams_data")
    demo_dir.mkdir(exist_ok=True)
    demo_df.to_csv(demo_dir / "Synthetic_Demo_Land_CAMS.csv", index=False)
    print("\nDemo data saved. Running full pipeline on it...")
    run_full_pipeline(str(demo_dir), "/home/workdir/artifacts/demo_reports", min_years=3, recent_window_years=12)

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAMS Cognitive Pipeline + Mind Reader")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo + full pipeline on it")
    parser.add_argument("--input", default="data/cleaned", help="Input directory of CSVs")
    parser.add_argument("--output", default="reports/cams_cognitive", help="Output directory")
    parser.add_argument("--society", help="Run focused mind-read on one society (requires data)")
    parser.add_argument("--recent", type=int, default=15, help="Recent window years for mind-read")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.society:
        # Placeholder for focused single-society run (extend with actual data load)
        print(f"Focused mind-read for {args.society} not yet wired to real data. Use --demo or full pipeline first.")
    else:
        run_full_pipeline(args.input, args.output, recent_window_years=args.recent)