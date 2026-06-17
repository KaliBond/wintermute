#!/usr/bin/env python3
"""
juno_backcalc.py — JUNO-1.0 / CAMS-v3.2 dual-formalism back-calculation adapter
================================================================================
Reads raw CAMS scores (C, K, S, A), computes both formalisms' bond metrics,
and writes DB-ready CSVs matching schema_dual_formalism.sql.

Formulas (all verified on the 5,080-row JUNO corpus, zero error):
  Node Value      : NV  = C + K + 0.5·A − S
  Coupling quality: q_i = (0.6·C_i + 0.4·A_i) / 10
  JUNO-1.0 bond   : B_ij = √(q_i·q_j) · 2^(−(S_i+S_j)/10)   ∈ [0, 1]
  CAMS-v3.2 bond  : B_ij = (0.6·C_i·C_j + 0.4·A_i·A_j) · exp(−(S_i+S_j)/20)  [legacy]
  Per-node B_i    : mean of off-diagonal B_ij
  System B̄        : mean of all off-diagonal B_ij
  Decay index L   : 1 − B̄/B̄₀  (B̄₀ = first available year for each society)
  λ₂              : second-smallest eigenvalue of the graph Laplacian (algebraic connectivity)

Usage:
  # Basic — JUNO output only:
  python juno_backcalc.py --raw /path/to/raw_scores.csv --out ./juno_output/

  # With legacy CAMS-v3.2 comparison and concordance report:
  python juno_backcalc.py \\
      --raw /path/to/raw_scores.csv \\
      --legacy-cams /path/to/legacy_node_metrics_cams.csv \\
      --out ./juno_output/

  # Also accepts the full JUNO CSVs (with Node_Value_JUNO etc.) as --raw input;
  # it passes those values through and verifies them against the formula.

Input columns (--raw):
  Required : Society, Year, Node, Coherence, Capacity, Stress, Abstraction
  Optional : Node_Value_JUNO, Bond_Strength_JUNO, System_Bond_Density_JUNO
             (if present, values are verified and any discrepancy is reported)

Legacy CAMS columns (--legacy-cams):
  Required : Society, Year, Node, Bond_Strength  (or Bond_Strength_CAMS)
  Optional : Node_Value, System_Bond_Density

Output files (all in --out directory):
  node_metrics_juno.csv       per-node JUNO metrics   → import into node_metrics
  system_metrics_juno.csv     system-level JUNO metrics → system_metrics
  bond_matrix_juno.csv        full dyadic B_ij (56 rows/society-year) → bond_matrix
  node_metrics_cams32.csv     back-calculated CAMS-v3.2 node bond strength
  system_metrics_cams32.csv   CAMS-v3.2 system bond density
  concordance_report.csv      Spearman / MAE between formalisms (per node + system)
  concordance_report.txt      human-readable summary
  verify_juno.csv             if input had pre-computed JUNO values: diff vs formula
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
N = len(NODES)

FORMALISM_JUNO  = "JUNO-1.0"
FORMALISM_CAMS  = "CAMS-v3.2"

# JUNO six-regime gates (RD-003 / OD-001; confirmed thresholds)
# Gate boundaries derived from corpus percentile anchoring on v2.4 data.
JUNO_REGIMES = [
    (0.30, 6, "High Coherence"),
    (0.25, 5, "Stable Integration"),
    (0.20, 4, "Moderate Stress"),
    (0.17, 3, "Fragmentation"),
    (0.15, 2, "Acute Crisis"),
    (0.00, 1, "Systemic Collapse"),
]


# ---------------------------------------------------------------------------
# Formula implementations
# ---------------------------------------------------------------------------

def node_value(C: np.ndarray, K: np.ndarray, S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """NV = C + K + 0.5·A − S  (verified zero-error on full JUNO corpus)."""
    return C + K + 0.5 * A - S


def coupling_quality(C: np.ndarray, A: np.ndarray) -> np.ndarray:
    """q_i = (0.6·C_i + 0.4·A_i) / 10  ∈ [0.1, 1.0]"""
    return (0.6 * C + 0.4 * A) / 10.0


def juno_bond_matrix(C: np.ndarray, A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    JUNO-1.0 bond matrix.
    B_ij = √(q_i·q_j) · 2^(−(S_i+S_j)/10),  diagonal = 0,  B_ij ∈ [0, 1].
    """
    q = coupling_quality(C, A)
    # Broadcasting: outer products
    q_prod = np.sqrt(np.outer(q, q))                      # √(q_i·q_j)
    s_sum  = S[:, None] + S[None, :]                       # S_i + S_j
    decay  = np.exp2(-s_sum / 10.0)                        # 2^(−(S_i+S_j)/10)
    B = q_prod * decay
    np.fill_diagonal(B, 0.0)
    return B


def cams32_bond_matrix(C: np.ndarray, A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    CAMS-v3.2 legacy exponential bond matrix (retired per RD-003, kept for comparison).
    B_ij = (0.6·C_i·C_j + 0.4·A_i·A_j) · exp(−(S_i+S_j)/20),  diagonal = 0.
    """
    linear = 0.6 * np.outer(C, C) + 0.4 * np.outer(A, A)
    s_sum  = S[:, None] + S[None, :]
    decay  = np.exp(-s_sum / 20.0)
    B = linear * decay
    np.fill_diagonal(B, 0.0)
    return B


def per_node_bond_strength(B: np.ndarray) -> np.ndarray:
    """Mean of each row's off-diagonal entries = per-node bond strength."""
    return B.sum(axis=1) / (N - 1)


def system_bond_density(B: np.ndarray) -> float:
    """Mean of all off-diagonal B_ij = system bond density B̄."""
    return B.sum() / (N * (N - 1))


def lambda2(B: np.ndarray) -> float:
    """
    Algebraic connectivity = second-smallest eigenvalue of the graph Laplacian.
    L = diag(B·1) − B   (degree matrix minus adjacency/weight matrix).
    λ₂ ordering is scale-invariant; useful for connectivity collapse detection.
    """
    degree = B.sum(axis=1)
    L = np.diag(degree) - B
    eigvals = np.linalg.eigvalsh(L)
    eigvals.sort()
    return float(eigvals[1])  # λ₁ ≈ 0 (connected graph), λ₂ is what we want


def juno_regime(sbd: float) -> tuple[int, str]:
    """Map system bond density to JUNO phase integer and regime label."""
    for threshold, phase, label in JUNO_REGIMES:
        if sbd >= threshold:
            return phase, label
    return 1, "Systemic Collapse"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class BackCalcPipeline:
    """
    Loads raw scores, computes both formalisms, optionally compares with
    legacy CAMS data, and writes DB-ready CSV outputs.
    """

    def __init__(self, raw_path: str, legacy_cams_path: str | None, out_dir: str):
        self.raw_path        = raw_path
        self.legacy_cams_path = legacy_cams_path
        self.out_dir         = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.raw:     pd.DataFrame | None = None
        self.nm_juno: pd.DataFrame | None = None
        self.sm_juno: pd.DataFrame | None = None
        self.bm_juno: pd.DataFrame | None = None
        self.nm_cams: pd.DataFrame | None = None
        self.sm_cams: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> "BackCalcPipeline":
        print(f"[load] Reading raw scores: {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        df.columns = df.columns.str.strip()

        required = {"Society", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"}
        missing  = required - set(df.columns)
        if missing:
            sys.exit(f"[error] Missing columns in --raw CSV: {missing}")

        df["Year"] = df["Year"].astype(int)
        df["Coherence"]   = df["Coherence"].astype(float)
        df["Capacity"]    = df["Capacity"].astype(float)
        df["Stress"]      = df["Stress"].astype(float)
        df["Abstraction"] = df["Abstraction"].astype(float)

        # Filter to known nodes only
        unknown = set(df["Node"].unique()) - set(NODES)
        if unknown:
            warnings.warn(f"[warn] Unknown nodes dropped: {unknown}")
            df = df[df["Node"].isin(NODES)]

        societies = sorted(df["Society"].unique())
        years     = sorted(df["Year"].unique())
        print(f"       {len(df):,} rows  |  {len(societies)} societies  |  "
              f"years {years[0]}–{years[-1]}  |  {len(societies)*len(years)*N:,} expected slots")
        self.raw = df
        return self

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self) -> "BackCalcPipeline":
        print("[compute] Running JUNO-1.0 and CAMS-v3.2 formulas …")
        nm_juno_rows, sm_juno_rows, bm_juno_rows = [], [], []
        nm_cams_rows, sm_cams_rows               = [], []

        # Baseline B̄₀ per society (first year with complete 8-node data)
        baseline_juno: dict[str, float] = {}
        baseline_cams: dict[str, float] = {}

        groups = self.raw.groupby(["Society", "Year"])
        skipped = 0

        for (society, year), g in groups:
            g = g.set_index("Node").reindex(NODES)
            if g[["Coherence", "Capacity", "Stress", "Abstraction"]].isna().any().any():
                skipped += 1
                continue

            C = g["Coherence"].values.astype(float)
            K = g["Capacity"].values.astype(float)
            S = g["Stress"].values.astype(float)
            A = g["Abstraction"].values.astype(float)

            nv = node_value(C, K, S, A)

            # --- JUNO-1.0 ---
            Bj = juno_bond_matrix(C, A, S)
            bs_juno = per_node_bond_strength(Bj)
            sbd_juno = system_bond_density(Bj)
            lam2_juno = lambda2(Bj)

            if society not in baseline_juno:
                baseline_juno[society] = sbd_juno
            B0_juno = baseline_juno[society]
            decay_j = 1.0 - sbd_juno / B0_juno if B0_juno > 0 else 0.0

            phase_j, label_j = juno_regime(sbd_juno)
            esch = (K - S).mean()  # system-level ESCH activation (capacity − stress)

            for i, node in enumerate(NODES):
                nm_juno_rows.append({
                    "Society": society, "Year": year, "Node": node,
                    "Formalism": FORMALISM_JUNO,
                    "Node_Value": round(nv[i], 4),
                    "Bond_Strength": round(bs_juno[i], 4),
                })
            sm_juno_rows.append({
                "Society": society, "Year": year,
                "Formalism": FORMALISM_JUNO,
                "System_Bond_Density": round(sbd_juno, 4),
                "System_Bond_Density_0": round(B0_juno, 4),
                "Decay_Index": round(decay_j, 4),
                "Lambda2_Connectivity": round(lam2_juno, 4),
                "Phase": phase_j,
                "Regime_Label": label_j,
                "ESCH_Activation": round(esch, 4),
            })
            for i, fn in enumerate(NODES):
                for j, tn in enumerate(NODES):
                    if i != j:
                        bm_juno_rows.append({
                            "Society": society, "Year": year,
                            "Formalism": FORMALISM_JUNO,
                            "From_Node": fn, "To_Node": tn,
                            "B_ij": round(Bj[i, j], 4),
                        })

            # --- CAMS-v3.2 (legacy) ---
            Bc = cams32_bond_matrix(C, A, S)
            bs_cams = per_node_bond_strength(Bc)
            sbd_cams = system_bond_density(Bc)
            lam2_cams = lambda2(Bc)

            if society not in baseline_cams:
                baseline_cams[society] = sbd_cams
            B0_cams = baseline_cams[society]
            decay_c = 1.0 - sbd_cams / B0_cams if B0_cams > 0 else 0.0

            for i, node in enumerate(NODES):
                nm_cams_rows.append({
                    "Society": society, "Year": year, "Node": node,
                    "Formalism": FORMALISM_CAMS,
                    "Node_Value": None,            # NV formula is JUNO-native; see note
                    "Bond_Strength": round(bs_cams[i], 4),
                })
            sm_cams_rows.append({
                "Society": society, "Year": year,
                "Formalism": FORMALISM_CAMS,
                "System_Bond_Density": round(sbd_cams, 4),
                "System_Bond_Density_0": round(B0_cams, 4),
                "Decay_Index": round(decay_c, 4),
                "Lambda2_Connectivity": round(lam2_cams, 4),
                "Phase": None,    # CAMS-v3.2 phase bands deprecated per OD-001
                "Regime_Label": "legacy_deprecated",
                "ESCH_Activation": round(esch, 4),
            })

        if skipped:
            print(f"       [warn] {skipped} society-year groups skipped (incomplete node set)")

        self.nm_juno = pd.DataFrame(nm_juno_rows)
        self.sm_juno = pd.DataFrame(sm_juno_rows)
        self.bm_juno = pd.DataFrame(bm_juno_rows)
        self.nm_cams = pd.DataFrame(nm_cams_rows)
        self.sm_cams = pd.DataFrame(sm_cams_rows)

        n_sy = len(sm_juno_rows)
        print(f"       Done: {n_sy:,} society-years  |  {len(nm_juno_rows):,} node-metrics  "
              f"|  {len(bm_juno_rows):,} bond-matrix rows")
        return self

    # ------------------------------------------------------------------
    # Verify (if input already had JUNO values)
    # ------------------------------------------------------------------

    def verify_existing_juno(self) -> "BackCalcPipeline":
        pre_cols = {"Node_Value_JUNO", "Bond_Strength_JUNO", "System_Bond_Density_JUNO"}
        if not pre_cols.issubset(self.raw.columns):
            return self

        print("[verify] Input has pre-computed JUNO columns — checking against formula …")
        merged = self.raw.merge(
            self.nm_juno[["Society","Year","Node","Bond_Strength"]],
            on=["Society","Year","Node"], suffixes=("","_calc")
        )
        merged = merged.merge(
            self.sm_juno[["Society","Year","System_Bond_Density"]],
            on=["Society","Year"], suffixes=("","_calc")
        )

        diffs = pd.DataFrame({
            "Society":  merged["Society"],
            "Year":     merged["Year"],
            "Node":     merged["Node"],
            "NV_csv":   merged["Node_Value_JUNO"],
            "NV_calc":  merged["Society"].map(str),  # placeholder — compute inline
            "BS_csv":   merged["Bond_Strength_JUNO"],
            "BS_calc":  merged["Bond_Strength"],
            "SBD_csv":  merged["System_Bond_Density_JUNO"],
            "SBD_calc": merged["System_Bond_Density"],
        })
        # Node_Value
        C = merged["Coherence"].values
        K = merged["Capacity"].values
        S = merged["Stress"].values
        A = merged["Abstraction"].values
        diffs["NV_calc"] = (C + K + 0.5*A - S).round(4)
        diffs["NV_diff"] = (merged["Node_Value_JUNO"].values - diffs["NV_calc"].values).round(6)
        diffs["BS_diff"] = (diffs["BS_csv"] - diffs["BS_calc"]).round(6)
        diffs["SBD_diff"] = (diffs["SBD_csv"] - diffs["SBD_calc"]).round(6)

        max_nv  = diffs["NV_diff"].abs().max()
        max_bs  = diffs["BS_diff"].abs().max()
        max_sbd = diffs["SBD_diff"].abs().max()
        print(f"       Max |Δ| Node_Value:          {max_nv:.2e}")
        print(f"       Max |Δ| Bond_Strength:       {max_bs:.2e}")
        print(f"       Max |Δ| System_Bond_Density: {max_sbd:.2e}")

        if max(max_nv, max_bs, max_sbd) > 1e-3:
            print("       [WARN] Discrepancies > 0.001 found — check verify_juno.csv")
            diffs[diffs[["NV_diff","BS_diff","SBD_diff"]].abs().max(axis=1) > 1e-3].to_csv(
                self.out_dir / "verify_juno.csv", index=False
            )
        else:
            print("       All values match (max error < 0.001). Formulas consistent.")
        return self

    # ------------------------------------------------------------------
    # Concordance (JUNO vs CAMS-v3.2 on the same corpus)
    # ------------------------------------------------------------------

    def concordance(self) -> "BackCalcPipeline":
        print("[concordance] Computing JUNO-1.0 vs CAMS-v3.2 agreement …")

        # --- Per-node bond strength concordance ---
        nm_j = self.nm_juno[["Society","Year","Node","Bond_Strength"]].rename(
            columns={"Bond_Strength": "BS_juno"})
        nm_c = self.nm_cams[["Society","Year","Node","Bond_Strength"]].rename(
            columns={"Bond_Strength": "BS_cams"})
        nm_both = nm_j.merge(nm_c, on=["Society","Year","Node"])

        rows = []
        # Overall
        rho, _ = spearmanr(nm_both["BS_juno"], nm_both["BS_cams"])
        mae    = (nm_both["BS_juno"] - nm_both["BS_cams"]).abs().mean()
        rows.append({"Metric":"bond_strength", "Node":"ALL", "Spearman_rho":round(rho,4),
                     "MAE":round(mae,4), "N":len(nm_both)})
        # Per-node
        for node in NODES:
            sub = nm_both[nm_both["Node"] == node]
            rho, _ = spearmanr(sub["BS_juno"], sub["BS_cams"])
            mae    = (sub["BS_juno"] - sub["BS_cams"]).abs().mean()
            rows.append({"Metric":"bond_strength", "Node":node, "Spearman_rho":round(rho,4),
                         "MAE":round(mae,4), "N":len(sub)})

        # --- System bond density concordance ---
        sm_j = self.sm_juno[["Society","Year","System_Bond_Density","Decay_Index","Lambda2_Connectivity"]].rename(
            columns={"System_Bond_Density":"SBD_juno","Decay_Index":"DI_juno","Lambda2_Connectivity":"L2_juno"})
        sm_c = self.sm_cams[["Society","Year","System_Bond_Density","Decay_Index","Lambda2_Connectivity"]].rename(
            columns={"System_Bond_Density":"SBD_cams","Decay_Index":"DI_cams","Lambda2_Connectivity":"L2_cams"})
        sm_both = sm_j.merge(sm_c, on=["Society","Year"])

        rho_sbd, _ = spearmanr(sm_both["SBD_juno"], sm_both["SBD_cams"])
        mae_sbd    = (sm_both["SBD_juno"] - sm_both["SBD_cams"]).abs().mean()
        rows.append({"Metric":"system_bond_density","Node":"ALL","Spearman_rho":round(rho_sbd,4),
                     "MAE":round(mae_sbd,4),"N":len(sm_both)})

        rho_di, _ = spearmanr(sm_both["DI_juno"], sm_both["DI_cams"])
        rows.append({"Metric":"decay_index","Node":"ALL","Spearman_rho":round(rho_di,4),
                     "MAE":None,"N":len(sm_both)})

        rho_l2, _ = spearmanr(sm_both["L2_juno"], sm_both["L2_cams"])
        rows.append({"Metric":"lambda2_connectivity","Node":"ALL","Spearman_rho":round(rho_l2,4),
                     "MAE":None,"N":len(sm_both)})

        conc = pd.DataFrame(rows)
        conc.to_csv(self.out_dir / "concordance_report.csv", index=False)

        # Human-readable text report
        lines = [
            "JUNO-1.0 vs CAMS-v3.2 Concordance Report",
            "=" * 50,
            f"Societies  : {sorted(self.raw['Society'].unique())}",
            f"Years      : {self.raw['Year'].min()}–{self.raw['Year'].max()}",
            f"Obs (node) : {len(nm_both):,}",
            "",
            "Bond Strength (per-node):",
            f"  Overall  Spearman ρ = {conc.loc[conc.Node=='ALL','Spearman_rho'].iloc[0]:.4f}  "
            f"MAE = {conc.loc[conc.Node=='ALL','MAE'].iloc[0]:.4f}",
            "  Note: MAE is cross-scale (JUNO ∈[0,1], CAMS-v3.2 ~[0.7,64]); Spearman is scale-invariant.",
            "",
            "Per-node Spearman ρ:",
        ]
        for node in NODES:
            r = conc[(conc["Metric"]=="bond_strength") & (conc["Node"]==node)].iloc[0]
            lines.append(f"  {node:<10} ρ = {r.Spearman_rho:.4f}  MAE = {r.MAE:.4f}")
        lines += [
            "",
            "System-level metrics (Spearman ρ, scale-invariant):",
            f"  System Bond Density  : {rho_sbd:.4f}",
            f"  Decay Index          : {rho_di:.4f}",
            f"  λ₂ Connectivity      : {rho_l2:.4f}",
            "",
            "Interpretation:",
            "  ρ > 0.95 → formalisms agree on rank ordering (same crises, same hierarchy).",
            "  Large MAE is expected and harmless: the two live on different scales.",
            "  Transitions and phase detections are driven by rank order, not absolute values.",
            "  Use v_bond_compare and v_system_compare SQL views for site comparison UI.",
        ]

        # Append per-society SBD concordance
        lines += ["", "Per-society SBD concordance:"]
        for soc in sorted(sm_both["Society"].unique()):
            sub = sm_both[sm_both["Society"]==soc]
            if len(sub) > 2:
                rho_s, _ = spearmanr(sub["SBD_juno"], sub["SBD_cams"])
                lines.append(f"  {soc:<20} ρ = {rho_s:.4f}  n={len(sub)}")

        # Optional: compare with externally-supplied legacy CAMS data
        if self.legacy_cams_path:
            lines += ["", f"External legacy CAMS file: {self.legacy_cams_path}"]
            try:
                leg = pd.read_csv(self.legacy_cams_path)
                leg.columns = leg.columns.str.strip()
                bs_col = next((c for c in leg.columns if "bond" in c.lower()), None)
                if bs_col and {"Society","Year","Node"}.issubset(leg.columns):
                    leg_merge = nm_both.merge(
                        leg[["Society","Year","Node",bs_col]].rename(columns={bs_col:"BS_leg"}),
                        on=["Society","Year","Node"]
                    )
                    rho_ext_j, _ = spearmanr(leg_merge["BS_juno"], leg_merge["BS_leg"])
                    rho_ext_c, _ = spearmanr(leg_merge["BS_cams"], leg_merge["BS_leg"])
                    lines.append(f"  External vs JUNO-1.0  ρ = {rho_ext_j:.4f}")
                    lines.append(f"  External vs CAMS-v3.2 ρ = {rho_ext_c:.4f}")
                else:
                    lines.append("  [warn] Could not identify bond strength column in legacy file.")
            except Exception as e:
                lines.append(f"  [warn] Could not read legacy file: {e}")

        report_path = self.out_dir / "concordance_report.txt"
        report_path.write_text("\n".join(lines) + "\n")
        print(f"       Concordance: SBD Spearman ρ = {rho_sbd:.4f}  "
              f"Bond Spearman ρ = {conc.loc[conc.Node=='ALL','Spearman_rho'].iloc[0]:.4f}")
        return self

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    def write(self) -> "BackCalcPipeline":
        print(f"[write] Saving CSV outputs to {self.out_dir}/")

        # node_metrics_juno.csv
        self.nm_juno.to_csv(self.out_dir / "node_metrics_juno.csv", index=False)
        print(f"        node_metrics_juno.csv     {len(self.nm_juno):>8,} rows")

        # system_metrics_juno.csv
        self.sm_juno.to_csv(self.out_dir / "system_metrics_juno.csv", index=False)
        print(f"        system_metrics_juno.csv   {len(self.sm_juno):>8,} rows")

        # bond_matrix_juno.csv
        self.bm_juno.to_csv(self.out_dir / "bond_matrix_juno.csv", index=False)
        print(f"        bond_matrix_juno.csv      {len(self.bm_juno):>8,} rows")

        # node_metrics_cams32.csv
        self.nm_cams.to_csv(self.out_dir / "node_metrics_cams32.csv", index=False)
        print(f"        node_metrics_cams32.csv   {len(self.nm_cams):>8,} rows")

        # system_metrics_cams32.csv
        self.sm_cams.to_csv(self.out_dir / "system_metrics_cams32.csv", index=False)
        print(f"        system_metrics_cams32.csv {len(self.sm_cams):>8,} rows")

        # DB-import helper: raw_scores (formalism-agnostic)
        raw_out_cols = ["Society", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"]
        self.raw[raw_out_cols].to_csv(self.out_dir / "raw_scores_import.csv", index=False)
        print(f"        raw_scores_import.csv     {len(self.raw):>8,} rows")

        print("[done]")
        return self

    # ------------------------------------------------------------------
    # Convenience: run everything
    # ------------------------------------------------------------------

    def run(self) -> "BackCalcPipeline":
        return (
            self.load()
                .compute()
                .verify_existing_juno()
                .concordance()
                .write()
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JUNO-1.0 / CAMS-v3.2 back-calculation adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw", required=True, metavar="CSV",
        help="Raw scores CSV (columns: Society, Year, Node, Coherence, Capacity, Stress, Abstraction). "
             "Also accepts existing JUNO CSVs with pre-computed columns — these are verified.",
    )
    parser.add_argument(
        "--legacy-cams", default=None, metavar="CSV",
        help="Optional: externally-computed CAMS node metrics CSV for concordance comparison.",
    )
    parser.add_argument(
        "--out", default="./juno_output", metavar="DIR",
        help="Output directory (created if needed). Default: ./juno_output",
    )
    parser.add_argument(
        "--no-bond-matrix", action="store_true",
        help="Skip writing bond_matrix_juno.csv (56 rows per society-year — large for long runs).",
    )
    args = parser.parse_args()

    pipeline = BackCalcPipeline(
        raw_path=args.raw,
        legacy_cams_path=args.legacy_cams,
        out_dir=args.out,
    )
    pipeline.run()

    if args.no_bond_matrix:
        bm_path = pipeline.out_dir / "bond_matrix_juno.csv"
        if bm_path.exists():
            bm_path.unlink()
            print("[info] bond_matrix_juno.csv removed per --no-bond-matrix flag")


if __name__ == "__main__":
    main()
