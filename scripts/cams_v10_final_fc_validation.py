#!/usr/bin/env python3
"""
cams_v10_final_fc_validation.py
================================
Validate CAMS v1.0-Final recomputed series against pre-registered
falsification criteria (FC1–FC6) using the primary in-range corpus.

Run from wintermute/:
    python scripts/cams_v10_final_fc_validation.py

Reads:
  data/v1.0_final_recompute/series/*_Bij_v10.csv
  data/v1.0_final_recompute/FG_corpus.csv
  data/v1.0_final_recompute/manifest.json

Writes:
  data/v1.0_final_recompute/FC_VALIDATION_REPORT.md
  data/v1.0_final_recompute/fc_validation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = ROOT / "data" / "v1.0_final_recompute" / "series"
FG_PATH = ROOT / "data" / "v1.0_final_recompute" / "FG_corpus.csv"
MANIFEST_PATH = ROOT / "data" / "v1.0_final_recompute" / "manifest.json"
REPORT_PATH = ROOT / "data" / "v1.0_final_recompute" / "FC_VALIDATION_REPORT.md"
JSON_PATH = ROOT / "data" / "v1.0_final_recompute" / "fc_validation.json"

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
CRISIS_REGIMES = {"Systemic crisis", "Freeze/Collapse", "Local Node Failure"}
STABLE_REGIMES = {"Stable adaptive"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_primary_series(manifest: dict) -> list[Path]:
    """Return paths for canonical_ens + cleaned_ens in-range series."""
    files = []
    for entry in manifest.get("series", []):
        if entry.get("range_tier") == "in_range" and entry.get("tier") in (
            "canonical_ens",
            "cleaned_ens",
        ):
            p = SERIES_DIR / entry["out_name"]
            if p.exists():
                files.append(p)
    return files


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Coherence", "Capacity", "Stress", "Abstraction", "Node Value", "Bond Strength"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pivot_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per (Society, Year), columns per node-metric."""
    required = ["Society", "Year", "Node", "Coherence", "Capacity", "Stress",
                "Abstraction", "Node Value", "Bond Strength"]
    df = df[required].dropna()
    wide = df.pivot(index=["Society", "Year"], columns="Node")
    wide.columns = [f"{m}_{n}" for m, n in wide.columns]
    return wide.reset_index()


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall tau and two-sided p-value."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return s, p


# ---------------------------------------------------------------------------
# FC1 — Stress-Capacity Anti-Correlation
# ---------------------------------------------------------------------------

def validate_fc1(df_long: pd.DataFrame) -> dict:
    """
    FC1: Stress and Capacity must be anti-correlated across nodes.
    Per-society-year Spearman ρ(Stress, Capacity) should be predominantly negative.
    """
    rhos = []
    for (soc, year), g in df_long.groupby(["Society", "Year"]):
        if len(g) < 8:
            continue
        r, p = stats.spearmanr(g["Stress"], g["Capacity"])
        rhos.append({"Society": soc, "Year": year, "rho": r, "p": p})

    rho_df = pd.DataFrame(rhos)
    median_rho = rho_df["rho"].median()
    mean_rho = rho_df["rho"].mean()
    pct_negative = (rho_df["rho"] < 0).mean() * 100
    # One-sample t-test against -0.5
    t_stat, t_p = stats.ttest_1samp(rho_df["rho"], -0.5)

    # Pooled node-level correlation
    pool_r, pool_p = stats.spearmanr(df_long["Stress"], df_long["Capacity"])

    return {
        "criterion": "FC1 — Stress-Capacity Anti-Correlation",
        "status": "PASS" if median_rho < -0.5 else "FAIL",
        "median_rho_per_year": round(median_rho, 4),
        "mean_rho_per_year": round(mean_rho, 4),
        "pct_negative": round(pct_negative, 2),
        "ttest_vs_minus0.5_t": round(t_stat, 4),
        "ttest_vs_minus0.5_p": round(t_p, 4),
        "pooled_rho": round(pool_r, 4),
        "pooled_p": round(pool_p, 4),
        "n_society_years": len(rho_df),
        "threshold": "median ρ < -0.5",
    }


# ---------------------------------------------------------------------------
# FC2 — Three-Tier Crisis Composite (computable subset)
# ---------------------------------------------------------------------------

def validate_fc2(fg: pd.DataFrame) -> dict:
    """
    FC2: Crisis detection via three-tier composite.
    Primary:   V_min < 4
    Secondary: |ΔV̄| > 12% (minor) or 20% (major)  [proxy for JUNO_v2.1]
    Acute:     σ_rate = |Δσ_V| > 0.5  (year-on-year change in V_std)
    """
    fg = fg.sort_values(["Society", "Year"]).copy()
    fg["dV_mean"] = fg.groupby("Society")["V_mean"].pct_change()
    fg["sigma_rate"] = fg.groupby("Society")["V_std"].diff().abs()

    crisis = fg[fg["regime"].isin(CRISIS_REGIMES)].copy()
    stable = fg[fg["regime"].isin(STABLE_REGIMES)].copy()

    # Primary trigger rate in crisis years
    primary_hit = (crisis["V_min"] < 4).mean() * 100 if len(crisis) else np.nan
    # False positive in stable years
    primary_fp = (stable["V_min"] < 4).mean() * 100 if len(stable) else np.nan

    # Secondary: ΔV̄ < -12%
    sec_hit = (crisis["dV_mean"] < -0.12).mean() * 100 if len(crisis) else np.nan
    sec_fp = (stable["dV_mean"] < -0.12).mean() * 100 if len(stable) else np.nan

    # Acute: sigma_rate > 0.5
    acute_hit = (crisis["sigma_rate"] > 0.5).mean() * 100 if len(crisis) else np.nan
    acute_fp = (stable["sigma_rate"] > 0.5).mean() * 100 if len(stable) else np.nan

    # Composite: any tier triggered
    fg["tier1"] = fg["V_min"] < 4
    fg["tier2"] = fg["dV_mean"] < -0.12
    fg["tier3"] = fg["sigma_rate"] > 0.5
    fg["any_tier"] = fg["tier1"] | fg["tier2"] | fg["tier3"]

    comp_hit = fg[fg["regime"].isin(CRISIS_REGIMES)]["any_tier"].mean() * 100
    comp_fp = fg[fg["regime"].isin(STABLE_REGIMES)]["any_tier"].mean() * 100

    return {
        "criterion": "FC2 — Three-Tier Crisis Composite (proxy)",
        "status": "PASS" if primary_hit > 50 else "FAIL",
        "note": "JUNO_v2.1 secondary tier omitted; using ΔV̄<-12% proxy",
        "primary_Vmin_lt4_hit_pct": round(primary_hit, 2),
        "primary_Vmin_lt4_fp_pct": round(primary_fp, 2),
        "secondary_dVmean_lt12_hit_pct": round(sec_hit, 2),
        "secondary_dVmean_lt12_fp_pct": round(sec_fp, 2),
        "acute_sigma_rate_gt0.5_hit_pct": round(acute_hit, 2),
        "acute_sigma_rate_gt0.5_fp_pct": round(acute_fp, 2),
        "composite_any_tier_hit_pct": round(comp_hit, 2),
        "composite_any_tier_fp_pct": round(comp_fp, 2),
        "n_crisis_years": len(crisis),
        "n_stable_years": len(stable),
    }


# ---------------------------------------------------------------------------
# FC3 — Common-Latent-Factor Structure (PCA)
# ---------------------------------------------------------------------------

def validate_fc3(df_wide: pd.DataFrame) -> dict:
    """
    FC3: 8-node Node Values should collapse to ~2-3 dimensions.
    PC1 should be a positive-loading common factor (system health).
    """
    cols = [f"Node Value_{n}" for n in NODES if f"Node Value_{n}" in df_wide.columns]
    if len(cols) < 8:
        return {"criterion": "FC3 — PCA", "status": "SKIP", "reason": "missing nodes"}

    X = df_wide[cols].dropna().values
    # Center and scale
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    ev = (S ** 2) / (len(Xs) - 1)
    ev_ratio = ev / ev.sum()

    pc1_loadings = Vt[0]
    all_positive = (pc1_loadings > 0).all()
    min_loading = pc1_loadings.min()

    # Effective dimensionality (Kaiser: EV > 1)
    kaiser_dim = (ev > 1).sum()

    return {
        "criterion": "FC3 — Common-Latent-Factor (PCA)",
        "status": "PASS" if ev_ratio[0] > 0.40 and all_positive else "FAIL",
        "pc1_variance_explained": round(ev_ratio[0], 4),
        "pc2_variance_explained": round(ev_ratio[1], 4),
        "pc3_variance_explained": round(ev_ratio[2], 4),
        "kaiser_effective_dim": int(kaiser_dim),
        "pc1_all_positive": bool(all_positive),
        "pc1_min_loading": round(min_loading, 4),
        "n_obs": len(X),
    }


# ---------------------------------------------------------------------------
# FC4 — Bond-Health Coupling
# ---------------------------------------------------------------------------

def validate_fc4(df_long: pd.DataFrame, fg: pd.DataFrame) -> dict:
    """
    FC4: Bond Strength and Node Value must be positively coupled.
    Per-node pooled correlation and system-level B_mean vs V_mean.
    """
    # Pooled per-node
    pool_r, pool_p = stats.pearsonr(df_long["Bond Strength"], df_long["Node Value"])

    # System-level
    sys_r, sys_p = stats.pearsonr(fg["B_mean"], fg["V_mean"])

    # Per-society correlation
    soc_rs = []
    for soc, g in fg.groupby("Society"):
        if len(g) < 5:
            continue
        r, p = stats.pearsonr(g["B_mean"], g["V_mean"])
        soc_rs.append(r)
    median_soc_r = np.median(soc_rs) if soc_rs else np.nan

    return {
        "criterion": "FC4 — Bond-Health Coupling",
        "status": "PASS" if pool_r > 0.50 and sys_r > 0.50 else "FAIL",
        "pooled_per_node_r": round(pool_r, 4),
        "pooled_per_node_p": round(pool_p, 4),
        "system_level_r": round(sys_r, 4),
        "system_level_p": round(sys_p, 4),
        "median_per_society_r": round(median_soc_r, 4),
        "n_societies_tested": len(soc_rs),
        "threshold": "ρ > 0.50",
    }


# ---------------------------------------------------------------------------
# FC5 — Shield Ranking Test
# ---------------------------------------------------------------------------

def validate_fc5(df_long: pd.DataFrame, fg: pd.DataFrame) -> dict:
    """
    FC5: Shield BS should not be the minimum in stable years;
    should rank in top half during adaptive periods.
    """
    # Merge regime info
    df = df_long.merge(fg[["Society", "Year", "regime"]], on=["Society", "Year"], how="left")
    stable = df[df["regime"].isin(STABLE_REGIMES)]

    if stable.empty:
        return {"criterion": "FC5 — Shield Ranking", "status": "SKIP", "reason": "no stable years"}

    # Per stable society-year, rank Shield BS among 8 nodes
    ranks = []
    for (soc, year), g in stable.groupby(["Society", "Year"]):
        if len(g) < 8:
            continue
        g = g.sort_values("Bond Strength", ascending=False).reset_index(drop=True)
        shield_row = g[g["Node"] == "Shield"]
        if shield_row.empty:
            continue
        rank = shield_row.index[0] + 1  # 1 = highest
        ranks.append(rank)

    median_rank = np.median(ranks) if ranks else np.nan
    pct_top_half = (np.array(ranks) <= 4).mean() * 100 if ranks else np.nan

    # Also: Shield BS vs system B_mean in stable years
    shield_stable = stable[stable["Node"] == "Shield"][["Society", "Year", "Bond Strength"]]
    sys_stable = stable.groupby(["Society", "Year"])["Bond Strength"].mean().reset_index(name="B_mean")
    merged = shield_stable.merge(sys_stable, on=["Society", "Year"])
    if len(merged) > 3:
        r, p = stats.pearsonr(merged["Bond Strength"], merged["B_mean"])
    else:
        r, p = np.nan, np.nan

    return {
        "criterion": "FC5 — Shield Ranking",
        "status": "PASS" if pct_top_half > 50 else "FAIL",
        "median_rank_in_stable_years": round(median_rank, 2),  # 1 = highest
        "pct_top_half": round(pct_top_half, 2),
        "shield_vs_system_B_r": round(r, 4),
        "shield_vs_system_B_p": round(p, 4),
        "n_stable_years_tested": len(ranks),
    }


# ---------------------------------------------------------------------------
# FC6 — λ₂ Degradation Prior to Crisis
# ---------------------------------------------------------------------------

def validate_fc6(fg: pd.DataFrame) -> dict:
    """
    FC6: λ₂ should degrade (decline) in the 3 years preceding crisis onset.
    Uses Mann-Kendall trend test on λ₂ in the 3-year window before each crisis.
    """
    fg = fg.sort_values(["Society", "Year"]).copy()
    fg["is_crisis"] = fg["regime"].isin(CRISIS_REGIMES)
    fg["crisis_onset"] = fg.groupby("Society")["is_crisis"].transform(
        lambda x: x & (~x.shift(1).fillna(False))
    )

    trends = []
    for soc in fg["Society"].unique():
        soc_df = fg[fg["Society"] == soc].copy()
        onsets = soc_df[soc_df["crisis_onset"]]["Year"].values
        for yr in onsets:
            pre = soc_df[(soc_df["Year"] >= yr - 3) & (soc_df["Year"] < yr)]["lambda2"].values
            if len(pre) >= 3:
                s, p = mann_kendall(pre)
                trends.append({
                    "Society": soc,
                    "crisis_year": int(yr),
                    "lambda2_pre": list(pre),
                    "mk_tau": s,
                    "mk_p": p,
                    "declining": s < 0 and p < 0.10,
                })

    if not trends:
        return {"criterion": "FC6 — λ₂ Degradation", "status": "SKIP", "reason": "no crisis onsets with 3-year pre-window"}

    tdf = pd.DataFrame(trends)
    pct_declining = tdf["declining"].mean() * 100

    return {
        "criterion": "FC6 — λ₂ Degradation Prior to Crisis",
        "status": "PASS" if pct_declining > 33 else "PARTIAL FAIL",
        "pct_declining_3yr_pre_crisis": round(pct_declining, 2),
        "n_crisis_onsets_tested": len(tdf),
        "median_mk_tau": round(tdf["mk_tau"].median(), 4),
        "note": "Threshold: >33% of crises show pre-decline (v3.2-R achieved 1/3)",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("CAMS v1.0-Final Falsification-Criteria Validation")
    print("=" * 55)

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    series_paths = load_primary_series(manifest)
    print(f"Primary in-range series: {len(series_paths)}")

    if not series_paths:
        print("No series to validate.")
        return 1

    # Load all long-form data
    all_long = []
    for p in series_paths:
        df = read_series(p)
        all_long.append(df)
    df_long = pd.concat(all_long, ignore_index=True)
    print(f"Total observations: {len(df_long):,} (node-years)")

    # Pivot for PCA
    df_wide = pivot_nodes(df_long)
    print(f"Complete society-years for PCA: {len(df_wide):,}")

    # Load FG corpus (in-range only)
    fg = pd.read_csv(FG_PATH)
    fg = fg[fg["range_tier"] == "in_range"].copy()
    # Keep only primary series
    primary_files = {p.name for p in series_paths}
    fg = fg[fg["series_file"].isin(primary_files)].copy()
    print(f"FG society-years (primary in-range): {len(fg):,}")

    results = []
    results.append(validate_fc1(df_long))
    results.append(validate_fc2(fg))
    results.append(validate_fc3(df_wide))
    results.append(validate_fc4(df_long, fg))
    results.append(validate_fc5(df_long, fg))
    results.append(validate_fc6(fg))

    # Summary
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    partials = sum(1 for r in results if r["status"] == "PARTIAL FAIL")
    skips = sum(1 for r in results if r["status"] == "SKIP")

    # Write JSON
    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write Markdown report
    lines = [
        "# CAMS v1.0-Final Falsification-Criteria Validation Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Primary series:** {len(series_paths)}",
        f"**Society-years (in-range):** {len(fg):,}",
        f"**Node-observations:** {len(df_long):,}",
        "",
        "## Summary",
        "",
        f"| Status | Count |",
        f"|--------|------:|",
        f"| PASS | {passes} |",
        f"| PARTIAL FAIL | {partials} |",
        f"| FAIL | {fails} |",
        f"| SKIP | {skips} |",
        "",
    ]

    for r in results:
        lines.append(f"## {r['criterion']}")
        lines.append("")
        lines.append(f"**Status:** {r['status']}")
        if "note" in r:
            lines.append(f"**Note:** {r['note']}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for k, v in r.items():
            if k in ("criterion", "status", "note", "lambda2_pre"):
                continue
            if isinstance(v, float):
                val = f"{v:.4f}"
            elif isinstance(v, list):
                val = str(v)[:60]
            else:
                val = str(v)
            lines.append(f"| {k} | {val} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Interpretation Notes",
        "",
        "- **FC1**: The Stress-Capacity anti-correlation is the bedrock thermodynamic "
        "signature. A median ρ < −0.5 across society-years is required for the framework "
        "to claim internal consistency.",
        "- **FC2**: JUNO_v2.1 character classification is not available in the raw "
        "recomputed series; the secondary tier uses ΔV̄ < −12% as a proxy. For full "
        "validation, run JUNO v1.2 regime classifier on the same corpus.",
        "- **FC3**: PCA dimensionality should be ≤3 for the 8-node architecture to be "
        "justified against a 1-factor null model (per Memory 45).",
        "- **FC4**: Bond-Health coupling > 0.50 is the minimum threshold; v3.2-R "
        "achieved >0.60.",
        "- **FC5**: Shield ranking tests whether the security node maintains structural "
        "prominence during stable periods. Collapse to bottom-half rank indicates "
        "Praetorian inversion.",
        "- **FC6**: λ₂ degradation is the weakest empirical signal (v3.2-R: 1/3 crises). "
        "It is retained as a falsification criterion but downgraded to secondary status.",
        "",
        "**Next step:** If any criterion FAILs, inspect the underlying data for "
        "(a) scoring drift, (b) node-mapping errors, or (c) formula implementation bugs.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"PASS: {passes}  PARTIAL: {partials}  FAIL: {fails}  SKIP: {skips}")
    print(f"Report: {REPORT_PATH}")
    print(f"JSON:   {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
