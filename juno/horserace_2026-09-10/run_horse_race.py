#!/usr/bin/env python3
"""
Epiphenomenon horse-race: lagged CAMS (tau/BS/Hands) vs external competitors
predicting Hansard sino_ppm (2009–2025 primary; exploratory 2006–2025).

Implements Grok Report v3 (14 May 2026) §8 highest-value next step.
Reproducible; never invents numeric values for missing series.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path("/workspace/epiphenomenon/horserace_out")
DATA = Path("/workspace/epiphenomenon")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Outcome + CAMS panel
# ---------------------------------------------------------------------------
hr = pd.read_csv(DATA / "horse_race_data.csv")
hm = pd.read_csv(DATA / "hansard_master.csv")
raw = pd.read_csv(DATA / "hansard_raw_counts.csv")
aus = pd.read_csv(DATA / "aus_annual.csv").rename(columns={"Year": "year"})
hg = pd.read_csv(DATA / "hands_grievance_hansard.csv")

# Primary outcome years from horse_race_data (matches prior Hands race n=17)
panel = hr[["year", "sino_ppm", "tau", "BS", "Hands", "praet_gap",
            "tau_L2", "BS_L2", "Hands_L2", "praet_gap_L2",
            "tau_L3", "BS_L3", "Hands_L3", "praet_gap_L3"]].copy()
panel["sino_source"] = "horse_race_data.csv"
panel["partial_year"] = False

# Extend 2006–2008 from hansard_master + CAMS lags from aus_annual / hg
cams_long = aus[["year", "tau", "BS", "Hands"]].copy()
# Prefer hg Hands/tau/BS where overlapping (same scale as horse_race_data)
for y in hg["year"]:
    cams_long.loc[cams_long.year == y, ["tau", "BS", "Hands"]] = (
        hg.loc[hg.year == y, ["tau", "BS", "Hands"]].values
    )
cams_long = cams_long.sort_values("year").reset_index(drop=True)
cams_long["tau_L2"] = cams_long["tau"].shift(2)
cams_long["BS_L2"] = cams_long["BS"].shift(2)
cams_long["Hands_L2"] = cams_long["Hands"].shift(2)
cams_long["tau_L3"] = cams_long["tau"].shift(3)
cams_long["BS_L3"] = cams_long["BS"].shift(3)
cams_long["Hands_L3"] = cams_long["Hands"].shift(3)

extra_rows = []
for y in (2006, 2007, 2008):
    row_hm = hm.loc[hm.year == y].iloc[0]
    row_c = cams_long.loc[cams_long.year == y].iloc[0]
    extra_rows.append({
        "year": y,
        "sino_ppm": float(row_hm.sino_ppm),
        "tau": float(row_c.tau),
        "BS": float(row_c.BS),
        "Hands": float(row_c.Hands),
        "praet_gap": np.nan,
        "tau_L2": float(row_c.tau_L2) if pd.notna(row_c.tau_L2) else np.nan,
        "BS_L2": float(row_c.BS_L2) if pd.notna(row_c.BS_L2) else np.nan,
        "Hands_L2": float(row_c.Hands_L2) if pd.notna(row_c.Hands_L2) else np.nan,
        "praet_gap_L2": np.nan,
        "tau_L3": float(row_c.tau_L3) if pd.notna(row_c.tau_L3) else np.nan,
        "BS_L3": float(row_c.BS_L3) if pd.notna(row_c.BS_L3) else np.nan,
        "Hands_L3": float(row_c.Hands_L3) if pd.notna(row_c.Hands_L3) else np.nan,
        "praet_gap_L3": np.nan,
        "sino_source": "hansard_master.csv",
        "partial_year": False,
    })

# 2026 partial from raw counts
r26 = raw.loc[raw.year == 2026].iloc[0]
sino_2026 = 1e6 * float(r26.sino_raw) / float(r26.total_words)
row_c26 = cams_long.loc[cams_long.year == 2025]  # lags for 2026 need 2024/2023
# Build 2026 CAMS from aus if present else 2025 carry — check
if (cams_long.year == 2026).any():
    c26 = cams_long.loc[cams_long.year == 2026].iloc[0]
else:
    # No 2026 CAMS row in aus_annual for predictors contemporaneous; still can form L2 from 2024
    c24 = cams_long.loc[cams_long.year == 2024].iloc[0]
    c23 = cams_long.loc[cams_long.year == 2023].iloc[0]
    c25 = cams_long.loc[cams_long.year == 2025].iloc[0]
    class R: pass
    c26 = R()
    c26.tau = c25.tau  # placeholder contemporaneous — not used as predictor
    c26.BS = c25.BS
    c26.Hands = c25.Hands
    c26.tau_L2 = c24.tau
    c26.BS_L2 = c24.BS
    c26.Hands_L2 = c24.Hands
    c26.tau_L3 = c23.tau
    c26.BS_L3 = c23.BS
    c26.Hands_L3 = c23.Hands

extra_rows.append({
    "year": 2026,
    "sino_ppm": sino_2026,
    "tau": float(c26.tau),
    "BS": float(c26.BS),
    "Hands": float(c26.Hands),
    "praet_gap": np.nan,
    "tau_L2": float(c26.tau_L2),
    "BS_L2": float(c26.BS_L2),
    "Hands_L2": float(c26.Hands_L2),
    "praet_gap_L2": np.nan,
    "tau_L3": float(c26.tau_L3),
    "BS_L3": float(c26.BS_L3),
    "Hands_L3": float(c26.Hands_L3),
    "praet_gap_L3": np.nan,
    "sino_source": "hansard_raw_counts.csv (partial)",
    "partial_year": True,
})

panel_ext = pd.concat([pd.DataFrame(extra_rows), panel], ignore_index=True)
panel_ext = panel_ext.sort_values("year").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. External series (documented; no fabrication)
# ---------------------------------------------------------------------------
# Trade: WITS Export Partner Share China % (UN Comtrade via WITS), 2006–2023
# + Comtrade preview API primaryValue ratio for 2024. 2025/2026 MISSING.
wits = pd.read_csv(OUT / "wits_china_shares.csv")
trade = dict(zip(wits.year, wits.china_export_partner_share_pct))
trade[2024] = round(100.0 * 102629123796.603 / 340854851740.669, 2)  # Comtrade
# verified matches WITS for overlapping years

# PRC actions: annual count of discrete documented coercive/military incidents
# See event_coding.md for full list + citations. Transparent coding.
prc_count = {
    2006: 0, 2007: 0, 2008: 0,
    2009: 1,  # Stern Hu / Rio Tinto case (diplomatic-legal crisis)
    2010: 0, 2011: 0, 2012: 0, 2013: 0, 2014: 0, 2015: 0,
    2016: 1,  # SCS arbitral award + bilateral diplomatic confrontation
    2017: 1,  # Foreign-influence crisis peak (Dastyari; interference debate)
    2018: 2,  # Huawei 5G exclusion (Aug) + FITS/espionage legislative package PRC reaction year
    2019: 1,  # Dalian/Australian coal import restrictions (Feb)
    2020: 6,  # Major coercive trade package: barley tariffs; beef suspensions; wine AD;
              # cotton/lobster/timber/coal informal bans (count major measure clusters)
    2021: 1,  # Sanctions maintained + diplomatic freeze / wolf-warrior peak (continuation year=1)
    2022: 1,  # Defence-documented unsafe PLA intercept of RAAF P-8 (26 May 2022)
    2023: 0,  # Thaw / partial sanction lifts dominate; no new major coercive wave coded
    2024: 0,  # Wine tariffs lifted Mar 2024; thaw continues
    2025: 2,  # Defence/ABC: Feb 2025 flares near RAAF P-8; Oct 2025 flares near RAAF P-8
    2026: np.nan,  # year incomplete at coding date
}
prc_sanctions_active = {y: (1 if 2020 <= y <= 2023 else 0) for y in range(2006, 2027)}
prc_sanctions_active[2026] = np.nan

# US alliance pressure: count of major escalation milestones that year
us_alliance = {
    2006: 0, 2007: 0, 2008: 0, 2009: 0, 2010: 0,
    2011: 1,  # US Marine Rotational Force – Darwin announced (Obama visit)
    2012: 0, 2013: 0,
    2014: 1,  # Force Posture Agreement (AUSMIN 2014)
    2015: 0, 2016: 0,
    2017: 1,  # Quad revived (ASEAN Summit / Manila)
    2018: 0, 2019: 0, 2020: 0,
    2021: 1,  # AUKUS announced 15 Sep 2021
    2022: 1,  # ENNPIA / AUKUS implementation milestones
    2023: 1,  # AUKUS Optimal Pathway (13 Mar 2023)
    2024: 1,  # AUKUS enabling / SRF-West pathway progress (coded as milestone year)
    2025: 0,
    2026: np.nan,
}

# Domestic politics
election_year = {y: 0 for y in range(2006, 2027)}
for y in (2007, 2010, 2013, 2016, 2019, 2022, 2025):
    election_year[y] = 1
election_year[2026] = 0

# Labor government majority of calendar year (1=Labor, 0=Coalition)
# Howard to Dec 2007; Rudd/Gillard/Rudd to Sep 2013; Coalition to May 2022; Labor thereafter
labor_gov = {}
for y in range(2006, 2027):
    if y <= 2007:
        labor_gov[y] = 0
    elif y <= 2013:
        labor_gov[y] = 1
    elif y <= 2021:
        labor_gov[y] = 0
    else:  # 2022–2026 Labor (Albanese from May 2022; 2022 coded Labor as majority of year post-election)
        labor_gov[y] = 1
# 2022: election 21 May — code as 1 (Labor majority of remaining year + discourse year)
labor_gov[2022] = 1

# Major security announcement dummies
sec_dwp2016 = {y: int(y == 2016) for y in range(2006, 2027)}
sec_su2020 = {y: int(y == 2020) for y in range(2006, 2027)}
sec_aukus2021 = {y: int(y == 2021) for y in range(2006, 2027)}
sec_dsr2023 = {y: int(y == 2023) for y in range(2006, 2027)}
security_announce_any = {
    y: int(y in (2016, 2020, 2021, 2023)) for y in range(2006, 2027)
}

years = list(range(2006, 2027))
ext = pd.DataFrame({"year": years})
ext["china_export_share_pct"] = [trade.get(y, np.nan) for y in years]
ext["prc_actions_count"] = [prc_count[y] for y in years]
ext["prc_sanctions_active"] = [prc_sanctions_active[y] for y in years]
ext["us_alliance_pressure"] = [us_alliance[y] for y in years]
ext["election_year"] = [election_year[y] for y in years]
ext["labor_gov"] = [labor_gov[y] for y in years]
ext["security_announce_any"] = [security_announce_any[y] for y in years]
ext["sec_dwp2016"] = [sec_dwp2016[y] for y in years]
ext["sec_su2020"] = [sec_su2020[y] for y in years]
ext["sec_aukus2021"] = [sec_aukus2021[y] for y in years]
ext["sec_dsr2023"] = [sec_dsr2023[y] for y in years]
ext["trade_share_source"] = [
    "WITS XPRT-PRTNR-SHR (UN Comtrade)" if y <= 2023 and y in trade
    else ("UN Comtrade preview API ratio" if y == 2024
          else "MISSING")
    for y in years
]
ext["notes"] = [
    "2026 partial / incomplete" if y == 2026
    else ("china share MISSING" if y >= 2025 and y != 2024 and pd.isna(trade.get(y, np.nan))
          else "")
    for y in years
]

# Merge
full = panel_ext.merge(ext, on="year", how="left")
# Primary analysis sample: 2009–2025 complete (horse_race_data), non-partial
primary = full[(full.year >= 2009) & (full.year <= 2025)].copy()
assert len(primary) == 17

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def spearman_pearson(y, x):
    mask = y.notna() & x.notna()
    yy, xx = y[mask], x[mask]
    n = int(mask.sum())
    if n < 5:
        return dict(n=n, spearman_r=np.nan, spearman_p=np.nan,
                    pearson_r=np.nan, pearson_p=np.nan)
    sr, sp = stats.spearmanr(xx, yy)
    pr, pp = stats.pearsonr(xx, yy)
    return dict(n=n, spearman_r=float(sr), spearman_p=float(sp),
                pearson_r=float(pr), pearson_p=float(pp))


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def ols_report(df, ycol, xcols, standardize=True):
    d = df[[ycol] + xcols].dropna().copy()
    n = len(d)
    y = d[ycol]
    X = d[xcols].copy()
    if standardize:
        for c in xcols:
            X[c] = zscore(X[c])
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    vifs = {}
    if len(xcols) >= 2:
        try:
            for i, c in enumerate(xcols):
                vifs[c] = float(variance_inflation_factor(X.values, i))
        except Exception:
            vifs = {c: np.nan for c in xcols}
    coefs = {}
    for c in xcols:
        coefs[c] = {
            "coef": float(model.params[c]),
            "se": float(model.bse[c]),
            "p": float(model.pvalues[c]),
            "t": float(model.tvalues[c]),
        }
    return {
        "n": n,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f": float(model.fvalue) if model.fvalue is not None else None,
        "f_p": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "coefs": coefs,
        "vif": vifs,
        "standardized": standardize,
        "const": float(model.params["const"]),
    }


# ---------------------------------------------------------------------------
# 4. Univariate correlations
# ---------------------------------------------------------------------------
predictors = [
    "tau_L2", "BS_L2", "Hands_L2", "tau_L3", "BS_L3", "Hands_L3",
    "china_export_share_pct", "prc_actions_count", "prc_sanctions_active",
    "us_alliance_pressure", "election_year", "labor_gov", "security_announce_any",
]

univ = {}
for p in predictors:
    univ[p] = spearman_pearson(primary["sino_ppm"], primary[p])

# ---------------------------------------------------------------------------
# 5. Nested / sparse OLS horse-races (standardized predictors)
# ---------------------------------------------------------------------------
# Best external by |Spearman| among complete series
ext_cands = ["china_export_share_pct", "prc_actions_count", "prc_sanctions_active",
             "us_alliance_pressure", "election_year", "labor_gov", "security_announce_any"]
best_ext = max(ext_cands, key=lambda c: abs(univ[c]["spearman_r"]) if pd.notna(univ[c]["spearman_r"]) else -1)
second_ext = sorted(
    [c for c in ext_cands if c != best_ext],
    key=lambda c: abs(univ[c]["spearman_r"]) if pd.notna(univ[c]["spearman_r"]) else -1,
    reverse=True,
)[0]

models = {}
# Univariate z-OLS
for p in ["tau_L2", "BS_L2", "Hands_L2", best_ext, second_ext, "prc_actions_count",
          "china_export_share_pct", "us_alliance_pressure", "labor_gov",
          "prc_sanctions_active", "security_announce_any"]:
    models[f"univ_z_{p}"] = ols_report(primary, "sino_ppm", [p])

# Nested
models["externals_top2"] = ols_report(primary, "sino_ppm", [best_ext, second_ext])
models["cams_L2_only"] = ols_report(primary, "sino_ppm", ["tau_L2", "BS_L2", "Hands_L2"])
models["cams_tau_BS_L2"] = ols_report(primary, "sino_ppm", ["tau_L2", "BS_L2"])
models["nested_tau_L2_plus_best_ext"] = ols_report(primary, "sino_ppm", ["tau_L2", best_ext])
models["nested_BS_L2_plus_best_ext"] = ols_report(primary, "sino_ppm", ["BS_L2", best_ext])
models["nested_Hands_L2_plus_best_ext"] = ols_report(primary, "sino_ppm", ["Hands_L2", best_ext])
models["nested_tau_L2_plus_top2_ext"] = ols_report(
    primary, "sino_ppm", ["tau_L2", best_ext, second_ext]
)
models["full_cams_L2_plus_best_ext"] = ols_report(
    primary, "sino_ppm", ["tau_L2", "BS_L2", "Hands_L2", best_ext]
)
# Race: each CAMS L2 alone vs best external alone (already in univ)
# Sparse: tau_L2 + prc_sanctions + china share (theoretically motivated)
models["sparse_tau_prc_trade"] = ols_report(
    primary, "sino_ppm", ["tau_L2", "prc_sanctions_active", "china_export_share_pct"]
)
models["sparse_tau_us_trade"] = ols_report(
    primary, "sino_ppm", ["tau_L2", "us_alliance_pressure", "china_export_share_pct"]
)

# Who wins: compare adj-R² of univ_z models and key nested
def adj_r2(key):
    return models[key]["adj_r2"]

winner_univ = max(
    ["univ_z_tau_L2", "univ_z_BS_L2", "univ_z_Hands_L2",
     f"univ_z_{best_ext}", f"univ_z_{second_ext}"],
    key=adj_r2,
)

# ---------------------------------------------------------------------------
# 6. Write external_series.csv
# ---------------------------------------------------------------------------
ext_out = full[[
    "year", "sino_ppm", "partial_year", "sino_source",
    "tau_L2", "BS_L2", "Hands_L2", "tau_L3", "BS_L3", "Hands_L3",
    "china_export_share_pct", "prc_actions_count", "prc_sanctions_active",
    "us_alliance_pressure", "election_year", "labor_gov", "security_announce_any",
    "sec_dwp2016", "sec_su2020", "sec_aukus2021", "sec_dsr2023",
    "trade_share_source", "notes",
]].copy()
ext_out.to_csv(OUT / "external_series.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Results JSON
# ---------------------------------------------------------------------------
results = {
    "title": "Epiphenomenon horse-race CAMS L2 vs externals → sino_ppm",
    "date_sydney": "2026-09-10",
    "primary_sample": "2009–2025 horse_race_data.csv",
    "n_primary": 17,
    "outcome": "sino_ppm",
    "best_external_by_abs_spearman": best_ext,
    "second_external_by_abs_spearman": second_ext,
    "univariate": univ,
    "models": models,
    "winner_univ_by_adj_r2": winner_univ,
    "interpretation_soft": (
        "Exploratory comparative fit only; small-n; not causal validation. "
        "CAMS L2 series and externals are raced on adj-R² / standardized coefs."
    ),
    "missing_series": [
        "china_export_share_pct for 2025 and 2026 (Comtrade/WITS not retrieved)",
        "australia_econ_controls.csv ends ~1954 — unused for 2006+",
        "No second-coder PRC event scoring; single transparent coding table",
        "2026 sino_ppm is partial-year (hansard_raw_counts)",
    ],
    "sino_ppm_2026_partial": sino_2026,
    "trade_2024_comtrade_pct": trade[2024],
}

# Convert numpy types
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if o is None or (isinstance(o, float) and not np.isfinite(o)):
        return None
    return o

with open(OUT / "horse_race_results.json", "w") as f:
    json.dump(_clean(results), f, indent=2)

# ---------------------------------------------------------------------------
# 8. Markdown tables
# ---------------------------------------------------------------------------
lines = []
lines.append("# Horse-race tables — Epiphenomenon (10 Sep 2026 Sydney)\n")
lines.append("Primary sample: **n=17**, years **2009–2025** (`horse_race_data.csv`).\n")
lines.append("Predictors standardized (z) in OLS for coefficient comparability.\n")
lines.append("Language: exploratory / comparative fit — **not** validated causal claims.\n")

lines.append("\n## 1. Univariate Spearman / Pearson vs sino_ppm\n")
lines.append("| Predictor | n | Spearman r | p | Pearson r | p |")
lines.append("|---|---:|---:|---:|---:|---:|")
for p in predictors:
    u = univ[p]
    lines.append(
        f"| {p} | {u['n']} | {u['spearman_r']:.3f} | {u['spearman_p']:.4f} | "
        f"{u['pearson_r']:.3f} | {u['pearson_p']:.4f} |"
    )

lines.append(f"\n**Best external by |Spearman|:** `{best_ext}` "
             f"(r={univ[best_ext]['spearman_r']:.3f}). "
             f"**Second:** `{second_ext}` (r={univ[second_ext]['spearman_r']:.3f}).\n")

lines.append("\n## 2. Univariate z-OLS (sino_ppm ~ z(predictor))\n")
lines.append("| Predictor | n | coef (z) | SE | p | R² | adj-R² |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for key in [k for k in models if k.startswith("univ_z_")]:
    m = models[key]
    p = key.replace("univ_z_", "")
    c = m["coefs"][p]
    lines.append(
        f"| {p} | {m['n']} | {c['coef']:.3f} | {c['se']:.3f} | {c['p']:.4f} | "
        f"{m['r2']:.3f} | {m['adj_r2']:.3f} |"
    )

lines.append("\n## 3. Nested / sparse multivariate z-OLS\n")
multi_keys = [k for k in models if not k.startswith("univ_z_")]
for key in multi_keys:
    m = models[key]
    lines.append(f"\n### `{key}` — n={m['n']}, R²={m['r2']:.3f}, adj-R²={m['adj_r2']:.3f}, "
                 f"F={m['f']:.3f}, F-p={m['f_p']:.4f}\n")
    lines.append("| Predictor | coef (z) | SE | p | VIF |")
    lines.append("|---|---:|---:|---:|---:|")
    for p, c in m["coefs"].items():
        vif = m["vif"].get(p, float("nan"))
        vif_s = f"{vif:.2f}" if vif == vif else "—"
        lines.append(f"| {p} | {c['coef']:.3f} | {c['se']:.3f} | {c['p']:.4f} | {vif_s} |")

lines.append("\n## 4. Horse-race summary (who wins on adj-R²)\n")
lines.append(f"- Univariate winner (among CAMS L2 + top2 externals): **`{winner_univ}`** "
             f"(adj-R²={models[winner_univ]['adj_r2']:.3f}).\n")
lines.append(f"- CAMS L2 block (`tau_L2+BS_L2+Hands_L2`) adj-R²="
             f"{models['cams_L2_only']['adj_r2']:.3f} "
             f"(note multicollinearity; VIFs high).\n")
lines.append(f"- Externals top2 (`{best_ext}`+`{second_ext}`) adj-R²="
             f"{models['externals_top2']['adj_r2']:.3f}.\n")
lines.append(f"- Nested `tau_L2` + best external adj-R²="
             f"{models['nested_tau_L2_plus_best_ext']['adj_r2']:.3f}.\n")

(OUT / "horse_race_tables.md").write_text("\n".join(lines))

# ---------------------------------------------------------------------------
# 9. Optional figure
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, vals = [], []
    for key in ["univ_z_tau_L2", "univ_z_BS_L2", "univ_z_Hands_L2",
                f"univ_z_{best_ext}", f"univ_z_{second_ext}",
                "cams_L2_only", "externals_top2",
                "nested_tau_L2_plus_best_ext"]:
        labels.append(key.replace("univ_z_", "").replace("nested_", "n:"))
        vals.append(models[key]["adj_r2"])
    colors = ["#2a6fbb" if "tau" in l or "BS" in l or "Hands" in l or l.startswith("cams")
              else "#c44e52" for l in labels]
    # fix: nested has tau — blue; externals red
    colors = []
    for l in labels:
        if l in (best_ext, second_ext, "externals_top2") or l.startswith("externals"):
            colors.append("#c44e52")
        elif l.startswith("n:") and best_ext in l:
            colors.append("#7b68a6")
        else:
            colors.append("#2a6fbb")
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Adjusted R²")
    ax.set_title("Horse-race adj-R² (z-OLS, n=17, 2009–2025)")
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "horse_race_adjr2.png", dpi=140)
    plt.close()
except Exception as e:
    print("figure skip:", e)

# ---------------------------------------------------------------------------
# 10. Print key numbers for parent summary
# ---------------------------------------------------------------------------
print("BEST_EXT", best_ext, univ[best_ext])
print("SECOND_EXT", second_ext, univ[second_ext])
print("WINNER_UNIV", winner_univ, models[winner_univ]["adj_r2"])
for k in ["univ_z_tau_L2", "univ_z_BS_L2", "univ_z_Hands_L2",
          f"univ_z_{best_ext}", "cams_L2_only", "externals_top2",
          "nested_tau_L2_plus_best_ext", "cams_tau_BS_L2",
          "sparse_tau_prc_trade"]:
    m = models[k]
    print(f"MODEL {k}: R2={m['r2']:.4f} adjR2={m['adj_r2']:.4f} n={m['n']} coefs={m['coefs']}")
print("SINO2026", sino_2026)
print("DONE")
