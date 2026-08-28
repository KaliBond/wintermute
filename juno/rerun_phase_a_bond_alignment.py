#!/usr/bin/env python3
"""Re-run Phase A tests that slice Bond_Strength_Calc by Node.

Compares:
  - Bond_Strength_Calc         (aligned, post 2026-08-28 fix)
  - Bond_Strength_Calc_legacy  (misaligned column introduced 2026-08-20)

Empirical note: the 36-society panel at Phase A (git 4289140) already stored
node-aligned Bond_Strength_Calc; it matches current Bond_Strength_Calc
(max |diff| 5e-5). Legacy is the Aug 20 scramble, not the published Phase A
column.

External inputs (downloaded if missing):
  - SIPRI Share of GDP sheet (sipri.org Excel, 1949-2025)
  - World Bank MS.MIL.XPND.GD.ZS
  - in-repo juno/vdem_core_v16.csv

Restricts to the original JUNO_36 panel (excludes the 12 societies added
2026-08-20) so numbers are comparable to PhaseA_Validation_Report_2026-08-16.md.
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JUNO_PATH = HERE / "JUNO_Unified_Dataset.csv"
VDEM_PATH = HERE / "vdem_core_v16.csv"
DATA_DIR = Path("/tmp/phasea_data")
OUT_CSV = HERE / "phase_a_bond_rerun_results_2026-08-28.csv"

NEW12 = {
    "Democratic Republic of the Congo",
    "Egypt",
    "Ethiopia",
    "Mauritania",
    "Nigeria",
    "Cambodia",
    "Mongolia",
    "New Zealand",
    "Greece",
    "Philippines",
    "Latvia",
    "Laos",
}

# JUNO name -> external name
SIPRI_NAME = {
    "USA": "United States of America",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Turkey": "Türkiye",
    "Hong Kong": "Hong Kong",
}
OWID_NAME = {
    "USA": "United States",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Turkey": "Turkey",
}
WB_NAME = {
    "USA": "United States",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Turkey": "Turkiye",
    "Hong Kong": "Hong Kong SAR, China",
    "Iran": "Iran, Islamic Rep.",
    "Russia": "Russian Federation",
    "Venezuela": "Venezuela, RB",
    "Syria": "Syrian Arab Republic",
    "Egypt": "Egypt, Arab Rep.",
}
VDEM_NAME = {
    "USA": "United States of America",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Turkey": "Türkiye",
}

# Region headers in the SIPRI Excel (not countries)
SIPRI_SKIP = {
    "Africa", "North Africa", "sub-Saharan Africa", "Americas",
    "Central America and the Caribbean", "North America", "South America",
    "Asia & Oceania", "Oceania", "South Asia", "East Asia", "South East Asia",
    "Central Asia", "Europe", "Central Europe", "Eastern Europe",
    "Western Europe", "European Union", "Middle East",
}


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return ""
    if p == 0 or p < 1e-300:
        return "0"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4g}"


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(len(x))
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": np.nan, "p": np.nan, "n": n}
    r, p = stats.pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": n}


def demean_within(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c + "_dm"] = out.groupby(group)[c].transform(lambda s: s - s.mean())
    return out


def to_numeric_sipri(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) if np.isfinite(v) else np.nan
    s = str(v).strip()
    if s in {"", "...", ". .", "xxx", "xxx ", ".", "..", "NA", "nan"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan



def ensure_external_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "SIPRI-Milex-data-1949-2025_v1.2.xlsx":
            "https://www.sipri.org/sites/default/files/SIPRI-Milex-data-1949-2025_v1.2.xlsx",
        "military-spending-as-a-share-of-gdp-sipri.csv":
            "https://ourworldindata.org/grapher/military-spending-as-a-share-of-gdp-sipri.csv?v=1&csvType=full&useColumnShortNames=false",
        "wb_milex.json":
            "https://api.worldbank.org/v2/country/all/indicator/MS.MIL.XPND.GD.ZS?format=json&per_page=20000",
    }
    for name, url in files.items():
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)


def load_juno() -> pd.DataFrame:
    df = pd.read_csv(JUNO_PATH, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Society"].isin(set(df["Society"].unique()) - NEW12)].copy()
    return df


def original36(df: pd.DataFrame) -> list[str]:
    return sorted(df["Society"].unique())


def load_sipri_share_of_gdp() -> pd.DataFrame:
    path = DATA_DIR / "SIPRI-Milex-data-1949-2025_v1.2.xlsx"
    raw = pd.read_excel(path, sheet_name="Share of GDP", header=None)
    header_row = 5
    years = [int(y) for y in raw.iloc[header_row, 2:].tolist()]
    records = []
    for i in range(header_row + 2, len(raw)):
        name = raw.iat[i, 0]
        if pd.isna(name):
            continue
        name = str(name).strip()
        if name in SIPRI_SKIP or name == "":
            continue
        for j, yr in enumerate(years):
            val = to_numeric_sipri(raw.iat[i, 2 + j])
            if np.isfinite(val):
                records.append({"sipri_name": name, "Year": yr, "milex_gdp": val})
    return pd.DataFrame(records)


def load_owid() -> pd.DataFrame:
    path = DATA_DIR / "military-spending-as-a-share-of-gdp-sipri.csv"
    ow = pd.read_csv(path)
    ow = ow.rename(columns={
        "Entity": "owid_name",
        "Year": "Year",
        "Military expenditure (% of GDP)": "milex_gdp_pct",
    })
    ow["Year"] = pd.to_numeric(ow["Year"], errors="coerce")
    ow["milex_gdp"] = pd.to_numeric(ow["milex_gdp_pct"], errors="coerce") / 100.0
    return ow.dropna(subset=["milex_gdp"])[["owid_name", "Year", "milex_gdp"]]


def load_wb() -> pd.DataFrame:
    path = DATA_DIR / "wb_milex.json"
    with path.open() as f:
        blob = json.load(f)
    recs = []
    for r in blob[1]:
        if r.get("value") is None:
            continue
        recs.append({
            "wb_name": r["country"]["value"],
            "iso3": r.get("countryiso3code"),
            "Year": int(r["date"]),
            "milex_gdp": float(r["value"]) / 100.0,  # WB is percent
        })
    return pd.DataFrame(recs)


def load_vdem() -> pd.DataFrame:
    v = pd.read_csv(VDEM_PATH)
    v["year"] = pd.to_numeric(v["year"], errors="coerce")
    v = v.rename(columns={"year": "Year", "country_name": "vdem_name"})
    return v


def map_name(soc: str, table: dict) -> str:
    return table.get(soc, soc)


def shield_panel(juno: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Society", "Year", "Bond_Strength_Calc", "Bond_Strength_Calc_legacy",
        "SBD_Calc", "Lambda2_Calc", "Phase_Calc", "Node_Value_Calc",
    ]
    sh = juno[juno["Node"] == "Shield"][cols].copy()
    sh["Year"] = sh["Year"].astype(int)
    return sh


def merge_ext(shield: pd.DataFrame, ext: pd.DataFrame, name_col: str, table: dict) -> pd.DataFrame:
    s = shield.copy()
    s["ext_name"] = s["Society"].map(lambda x: map_name(x, table))
    m = s.merge(ext, left_on=["ext_name", "Year"], right_on=[name_col, "Year"], how="inner")
    return m


def corr_pair(df: pd.DataFrame, xcol: str, ycol: str, society_col: str = "Society"):
    xs = pearson(df[xcol], df[ycol])
    d = demean_within(df[[society_col, xcol, ycol]].dropna(), society_col, [xcol, ycol])
    ws = pearson(d[xcol + "_dm"], d[ycol + "_dm"])
    return {
        "cross_r": xs["r"], "cross_p": xs["p"], "cross_n": xs["n"],
        "within_r": ws["r"], "within_p": ws["p"], "within_n": ws["n"],
        "n_soc": int(df[society_col].nunique()),
    }


def q_i(C, A):
    return (0.6 * C + 0.4 * A) / 10.0


def w_i(C, A, S):
    q = q_i(C, A)
    if q <= 0:
        return 0.0
    return math.sqrt(q) * (2.0 ** (-S / 10.0))


def pairwise_bond(row_i, row_j):
    wi = w_i(row_i["Coherence"], row_i["Abstraction"], row_i["Stress"])
    wj = w_i(row_j["Coherence"], row_j["Abstraction"], row_j["Stress"])
    b = wi * wj
    return min(1.0, max(0.0, b))


def add_row(rows, test, metric, value, p="", n="", note=""):
    rows.append({
        "test": test,
        "metric": metric,
        "value": value if value == value else "",
        "p": p,
        "n": n,
        "note": note,
    })


def main() -> int:
    ensure_external_files()
    juno = load_juno()
    socs = original36(juno)
    print(f"JUNO_36 societies: {len(socs)}")
    print(" ", ", ".join(socs))
    print(f" rows={len(juno)}  society-years={juno.groupby(['Society','Year']).ngroups}")

    vintage_path = DATA_DIR / "JUNO_phaseA_vintage.csv"
    if vintage_path.exists():
        old = pd.read_csv(vintage_path, low_memory=False)
        key = ["Society", "Year", "Node"]
        o = old[key + ["Bond_Strength_Calc"]].rename(columns={"Bond_Strength_Calc": "bond_phaseA"})
        n = juno[key + ["Bond_Strength_Calc", "Bond_Strength_Calc_legacy"]]
        vm = o.merge(n, on=key)
        shv = vm[vm.Node == "Shield"]
        print("Phase A vintage vs current (original 36 node-rows):", len(vm))
        print(f"  max|phaseA-aligned|={((vm.bond_phaseA-vm.Bond_Strength_Calc).abs().max()):.2e}  "
              f"frac>|1e-4|={((vm.bond_phaseA-vm.Bond_Strength_Calc).abs()>1e-4).mean():.4f}")
        print(f"  Shield frac phaseA!=legacy (>1e-4)={((shv.bond_phaseA-shv.Bond_Strength_Calc_legacy).abs()>1e-4).mean():.4f}")

    shield = shield_panel(juno)
    rows = []

    # ------------------------------------------------------------------
    # T1 SIPRI / WB
    # ------------------------------------------------------------------
    sipri = load_sipri_share_of_gdp()
    print(f"\nSIPRI Share of GDP long: {len(sipri)} country-years")
    m_sip = merge_ext(shield, sipri, "sipri_name", SIPRI_NAME)
    unmatched = sorted(set(socs) - set(m_sip["Society"].unique()))
    print(f"SIPRI merge: n={len(m_sip)} societies={m_sip.Society.nunique()} unmatched={unmatched}")
    print(f"  years {m_sip.Year.min()}-{m_sip.Year.max()}")

    for bond_col, tag in [
        ("Bond_Strength_Calc", "aligned"),
        ("Bond_Strength_Calc_legacy", "legacy"),
    ]:
        r = corr_pair(m_sip, bond_col, "milex_gdp")
        print(f"  T1 SIPRI {tag}: cross r={r['cross_r']:.4f} p={fmt_p(r['cross_p'])} n={r['cross_n']} "
              f"within r={r['within_r']:.4f} p={fmt_p(r['within_p'])} n_soc={r['n_soc']}")
        add_row(rows, f"T1_SIPRI_Shield_{tag}", "pearson_r_cross_sectional",
                round(r["cross_r"], 4), fmt_p(r["cross_p"]), r["cross_n"],
                f"n_soc={r['n_soc']}")
        add_row(rows, f"T1_SIPRI_Shield_{tag}", "pearson_r_within_society",
                round(r["within_r"], 4), fmt_p(r["within_p"]), r["within_n"],
                f"n_soc={r['n_soc']}")
        add_row(rows, f"T1_SIPRI_Shield_{tag}", "n_societies", r["n_soc"], "", r["cross_n"])

    # OWID as robustness (same SIPRI, different packaging)
    try:
        owid = load_owid()
        m_ow = merge_ext(shield, owid, "owid_name", OWID_NAME)
        print(f"OWID merge: n={len(m_ow)} societies={m_ow.Society.nunique()}")
        for bond_col, tag in [
            ("Bond_Strength_Calc", "aligned"),
            ("Bond_Strength_Calc_legacy", "legacy"),
        ]:
            r = corr_pair(m_ow, bond_col, "milex_gdp")
            print(f"  T1 OWID {tag}: cross r={r['cross_r']:.4f} p={fmt_p(r['cross_p'])} n={r['cross_n']} "
                  f"within r={r['within_r']:.4f} p={fmt_p(r['within_p'])}")
            add_row(rows, f"T1_OWID_SIPRI_Shield_{tag}", "pearson_r_cross_sectional",
                    round(r["cross_r"], 4), fmt_p(r["cross_p"]), r["cross_n"])
            add_row(rows, f"T1_OWID_SIPRI_Shield_{tag}", "pearson_r_within_society",
                    round(r["within_r"], 4), fmt_p(r["within_p"]), r["within_n"])
    except Exception as e:
        print("OWID skip:", e)

    wb = load_wb()
    m_wb = merge_ext(shield, wb, "wb_name", WB_NAME)
    print(f"WB merge: n={len(m_wb)} societies={m_wb.Society.nunique()} "
          f"years {m_wb.Year.min()}-{m_wb.Year.max()}")
    unmatched_wb = sorted(set(socs) - set(m_wb["Society"].unique()))
    print(f"  unmatched={unmatched_wb}")
    for bond_col, tag in [
        ("Bond_Strength_Calc", "aligned"),
        ("Bond_Strength_Calc_legacy", "legacy"),
    ]:
        r = corr_pair(m_wb, bond_col, "milex_gdp")
        print(f"  T1 WB {tag}: cross r={r['cross_r']:.4f} p={fmt_p(r['cross_p'])} n={r['cross_n']} "
              f"within r={r['within_r']:.4f} p={fmt_p(r['within_p'])} n_soc={r['n_soc']}")
        add_row(rows, f"T1_WB_Shield_{tag}", "pearson_r_cross_sectional",
                round(r["cross_r"], 4), fmt_p(r["cross_p"]), r["cross_n"],
                f"n_soc={r['n_soc']}")
        add_row(rows, f"T1_WB_Shield_{tag}", "pearson_r_within_society",
                round(r["within_r"], 4), fmt_p(r["within_p"]), r["within_n"],
                f"n_soc={r['n_soc']}")

    # ------------------------------------------------------------------
    # T2 Hands rank / H-S bond (mostly Node_Value / pairwise, not Bond_Strength)
    # ------------------------------------------------------------------
    post = juno[(juno["Year"] >= 1990) & (juno["Year"] <= 2026)].copy()
    # rank by Node_Value within society-year; 1=lowest as original
    post["nv_rank"] = post.groupby(["Society", "Year"])["Node_Value_Calc"].rank(
        method="average", ascending=True
    )
    hands = post[post["Node"] == "Hands"]
    stewards = post[post["Node"] == "Stewards"]
    print(f"\nT2 post-1990 society-years Hands n={len(hands)} soc={hands.Society.nunique()}")
    print(f"  Hands mean rank (1=lowest): {hands.nv_rank.mean():.3f}")
    print(f"  Stewards mean rank: {stewards.nv_rank.mean():.3f}")
    add_row(rows, "T2_ILO_HandsStewards", "hands_mean_rank_of_8",
            round(float(hands.nv_rank.mean()), 3), "", len(hands), "Node_Value rank; unaffected")
    add_row(rows, "T2_ILO_HandsStewards", "stewards_mean_rank_of_8",
            round(float(stewards.nv_rank.mean()), 3), "", len(stewards), "Node_Value rank; unaffected")

    # Pairwise Hands-Stewards v1.2 bond vs SBD
    idx = post.set_index(["Society", "Year", "Node"])
    hs_recs = []
    for (soc, yr), g in post.groupby(["Society", "Year"]):
        try:
            h = idx.loc[(soc, yr, "Hands")]
            s = idx.loc[(soc, yr, "Stewards")]
        except KeyError:
            continue
        b_hs = pairwise_bond(h, s)
        sbd = float(h["SBD_Calc"])
        hs_recs.append({
            "Society": soc, "Year": int(yr), "b_hs": b_hs, "sbd": sbd,
            "bond_h_al": float(h["Bond_Strength_Calc"]),
            "bond_s_al": float(s["Bond_Strength_Calc"]),
            "bond_h_lg": float(h["Bond_Strength_Calc_legacy"]),
            "bond_s_lg": float(s["Bond_Strength_Calc_legacy"]),
        })
    hs = pd.DataFrame(hs_recs)
    hs["hs_vs_sbd"] = hs["b_hs"] / hs["sbd"]
    hs["hs_mean_al"] = 0.5 * (hs["bond_h_al"] + hs["bond_s_al"])
    hs["hs_mean_lg"] = 0.5 * (hs["bond_h_lg"] + hs["bond_s_lg"])
    print(f"  pairwise B_HS / SBD mean: {hs.hs_vs_sbd.mean():.3f}  (published 0.88)")
    print(f"  pairwise B_HS mean: {hs.b_hs.mean():.4f}")
    print(f"  mean of Hands+Stewards Bond_Strength aligned/SBD: {(hs.hs_mean_al/hs.sbd).mean():.3f}")
    print(f"  mean of Hands+Stewards Bond_Strength legacy/SBD: {(hs.hs_mean_lg/hs.sbd).mean():.3f}")
    d90 = hs[(hs.Year >= 1990) & (hs.Year <= 1999)]
    d20 = hs[(hs.Year >= 2020) & (hs.Year <= 2026)]
    print(f"  B_HS 1990s mean={d90.b_hs.mean():.4f}  2020s mean={d20.b_hs.mean():.4f}")
    print(f"  NV Hands 1990s={post[(post.Node=='Hands')&(post.Year.between(1990,1999))].Node_Value_Calc.mean():.2f}"
          f"  2020s={post[(post.Node=='Hands')&(post.Year.between(2020,2026))].Node_Value_Calc.mean():.2f}")
    add_row(rows, "T2_ILO_HandsStewards", "hs_pairwise_vs_sbd",
            round(float(hs.hs_vs_sbd.mean()), 3), "", len(hs), "v1.2 pairwise B_ij / SBD")
    add_row(rows, "T2_ILO_HandsStewards", "hs_meanbond_aligned_vs_sbd",
            round(float((hs.hs_mean_al / hs.sbd).mean()), 3), "", len(hs))
    add_row(rows, "T2_ILO_HandsStewards", "hs_meanbond_legacy_vs_sbd",
            round(float((hs.hs_mean_lg / hs.sbd).mean()), 3), "", len(hs))

    # ------------------------------------------------------------------
    # T3 V-Dem
    # ------------------------------------------------------------------
    vdem = load_vdem()
    sh = shield.copy()
    sh["vdem_name"] = sh["Society"].map(lambda x: map_name(x, VDEM_NAME))
    mv_all = sh.merge(vdem, left_on=["vdem_name", "Year"], right_on=["vdem_name", "Year"], how="inner")
    print(f"\nT3 V-Dem merge: n={len(mv_all)} soc={mv_all.Society.nunique()} years {mv_all.Year.min()}-{mv_all.Year.max()}")
    unmatched_v = sorted(set(socs) - set(mv_all["Society"].unique()))
    print(f"  unmatched={unmatched_v}")
    # Published T3c used the v2x_libdem-complete panel (n=3806)
    mv = mv_all.dropna(subset=["v2x_libdem"]).copy()
    print(f"  libdem-complete panel n={len(mv)} (published n=3806)")

    indicators = ["v2x_rule", "v2x_neopat", "v2x_liberal", "v2x_civlib", "v2x_libdem"]
    for bond_col, tag in [
        ("Bond_Strength_Calc", "aligned"),
        ("Bond_Strength_Calc_legacy", "legacy"),
    ]:
        print(f"  --- Shield Bond {tag} ---")
        for ind in indicators:
            sub = mv.dropna(subset=[bond_col, ind])
            r = corr_pair(sub, bond_col, ind)
            print(f"    {ind}: cross r={r['cross_r']:.4f} p={fmt_p(r['cross_p'])} n={r['cross_n']} "
                  f"within r={r['within_r']:.4f} p={fmt_p(r['within_p'])}")
            add_row(rows, f"T3c_VDem_Shield_{tag}", f"r_{ind}_crosssec",
                    round(r["cross_r"], 4), fmt_p(r["cross_p"]), r["cross_n"])
            add_row(rows, f"T3c_VDem_Shield_{tag}", f"r_{ind}_within",
                    round(r["within_r"], 4), fmt_p(r["within_p"]), r["within_n"])

        # Australia depth
        aus = mv[mv.Society == "Australia"].dropna(subset=[bond_col])
        print(f"    Australia n={len(aus)}")
        for ind in ["v2x_liberal", "v2x_neopat", "v2x_rule", "v2xnp_client"]:
            sub = aus.dropna(subset=[ind])
            rr = pearson(sub[bond_col], sub[ind])
            print(f"      AU {ind}: r={rr['r']:.4f} n={rr['n']}")
            add_row(rows, f"T3a_Australia_Shield_{tag}", f"r_{ind}",
                    round(rr["r"], 4) if rr["r"] == rr["r"] else "", fmt_p(rr["p"]), rr["n"])

        # per-society r vs v2x_libdem
        soc_r = []
        for soc, g in mv.groupby("Society"):
            g = g.dropna(subset=[bond_col, "v2x_libdem"])
            rr = pearson(g[bond_col], g["v2x_libdem"])
            if rr["n"] >= 5 and rr["r"] == rr["r"]:
                soc_r.append((soc, rr["r"], rr["p"], rr["n"]))
        soc_r.sort(key=lambda t: -t[1])
        rs = [t[1] for t in soc_r]
        n_pos = sum(1 for x in rs if x > 0)
        n_sig = sum(1 for t in soc_r if t[2] < 0.05)
        print(f"    per-soc libdem: median r={np.median(rs):.3f} frac_pos={n_pos/len(rs):.3f} "
              f"frac_p<0.05={n_sig/len(rs):.3f} n_soc={len(rs)}")
        print("      top+:", ", ".join(f"{s} {r:+.2f}" for s, r, p, n in soc_r[:7]))
        print("      bottom-:", ", ".join(f"{s} {r:+.2f}" for s, r, p, n in soc_r[-5:]))
        add_row(rows, f"T3c_VDem_Shield_{tag}", "median_within_soc_r_libdem",
                round(float(np.median(rs)), 3), "", len(rs))
        add_row(rows, f"T3c_VDem_Shield_{tag}", "frac_positive_soc_r_libdem",
                round(n_pos / len(rs), 3), "", len(rs))
        add_row(rows, f"T3c_VDem_Shield_{tag}", "frac_p05_soc_r_libdem",
                round(n_sig / len(rs), 3), "", len(rs))
        for s, r, p, n in soc_r:
            add_row(rows, f"T3c_per_society_{tag}", f"r_libdem_{s}",
                    round(r, 3), fmt_p(p), n)

    # T3b Lambda2 by Phase — should match published (unaffected)
    # Use year-level (one row per society-year); Shield row is fine (Lambda2 is year-level)
    lam = shield.dropna(subset=["Lambda2_Calc", "Phase_Calc"]).copy()
    lam["phase_i"] = pd.to_numeric(lam["Phase_Calc"], errors="coerce").round().astype("Int64")
    lam = lam.dropna(subset=["phase_i"]).copy()
    lam["phase_i"] = lam["phase_i"].astype(int)
    print("\nT3b Lambda2 by Phase (year-level, should match published):")
    groups = []
    for ph in range(1, 7):
        sub = lam[lam.phase_i == ph]["Lambda2_Calc"]
        print(f"  Phase {ph}: mean={sub.mean():.3f} std={sub.std():.3f} n={len(sub)}")
        groups.append(sub.values)
        add_row(rows, "T3b_Lambda2_by_Phase", f"lambda2_phase{ph}",
                round(float(sub.mean()), 3), "", len(sub), "unaffected")
    F, p = stats.f_oneway(*groups)
    print(f"  ANOVA F={F:.1f} p={fmt_p(p)} n={len(lam)}")
    add_row(rows, "T3b_Lambda2_by_Phase", "ANOVA_F", round(float(F), 1), fmt_p(p), len(lam), "unaffected")

    # T3c ANOVA libdem by Phase (unaffected — uses Phase not Bond)
    lib = mv.dropna(subset=["v2x_libdem", "Phase_Calc"]).copy()
    lib["phase_i"] = pd.to_numeric(lib["Phase_Calc"], errors="coerce").round().astype("Int64")
    lib = lib.dropna(subset=["phase_i"]).copy()
    lib["phase_i"] = lib["phase_i"].astype(int)
    print("T3c v2x_libdem by Phase (should match published):")
    groups = []
    for ph in range(1, 7):
        sub = lib[lib.phase_i == ph]["v2x_libdem"]
        print(f"  Phase {ph}: mean={sub.mean():.3f} n={len(sub)}")
        groups.append(sub.values)
        add_row(rows, "T3c_libdem_by_Phase", f"phase{ph}_mean_libdem",
                round(float(sub.mean()), 3), "", len(sub), "unaffected")
    F, p = stats.f_oneway(*groups)
    print(f"  ANOVA F={F:.1f} p={fmt_p(p)} n={len(lib)}")
    add_row(rows, "T3c_libdem_by_Phase", "ANOVA_F", round(float(F), 1), fmt_p(p), len(lib), "unaffected")

    # Lambda2 x libdem within-society (unaffected)
    d = demean_within(mv.dropna(subset=["Lambda2_Calc", "v2x_libdem"]), "Society",
                      ["Lambda2_Calc", "v2x_libdem"])
    rr = pearson(d["Lambda2_Calc_dm"], d["v2x_libdem_dm"])
    print(f"  Lambda2 x libdem within: r={rr['r']:.4f} p={fmt_p(rr['p'])} n={rr['n']}")
    add_row(rows, "T3c_Lambda2_libdem", "r_within", round(rr["r"], 4), fmt_p(rr["p"]), rr["n"],
            "unaffected")

    # ------------------------------------------------------------------
    # T5 France Archive / Denmark triangle (Node_Value / pairwise; not Bond-by-node)
    # ------------------------------------------------------------------
    fr = juno[(juno.Society == "France") & (juno.Year >= 1800) & (juno.Year <= 2026)]
    fr_arch = fr[fr.Node == "Archive"]
    sys_mean = fr.groupby("Year")["Node_Value_Calc"].mean()
    arch_mean = fr_arch.set_index("Year")["Node_Value_Calc"]
    premium = (arch_mean - sys_mean).mean()
    # rank 1=top for T5
    fr = fr.copy()
    fr["rank_top"] = fr.groupby("Year")["Node_Value_Calc"].rank(method="min", ascending=False)
    arch_top3 = (fr[fr.Node == "Archive"]["rank_top"] <= 3).mean()
    print(f"\nT5 France Archive 1800-2026: NV mean={fr_arch.Node_Value_Calc.mean():.2f} "
          f"sys mean={fr.Node_Value_Calc.mean():.2f} premium={premium:.3f} top3={100*arch_top3:.1f}%")
    add_row(rows, "T5_France_Archive", "archive_premium_over_system_1800_2026",
            round(float(premium), 3), "", fr_arch.Year.nunique(), "Node_Value; unaffected")
    add_row(rows, "T5_France_Archive", "archive_pct_years_top3",
            round(float(100 * arch_top3), 1), "", fr_arch.Year.nunique(), "Node_Value; unaffected")

    dk = juno[(juno.Society == "Denmark") & (juno.Year >= 1900)]
    print("T5 Denmark post-1900 mean NV / rank(1=top):")
    dk2 = dk.copy()
    dk2["rank_top"] = dk2.groupby("Year")["Node_Value_Calc"].rank(method="average", ascending=False)
    for node in ["Helm", "Craft", "Stewards"]:
        sub = dk2[dk2.Node == node]
        print(f"  {node}: NV={sub.Node_Value_Calc.mean():.2f} rank={sub.rank_top.mean():.2f}")

    # pairwise v1.2 bonds Denmark
    dki = dk.set_index(["Year", "Node"])
    tri = {"Helm-Stewards": [], "Helm-Craft": [], "Stewards-Craft": [],
           "Hands-Helm": [], "Hands-Craft": []}
    years = sorted(dk.Year.unique())
    for yr in years:
        try:
            nodes = {n: dki.loc[(yr, n)] for n in
                     ["Helm", "Stewards", "Craft", "Hands"]}
        except KeyError:
            continue
        tri["Helm-Stewards"].append(pairwise_bond(nodes["Helm"], nodes["Stewards"]))
        tri["Helm-Craft"].append(pairwise_bond(nodes["Helm"], nodes["Craft"]))
        tri["Stewards-Craft"].append(pairwise_bond(nodes["Stewards"], nodes["Craft"]))
        tri["Hands-Helm"].append(pairwise_bond(nodes["Hands"], nodes["Helm"]))
        tri["Hands-Craft"].append(pairwise_bond(nodes["Hands"], nodes["Craft"]))
    hsc = np.mean(tri["Helm-Stewards"] + tri["Helm-Craft"] + tri["Stewards-Craft"])
    lab = np.mean(tri["Hands-Helm"] + tri["Hands-Craft"])
    print(f"  v1.2 pairwise HSC mean={hsc:.4f} labour periphery={lab:.4f} (scale 0-1; published used different raw formula ~47)")
    add_row(rows, "T5_Denmark_HSC", "hsc_triangle_mean_bond_v12",
            round(float(hsc), 4), "", len(years), "v1.2 pairwise; not the published ~48.6 scale")
    add_row(rows, "T5_Denmark_HSC", "labour_periphery_mean_bond_v12",
            round(float(lab), 4), "", len(years))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
