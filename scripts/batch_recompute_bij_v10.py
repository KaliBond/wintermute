#!/usr/bin/env python3
"""
batch_recompute_bij_v10.py
==========================
Batch-recompute all eligible CAMS CSV series under the canonical v1.0-Final
B_ij formula (cams_framework_v2_4.py):

    V_i  = C + K - S + 0.5*A
    q_i  = (0.6*C + 0.4*A) / 10
    B_ij = sqrt(q_i * q_j) * 2**(-(S_i + S_j)/10)   ∈ [0, 1]  (for C,A,S ∈ [1,10])
    per-node BS = mean of 7 off-diagonal edges

Writes:
  data/v1.0_final_recompute/series/*.csv   — recomputed node-year tables
  data/v1.0_final_recompute/FG_corpus.csv  — F_G + regime per society-year
  data/v1.0_final_recompute/RECOMPUTE_REPORT.md
  data/v1.0_final_recompute/manifest.json

Run from wintermute/:
    python scripts/batch_recompute_bij_v10.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cams_framework_v2_4 import (  # noqa: E402
    VERSION,
    compute_derived_columns,
    phase_state,
    classify_regime,
    NODES,
)

OUT_DIR = ROOT / "data" / "v1.0_final_recompute"
SERIES_DIR = OUT_DIR / "series"
REPORT_PATH = OUT_DIR / "RECOMPUTE_REPORT.md"
FG_PATH = OUT_DIR / "FG_corpus.csv"
MANIFEST_PATH = OUT_DIR / "manifest.json"

STANDARD_NODES = set(NODES)

# Preference rank when multiple legacy labels map to the same CAMS node
# (lower = preferred). Used when a file has both "Trades/Professions" and
# "Trades/Prof." in the same year.
NODE_PREFERENCE = {
    "helm": 0,
    "executive": 1,
    "shield": 0,
    "army": 1,
    "military": 2,
    "lore": 0,
    "priests": 1,
    "priesthood / knowledge workers": 1,
    "priesthood": 2,
    "knowledge workers": 2,
    "stewards": 0,
    "property": 1,
    "property owners": 1,
    "craft": 0,
    "trades/professions": 1,
    "trades / professions": 1,
    "trades/prof.": 2,
    "trades professions": 2,
    "trades /": 3,
    "tradesprofessions": 3,
    "hands": 0,
    "proletariat": 1,
    "labour": 2,
    "archive": 0,
    "state memory": 1,
    "statememory": 1,
    "flow": 0,
    "shopkeepers/merchants": 1,
    "shopkeepers / merchants": 1,
    "merchants / shopkeepers": 1,
    "merchants": 2,
    "shopkeepers": 2,
    "commerce": 3,
}

NODE_MAP = {
    "executive": "Helm",
    "helm": "Helm",
    "army": "Shield",
    "shield": "Shield",
    "military": "Shield",
    "priests": "Lore",
    "priesthood / knowledge workers": "Lore",
    "priesthood": "Lore",
    "knowledge workers": "Lore",
    "lore": "Lore",
    "property owners": "Stewards",
    "property": "Stewards",
    "stewards": "Stewards",
    "trades/professions": "Craft",
    "trades / professions": "Craft",
    "trades/prof.": "Craft",
    "trades professions": "Craft",
    "trades /": "Craft",
    "tradesprofessions": "Craft",
    "craft": "Craft",
    "proletariat": "Hands",
    "hands": "Hands",
    "labour": "Hands",
    "state memory": "Archive",
    "statememory": "Archive",
    "archive": "Archive",
    "shopkeepers/merchants": "Flow",
    "shopkeepers / merchants": "Flow",
    "merchants / shopkeepers": "Flow",
    "merchants": "Flow",
    "shopkeepers": "Flow",
    "flow": "Flow",
    "commerce": "Flow",
}

RAW_COLS = ["Coherence", "Capacity", "Stress", "Abstraction"]


def map_node(raw) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s in STANDARD_NODES:
        return s
    return NODE_MAP.get(s.lower())


def node_pref(raw) -> int:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 99
    s = str(raw).strip()
    if s in STANDARD_NODES:
        return 0
    return NODE_PREFERENCE.get(s.lower(), 50)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace("_", " ").strip()
        if cl == "nation":
            col_map[col] = "Society"
        elif cl in ("node value", "nodevalue"):
            col_map[col] = "Node Value"
        elif cl in ("bond strength", "bondstrength"):
            col_map[col] = "Bond Strength"
        elif cl == "node":
            col_map[col] = "Node"
        elif cl == "year":
            col_map[col] = "Year"
        elif cl == "society":
            col_map[col] = "Society"
        elif cl == "coherence":
            col_map[col] = "Coherence"
        elif cl == "capacity":
            col_map[col] = "Capacity"
        elif cl == "stress":
            col_map[col] = "Stress"
        elif cl == "abstraction":
            col_map[col] = "Abstraction"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def society_key_from_filename(name: str) -> str:
    stem = Path(name).stem
    for token in (
        "_ENS_",
        "_ENV_",
        "_ENS",
        "_ENV",
        "_cleaned",
        "_CAMS5_ensemble",
        "_CAMS5_calc",
        "_cam5_",
        "_CAMS_",
        "_CAMS",
    ):
        if token in stem:
            stem = stem.split(token)[0]
            break
    parts = stem.split("_")
    keep = []
    for p in parts:
        if p.isdigit() and len(p) == 4:
            break
        keep.append(p)
    key = "_".join(keep) if keep else stem
    return key.lower().replace(" ", "_")


def safe_slug(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_")


def prepare_series(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Return raw 8-node long table ready for compute_derived_columns."""
    df = normalise_columns(df)

    if "Society" not in df.columns:
        df.insert(0, "Society", society_key_from_filename(source_name))

    missing = [c for c in ["Year", "Node"] + RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.copy()
    df["_raw_node"] = df["Node"]
    df["_pref"] = df["Node"].map(node_pref)
    df["Node"] = df["Node"].map(map_node)
    df = df.dropna(subset=["Node"])

    for c in RAW_COLS + ["Year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=RAW_COLS + ["Year", "Society"])
    df["Year"] = df["Year"].astype(int)
    df["Society"] = df["Society"].astype(str).str.strip()

    # Prefer better node aliases, then last row
    df = df.sort_values(["Society", "Year", "Node", "_pref"])
    df = df.drop_duplicates(subset=["Society", "Year", "Node"], keep="first")

    # Keep only complete 8-node society-years
    counts = df.groupby(["Society", "Year"])["Node"].nunique()
    good = counts[counts == 8].index
    df = df.set_index(["Society", "Year"]).loc[good].reset_index()

    if df.empty:
        raise ValueError("no complete 8-node society-years after mapping")

    # If multi-society file, keep the dominant society only (unless multi is intentional)
    societies = df["Society"].unique()
    if len(societies) > 1:
        # Prefer the society named in the filename when present
        key = society_key_from_filename(source_name).replace("_", " ").lower()
        match = [s for s in societies if key in s.lower() or s.lower() in key]
        if match:
            df = df[df["Society"] == match[0]]
        else:
            # keep largest
            top = df["Society"].value_counts().index[0]
            df = df[df["Society"] == top]

    out = df[["Society", "Year", "Node"] + RAW_COLS].copy()
    counts = out.groupby(["Society", "Year"])["Node"].nunique()
    good = counts[counts == 8].index
    out = out.set_index(["Society", "Year"]).loc[good].reset_index()
    if out.empty:
        raise ValueError("no complete rows after society filter")
    return out


def score_range_tier(df: pd.DataFrame) -> str:
    """canonical_range if all C/K/S/A in [1,10], else dev_range."""
    ok = True
    for c in RAW_COLS:
        s = df[c]
        if (s < 1).any() or (s > 10).any():
            ok = False
            break
    return "in_range" if ok else "out_of_range"


def collect_sources() -> list[dict]:
    sources: list[dict] = []

    nations = sorted((ROOT / "data" / "nations").glob("*_ENS.csv"))
    for p in nations:
        sources.append(
            {
                "path": p,
                "key": society_key_from_filename(p.name),
                "tier": "canonical_ens",
                "priority": 1,
            }
        )

    cleaned = ROOT / "cleaned_datasets"
    if cleaned.exists():
        ens = sorted(cleaned.glob("*_ENS_*.csv"))
        other = sorted(
            p
            for p in cleaned.glob("*.csv")
            if "_ENV_" not in p.name and "_ENS_" not in p.name
        )
        nations_keys = {s["key"] for s in sources}
        for p in ens:
            key = society_key_from_filename(p.name)
            if key in nations_keys:
                # already covered by data/nations ENS — skip duplicate
                continue
            sources.append(
                {
                    "path": p,
                    "key": key,
                    "tier": "cleaned_ens",
                    "priority": 2,
                }
            )
            nations_keys.add(key)

        for p in other:
            key = society_key_from_filename(p.name)
            # Always include as separate series file (unique by source stem)
            sources.append(
                {
                    "path": p,
                    "key": key,
                    "tier": "usp_or_legacy",
                    "priority": 3,
                }
            )

    return sources


def recompute_one(src: dict) -> dict:
    path: Path = src["path"]
    df_raw = pd.read_csv(path)
    prepared = prepare_series(df_raw, path.name)
    range_tier = score_range_tier(prepared)
    scored = compute_derived_columns(prepared)

    society = str(scored["Society"].iloc[0])
    out_name = f"{safe_slug(path.stem)}_Bij_v10.csv"
    out_path = SERIES_DIR / out_name
    scored = scored.sort_values(["Society", "Year", "Node"])
    cols = ["Society", "Year", "Node"] + RAW_COLS + ["Node Value", "Bond Strength"]
    scored[cols].to_csv(out_path, index=False)

    fg_rows = []
    for (soc, year), g in scored.groupby(["Society", "Year"]):
        g = g.set_index("Node").reindex(NODES)
        if g[RAW_COLS].isna().any().any():
            continue
        C = g["Coherence"].values.astype(float)
        K = g["Capacity"].values.astype(float)
        S = g["Stress"].values.astype(float)
        A = g["Abstraction"].values.astype(float)
        phi = phase_state(C, K, S, A)
        regime = classify_regime(phi)
        fg_rows.append(
            {
                "Society": soc,
                "Year": int(year),
                "series_file": out_name,
                "tier": src["tier"],
                "range_tier": range_tier,
                "source": path.name,
                "V_mean": phi["V_mean"],
                "V_std": phi["V_std"],
                "V_min": phi["V_min"],
                "B_mean": phi["B_mean"],
                "lambda2": phi["lambda2"],
                "sigma_min": phi["sigma_min"],
                "regime": regime,
            }
        )

    bs = scored["Bond Strength"]
    nv = scored["Node Value"]
    return {
        "key": src["key"],
        "society": society,
        "tier": src["tier"],
        "range_tier": range_tier,
        "source": str(path.relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
        "out_name": out_name,
        "rows": int(len(scored)),
        "years": int(scored["Year"].nunique()),
        "year_min": int(scored["Year"].min()),
        "year_max": int(scored["Year"].max()),
        "bs_min": float(bs.min()),
        "bs_max": float(bs.max()),
        "nv_min": float(nv.min()),
        "nv_max": float(nv.max()),
        "fg_rows": fg_rows,
        "ok": True,
        "error": None,
    }


def main() -> int:
    # Clean previous series outputs for a fresh write
    if SERIES_DIR.exists():
        for old in SERIES_DIR.glob("*.csv"):
            old.unlink()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)

    sources = collect_sources()
    print(f"CAMS Framework v{VERSION} — batch recompute under canonical B_ij")
    print(f"Sources queued: {len(sources)}")
    print(f"Output: {OUT_DIR}")

    results = []
    all_fg = []
    errors = []

    for i, src in enumerate(sources, 1):
        try:
            r = recompute_one(src)
            results.append(r)
            all_fg.extend(r["fg_rows"])
            flag = "OK " if r["range_tier"] == "in_range" else "DEV"
            print(
                f"[{i:03d}/{len(sources)}] {flag} {r['society']:28s} "
                f"years={r['years']:4d}  BS=[{r['bs_min']:.3f},{r['bs_max']:.3f}]  "
                f"← {src['path'].name}"
            )
        except Exception as e:
            err = {
                "key": src["key"],
                "source": str(src["path"]),
                "tier": src["tier"],
                "ok": False,
                "error": str(e),
            }
            errors.append(err)
            results.append(err)
            print(f"[{i:03d}/{len(sources)}] ERR {src['path'].name}: {e}")

    if all_fg:
        fg_df = pd.DataFrame(all_fg).sort_values(["Society", "Year", "series_file"])
        fg_df.to_csv(FG_PATH, index=False)
    else:
        fg_df = pd.DataFrame()

    ok_results = [r for r in results if r.get("ok")]
    in_range = [r for r in ok_results if r.get("range_tier") == "in_range"]
    out_range = [r for r in ok_results if r.get("range_tier") == "out_of_range"]
    n_ok = len(ok_results)
    n_err = len(errors)
    n_sy = len(fg_df) if len(fg_df) else 0
    n_sy_in = int((fg_df["range_tier"] == "in_range").sum()) if n_sy else 0

    # Primary comparison corpus: one series per society from canonical ENS
    # (priority 1/2), in-range only
    primary = [r for r in ok_results if r["tier"] in ("canonical_ens", "cleaned_ens") and r["range_tier"] == "in_range"]

    manifest = {
        "formula_version": VERSION,
        "formula": {
            "V_i": "C + K - S + 0.5*A",
            "q_i": "(0.6*C + 0.4*A) / 10",
            "B_ij": "sqrt(q_i*q_j) * 2**(-(S_i+S_j)/10)",
            "per_node_BS": "mean of 7 off-diagonal B_ij",
            "T_ij": "fully connected prior (T=1)",
            "laplacian": "raw L = D - W",
            "source_doc": "papers/CAMS_v1.0-Final_Framework_Conclusion.md",
        },
        "n_series_ok": n_ok,
        "n_series_in_range": len(in_range),
        "n_series_out_of_range": len(out_range),
        "n_series_error": n_err,
        "n_primary_ens": len(primary),
        "n_society_years": n_sy,
        "n_society_years_in_range": n_sy_in,
        "series": [{k: v for k, v in r.items() if k != "fg_rows"} for r in results],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# CAMS v1.0-Final Batch Recompute Report",
        "",
        f"**Engine:** `cams_framework_v2_4.py` (v{VERSION})",
        f"**Source doc:** `papers/CAMS_v1.0-Final_Framework_Conclusion.md`",
        "",
        "## Canonical operators",
        "",
        "```",
        "V_i  = C + K - S + 0.5*A",
        "q_i  = (0.6*C + 0.4*A) / 10",
        "B_ij = sqrt(q_i * q_j) * 2**(-(S_i + S_j)/10)   ∈ [0, 1]  for scores in [1,10]",
        "BS_i = mean_j≠i B_ij",
        "F_G  = (V̄, V_std, V_min, B̄, λ₂, σ_min)",
        "```",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Series recomputed OK | {n_ok} |",
        f"| In-range (C/K/S/A ∈ [1,10]) | {len(in_range)} |",
        f"| Out-of-range (DEV / legacy Stress) | {len(out_range)} |",
        f"| Primary ENS corpus (cross-society) | {len(primary)} |",
        f"| Series failed | {n_err} |",
        f"| Society-years (all) | {n_sy} |",
        f"| Society-years (in-range) | {n_sy_in} |",
        f"| Output directory | `data/v1.0_final_recompute/series/` |",
        f"| F_G corpus | `data/v1.0_final_recompute/FG_corpus.csv` |",
        "",
        "### Primary ENS series (cross-society comparison corpus)",
        "",
        "| Society | Years | Span | BS min | BS max | Source |",
        "|---------|------:|------|-------:|-------:|--------|",
    ]
    for r in sorted(primary, key=lambda x: x["society"]):
        lines.append(
            f"| {r['society']} | {r['years']} | {r['year_min']}–{r['year_max']} | "
            f"{r['bs_min']:.4f} | {r['bs_max']:.4f} | `{Path(r['source']).name}` |"
        )

    lines += [
        "",
        "### All recomputed series",
        "",
        "| Society | Tier | Range | Years | Span | BS min | BS max | Source |",
        "|---------|------|-------|------:|------|-------:|-------:|--------|",
    ]
    for r in sorted(ok_results, key=lambda x: (x["society"], x["source"])):
        lines.append(
            f"| {r['society']} | {r['tier']} | {r['range_tier']} | {r['years']} | "
            f"{r['year_min']}–{r['year_max']} | {r['bs_min']:.4f} | {r['bs_max']:.4f} | "
            f"`{Path(r['source']).name}` |"
        )

    if errors:
        lines += ["", "## Errors", ""]
        for e in errors:
            lines.append(f"- `{Path(e['source']).name}`: {e['error']}")

    if n_sy:
        fg_in = fg_df[fg_df["range_tier"] == "in_range"] if "range_tier" in fg_df.columns else fg_df
        if len(fg_in):
            reg = fg_in["regime"].value_counts()
            lines += [
                "",
                "## Regime distribution (in-range society-years only)",
                "",
                "| Regime | Count | Share |",
                "|--------|------:|------:|",
            ]
            for name, cnt in reg.items():
                lines.append(f"| {name} | {cnt} | {100 * cnt / len(fg_in):.1f}% |")

            bs_ok = (fg_in["B_mean"] >= 0).all() and (fg_in["B_mean"] <= 1).all()
            lines += [
                "",
                "## Sanity checks (in-range)",
                "",
                f"- B̄ ∈ [0,1]: **{'PASS' if bs_ok else 'FAIL'}** "
                f"(range [{fg_in['B_mean'].min():.4f}, {fg_in['B_mean'].max():.4f}])",
                f"- λ₂ range: [{fg_in['lambda2'].min():.4f}, {fg_in['lambda2'].max():.4f}]",
                f"- V̄ range: [{fg_in['V_mean'].min():.2f}, {fg_in['V_mean'].max():.2f}]",
                f"- σ_min range: [{fg_in['sigma_min'].min():.3f}, {fg_in['sigma_min'].max():.3f}]",
            ]

    lines += [
        "",
        "## Notes",
        "",
        "- Per-node Bond Strength is the **mean of the 7 off-diagonal** B_ij edges.",
        "- Legacy Army/Executive/… labels mapped to the 8-node CAMS set; dual aliases "
        "(e.g. Trades/Prof. vs Trades/Professions) resolved by preference rank.",
        "- Incomplete society-years (≠ 8 nodes after mapping) are dropped.",
        "- ENV (variance) files are not recomputed.",
        "- **Out-of-range** series (negative Stress legacy scale, C>10, etc.) are "
        "recomputed for archival completeness but **must not** be used for "
        "cross-society F_G threshold recalibration — B_ij can exceed 1 when S < 0.",
        "- Primary corpus for cross-society work = `canonical_ens` + `cleaned_ens` "
        "with `range_tier=in_range`.",
        "- Next critical-path step: F_G threshold recalibration on the primary corpus.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(
        f"Done. {n_ok} series OK ({len(in_range)} in-range, {len(out_range)} DEV), "
        f"{n_err} errors, {n_sy} society-years ({n_sy_in} in-range)."
    )
    print(f"Primary ENS corpus: {len(primary)} series")
    print(f"Report: {REPORT_PATH}")
    print(f"F_G:    {FG_PATH}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
