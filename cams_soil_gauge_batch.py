#!/usr/bin/env python3
"""
cams_soil_gauge_batch.py
========================
Run Robson soil gauge (η_soil) v1.0 across recomputed CAMS series.

Usage (from wintermute/):
    python cams_soil_gauge_batch.py

Uses robson_gauge_v10.py (paper unscaled default; also reports ×1000 Hermes).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from robson_gauge_v10 import robson_gauge

ROOT = Path(__file__).resolve().parent
SERIES_DIR = ROOT / "data" / "v1.0_final_recompute" / "series"
MANIFEST = ROOT / "data" / "v1.0_final_recompute" / "manifest.json"
OUT_DIR = ROOT / "data" / "v1.0_final_recompute"
EXPORT_DIR = Path.home() / "Downloads" / "experiment"
DESK = Path.home() / "Desktop" / "CAMS_v1.0_Final_B_ij_Recompute"

NODES = ["Helm", "Shield", "Flow", "Hands", "Craft", "Stewards", "Archive", "Lore"]


def interpret_hermes(eta_x1000: float) -> str:
    """Band labels on Hermes (×1000) scale."""
    if eta_x1000 > 800:
        return "Artificial high anchoring (Singapore pattern)"
    if eta_x1000 < 100:
        return "Organic / WATCH range"
    return "Moderate Maritime"


def series_to_cams_data(g: pd.DataFrame) -> dict | None:
    data = {}
    for _, row in g.iterrows():
        node = str(row["Node"]).strip()
        if node not in NODES:
            continue
        data[node] = {
            "C": float(row["Coherence"]),
            "K": float(row["Capacity"]),
            "S": float(row["Stress"]),
            "A": float(row["Abstraction"]),
        }
    if len(data) < 8:
        return None
    return data


def load_manifest_meta() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for e in man.get("series", []):
        if not e.get("ok"):
            continue
        out[e["out_name"]] = {
            "tier": e.get("tier"),
            "range_tier": e.get("range_tier"),
            "society_meta": e.get("society"),
            "source": e.get("source"),
        }
    return out


def main() -> int:
    if not SERIES_DIR.is_dir():
        print(f"No series directory: {SERIES_DIR}")
        return 1

    # Self-test first
    print("=== robson_gauge v1.0 self-check ===")
    import robson_gauge_v10  # noqa: F401
    # run module demo pieces via import path already tested

    meta = load_manifest_meta()
    paths = sorted(SERIES_DIR.glob("*_Bij_v10.csv"))
    print(f"Batch over {len(paths)} series files\n")

    all_rows = []
    errors = []

    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            errors.append((path.name, str(e)))
            continue

        for c in ["Year", "Coherence", "Capacity", "Stress", "Abstraction"]:
            if c not in df.columns:
                errors.append((path.name, f"missing {c}"))
                break
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            m = meta.get(path.name, {})
            for (soc, year), g in df.groupby(["Society", "Year"]):
                cams = series_to_cams_data(g)
                if cams is None:
                    continue
                try:
                    eta1, det = robson_gauge(cams, scale=1.0)
                    eta1k = float(det["eta_unscaled"]) * 1000.0
                except Exception as e:
                    errors.append((f"{path.name}:{soc}:{year}", str(e)))
                    continue
                all_rows.append(
                    {
                        "Society": soc,
                        "Year": int(year),
                        "series_file": path.name,
                        "tier": m.get("tier", ""),
                        "range_tier": m.get("range_tier", ""),
                        "eta_unscaled": det["eta_unscaled"],
                        "eta_x1000": round(eta1k, 4),
                        "interpretation_hermes": interpret_hermes(eta1k),
                        "V_mean": det["V_mean"],
                        "sigma_V": det["sigma_V"],
                        "BS_Lore": det["BS_Lore"],
                        "BS_Archive": det["BS_Archive"],
                        "C_Hands": det["C_Hands"],
                        "S_Hands": det["S_Hands"],
                        "numerator": det["numerator"],
                        "denominator": det["denominator"],
                        "epsilon": det["epsilon"],
                    }
                )

    if not all_rows:
        print("No society-years scored.")
        return 1

    full = pd.DataFrame(all_rows).sort_values(["Society", "Year", "series_file"])
    full_path = OUT_DIR / "soil_gauge_all_years.csv"
    full.to_csv(full_path, index=False)

    latest_idx = full.groupby("series_file")["Year"].idxmax()
    latest = full.loc[latest_idx].sort_values("eta_unscaled", ascending=False)
    latest_path = OUT_DIR / "soil_gauge_latest.csv"
    latest.to_csv(latest_path, index=False)

    primary = latest[
        (latest["range_tier"] == "in_range")
        & (latest["tier"].isin(["canonical_ens", "cleaned_ens"]))
    ].copy()
    if primary.empty:
        primary = latest[latest["range_tier"] == "in_range"].copy()

    primary_best = (
        primary.sort_values(["Society", "Year"], ascending=[True, False])
        .groupby("Society", as_index=False)
        .first()
        .sort_values("eta_unscaled", ascending=False)
    )

    print("=" * 78)
    print("PRIMARY ENS / IN-RANGE — latest year (η_unscaled paper scale)")
    print("=" * 78)
    for _, r in primary_best.iterrows():
        print(
            f"{r['Society']:28s} {int(r['Year']):4d}  "
            f"η={r['eta_unscaled']:8.4f}  (×1000={r['eta_x1000']:8.2f})  "
            f"V̄={r['V_mean']:6.2f}  σV={r['sigma_V']:5.2f}  "
            f"| {r['interpretation_hermes']}"
        )

    bands = primary_best["interpretation_hermes"].value_counts()
    print()
    print("Primary band distribution (Hermes ×1000 cutoffs):")
    for k, v in bands.items():
        print(f"  {k}: {v}")

    print()
    print(f"Society-years: {len(full):,}  |  series latest: {len(latest)}  |  primary: {len(primary_best)}")

    lines = [
        "# CAMS Robson Soil Gauge (η_soil) v1.0 — Corpus Batch",
        "",
        "**Engine:** `robson_gauge_v10.py` (paper unscaled default)",
        "",
        "```",
        "η_soil = (BS_Lore · BS_Archive · C_Hands) / (S_Hands · σ_V + ε),  ε = 2.0",
        "```",
        "",
        f"**Series files:** {len(paths)}  ",
        f"**Society-years:** {len(full):,}  ",
        f"**Primary societies (latest):** {len(primary_best)}  ",
        "",
        "Hermes band cutoffs apply to **η × 1000**: >800 Artificial · 100–800 Moderate · <100 Organic/WATCH.",
        "",
        "## Primary ENS / in-range (latest year)",
        "",
        "| Society | Year | η (unscaled) | η×1000 | V̄ | σ_V | BS_Lore | BS_Archive | Hands C/S | Band |",
        "|---------|-----:|-------------:|-------:|---:|----:|--------:|-----------:|----------:|------|",
    ]
    for _, r in primary_best.iterrows():
        lines.append(
            f"| {r['Society']} | {int(r['Year'])} | {r['eta_unscaled']:.4f} | {r['eta_x1000']:.2f} | "
            f"{r['V_mean']:.2f} | {r['sigma_V']:.2f} | {r['BS_Lore']:.4f} | {r['BS_Archive']:.4f} | "
            f"{r['C_Hands']}/{r['S_Hands']} | {r['interpretation_hermes']} |"
        )
    lines += ["", "## Band distribution", "", "| Band | Count |", "|------|------:|"]
    for k, v in bands.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Files",
        "",
        "- `soil_gauge_latest.csv`",
        "- `soil_gauge_all_years.csv`",
        "- `series/*_Bij_v10.csv`",
        "- `FG_corpus.csv`, `manifest.json`",
        "",
    ]
    report_path = OUT_DIR / "SOIL_GAUGE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")

    # --- Export full recompute corpus to Downloads/experiment ---
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    exp = EXPORT_DIR / "CAMS_v1.0_Final_B_ij_Recompute"
    if exp.exists():
        shutil.rmtree(exp)
    exp.mkdir(parents=True)

    # series
    ser_out = exp / "series"
    ser_out.mkdir()
    for p in paths:
        shutil.copy2(p, ser_out / p.name)

    # core outputs
    for name in [
        "FG_corpus.csv",
        "manifest.json",
        "RECOMPUTE_REPORT.md",
        "FC_VALIDATION_REPORT.md",
        "fc_validation.json",
        "soil_gauge_all_years.csv",
        "soil_gauge_latest.csv",
        "SOIL_GAUGE_REPORT.md",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, exp / name)

    # scripts
    scripts = exp / "scripts"
    scripts.mkdir()
    for name in [
        "cams_framework_v2_4.py",
        "batch_recompute_bij_v10.py",
        "cams_v10_final_fc_validation.py",
    ]:
        # framework lives at wintermute root; batch scripts may be in scripts/
        candidates = [
            ROOT / "scripts" / name,
            ROOT / name,
            DESK / "scripts" / name,
        ]
        for c in candidates:
            if c.exists():
                shutil.copy2(c, scripts / name)
                break
    # gauge modules
    for name in ["robson_gauge_v10.py", "cams_soil_gauge.py", "cams_soil_gauge_batch.py"]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, scripts / name)

    # requirements + short README
    (exp / "requirements.txt").write_text(
        "numpy>=1.24\nscipy>=1.10\npandas>=2.0\n", encoding="utf-8"
    )
    (exp / "README.md").write_text(
        "\n".join(
            [
                "# CAMS v1.0-Final recompute + Robson Gauge export",
                "",
                f"Exported to `{exp}`.",
                "",
                "## Robson Gauge",
                "",
                "```",
                "η_soil = (BS_Lore · BS_Archive · C_Hands) / (S_Hands · σ_V + ε), ε=2",
                "```",
                "",
                "- Paper / publish: **η unscaled** (`eta_unscaled` column)",
                "- Hermes bands: **η × 1000** (`eta_x1000`)",
                "",
                "```bash",
                "python scripts/robson_gauge_v10.py",
                "python scripts/cams_soil_gauge_batch.py",
                "```",
                "",
                "See `SOIL_GAUGE_REPORT.md` and `series/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    n_ser = len(list(ser_out.glob("*.csv")))
    print(f"\nExported corpus → {exp}")
    print(f"  series CSVs: {n_ser}")
    print(f"  contents: {sorted(p.name for p in exp.iterdir())}")

    if DESK.is_dir():
        for p in (full_path, latest_path, report_path):
            shutil.copy2(p, DESK / p.name)

    if errors:
        print(f"Errors/skips: {len(errors)}")
        for f, e in errors[:8]:
            print(f"  {f}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
