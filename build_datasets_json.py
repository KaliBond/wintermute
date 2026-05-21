"""
build_datasets_json.py
Scans cleaned_datasets/, reads each CSV, extracts metadata, writes datasets.json.
Run from repo root: python build_datasets_json.py
"""
import csv
import json
import os
import re
from datetime import date

CORPUS_DIR = "cleaned_datasets"
OUTPUT = "datasets.json"
GITHUB_BASE = "https://github.com/KaliBond/wintermute/blob/main/cleaned_datasets"

REGION_MAP = {
    "USA": "North America", "Canada": "North America", "WorldCom": "Global",
    "England": "Europe", "UK": "Europe", "France": "Europe",
    "Germany": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Netherlands": "Europe", "Denmark": "Europe", "Norway": "Europe",
    "Sweden": "Europe", "Russia": "Europe",
    "Australia": "Asia-Pacific", "IndigenousAustralia": "Asia-Pacific",
    "New_Zealand": "Asia-Pacific", "NewZealand": "Asia-Pacific",
    "Japan": "Asia-Pacific", "China": "Asia-Pacific",
    "Hong_Kong": "Asia-Pacific", "Hongkong": "Asia-Pacific",
    "India": "Asia-Pacific", "Indonesia": "Asia-Pacific",
    "Singapore": "Asia-Pacific", "Thailand": "Asia-Pacific",
    "Pakistan": "Asia-Pacific",
    "Iran": "Middle East", "Iraq": "Middle East", "Israel": "Middle East",
    "Lebanon": "Middle East", "Saudi_Arabia": "Middle East", "Syria": "Middle East",
    "Afghanistan": "Middle East",
    "Argentina": "South America",
    "LatimVetus": "Historical", "Rome": "Historical",
    "New_Rome": "Historical", "Latium": "Historical",
    "SpaceX": "Non-State",
}

def detect_family(fname):
    f = fname.lower()
    if "envelope" in f or f.endswith("_sd_cleaned.csv"): return "envelope"
    if "camnations5" in f: return "CAMNATIONS5"
    if "cams5_ensemble_mean" in f: return "CAMS5 ensemble mean"
    if "cams5_ensemble" in f: return "CAMS5 ensemble"
    if "cams5_calc" in f or "_calc_" in f: return "CAMS5 calculated"
    if "cam5" in f: return "CAM5"
    if "_gem_" in f or "_gem_" in f: return "GEM"
    if "_claude_" in f: return "Claude"
    if "recalculated" in f: return "recalculated"
    if "highres" in f or "high_res" in f: return "high-res"
    if "reconstructed" in f: return "reconstructed"
    if "maximum" in f: return "maximum"
    if "manual" in f: return "manual"
    if "master" in f: return "master"
    if "marker" in f: return "MARKER ensemble"
    return "standard"

def detect_society(fname):
    # Try to pull the Society column value first; fall back to filename
    known = [
        "IndigenousAustralia", "Australia", "Argentina", "Canada", "China",
        "Denmark", "England", "France", "Germany", "Hong_Kong", "Hongkong",
        "India", "Indonesia", "Iran", "Iraq", "Israel", "Italy", "Japan",
        "LatimVetus", "Latium", "Lebanon", "Netherlands", "NewZealand",
        "New_Zealand", "New_Rome", "Norway", "Pakistan", "Rome", "Russia",
        "Saudi_Arabia", "Singapore", "SpaceX", "Spain", "Sweden", "Syria",
        "Thailand", "UK", "USA", "Usa", "WorldCom", "Afghanistan",
    ]
    for s in known:
        if fname.startswith(s) or fname.upper().startswith(s.upper()):
            return s
    # fallback: first segment before underscore
    return fname.split("_")[0]

def read_csv_meta(filepath):
    """Return (society_col_val, year_min, year_max, row_count, columns)."""
    try:
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = [c.strip() for c in (reader.fieldnames or [])]
            years = []
            society_val = None
            count = 0
            for row in reader:
                count += 1
                # year column
                yr_key = next((k for k in row if k.strip().lower() == "year"), None)
                if yr_key:
                    try:
                        years.append(int(float(row[yr_key])))
                    except (ValueError, TypeError):
                        pass
                # society column
                if society_val is None:
                    soc_key = next((k for k in row if k.strip().lower() in ("society","nation")), None)
                    if soc_key:
                        society_val = row[soc_key].strip()
        return society_val, (min(years) if years else None), (max(years) if years else None), count, cols
    except Exception as e:
        return None, None, None, 0, []

def region_for(society):
    for key, reg in REGION_MAP.items():
        if society and (society.lower().startswith(key.lower()) or key.lower() in society.lower()):
            return reg
    return "Other"

def has_col(cols, *names):
    lc = [c.lower().strip() for c in cols]
    return any(n.lower() in lc for n in names)

datasets = []
total_records = 0

files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith(".csv"))
print(f"Scanning {len(files)} CSV files…")

for fname in files:
    path = os.path.join(CORPUS_DIR, fname)
    soc_col, yr_min, yr_max, count, cols = read_csv_meta(path)

    # Society name: prefer CSV column, else infer from filename
    society = soc_col or detect_society(fname)
    # Normalise common variants
    society = society.replace("Hong Kong", "Hong Kong")

    family = detect_family(fname)
    is_envelope = family == "envelope"

    entry = {
        "filename": fname,
        "society": society,
        "year_start": yr_min,
        "year_end": yr_max,
        "records": count,
        "family": family,
        "is_envelope": is_envelope,
        "has_bond_strength": has_col(cols, "Bond Strength", "Bond strength"),
        "has_node_value": has_col(cols, "Node Value", "Node value"),
        "region": region_for(society),
        "github_url": f"{GITHUB_BASE}/{fname}",
    }
    datasets.append(entry)
    total_records += count
    yr_str = f"{yr_min}–{yr_max}" if yr_min else "?"
    print(f"  {society:25s} {yr_str:15s} {count:5d} rows  [{family}]")

# Sort: region → society → year_start
REGION_ORDER = ["North America","South America","Europe","Asia-Pacific","Middle East","Historical","Non-State","Global","Other"]
datasets.sort(key=lambda d: (
    REGION_ORDER.index(d["region"]) if d["region"] in REGION_ORDER else 99,
    d["society"].lower(),
    d["year_start"] or 0,
))

out = {
    "generated": str(date.today()),
    "total_files": len(datasets),
    "total_records": total_records,
    "datasets": datasets,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"\nWrote {OUTPUT}: {len(datasets)} datasets, {total_records:,} total records")
