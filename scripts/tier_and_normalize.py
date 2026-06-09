"""
scripts/tier_and_normalize.py
Classify every CSV in cleaned_datasets/ into CANONICAL / USP / DEV,
normalise column names, and write to staging directories:

  data/v2.3_input/canonical/   <- raw scores ready for batch_score()
  data/v2.3_input/usp/         <- raw scores ready for batch_score()
  data/v2.3/dev/               <- archived as-is (no recalc)

ENV files (SD/variance companions) are skipped here — handle separately.

Run from wintermute/:
    python scripts/tier_and_normalize.py
"""

import pathlib
import shutil
import pandas as pd

STANDARD_NODES = {
    "Helm", "Shield", "Lore", "Stewards",
    "Craft", "Hands", "Archive", "Flow",
}
REQUIRED_RAW = ["Coherence", "Capacity", "Stress", "Abstraction"]

INPUT_DIR   = pathlib.Path("cleaned_datasets")
CANON_STAGE = pathlib.Path("data/v2.3_input/canonical")
USP_STAGE   = pathlib.Path("data/v2.3_input/usp")
DEV_OUT     = pathlib.Path("data/v2.3/dev")

for d in [CANON_STAGE, USP_STAGE, DEV_OUT]:
    d.mkdir(parents=True, exist_ok=True)

results = {"canonical": [], "usp": [], "dev": [], "env_skip": [], "error": []}

for csv_path in sorted(INPUT_DIR.glob("*.csv")):

    # --- ENV files: badly ingested in cleaned_datasets (wrong headers) ---
    if "_ENV_" in csv_path.name:
        results["env_skip"].append(csv_path.name)
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        results["error"].append((csv_path.name, str(e)))
        continue

    # --- Normalise column names ---
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "nation":
            col_map[col] = "Society"
        elif cl == "node value":
            col_map[col] = "Node Value"
        elif cl == "bond strength":
            col_map[col] = "Bond Strength"
    if col_map:
        df = df.rename(columns=col_map)

    # --- Inject Society from filename if missing ---
    if "Society" not in df.columns:
        society_name = csv_path.stem.replace("_cleaned", "")
        df.insert(0, "Society", society_name)

    # --- Required columns check ---
    missing = [c for c in ["Society", "Year", "Node"] + REQUIRED_RAW
               if c not in df.columns]
    if missing:
        results["error"].append((csv_path.name, f"missing columns: {missing}"))
        continue

    # --- Node name eligibility ---
    nodes_found = set(df["Node"].dropna().unique())
    if not nodes_found.issubset(STANDARD_NODES):
        non_std = nodes_found - STANDARD_NODES
        reason = f"non-standard nodes: {sorted(non_std)}"
        results["dev"].append((csv_path.name, reason))
        shutil.copy(csv_path, DEV_OUT / csv_path.name)
        continue

    # --- Score range eligibility ---
    stress    = pd.to_numeric(df["Stress"],    errors="coerce").dropna()
    coherence = pd.to_numeric(df["Coherence"], errors="coerce").dropna()
    if (stress < 1).any() or (stress > 10).any() or (coherence > 10).any():
        reason = (f"Stress=[{stress.min():.1f},{stress.max():.1f}] "
                  f"C_max={coherence.max():.1f}")
        results["dev"].append((csv_path.name, reason))
        shutil.copy(csv_path, DEV_OUT / csv_path.name)
        continue

    # --- Node count per society-year ---
    counts = df.groupby(["Society", "Year"])["Node"].count()
    bad = counts[counts != 8]
    if not bad.empty:
        reason = f"{len(bad)} society-years with node count != 8"
        results["dev"].append((csv_path.name, reason))
        shutil.copy(csv_path, DEV_OUT / csv_path.name)
        continue

    # --- Keep only raw score columns for recalc input ---
    out = df[["Society", "Year", "Node"] + REQUIRED_RAW].copy()

    # --- Assign tier ---
    is_canonical = "_ENS_" in csv_path.name
    if is_canonical:
        out.to_csv(CANON_STAGE / csv_path.name, index=False)
        results["canonical"].append(csv_path.name)
    else:
        out.to_csv(USP_STAGE / csv_path.name, index=False)
        results["usp"].append(csv_path.name)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"CANONICAL  ({len(results['canonical'])} files)  →  {CANON_STAGE}")
for f in results["canonical"]:
    print(f"  {f}")

print(f"\nUSP  ({len(results['usp'])} files)  →  {USP_STAGE}")
for f in results["usp"]:
    print(f"  {f}")

print(f"\nDEV  ({len(results['dev'])} files)  →  {DEV_OUT}")
for f, reason in results["dev"]:
    print(f"  {f}  [{reason}]")

print(f"\nENV skipped  ({len(results['env_skip'])} files — copy from cam5/ separately)")
for f in results["env_skip"]:
    print(f"  {f}")

if results["error"]:
    print(f"\nERRORS  ({len(results['error'])})")
    for f, e in results["error"]:
        print(f"  {f}: {e}")

# --- Tier manifest for DEV ---
manifest = DEV_OUT / "TIER_MANIFEST.md"
with open(manifest, "w") as fh:
    fh.write("# DEV Tier Manifest\n\n")
    fh.write("Files archived here have Stress or Coherence values outside\n")
    fh.write("the canonical 1–10 range, non-standard node names, or incomplete\n")
    fh.write("society-year groups.  Not eligible for v2.4 recomputation.\n\n")
    for f, reason in results["dev"]:
        fh.write(f"- `{f}`: {reason}\n")
print(f"\nManifest → {manifest}")
