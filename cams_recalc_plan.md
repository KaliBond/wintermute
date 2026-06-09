# CAMS v2.3 Derivatives Recalculation Plan

## Data Tiers

### CANONICAL
Ensemble mean files produced by `/camnations5` (5-scorer pipeline).
Identified by `_ENS_` in filename. These are the authoritative datasets.
Node Value and Bond Strength are recalculated from their C/K/S/A means.

Companion ENV files (inter-scorer variance: SD_Coherence, SD_Capacity, etc.)
are **not** recalculated — they are copied as-is. Note: ENV files in
`cleaned_datasets/` were badly ingested (wrong headers, empty values);
source them from `cam5/` originals instead.

### USP — Useful Single Passes
Single-pass datasets with valid 1–10 scores on all four dimensions and
standard CAMS node names. Scientifically useful for comparison and
longitudinal analysis, but not canonical.

### DEV — Development / Legacy
Datasets with negative Stress integers or scores outside [1, 10].
Archived without recalculation. Useful as historical record and for
understanding early methodology.

---

## Formulas (cams_framework_v2_3.py — do not deviate)

```
Node Value:    V_i = C + K + (A / 2) - S
Bond Strength: Bij = sqrt(max(Vi + 8, 0) * max(Vj + 8, 0)) / 32
Per-node BS  = mean(Bij) across all 7 pairwise bonds in the society-year
```

---

## Output Directory Structure

```
data/
  v2.3/
    canonical/    ← recalculated ENS files + copied ENV files
    usp/          ← recalculated USP files
    dev/          ← archived DEV files (no recalc, with tier manifest)
    README.md     ← existing, update with tier summary
```

---

## Step 1 — Write the tiering + normalization script

Create `scripts/tier_and_normalize.py`:

```python
"""
tier_and_normalize.py

Classifies every CSV in cleaned_datasets/ into CANONICAL / USP / DEV,
normalises column names, and writes to staging directories:
  data/v2.3_input/canonical/   ← raw scores ready for batch_score()
  data/v2.3_input/usp/         ← raw scores ready for batch_score()
  data/v2.3/dev/               ← archived as-is (no recalc)

ENV files from cleaned_datasets are noted as badly ingested; real ENV
files are sourced from cam5/ (see Step 3).

Usage:
    python scripts/tier_and_normalize.py
"""

import pathlib, shutil
import pandas as pd

STANDARD_NODES = {
    "Helm", "Shield", "Lore", "Stewards",
    "Craft", "Hands", "Archive", "Flow"
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
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        results["error"].append((csv_path.name, str(e)))
        continue

    # --- Detect ENV files (SD/variance companions) ---
    # Badly ingested in cleaned_datasets — skip here, handle in Step 3
    if "_ENV_" in csv_path.name or csv_path.name.endswith("_ENV_cleaned.csv"):
        results["env_skip"].append(csv_path.name)
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

    # Inject Society from filename if missing
    if "Society" not in df.columns:
        society_name = csv_path.stem.replace("_cleaned", "")
        df.insert(0, "Society", society_name)

    # --- Check required columns ---
    missing = [c for c in ["Society", "Year", "Node"] + REQUIRED_RAW
               if c not in df.columns]
    if missing:
        results["error"].append((csv_path.name, f"missing: {missing}"))
        continue

    # --- Check node names ---
    nodes_found = set(df["Node"].dropna().unique())
    if not nodes_found.issubset(STANDARD_NODES):
        non_std = nodes_found - STANDARD_NODES
        results["dev"].append((csv_path.name, f"non-standard nodes: {non_std}"))
        shutil.copy(csv_path, DEV_OUT / csv_path.name)
        continue

    # --- Determine tier by Stress range ---
    stress = df["Stress"].dropna()
    coherence = df["Coherence"].dropna()
    is_dev = (stress < 1).any() or (stress > 10).any() or (coherence > 10).any()

    if is_dev:
        results["dev"].append((csv_path.name,
            f"Stress=[{stress.min():.1f},{stress.max():.1f}] "
            f"C_max={coherence.max():.1f}"))
        shutil.copy(csv_path, DEV_OUT / csv_path.name)
        continue

    # --- Keep only raw score columns for recalc input ---
    out = df[["Society", "Year", "Node"] + REQUIRED_RAW].copy()

    # --- Assign to CANONICAL or USP ---
    is_canonical = "_ENS_" in csv_path.name

    if is_canonical:
        out.to_csv(CANON_STAGE / csv_path.name, index=False)
        results["canonical"].append(csv_path.name)
    else:
        out.to_csv(USP_STAGE / csv_path.name, index=False)
        results["usp"].append(csv_path.name)

# --- Print summary ---
print(f"\n{'='*60}")
print(f"CANONICAL ({len(results['canonical'])} files) → {CANON_STAGE}")
for f in results["canonical"]:
    print(f"  {f}")

print(f"\nUSP ({len(results['usp'])} files) → {USP_STAGE}")
for f in results["usp"]:
    print(f"  {f}")

print(f"\nDEV ({len(results['dev'])} files) → {DEV_OUT}")
for f, reason in results["dev"]:
    print(f"  {f}  [{reason}]")

print(f"\nENV skipped ({len(results['env_skip'])} files — handle in Step 3)")
for f in results["env_skip"]:
    print(f"  {f}")

if results["error"]:
    print(f"\nERRORS ({len(results['error'])})")
    for f, e in results["error"]:
        print(f"  {f}: {e}")

# Write tier manifest to data/v2.3/dev/
manifest_path = DEV_OUT / "TIER_MANIFEST.md"
with open(manifest_path, "w") as fh:
    fh.write("# DEV Tier Manifest\n\n")
    fh.write("Files archived here have Stress or Coherence values outside the\n")
    fh.write("canonical 1–10 range, or non-standard node names.\n\n")
    for f, reason in results["dev"]:
        fh.write(f"- `{f}`: {reason}\n")
print(f"\nManifest written to {manifest_path}")
```

---

## Step 2 — Run batch recalculation for CANONICAL and USP

```bash
# Recalculate CANONICAL derivatives
python cams_framework_v2_3.py data/v2.3_input/canonical/ data/v2.3/canonical/

# Recalculate USP derivatives
python cams_framework_v2_3.py data/v2.3_input/usp/ data/v2.3/usp/
```

---

## Step 3 — Source real ENV files from cam5/

The ENV files in `cleaned_datasets/` were badly ingested (ENS-style headers,
all values empty). The real ENV files with SD/variance data live in `cam5/`.

```bash
python - <<'EOF'
import pathlib, shutil

CAM5_DIR   = pathlib.Path("../cam5")   # adjust if cam5 is elsewhere
CANON_DIR  = pathlib.Path("data/v2.3/canonical")
CANON_DIR.mkdir(exist_ok=True)

# Match cam5 envelope files to canonical nations
# cam5 naming convention: <Society>_CAMS5_envelope_*.csv or <Society>_CAMS_envelope.csv
copied = []
for env_path in sorted(CAM5_DIR.glob("*envelope*")):
    dest = CANON_DIR / env_path.name
    shutil.copy(env_path, dest)
    copied.append(env_path.name)

print(f"Copied {len(copied)} ENV files to {CANON_DIR}")
for f in copied:
    print(f"  {f}")
EOF
```

**Note:** Confirm `CAM5_DIR` path before running. If cam5 is at a different
relative location, adjust accordingly.

---

## Step 4 — Verify outputs

```bash
python cams_v23_tests.py
```

Range sanity check:

```bash
python - <<'EOF'
import pathlib, pandas as pd

errors = []
for tier, d in [("canonical", "data/v2.3/canonical"), ("usp", "data/v2.3/usp")]:
    csv_files = list(pathlib.Path(d).glob("*.csv"))
    print(f"\n{tier.upper()}: {len(csv_files)} files")
    for p in csv_files:
        df = pd.read_csv(p)
        if "Node Value" not in df.columns:
            continue  # ENV file, skip
        if df["Node Value"].isna().any():
            errors.append(f"{p.name}: NaN in Node Value")
        if not ((df["Node Value"] > -10) & (df["Node Value"] < 40)).all():
            errors.append(f"{p.name}: Node Value out of range")
        if not ((df["Bond Strength"] >= 0) & (df["Bond Strength"] <= 1)).all():
            errors.append(f"{p.name}: Bond Strength out of [0,1]")

if errors:
    print("\nERRORS:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nAll checks passed.")
EOF
```

---

## Step 5 — Commit

```bash
git add data/v2.3/canonical/ data/v2.3/usp/ data/v2.3/dev/ scripts/tier_and_normalize.py
git commit -m "recalc: tier + recalculate CAMS v2.3 derivatives

Tiers:
  CANONICAL — camnations5 ENS files, authoritative
  USP       — valid single-pass files (Stress 1-10, standard nodes)
  DEV       — legacy files with negative Stress or out-of-range scores

Pipeline:
  scripts/tier_and_normalize.py  → classifies and stages inputs
  cams_framework_v2_3.py         → batch_score() canonical formulas
  cam5/ ENV files                → copied as-is to canonical/

Formulas: V = C + K + A/2 - S
          Bij = sqrt(max(Vi+8,0)*max(Vj+8,0)) / 32"
```

---

## Notes for Claude Code

- **Do not modify `cams_framework_v2_3.py`** — it is the canonical formula source.
- Staging dirs (`data/v2.3_input/`) are temporary; delete after commit.
- ENV files contain inter-scorer variance (SD columns), not raw scores.
  They are companions to ENS files and bypass the recalc pipeline.
- `MARKER_USA_1900_2026_ENSEMBLE_MEAN_cleaned.csv` is classified USP
  (not ENS naming convention); review manually if canonical status is needed.
- `Germany_cleaned.csv` has S_min=0.0 — technically borderline. The script
  classifies it USP; flag for manual review.
- `Argentina` has multiple overlapping files (CAMS5_calc, CAMS5_ensemble,
  cam5, ENS). Only the ENS file goes to canonical; the others go to USP.
  Deduplicate the output manually if needed.
