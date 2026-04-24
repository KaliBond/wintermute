"""
Ingest 7 new CSV files into cleaned_datasets/ and update cams_index.json.
5-agent ensemble files (cam5/camican5) are noted in provenance.
For France 1800-1830, both GEM and Claude versions are saved; GEM used for index.
"""

import csv
import json
import shutil
import os
from pathlib import Path

DOCS = Path("C:/Users/julie/OneDrive/Documents")
CLEANED = Path("C:/Users/julie/wintermute/cleaned_datasets")
INDEX_PATH = CLEANED / "cams_index.json"

def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Normalize "Entity" → "Society"
    if rows and 'Entity' in rows[0] and 'Society' not in rows[0]:
        for row in rows:
            row['Society'] = row.pop('Entity')
    return rows

def rows_to_index_entries(rows):
    """Returns dict: {year_str: {node: [C,K,S,A]}}"""
    entries = {}
    for row in rows:
        year = str(int(float(row['Year'])))
        node = row['Node'].strip()
        vals = [
            float(row['Coherence']),
            float(row['Capacity']),
            float(row['Stress']),
            float(row['Abstraction']),
        ]
        entries.setdefault(year, {})[node] = vals
    return entries

def write_cleaned_csv(rows, dest_path, society_name=None):
    fieldnames = ['Society', 'Year', 'Node', 'Coherence', 'Capacity',
                  'Stress', 'Abstraction', 'Node Value', 'Bond Strength']
    # Normalise field names from source
    with open(dest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            # Remap old field names if needed
            out = {
                'Society': row.get('Society', society_name or ''),
                'Year': row.get('Year', ''),
                'Node': row.get('Node', ''),
                'Coherence': row.get('Coherence', ''),
                'Capacity': row.get('Capacity', ''),
                'Stress': row.get('Stress', ''),
                'Abstraction': row.get('Abstraction', ''),
                'Node Value': row.get('Node Value', ''),
                'Bond Strength': row.get('Bond Strength', ''),
            }
            writer.writerow(out)

# Load existing index
with open(INDEX_PATH, 'r', encoding='utf-8') as f:
    index = json.load(f)

# Define files to process
# (src_filename, dest_stem, society_key_in_index, use_for_index, provenance_note)
files = [
    (
        "AUSTRALIA_camican5_24_4_26.CSV",
        "Australia_cam5_1996_2026_cleaned.csv",
        "Australia",
        True,
        "5-agent ensemble"
    ),
    (
        "USA_cam5_24_4_26.csv",
        "USA_cam5_1996_2026_cleaned.csv",
        "USA",
        True,
        "5-agent ensemble"
    ),
    (
        "Russia_Claude_1800-1830.csv",
        "Russia_1800_1830_claude_cleaned.csv",
        "Russia",
        True,
        "Claude"
    ),
    (
        "Spain_1800_1830.csv",
        "Spain_1800_1830_cleaned.csv",
        "Spain",
        True,
        None
    ),
    (
        "spacex_camican5_24_4_26.csv",
        "SpaceX_cam5_2006_2026_cleaned.csv",
        "SpaceX",
        True,
        "5-agent ensemble"
    ),
    (
        "france_1800_1830_GEM.csv",
        "France_1800_1830_GEM_cleaned.csv",
        "France",
        True,   # GEM used for index
        "Google Gemini"
    ),
    (
        "france_1800_1830_claude.csv",
        "France_1800_1830_claude_cleaned.csv",
        "France",
        False,  # Claude version saved but not used for index
        "Claude"
    ),
]

for src_name, dest_name, society_key, use_for_index, provenance in files:
    src = DOCS / src_name
    dest = CLEANED / dest_name

    print(f"Processing {src_name} -> {dest_name}")
    rows = read_csv_rows(src)
    print(f"  {len(rows)} rows, provenance: {provenance or 'standard'}")

    write_cleaned_csv(rows, dest)

    if use_for_index:
        entries = rows_to_index_entries(rows)
        print(f"  Merging {len(entries)} years into index['{society_key}']")

        if society_key not in index:
            index[society_key] = {}

        for year, nodes in entries.items():
            if year not in index[society_key]:
                index[society_key][year] = {}
            index[society_key][year].update(nodes)

# Write updated index
with open(INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(index, f, separators=(',', ':'))

print(f"\nDone. Index societies: {sorted(index.keys())}")
print(f"Index written to {INDEX_PATH}")
