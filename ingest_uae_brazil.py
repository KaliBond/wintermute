"""
Add UAE and Brazil to cleaned_datasets/cams_index.json.
Completes the 22-new-society integration into CAMS Explorer / Interpreter / Zeitgeist
(the other 20 were already merged in earlier commits).
"""

import csv
import json
from pathlib import Path

CLEANED = Path("data/cleaned")
INDEX_PATH = Path("cleaned_datasets/cams_index.json")

SPURIOUS_NODES = {'Node', 'node', ''}

FILES = [
    ("uae_gem_marc.csv", "UAE"),
    ("brazil_grok_jan.csv", "Brazil"),
]


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def rows_to_index_entries(rows):
    """Returns dict: {year_str: {node: [C,K,S,A]}}, skipping spurious/malformed rows."""
    entries = {}
    skipped = 0
    for row in rows:
        node = (row.get('Node') or '').strip()
        if node in SPURIOUS_NODES:
            skipped += 1
            continue
        try:
            year = str(int(float(row['Year'])))
            vals = [
                float(row['Coherence']),
                float(row['Capacity']),
                float(row['Stress']),
                float(row['Abstraction']),
            ]
        except (ValueError, TypeError, KeyError):
            skipped += 1
            continue
        entries.setdefault(year, {})[node] = vals
    return entries, skipped


with open(INDEX_PATH, 'r', encoding='utf-8-sig') as f:
    index = json.load(f)

for src_name, society_key in FILES:
    src = CLEANED / src_name
    rows = read_csv_rows(src)
    entries, skipped = rows_to_index_entries(rows)
    print(f"{src_name} -> '{society_key}': {len(entries)} years, {skipped} rows skipped")

    if society_key not in index:
        index[society_key] = {}
    for year, nodes in entries.items():
        index[society_key].setdefault(year, {}).update(nodes)

with open(INDEX_PATH, 'w', encoding='utf-8-sig') as f:
    json.dump(index, f, separators=(',', ':'))

print(f"\nDone. Index now has {len(index)} societies.")
