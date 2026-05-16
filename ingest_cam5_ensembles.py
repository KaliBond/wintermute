"""
Ingest cam5 ensemble mean CSVs from Desktop into cleaned_datasets/,
update cams_index.json with ensemble mean entries,
and inject new society DATA into all three telescope HTML files.

Run from: C:/Users/julie/wintermute/
"""

import csv, json, re, shutil
from pathlib import Path

SRC  = Path("C:/Users/julie/Desktop/HariSeldon/cam5")
CLEANED = Path("C:/Users/julie/wintermute/cleaned_datasets")
REPO = Path("C:/Users/julie/wintermute")
INDEX_PATH = CLEANED / "cams_index.json"

SLOW_NODES = ['Lore', 'Archive', 'Helm', 'Stewards']
FAST_NODES = ['Shield', 'Craft', 'Hands', 'Flow']
MAX_BOND   = 20.0   # pairwise bond ceiling: [(10+10)*0.6+(10+10)*0.4]/(1+0)=20


# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_cleaned(rows, dest):
    fields = ['Society','Year','Node','Coherence','Capacity','Stress',
              'Abstraction','Node Value','Bond Strength']
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k,'') for k in fields})

def rows_to_index(rows):
    """Returns {year_str: {node: [C,K,S,A]}}"""
    out = {}
    for row in rows:
        yr   = str(int(float(row['Year'])))
        node = row['Node'].strip()
        vals = [float(row['Coherence']), float(row['Capacity']),
                float(row['Stress']),    float(row['Abstraction'])]
        out.setdefault(yr, {})[node] = vals
    return out

def rows_to_telescope(rows):
    """Returns {year_str: {nodes:{...}, b, dominant, weakest, praetorian, cog_gap, sk, ms, mk}}"""
    by_year = {}
    for row in rows:
        yr   = str(int(float(row['Year'])))
        node = row['Node'].strip()
        by_year.setdefault(yr, {})[node] = {
            'C': round(float(row['Coherence']),  3),
            'K': round(float(row['Capacity']),   3),
            'S': round(float(row['Stress']),     3),
            'A': round(float(row['Abstraction']),3),
            'V': round(float(row['Node Value']), 3),
            'b': round(float(row['Bond Strength']) / MAX_BOND, 3),
        }

    result = {}
    for yr, nodes in sorted(by_year.items(), key=lambda x: int(x[0])):
        if len(nodes) < 8:
            continue  # skip incomplete year records

        sys_b   = round(sum(n['b'] for n in nodes.values()) / len(nodes), 3)
        mean_S  = sum(n['S'] for n in nodes.values()) / len(nodes)
        mean_K  = sum(n['K'] for n in nodes.values()) / len(nodes)
        sk      = round(mean_S / mean_K if mean_K else 0, 3)
        dom     = max(nodes, key=lambda k: nodes[k]['V'])
        wk      = min(nodes, key=lambda k: nodes[k]['V'])
        praet   = round(nodes.get('Shield',{}).get('b',0) - nodes.get('Helm',{}).get('b',0), 3)
        slow_b  = [nodes[n]['b'] for n in SLOW_NODES if n in nodes]
        fast_b  = [nodes[n]['b'] for n in FAST_NODES if n in nodes]
        cog_gap = round((sum(slow_b)/len(slow_b) - sum(fast_b)/len(fast_b))
                        if slow_b and fast_b else 0, 3)

        result[yr] = {
            'nodes':     nodes,
            'b':         sys_b,
            'dominant':  dom,
            'weakest':   wk,
            'praetorian':praet,
            'cog_gap':   cog_gap,
            'sk':        sk,
            'ms':        round(mean_S, 2),
            'mk':        round(mean_K, 2),
        }
    return result


# ── 1. FILES TO COPY FROM DESKTOP → cleaned_datasets ─────────────────────────
# All cam5 CSVs go into the repo regardless; only ensemble means go into index.

copy_all = [
    # (src filename, dest filename)
    ('Australia_CAMS5_ensemble_mean_1875_2026.csv', 'Australia_CAMS5_ensemble_mean_1875_2026_cleaned.csv'),
    ('Canada_CAMS5_ensemble_mean_1850_2026.csv',    'Canada_CAMS5_ensemble_mean_1850_2026_cleaned.csv'),
    ('Canada_CAMS5_envelope_1850_2026.csv',         'Canada_CAMS5_envelope_1850_2026_cleaned.csv'),
    ('China_CAMNATIONS5_1800_2025_EnsembleMean.csv','China_CAMNATIONS5_1800_2025_EnsembleMean_cleaned.csv'),
    ('China_CAMS5_envelope_1800_2025.csv',          'China_CAMS5_envelope_1800_2025_cleaned.csv'),
    ('LatimVetus_ensemble.csv',                     'LatimVetus_ensemble_cleaned.csv'),
    ('latium_envelope.csv',                         'latium_envelope_cleaned.csv'),
    ('Norway_CAMS5_ensemble_mean_1880_2026.csv',    'Norway_CAMS5_ensemble_mean_1880_2026_cleaned.csv'),
    ('Norway_CAMS5_envelope_1880_2026.csv',         'Norway_CAMS5_envelope_1880_2026_cleaned.csv'),
    ('Argentina_CAMS5_ensemble_mean_scores.csv',    'Argentina_CAMS5_ensemble_mean_scores_cleaned.csv'),
]

for src_name, dest_name in copy_all:
    src  = SRC / src_name
    dest = CLEANED / dest_name
    if not src.exists():
        print(f"  SKIP (not found): {src_name}")
        continue
    if dest.exists():
        print(f"  EXISTS: {dest_name}")
        continue
    rows = read_csv(src)
    write_cleaned(rows, dest)
    print(f"  Copied: {src_name} -> {dest_name}  ({len(rows)} rows)")


# ── 2. ENSEMBLE MEAN ENTRIES FOR index + telescope ────────────────────────────

# Each tuple: (cleaned_csv_filename, index_key, meta_for_telescope)
ENSEMBLE_MEANS = [
    (
        'Australia_CAMS5_ensemble_mean_1875_2026_cleaned.csv',
        'Australia (cam5 ensemble)',
        {'era': '1875–2026', 'baseline': 'Australia — 5-agent ensemble mean', 'color': '#20a0b0'},
    ),
    (
        'Canada_CAMS5_ensemble_mean_1850_2026_cleaned.csv',
        'Canada (cam5 ensemble)',
        {'era': '1850–2026', 'baseline': 'Canada — 5-agent ensemble mean', 'color': '#c06080'},
    ),
    (
        'China_CAMNATIONS5_1800_2025_EnsembleMean_cleaned.csv',
        'China (cam5 ensemble)',
        {'era': '1800–2025', 'baseline': 'China — CAMNations5 ensemble mean', 'color': '#d87030'},
    ),
    (
        'LatimVetus_ensemble_cleaned.csv',
        'Latium Vetus (ensemble)',
        {'era': '460–‐2010', 'baseline': 'Latium Vetus — ensemble mean', 'color': '#c87820'},
    ),
    (
        'Norway_CAMS5_ensemble_mean_1880_2026_cleaned.csv',
        'Norway (cam5 ensemble)',
        {'era': '1880–2026', 'baseline': 'Norway — 5-agent ensemble mean', 'color': '#2ca02c'},
    ),
    # Already in cleaned_datasets but not yet in index:
    (
        'Argentina_CAMS5_ensemble_1950_2026_cleaned.csv',
        'Argentina (cam5 ensemble)',
        {'era': '1950–2026', 'baseline': 'Argentina — 5-agent ensemble', 'color': '#9467bd'},
    ),
    (
        'Russia_cams5_ensemble_mean_cleaned.csv',
        'Russia (cam5 ensemble)',
        {'era': '1800–2026', 'baseline': 'Russia — cam5 ensemble mean', 'color': '#8c564b'},
    ),
    (
        'Thailand_CAMS5_ensemble_1850_2026_cleaned.csv',
        'Thailand (cam5 ensemble)',
        {'era': '1850–2026', 'baseline': 'Thailand — 5-agent ensemble', 'color': '#e377c2'},
    ),
    (
        'UK_CAMNATIONS5_ensemble_mean_cleaned.csv',
        'United Kingdom (cam5 ensemble)',
        {'era': '1800–2026', 'baseline': 'UK — CAMNations5 ensemble mean', 'color': '#6070c0'},
    ),
    (
        'germany_cams5_ensemble_mean_cleaned.csv',
        'Germany (cam5 ensemble)',
        {'era': '1800–2026', 'baseline': 'Germany — cam5 ensemble mean', 'color': '#90b030'},
    ),
]


# ── 3. UPDATE cams_index.json ─────────────────────────────────────────────────

with open(INDEX_PATH, encoding='utf-8') as f:
    index = json.load(f)

telescope_societies = {}   # key → telescope DATA dict (for step 4)

for cleaned_name, idx_key, meta in ENSEMBLE_MEANS:
    path = CLEANED / cleaned_name
    if not path.exists():
        print(f"  SKIP index (file missing): {cleaned_name}")
        continue

    rows = read_csv(path)
    if not rows:
        print(f"  SKIP index (empty): {cleaned_name}")
        continue

    # Check if CSV has V and Bond columns
    has_vb = 'Node Value' in rows[0] and 'Bond Strength' in rows[0]

    # Index update (C,K,S,A only)
    entries = rows_to_index(rows)
    if idx_key not in index:
        index[idx_key] = {}
    for yr, nodes in entries.items():
        index[idx_key].setdefault(yr, {}).update(nodes)
    print(f"  Index: '{idx_key}' — {len(entries)} years")

    # Telescope data (needs V and Bond)
    if has_vb:
        tel = rows_to_telescope(rows)
        telescope_societies[idx_key] = (tel, meta)
        print(f"  Telescope: '{idx_key}' — {len(tel)} years")
    else:
        print(f"  Telescope SKIP (no V/Bond): {cleaned_name}")

with open(INDEX_PATH, 'w', encoding='utf-8') as f:
    json.dump(index, f, separators=(',', ':'))
print(f"\nIndex written — {len(index)} societies total")


# ── 4. INJECT INTO TELESCOPE HTML FILES ───────────────────────────────────────

TELESCOPE_FILES = [
    REPO / 'test' / 'telescope_1.html',
    REPO / 'test' / 'telescope_2.html',
    REPO / 'test' / 'telescope_standalone.html',
]

# Colours for new societies (short display labels for buttons)
LABEL_MAP = {
    'Australia (cam5 ensemble)':     'AUS (ens)',
    'Canada (cam5 ensemble)':        'CAN (ens)',
    'China (cam5 ensemble)':         'China (ens)',
    'Latium Vetus (ensemble)':       'Latium Vetus',
    'Norway (cam5 ensemble)':        'Norway (ens)',
    'Argentina (cam5 ensemble)':     'ARG (ens)',
    'Russia (cam5 ensemble)':        'Russia (ens)',
    'Thailand (cam5 ensemble)':      'Thai (ens)',
    'United Kingdom (cam5 ensemble)':'UK (ens)',
    'Germany (cam5 ensemble)':       'GER (ens)',
}

def inject_telescope(html_path):
    txt = open(html_path, encoding='utf-8').read()

    # ── a) find the DATA JSON and parse it ───────────────────────────────
    data_match = re.search(r'(const DATA\s*=\s*)(\{.*?\})(;?\s*\n)', txt, re.DOTALL)
    if not data_match:
        print(f"  SKIP {html_path.name}: DATA not found")
        return

    prefix  = data_match.group(1)
    raw_json= data_match.group(2)
    suffix  = data_match.group(3)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"  SKIP {html_path.name}: JSON parse error: {e}")
        return

    added = []
    for key, (years_data, _) in telescope_societies.items():
        label = LABEL_MAP.get(key, key)
        if label in data:
            print(f"  EXISTS in {html_path.name}: {label}")
            continue
        data[label] = years_data
        added.append(label)

    if not added:
        print(f"  No new societies for {html_path.name}")
        return

    new_json = json.dumps(data, separators=(',', ':'))
    txt = txt[:data_match.start()] + prefix + new_json + suffix + txt[data_match.end():]

    # ── b) inject SOCIETY_META entries ───────────────────────────────────
    meta_match = re.search(r'(const SOCIETY_META\s*=\s*\{)(.*?)(\};)', txt, re.DOTALL)
    if meta_match:
        existing_meta = meta_match.group(2)
        new_entries = ''
        for key, (_, meta) in telescope_societies.items():
            label = LABEL_MAP.get(key, key)
            if f"'{label}'" in existing_meta or f'"{label}"' in existing_meta:
                continue
            new_entries += (
                f"\n  '{label}': "
                f"{{era:'{meta['era']}', baseline:'{meta['baseline']}', color:'{meta['color']}' }},"
            )
        if new_entries:
            txt = (txt[:meta_match.start(1)] +
                   meta_match.group(1) + existing_meta + new_entries +
                   meta_match.group(3) +
                   txt[meta_match.end():])

    open(html_path, 'w', encoding='utf-8').write(txt)
    print(f"  Updated {html_path.name}: added {added}")

for tf in TELESCOPE_FILES:
    if tf.exists():
        print(f"\nInjecting into {tf.name}...")
        inject_telescope(tf)
    else:
        print(f"  NOT FOUND: {tf}")

print("\nDone.")
