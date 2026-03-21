/**
 * build_index_from_clean.js
 * Processes positive-S CSVs from OneDrive Desktop/clean, Downloads, and Downloads/csv
 * Maps all node name variants → CAMS 3.0 standard names
 * Supports multiple datasets per nation via tagged society keys e.g. "China [Grok]"
 * Merges into existing cams_index.json (preserves manually calibrated data)
 */
const fs = require('fs');
const path = require('path');

const SRC   = 'C:/Users/julie/OneDrive/Desktop/clean/';
const DL    = 'C:/Users/julie/Downloads/';
const CSV   = 'C:/Users/julie/Downloads/csv/';
const INDEX = 'C:/Users/julie/wintermute/cleaned_datasets/cams_index.json';

// Files to process (positive-S only; row-level filter handles mixed files)
const FILES = [
    // ── OneDrive/Desktop/clean — ancient & regional ───────────────────────
    { src: SRC, file: 'Italy_CAMS_Cleaned.csv',      society: 'Italy' },
    { src: SRC, file: 'Palestine_CAMS_Cleaned.csv',  society: 'Palestine' },
    { src: SRC, file: 'ATHENS.CSV',                  society: 'Athens' },
    { src: SRC, file: 'SPARTA.CSV',                  society: 'Sparta' },
    { src: SRC, file: 'germany.csv',                 society: 'Germany' },
    { src: SRC, file: 'canada.csv',                  society: 'Canada' },

    // ── Downloads — primary sources ───────────────────────────────────────
    { src: DL,  file: 'Singapore_gem_d3c.csv',       society: 'Singapore' },
    { src: DL,  file: 'Japan 1850 2025.csv',         society: 'Japan' },
    { src: DL,  file: 'england.csv',                 society: 'England' },
    { src: DL,  file: 'Russia .csv',                 society: 'Russia' },
    { src: DL,  file: 'Saudi_Gem_Feb.csv',           society: 'Saudi Arabia' },
    { src: DL,  file: 'USA_Gem_2_Feb28.csv',         society: 'USA' },
    { src: DL,  file: 'China_Gem_Feb.csv',           society: 'China' },
    { src: DL,  file: 'Egypt_Gem_23Jan26.csv',       society: 'Egypt' },
    { src: DL,  file: 'phil_gem_dec.csv',            society: 'Philippines' },
    { src: DL,  file: 'LATIVA.CSV',                  society: 'Latvia' },
    { src: DL,  file: 'norway_gem_jan.csv',          society: 'Norway' },

    // ── Downloads/csv — primary (new nations + better sources) ───────────
    { src: CSV, file: 'Australian_Gem_Jan30.csv',                      society: 'Australia' },
    { src: CSV, file: 'Germany_gem_jan.csv',                           society: 'Germany' },
    { src: CSV, file: 'Iran 1900 -2025 Grok 12-1.csv',                 society: 'Iran' },
    { src: CSV, file: 'sweden_gem_jan26.csv',                          society: 'Sweden' },
    { src: CSV, file: 'Finland_Gem_21Jan.csv',                         society: 'Finland' },
    { src: CSV, file: 'Ukraine_gem_1930_Jan26_CLEAN (1).csv',          society: 'Ukraine' },
    { src: CSV, file: 'Venezuela_gem_dec1970_2025 (1).csv',            society: 'Venezuela' },
    { src: CSV, file: 'southafrica_nov (1) (1).csv',                   society: 'South Africa' },
    { src: CSV, file: 'ROME_new_gem_nov_CLEANED_RENAMED_FILLED (1).csv', society: 'Rome' },

    // ── Downloads/csv — alternative sources (tagged for comparison) ───────
    { src: CSV, file: 'australia_grok_feb (1).csv',                    society: 'Australia [Grok]' },
    { src: CSV, file: 'china_nov_new_gem_cleaned (1).csv',             society: 'China [Gem alt]' },
    { src: CSV, file: 'russia_gem_december (1).csv',                   society: 'Russia [Gem]' },
    { src: CSV, file: 'saudi_grok_february_cleaned (1).csv',           society: 'Saudi Arabia [Grok]' },
];

// Node name → CAMS 3.0 standard name
const NODE_MAP = {
    'executive':                           'Helm',
    'helm':                                'Helm',
    'army':                                'Shield',
    'shield':                              'Shield',
    'military':                            'Shield',
    'priests':                             'Lore',
    'priesthood / knowledge workers':      'Lore',
    'priesthood':                          'Lore',
    'knowledge workers':                   'Lore',
    'lore':                                'Lore',
    'property owners':                     'Stewards',
    'property':                            'Stewards',
    'stewards':                            'Stewards',
    'trades/professions':                  'Craft',
    'trades / professions':                'Craft',
    'trades/prof.':                        'Craft',
    'craft':                               'Craft',
    'proletariat':                         'Hands',
    'hands':                               'Hands',
    'labour':                              'Hands',
    'state memory':                        'Archive',
    'statememory':                         'Archive',
    'archive':                             'Archive',
    'shopkeepers/merchants':               'Flow',
    'shopkeepers / merchants':             'Flow',
    'merchants / shopkeepers':             'Flow',
    'merchants':                           'Flow',
    'shopkeepers':                         'Flow',
    'flow':                                'Flow',
    'commerce':                            'Flow',
};

function mapNode(raw) {
    return NODE_MAP[raw.toLowerCase().trim()] || null;
}

function parseCSV(filepath) {
    const raw = fs.readFileSync(filepath, 'utf8').replace(/^\uFEFF/, '');
    const lines = raw.split(/\r?\n/).filter(l => l.trim());
    const header = lines[0].split(',').map(h => h.trim().toLowerCase());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        const row = {};
        header.forEach((h, j) => row[h] = (cols[j] || '').trim());
        rows.push(row);
    }
    return rows;
}

function parseNum(v) {
    const n = parseFloat(v);
    return isNaN(n) ? null : n;
}

function clamp(v) {
    return Math.max(1, Math.min(10, Math.round(v * 2) / 2));
}

// Load existing index
const index = JSON.parse(fs.readFileSync(INDEX, 'utf8'));

let totalAdded = 0;
let totalSkipped = 0;

for (const { src, file, society } of FILES) {
    const filepath = path.join(src, file);
    if (!fs.existsSync(filepath)) {
        console.log(`SKIP (not found): ${file}`);
        continue;
    }

    const rows = parseCSV(filepath);
    let added = 0, skipped = 0;

    // Group by year
    const byYear = {};
    for (const row of rows) {
        // Get year (various column names)
        const yearRaw = row['year'] || row['period'] || '';
        if (!yearRaw) continue;
        const yearNum = Math.round(parseFloat(yearRaw));
        if (isNaN(yearNum)) continue;

        // Get node name
        const nodeName = row['node'] || '';
        const camsNode = mapNode(nodeName);
        if (!camsNode) {
            // skip 'Node' header row or unknown nodes
            continue;
        }

        // Get CKSA values
        const C = parseNum(row['coherence']);
        const K = parseNum(row['capacity']);
        const S = parseNum(row['stress']);
        const A = parseNum(row['abstraction']);

        if (C === null || K === null || S === null || A === null) continue;

        // Only accept positive S (skip negative-stress files entirely at row level)
        if (S < 0) { skipped++; continue; }

        if (!byYear[yearNum]) byYear[yearNum] = {};
        byYear[yearNum][camsNode] = [
            clamp(C), clamp(K), clamp(S), clamp(A)
        ];
    }

    // Merge into index — only write years that have all 8 nodes
    if (!index[society]) index[society] = {};

    for (const [yr, nodes] of Object.entries(byYear)) {
        const complete = Object.keys(nodes).length === 8;
        if (!complete) { skipped++; continue; }

        // Don't overwrite manually calibrated Germany 1931
        if (society === 'Germany' && yr === 1931 && index['Germany']?.['1931']) {
            console.log(`  Preserving manually calibrated Germany 1931`);
            continue;
        }

        index[society][String(yr)] = nodes;
        added++;
    }

    console.log(`${society} (${file}): ${added} years added, ${skipped} rows skipped`);
    totalAdded += added;
    totalSkipped += skipped;
}

fs.writeFileSync(INDEX, JSON.stringify(index));
console.log(`\nDone. Total years added: ${totalAdded}, rows skipped: ${totalSkipped}`);
console.log(`Nations in index: ${Object.keys(index).length}`);
console.log(`Nations: ${Object.keys(index).sort().join(', ')}`);
