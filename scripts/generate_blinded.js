/**
 * CAMS Blind Inference Pipeline — Data Preparation
 * Generates blinded_001.csv through blinded_005.csv
 *
 * For each target society:
 *  - Strips identity columns
 *  - Jitters years ±1 (fixed per-society seed for reproducibility)
 *  - Adds Gaussian noise σ=0.3 to C, K, S, A
 *  - Computes Vi and all 28 upper-triangle Bij per timestep
 *  - Outputs long-format CSV (Year, Node, C, K, S, A, Vi, B_X_Y × 28)
 *
 * Key file saved separately to blinded/BLINDING_KEY.json (not for release).
 */

'use strict';
const fs   = require('fs');
const path = require('path');

// ── Config ───────────────────────────────────────────────────────────────────

const SOCIETIES = ['China', 'USA', 'Singapore', 'Italy', 'Germany'];
const NODES     = ['Archive','Craft','Flow','Hands','Helm','Lore','Shield','Stewards'];
const SIGMA     = 0.3;
const MIN_YEAR  = 1900;

// Blinding assignment — randomised once, fixed here for reproducibility
// Shuffled from SOCIETIES using seed below; do not reorder after publication
const ASSIGNMENT = ['Germany','China','Italy','USA','Singapore'];
// blinded_001 = Germany, _002 = China, _003 = Italy, _004 = USA, _005 = Singapore

// ── Math ─────────────────────────────────────────────────────────────────────

// Box-Muller Gaussian (seeded via simple LCG)
function makePRNG(seed) {
    let s = seed >>> 0;
    return function() {
        s = (Math.imul(1664525, s) + 1013904223) >>> 0;
        const u1 = s / 4294967296;
        s = (Math.imul(1664525, s) + 1013904223) >>> 0;
        const u2 = s / 4294967296;
        return Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    };
}

function gaussian(rng, mu, sigma) {
    return mu + sigma * rng();
}

function vi(C, K, S, A) {
    return C + K + (A / 2) - S;
}

function bij(vi_i, vi_j) {
    return Math.sqrt(Math.max(vi_i + 8, 0) * Math.max(vi_j + 8, 0)) / 32;
}

function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
}

// ── CSV helpers ──────────────────────────────────────────────────────────────

// Pre-compute 28 upper-triangle node pairs
const PAIRS = [];
for (let i = 0; i < NODES.length; i++)
    for (let j = i + 1; j < NODES.length; j++)
        PAIRS.push([NODES[i], NODES[j]]);

function buildHeader() {
    const base = ['Year','Node','C','K','S','A','Vi'];
    const bcols = PAIRS.map(([a,b]) => `B_${a}_${b}`);
    return [...base, ...bcols].join(',');
}

function fmt(v, dp = 4) {
    return v.toFixed(dp);
}

// ── Main ─────────────────────────────────────────────────────────────────────

const rawPath = path.join(__dirname, '..', 'cleaned_datasets', 'cams_index.json');
const raw     = JSON.parse(fs.readFileSync(rawPath, 'utf8'));
const outDir  = path.join(__dirname, '..', 'blinded');

const key = {};

ASSIGNMENT.forEach((society, idx) => {
    const fileNum  = String(idx + 1).padStart(3, '0');
    const fileName = `blinded_${fileNum}.csv`;
    const outPath  = path.join(outDir, fileName);

    key[fileName] = society;

    const rng = makePRNG(0xDEAD0000 + idx * 0x1337);

    const socData = raw[society];
    if (!socData) { console.error(`Society not found: ${society}`); process.exit(1); }

    // Collect years ≥ MIN_YEAR, sorted
    const years = Object.keys(socData)
        .map(Number)
        .filter(y => y >= MIN_YEAR)
        .sort((a, b) => a - b);

    const lines = [buildHeader()];

    for (const yr of years) {
        // Year jitter ±1
        const jitter   = Math.round(rng() * 0.5);   // ~±1 via rounded normal
        const jitteredYr = yr + Math.sign(rng()) * (Math.abs(jitter) <= 1 ? Math.abs(jitter) : 1);

        const nodeData = socData[yr];

        // Noisy metrics & Vi per node
        const noisy = {};
        const viMap = {};
        for (const node of NODES) {
            if (!nodeData[node]) {
                // Fill missing nodes with neutral values
                noisy[node] = { C: 5, K: 5, S: 5, A: 5 };
            } else {
                const [C, K, S, A] = nodeData[node];
                noisy[node] = {
                    C: clamp(gaussian(rng, C, SIGMA), 1, 10),
                    K: clamp(gaussian(rng, K, SIGMA), 1, 10),
                    S: clamp(gaussian(rng, S, SIGMA), 1, 10),
                    A: clamp(gaussian(rng, A, SIGMA), 1, 10),
                };
            }
            viMap[node] = vi(noisy[node].C, noisy[node].K, noisy[node].S, noisy[node].A);
        }

        // 28 Bij values for this timestep
        const bijMap = {};
        for (const [a, b] of PAIRS) {
            bijMap[`B_${a}_${b}`] = bij(viMap[a], viMap[b]);
        }

        // One row per node
        for (const node of NODES) {
            const { C, K, S, A } = noisy[node];
            const Vval = viMap[node];
            const bijCols = PAIRS.map(([a,b]) => fmt(bijMap[`B_${a}_${b}`]));
            const row = [
                jitteredYr,
                node,
                fmt(C), fmt(K), fmt(S), fmt(A),
                fmt(Vval),
                ...bijCols
            ];
            lines.push(row.join(','));
        }
    }

    fs.writeFileSync(outPath, lines.join('\n') + '\n', 'utf8');
    console.log(`${fileName}  →  ${society}  (${years.length} years, ${(lines.length - 1)} rows)`);
});

// Save key (not for public release)
fs.writeFileSync(
    path.join(outDir, 'BLINDING_KEY.json'),
    JSON.stringify(key, null, 2) + '\n',
    'utf8'
);

console.log('\nBlinding key saved to blinded/BLINDING_KEY.json');
console.log('Do not commit BLINDING_KEY.json alongside the blinded CSVs.');
