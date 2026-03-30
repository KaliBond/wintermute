/**
 * CAMS Blind Inference Pipeline — Data Preparation
 * Generates blinded_001.csv through blinded_005.csv
 *
 * Format: Society (MarkerXXX), Year, Node, Coherence, Capacity, Stress,
 *         Abstraction, Node Value, Bond Strength
 *
 * Bond Strength per node = mean of all pairwise B_ij for that node,
 *   where B_ij = sqrt(max(Vi+8,0) * max(Vj+8,0)) / 8
 *   (divisor 8 gives values in the ~1–3 range typical of CAMS scored data)
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

// Blinding assignment — fixed for reproducibility; do not reorder after publication
const ASSIGNMENT = ['Germany','China','Italy','USA','Singapore'];
// blinded_001 = Germany, _002 = China, _003 = Italy, _004 = USA, _005 = Singapore

// Marker labels (shown in Society column instead of real names)
const MARKERS = ['Marker001','Marker002','Marker003','Marker004','Marker005'];

// ── Math ─────────────────────────────────────────────────────────────────────

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

function nodeValue(C, K, S, A) {
    return C + K + (A / 2) - S;
}

// Per-node Bond Strength = mean of all pairwise B_ij for that node
// B_ij = sqrt(max(Vi+8,0) * max(Vj+8,0)) / 8
function meanBondStrength(nodeIdx, viValues) {
    const vi_i = viValues[nodeIdx];
    const a    = Math.sqrt(Math.max(vi_i + 8, 0));
    let   sum  = 0;
    let   cnt  = 0;
    for (let j = 0; j < viValues.length; j++) {
        if (j === nodeIdx) continue;
        sum += a * Math.sqrt(Math.max(viValues[j] + 8, 0)) / 8;
        cnt++;
    }
    return cnt > 0 ? sum / cnt : 0;
}

function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
}

function fmt(v, dp = 3) {
    return v.toFixed(dp);
}

// ── Main ─────────────────────────────────────────────────────────────────────

const rawPath = path.join(__dirname, '..', 'cleaned_datasets', 'cams_index.json');
const raw     = JSON.parse(fs.readFileSync(rawPath, 'utf8'));
const outDir  = path.join(__dirname, '..', 'blinded');

const key = {};

ASSIGNMENT.forEach((society, idx) => {
    const marker   = MARKERS[idx];
    const fileNum  = String(idx + 1).padStart(3, '0');
    const fileName = `blinded_${fileNum}.csv`;
    const outPath  = path.join(outDir, fileName);

    key[fileName] = { society, marker };

    const rng = makePRNG(0xDEAD0000 + idx * 0x1337);

    const socData = raw[society];
    if (!socData) { console.error(`Society not found: ${society}`); process.exit(1); }

    const years = Object.keys(socData)
        .map(Number)
        .filter(y => y >= MIN_YEAR)
        .sort((a, b) => a - b);

    const header = 'Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength';
    const lines  = [header];

    for (const yr of years) {
        // Year jitter ±1
        const jitter     = Math.round(rng() * 0.5);
        const jitteredYr = yr + Math.sign(rng()) * (Math.abs(jitter) <= 1 ? Math.abs(jitter) : 1);

        const nodeData = socData[yr];

        // Noisy metrics
        const noisy = {};
        for (const node of NODES) {
            if (!nodeData[node]) {
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
        }

        // Node Values
        const viValues = NODES.map(n => nodeValue(noisy[n].C, noisy[n].K, noisy[n].S, noisy[n].A));

        // One row per node
        for (let i = 0; i < NODES.length; i++) {
            const node = NODES[i];
            const { C, K, S, A } = noisy[node];
            const Vi = viValues[i];
            const BS = meanBondStrength(i, viValues);
            const row = [
                marker,
                jitteredYr,
                node,
                fmt(C), fmt(K), fmt(S), fmt(A),
                fmt(Vi),
                fmt(BS)
            ];
            lines.push(row.join(','));
        }
    }

    fs.writeFileSync(outPath, lines.join('\n') + '\n', 'utf8');
    console.log(`${fileName}  →  ${society} (${marker})  (${years.length} years, ${lines.length - 1} rows)`);
});

fs.writeFileSync(
    path.join(outDir, 'BLINDING_KEY.json'),
    JSON.stringify(key, null, 2) + '\n',
    'utf8'
);

console.log('\nBlinding key saved to blinded/BLINDING_KEY.json');
console.log('Do not commit BLINDING_KEY.json alongside the blinded CSVs.');
