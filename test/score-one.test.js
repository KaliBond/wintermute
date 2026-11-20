// Tests for score-one.js. Run from the repo root: node --test test/score-one.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const S = require('../score-one.js');

const ROOT = path.join(__dirname, '..');
const req = { society: 'Australia', year: '1937', session: 'K7QX2M' };

function reply(rows, session = req.session) {
  return 'Here are the scores.\n\n```csv\nSociety,Year,Node,Coherence,Capacity,Stress,Abstraction\n'
    + rows.join('\n') + '\n```\n\nSESSION ' + session + '\n';
}
const VALID_ROWS = [
  'Australia,1937,Helm,7,6,4,6',
  'Australia,1937,Shield,6,6,3,5',
  'Australia,1937,Lore,7,7,3,7',
  'Australia,1937,Stewards,6,7,4,6',
  'Australia,1937,Craft,7,7,3,7',
  'Australia,1937,Hands,6,6,4,5',
  'Australia,1937,Archive,7,6,3,6',
  'Australia,1937,Flow,7,7,3,6',
];

// ── the three sample replies ──────────────────────────────────────────────
test('sample 1: valid reply is accepted and produces both CSVs', () => {
  const r = S.validateReply(reply(VALID_ROWS), req);
  assert.deepStrictEqual(r.problems, []);
  assert.deepStrictEqual(r.rowErrors, []);
  assert.strictEqual(r.ok, true);
  assert.deepStrictEqual(r.rows.map((x) => x.Node), S.NODES);

  const long = S.longCsv(r.rows, req).trim().split('\n');
  assert.strictEqual(long[0], 'Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength');
  assert.strictEqual(long.length, 9);
  // Helm: V = C + K - S + 0.5A = 7 + 6 - 4 + 3 = 12
  assert.match(long[1], /^Australia,1937,Helm,7,6,4,6,12\.000,0\.\d{5}$/);

  const wide = S.wideCsv(r.rows, req, 'Claude Opus 5.5', '2026-11-20').trim().split('\n');
  assert.strictEqual(wide.length, 2);
  assert.strictEqual(wide[0].split(',').length, 2 + 32 + 3);
  assert.strictEqual(wide[1].split(',').length, 2 + 32 + 3);
  assert.ok(wide[0].startsWith('Society,Year,Helm_C,Helm_K,Helm_S,Helm_A,Shield_C'));
  assert.ok(wide[1].startsWith('Australia,1937,7,6,4,6,6,6,3,5'));
  assert.ok(wide[1].endsWith('Claude Opus 5.5,K7QX2M,2026-11-20'));
});

test('sample 2: missing node is rejected with a clear message', () => {
  const r = S.validateReply(reply(VALID_ROWS.filter((l) => !l.includes(',Archive,'))), req);
  assert.strictEqual(r.ok, false);
  assert.ok(r.problems.includes('Expected exactly 8 data rows, found 7.'), r.problems.join(' | '));
  assert.ok(r.problems.includes('Missing node: Archive.'), r.problems.join(' | '));
});

test('sample 3: out-of-range score is rejected on its row', () => {
  const rows = VALID_ROWS.slice(); rows[4] = 'Australia,1937,Craft,7,11,3,0';
  const r = S.validateReply(reply(rows), req);
  assert.strictEqual(r.ok, false);
  assert.deepStrictEqual(r.problems, []);
  assert.strictEqual(r.rowErrors.length, 1);
  assert.strictEqual(r.rowErrors[0].row, 5);
  assert.strictEqual(r.rowErrors[0].node, 'Craft');
  assert.deepStrictEqual(r.rowErrors[0].messages, [
    'Capacity is "11"; must be a whole number from 1 to 10',
    'Abstraction is "0"; must be a whole number from 1 to 10',
  ]);
});

// ── further checks ────────────────────────────────────────────────────────
test('SESSION mismatch, wrong year, wrong society, duplicate node, CONTAMINATED', () => {
  assert.match(S.validateReply(reply(VALID_ROWS, 'ZZZZZZ'), req).problems[0], /expected K7QX2M, found ZZZZZZ/);
  assert.match(S.validateReply(reply(VALID_ROWS).replace(/SESSION.*\n/, ''), req).problems[0], /SESSION line missing/);
  const y = VALID_ROWS.slice(); y[0] = 'Australia,1938,Helm,7,6,4,6';
  assert.deepStrictEqual(S.validateReply(reply(y), req).rowErrors[0].messages, ['year is "1938"; expected 1937']);
  const s = VALID_ROWS.slice(); s[1] = 'Canada,1937,Shield,6,6,3,5';
  assert.deepStrictEqual(S.validateReply(reply(s), req).rowErrors[0].messages, ['society is "Canada"; expected "Australia"']);
  const d = VALID_ROWS.slice(); d[7] = 'Australia,1937,Helm,7,7,3,6';
  const rd = S.validateReply(reply(d), req);
  assert.deepStrictEqual(rd.rowErrors[0].messages, ['Helm appears more than once']);
  assert.ok(rd.problems.includes('Missing node: Flow.'));
  assert.match(S.validateReply('CONTAMINATED', req).problems[0], /CONTAMINATED/);
  // Markdown emphasis on the SESSION line is accepted, as in the Observatory
  assert.strictEqual(S.validateReply(reply(VALID_ROWS).replace('SESSION K7QX2M', '**SESSION: K7QX2M**'), req).ok, true);
});

test('prompt quotes Protocol v2.1 verbatim and carries the contamination check and SESSION code', () => {
  const md = fs.readFileSync(path.join(ROOT, 'CAMS_SCORING_PROTOCOL_V2_1.md'), 'utf8').replace(/\r\n/g, '\n');
  assert.strictEqual(S.PROTOCOL, md);
  const p = S.buildPrompt('Australia', '1937', 'K7QX2M');
  assert.ok(p.startsWith(md.trim()));
  assert.match(p, /CONTAMINATION CHECK/);
  assert.match(p, /\nSESSION K7QX2M\n$/);
  assert.match(S.newSession(), /^[A-HJKMNP-Z2-9]{6}$/);
});

test('parseCsv is the Observatory parser, unchanged', () => {
  const src = fs.readFileSync(path.join(ROOT, 'workers/citizen-observatory/src/validate.js'), 'utf8');
  const mine = fs.readFileSync(path.join(ROOT, 'score-one.js'), 'utf8');
  const grab = (s) => s.slice(s.indexOf('function parseCsv'), s.indexOf("return rows.filter((r) => !(r.length === 1 && r[0].trim() === ''));"))
    .split('\n').map((l) => l.trim()).join('\n');
  assert.strictEqual(grab(mine), grab(src));
});

test('Node Value and Bond Strength match the JUNO Calculator formulas', () => {
  const html = fs.readFileSync(path.join(ROOT, 'juno_calculator.html'), 'utf8');
  for (const f of ['q[n] = (0.6*sc[n].C + 0.4*sc[n].A)/10;', 'V[n] = sc[n].C + sc[n].K - sc[n].S + 0.5*sc[n].A;',
    'return Math.sqrt(q[n]*q[m]) * Math.pow(2, -(sc[n].S+sc[m].S)/10);',
    "let csv = 'Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength\\n';"]) {
    assert.ok(html.includes(f), 'Calculator no longer contains: ' + f);
  }
  // Independent recomputation for one node.
  const r = S.validateReply(reply(VALID_ROWS), req).rows;
  const q = r.map((x) => (0.6 * x.C + 0.4 * x.A) / 10);
  const b0 = q.slice(1).map((qj, k) => Math.sqrt(q[0] * qj) * Math.pow(2, -(r[0].S + r[k + 1].S) / 10));
  assert.strictEqual(S.nodeValues(r).BS.Helm.toFixed(10), (b0.reduce((a, b) => a + b) / 7).toFixed(10));
});
