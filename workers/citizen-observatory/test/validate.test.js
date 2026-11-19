// Run: npm test   (from workers/citizen-observatory)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateReply, METRICS, parseCsv } from '../src/validate.js';

const PROMPTS = JSON.parse(readFileSync(new URL('../src/prompts.json', import.meta.url), 'utf8'));
const a = PROMPTS['claude-1_batch3'];

function reply({ order = METRICS, session, score = '5', tier = 'B', note = 'Named archive evidence here', fence = '```csv', eol = '\n' } = {}) {
  const lines = ['case_id,arm,rater_id,metric,score,tier,note'];
  for (const c of a.cases) for (const m of order) lines.push(`${c},named,${a.rater},${m},${score},${tier},${note}`);
  return `${fence}\n${lines.join('\n')}\n\`\`\`\n${session ?? 'SESSION ' + a.session}`.replace(/\n/g, eol);
}
const ok = (r) => assert.deepEqual(validateReply(r, a).problems, []);
const bad = (r, re) => assert.ok(validateReply(r, a).problems.some((p) => re.test(p)), JSON.stringify(validateReply(r, a).problems));

test('metric order matches the prompt (C, K, A, S)', () => {
  assert.deepEqual(METRICS.slice(0, 4), ['C1', 'C2', 'C3', 'C4']);
  assert.deepEqual(METRICS.slice(8, 12), ['A1', 'A2', 'A3', 'A4']);
  assert.ok(a.prompt.indexOf('ACTIVITY') < a.prompt.indexOf('STRESS'));
});
test('a correct reply passes, with 128 rows', () => {
  const r = validateReply(reply(), a);
  assert.deepEqual(r.problems, []);
  assert.equal(r.rows.length, 128);
  assert.ok(r.warnings.some((w) => /No NA values/.test(w)));
});
test('plain fence, CRLF and quoted commas pass', () => {
  ok(reply({ fence: '```' }));
  ok(reply({ eol: '\r\n' }));
  ok(reply({ note: '"Censuses, tax rolls"' }));
});
test('SESSION with colon or Markdown emphasis passes (fixed)', () => {
  ok(reply({ session: `SESSION: ${a.session}` }));
  ok(reply({ session: `**SESSION ${a.session}**` }));
  ok(reply({ session: `**SESSION:** ${a.session}` }));
});
test('wrong or missing SESSION code fails', () => {
  bad(reply({ session: 'SESSION ZZZZZZ' }), /SESSION/);
  bad(reply({ session: '' }), /SESSION/);
});
test('all-NA reply is valid but flagged (fixed)', () => {
  const r = validateReply(reply({ score: 'NA', tier: 'NA' }), a);
  assert.deepEqual(r.problems, []);
  assert.ok(r.warnings.some((w) => /Every score is NA/.test(w)));
});
test('structural errors are rejected', () => {
  bad(reply({ tier: 'NA' }), /score\/tier invalid/);
  bad(reply({ score: '11' }), /score\/tier invalid/);
  bad(reply({ score: 'NA', tier: 'B' }), /NA score requires NA tier/);
  bad(reply({ note: '' }), /evidence note missing/);
  bad(reply({ note: Array(16).fill('w').join(' ') }), /longer than 15 words/);
  bad(reply({ order: [...METRICS].reverse() }), /assigned order/);
  bad(reply().replace(/claude-1,/g, 'gpt-2,'), /rater ID/);
  bad(reply().split('\n').slice(0, -3).join('\n') + '\n```\nSESSION ' + a.session, /128 data rows/);
  bad('no code block here at all, just a long enough piece of text', /No CSV code block/);
  bad(reply().replace(`${a.cases[0]},named`, `${a.cases[0]},named,extra`), /field count/);
});
test('CONTAMINATED is rejected with guidance', () => bad('CONTAMINATED', /fresh chat/));
test('parseCsv handles doubled quotes', () => {
  assert.deepEqual(parseCsv('a,"b ""q"" c",d'), [['a', 'b "q" c', 'd']]);
});
test('all 105 prompts carry the fields the Worker needs', () => {
  const ps = Object.values(PROMPTS);
  assert.equal(ps.length, 105);
  for (const p of ps) {
    assert.equal(p.cases.length, 8);
    assert.match(p.prompt, new RegExp(`SESSION ${p.session}`));
    assert.match(p.prompt, new RegExp(`rater_id is always: ${p.rater}`));
  }
});
