// Verifier v2, layer A: recompute the lab's quantities headlessly and compare against
// the audit's published outputs (see criteria.md). Exit 1 on any failure.
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '../../../shannon-laboratory.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) { console.error('FAIL: no script in lab'); process.exit(1); }
const els = {};
global.document = { getElementById: id => els[id] ?? (els[id] = { value: '42', textContent: '', innerHTML: '', addEventListener() {} }), createElement: () => ({ click() {}, remove() {}, style: {} }), body: { append() {} } };
global.URL = { createObjectURL: () => 'x', revokeObjectURL() {} };
global.setTimeout = f => 0;
const src = m[1].replace(/function fullRun\(\)[\s\S]*$/, '')
  + '\nglobal.__L={shannonOne,ablationFor,mutedEdges,simulateLong,columns,tePair,condTotalCorr,oInfo,memRunOnce,echoSweep};';
eval(src);
const L = global.__L;
const c = { gateA: .1, gateB: .9, authority: .75, evidence: .65, shock: 18, seed: 42, bins: 3, estRounds: 2000, flip: .04 };
let fails = 0;
const chk = (id, label, got, want, tol) => {
  const ok = Math.abs(got - want) <= tol;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${id} ${label}: got ${got.toFixed(4)}, want ${want} ±${tol}`);
  if (!ok) fails++;
};
const A = L.shannonOne(c.gateA, c), B = L.shannonOne(c.gateB, c);
chk('A1a', 'A envMI', A.envMI, 0.170, 0.002); chk('A1b', 'A up TE', A.up, 0.108, 0.002);
chk('A1c', 'A down TE', A.down, 0.017, 0.002); chk('A1d', 'A TC', A.tc, 5.455, 0.002);
chk('A1e', 'B envMI', B.envMI, 0.248, 0.002); chk('A1f', 'B up TE', B.up, 0.156, 0.002);
chk('A1g', 'B down TE', B.down, 0.011, 0.002); chk('A1h', 'B TC', B.tc, 5.803, 0.002);
const upMuteRun = gate => { const run = L.simulateLong(gate, c, [5, 4, 7]); const cols = L.columns(run, c.bins); return (L.tePair(cols[5], cols[0]) + L.tePair(cols[4], cols[0]) + L.tePair(cols[7], cols[0])) / 3; };
chk('A2a', 'all3-muted up baseline A', upMuteRun(c.gateA), 0.082, 0.004);
chk('A2b', 'all3-muted up baseline B', upMuteRun(c.gateB), 0.082, 0.004);
chk('A3a', 'A up|E', A.upE, 0.042, 0.004); chk('A3b', 'A down|E', A.downE, 0.055, 0.004);
chk('A3c', 'B up|E', B.upE, 0.072, 0.004); chk('A3d', 'B down|E', B.downE, 0.041, 0.004);
chk('A4a', 'A TC|E', A.tce, 3.733, 0.004); chk('A4b', 'B TC|E', B.tce, 3.824, 0.004);
chk('A5a', 'A O-info', A.oinfo, 3.897, 0.01); chk('A5b', 'B O-info', B.oinfo, 4.402, 0.01);
const pts = L.echoSweep(c);
const f0 = pts[0], f1 = pts[pts.length - 1];
chk('A6a', 'echo dTC|E (gate B)', f1.Btce - f0.Btce, 0.151, 0.02);
chk('A6b', 'echo dEnvMI (gate B)', f1.Benv - f0.Benv, -0.020, 0.01);
const content = (gate, slip) => L.memRunOnce(gate, c, { env: 'periodic', archive: 'calendar', slip }) - L.memRunOnce(gate, c, { env: 'periodic', archive: 'misaligned', slip });
chk('A7a', 'record content A (slip 0)', content(c.gateA, 0), 0.218, 0.05);
chk('A7b', 'record content B (slip 0)', content(c.gateB, 0), 0.085, 0.04);
const cc = content(c.gateA, 0), c4 = content(c.gateA, 0.04);
console.log(`${cc > c4 ? 'PASS' : 'FAIL'} A7c record content decays with slips: ${cc.toFixed(3)} -> ${c4.toFixed(3)}`);
if (!(cc > c4)) fails++;
chk('A8a', 'up null p at floor (A)', A.upNull.p, 0.005, 0.001);
chk('A8b', 'null floor printed', A.upNull.floor, 0.005, 0.0001);
console.log(fails ? `FAIL: ${fails} numeric checks failed` : 'ALL NUMERIC CHECKS PASS');
process.exit(fails ? 1 : 0);
