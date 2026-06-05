/* ================================================================
   CAMS · Compare — comparative dashboard for any two societies.
   Drives every chart off the REAL compute kernel (cams-compute.js)
   applied to the REAL ensemble data (cams-data.js). No fabricated
   indices: every series is V̄, σ_V, κ, λ₂, B(t), V_i, C/K/S/A or
   an attractor label straight out of computeAll().
   ================================================================ */

const NODES_CMP = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'];

const NODE_HEX_CMP = {
  Lore:'#2E8B6B', Archive:'#2F7A5C', Helm:'#C84C3E', Stewards:'#D98B2B',
  Shield:'#B0392E', Craft:'#3A6FA8', Hands:'#4B7CAE', Flow:'#6A8FB8',
};
const NODE_LAYER_CMP = {
  Lore:'Mythic', Archive:'Mythic',
  Helm:'Interface', Stewards:'Interface', Shield:'Interface',
  Craft:'Material', Hands:'Material', Flow:'Material',
};

// Mode (measure) colors from the design system
const MODE_HEX = { C:'#2F7ECC', K:'#2E8B6B', S:'#C84C3E', A:'#8A5BB8' };
const MODE_NAME = { C:'Coherence', K:'Capacity', S:'Stress', A:'Abstraction' };

// Attractor basin colors (match the Guide)
const ATTRACTOR_HEX = {
  'Re-synchronisation':'#5C8C72',
  'Oscillation':'#C29A28',
  'Buffering':'#C28856',
  'Fracture':'#C84C3E',
  'Thermodynamic Freeze':'#8B2E22',
};

// κ tier thresholds (v3.2-R)
const KAPPA_TIERS = [
  { name:'NOMINAL',  max:0.30, hex:'#5C8C72' },
  { name:'WATCH',    max:0.35, hex:'#7BAE96' },
  { name:'WARNING',  max:0.42, hex:'#C28856' },
  { name:'CRITICAL', max:0.57, hex:'#C84C3E' },
  { name:'EXTREME',  max:Infinity, hex:'#8B2E22' },
];
function kappaTierHex(k){
  for(const t of KAPPA_TIERS){ if(k < t.max) return t.hex; }
  return '#8B2E22';
}

// Two fixed comparison identities: A = teal, B = ochre.
const COL_A = '#1C8C99';   // society A (teal)
const COL_B = '#CC7A1F';   // society B (ochre)
const COL_A_FILL = 'rgba(28,140,153,0.10)';
const COL_B_FILL = 'rgba(204,122,31,0.10)';

const INK = '#1B2A33', INK5 = '#5A6E78', GRID = 'rgba(90,110,120,0.16)';

/* ---------- Series computation over the kernel ---------- */

const seriesCache = {};
function computeSeries(soc){
  if(seriesCache[soc]) return seriesCache[soc];
  const byYear = DATA[soc];
  if(!byYear) return null;
  const years = Object.keys(byYear).map(Number).sort((a,b)=>a-b);
  const out = {
    soc, years,
    meanV:[], sigmaV:[], kappa:[], kappaTier:[], lambda2:[], Bagg:[],
    helmV:[], attractor:[], alarm:[], tau:[], epsilon:[],
    nodes:{}, // node -> { Vi:[], C:[], K:[], S:[], A:[] }
  };
  NODES_CMP.forEach(n => out.nodes[n] = { Vi:[], C:[], K:[], S:[], A:[] });

  years.forEach(yr => {
    const snap = byYear[yr];
    const matrix = NODES_CMP.map(n => snap[n]); // [C,K,S,A] each
    const r = computeAll(matrix);
    out.meanV.push(r.meanV);
    out.sigmaV.push(r.sigmaV);
    out.kappa.push(r.kappa);
    out.kappaTier.push(r.kappaTier);
    out.lambda2.push(r.lambda2);
    out.Bagg.push(r.Bagg);
    out.helmV.push(r.V[0]);
    out.attractor.push(r.attractor);
    out.alarm.push(r.alarmConfidence);
    out.tau.push(r.tau);
    out.epsilon.push(r.epsilon);
    NODES_CMP.forEach((n,i) => {
      const nd = out.nodes[n];
      nd.Vi.push(r.V[i]);
      nd.C.push(matrix[i][0]); nd.K.push(matrix[i][1]);
      nd.S.push(matrix[i][2]); nd.A.push(matrix[i][3]);
    });
  });
  seriesCache[soc] = out;
  return out;
}

// value at a given year (or null)
function atYear(series, key, yr){
  if(!series) return null;
  const i = series.years.indexOf(yr);
  if(i < 0) return null;
  return series[key][i];
}
function nodeAtYear(series, node, yr){
  if(!series) return null;
  const i = series.years.indexOf(yr);
  if(i < 0) return null;
  const nd = series.nodes[node];
  return { Vi:nd.Vi[i], C:nd.C[i], K:nd.K[i], S:nd.S[i], A:nd.A[i] };
}
// nearest available year <= or closest
function nearestYear(series, yr){
  if(!series || !series.years.length) return null;
  return series.years.reduce((a,b)=> Math.abs(b-yr) < Math.abs(a-yr) ? b : a);
}

/* ---------- Chart.js plumbing ---------- */

let SOC_A = 'Australia', SOC_B = 'Argentina', YEAR = 2026;
const charts = {};

function fmt(v, d=2){
  if(v === null || v === undefined || isNaN(+v)) return '—';
  if(!isFinite(v)) return '∞';
  return (+v).toFixed(d);
}

// vertical "current year" rule
const yearRulePlugin = {
  id:'yearRule',
  afterDraw(chart){
    const yr = chart.config._yearRule;
    if(yr == null) return;
    const x = chart.scales.x; const ya = chart.scales.y;
    if(!x || !ya) return;
    if(yr < x.min || yr > x.max) return;
    const px = x.getPixelForValue(yr);
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(27,42,51,0.34)';
    ctx.lineWidth = 1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(px, ya.top); ctx.lineTo(px, ya.bottom); ctx.stroke();
    ctx.restore();
  }
};

function baseOpts(extra){
  const o = {
    responsive:true, maintainAspectRatio:false,
    animation:false,
    interaction:{ mode:'nearest', intersect:false },
    plugins:{
      legend:{ display:false },
      tooltip:{
        backgroundColor:'#FBF7EC', titleColor:INK, bodyColor:INK,
        borderColor:'#C9BE9E', borderWidth:1, padding:8,
        titleFont:{ family:"'Inter',sans-serif", weight:'700', size:11 },
        bodyFont:{ family:"'JetBrains Mono',monospace", size:11 },
      },
    },
    scales:{
      x:{ type:'linear', ticks:{ color:INK5, maxTicksLimit:8, font:{ family:"'JetBrains Mono',monospace", size:10 } },
          grid:{ color:GRID }, border:{ color:'#C9BE9E' } },
      y:{ ticks:{ color:INK5, maxTicksLimit:6, font:{ family:"'JetBrains Mono',monospace", size:10 } },
          grid:{ color:GRID }, border:{ color:'#C9BE9E' } },
    },
  };
  if(extra) deepMerge(o, extra);
  return o;
}
function deepMerge(t, s){
  for(const k in s){
    if(s[k] && typeof s[k]==='object' && !Array.isArray(s[k])){ t[k] = t[k]||{}; deepMerge(t[k], s[k]); }
    else t[k] = s[k];
  }
  return t;
}

function xRange(){
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  let lo = Infinity, hi = -Infinity;
  [sa,sb].forEach(s=>{ if(s){ lo=Math.min(lo,s.years[0]); hi=Math.max(hi,s.years[s.years.length-1]); }});
  return { lo, hi };
}

function lineData(series, key){
  if(!series) return [];
  return series.years.map((y,i)=>({ x:y, y:series[key][i] }));
}
function nodeLineData(series, node, metric){
  if(!series) return [];
  const arr = series.nodes[node][metric];
  return series.years.map((y,i)=>({ x:y, y:arr[i] }));
}

function makeLine(id, datasets, extra){
  const el = document.getElementById(id);
  if(!el) return null;
  if(charts[id]){ charts[id].destroy(); }
  const { lo, hi } = xRange();
  const opts = baseOpts(deepMerge({ scales:{ x:{ min:lo, max:hi } } }, extra||{}));
  const cfg = { type:'line', data:{ datasets }, options:opts, plugins:[yearRulePlugin] };
  cfg._yearRule = YEAR;
  charts[id] = new Chart(el, cfg);
  return charts[id];
}

const LINE_BASE = { backgroundColor:'transparent', borderWidth:2, pointRadius:0, tension:0.3 };
function dsA(series, key, label){ return Object.assign({ label:label||SOC_A, data:lineData(series,key), borderColor:COL_A }, LINE_BASE); }
function dsB(series, key, label){ return Object.assign({ label:label||SOC_B, data:lineData(series,key), borderColor:COL_B }, LINE_BASE); }
