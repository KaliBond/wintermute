/* ================================================================
   CAMS · Historical Telescope — embedded improved coordination view
   ================================================================ */

const TS_NODES = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'];

const TS_NODE_HEX = {
  Lore: '#2E8B6B', Archive: '#2F7A5C',
  Helm: '#C84C3E', Stewards: '#D98B2B', Shield: '#B0392E',
  Craft: '#3A6FA8', Hands: '#4B7CAE', Flow: '#6A8FB8',
};

/* ── Imperative force-directed coordination view ── */
function CoordView({ matrix, r, width, height }) {
  const W = r.W, V = r.V, sigma = r.sigma;
  const N = TS_NODES.length;
  const Vabs = V.map(Math.abs);
  const vMin = Math.min(...Vabs), vMax = Math.max(...Vabs, 1);
  const sizeMap = TS_NODES.map((_, i) => {
    const norm = vMax === vMin ? 0.5 : (Vabs[i] - vMin) / (vMax - vMin);
    return 18 + norm * 22;
  });
  const edges = React.useMemo(() => {
    const all = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) all.push({ i, j, w: W[i][j] });
    all.sort((a, b) => b.w - a.w);
    return all.slice(0, 12);
  }, [W]);

  const stateRef = React.useRef(null);
  if (!stateRef.current) {
    stateRef.current = TS_NODES.map((_, i) => {
      const a = (i / N) * Math.PI * 2 - Math.PI / 2;
      const r0 = Math.min(width, height) * 0.30;
      return { x: width/2 + Math.cos(a)*r0, y: height/2 + Math.sin(a)*r0, vx: 0, vy: 0 };
    });
  }
  const nodeRefs = React.useRef([]);
  const edgeRefs = React.useRef([]);
  const haloRefs = React.useRef([]);
  const tRef = React.useRef(0);
  const rafRef = React.useRef(null);
  const [hovered, setHovered] = React.useState(null);

  const inputsRef = React.useRef({ W, matrix, sigma, sizeMap, width, height, edges, hovered });
  inputsRef.current = { W, matrix, sigma, sizeMap, width, height, edges, hovered };

  React.useEffect(() => {
    const nodes = stateRef.current;
    const step = () => {
      tRef.current += 1;
      const t = tRef.current;
      const { W: cW, matrix: cM, sigma: cS, sizeMap: sz, width: w, height: h, edges: eds, hovered: hov } = inputsRef.current;
      const cx = w/2, cy = h/2;
      const motionScale = hov != null ? 0 : 1;
      const Lmin = Math.min(w, h) * 0.18;
      const Lmax = Math.min(w, h) * 0.50;

      const wList = [];
      for (let a=0;a<N;a++) for (let b=a+1;b<N;b++) wList.push(cW[a][b]);
      wList.sort((x,y) => y-x);
      const wThresh = wList[Math.min(11, wList.length-1)] || 0;

      for (let i=0;i<N;i++) {
        let fx=0, fy=0;
        for (let j=0;j<N;j++) {
          if (i===j) continue;
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const d = Math.hypot(dx,dy) + 0.001;
          const ww = cW[i][j];
          if (ww >= wThresh) {
            const L = Lmin + (Lmax-Lmin)*(1-ww);
            const k = 0.012 * (0.25 + ww);
            const f = k*(d - L);
            fx += (dx/d)*f; fy += (dy/d)*f;
          }
          const rep = 6500/(d*d);
          fx -= (dx/d)*rep; fy -= (dy/d)*rep;
        }
        fx += (cx - nodes[i].x) * 0.0028;
        fy += (cy - nodes[i].y) * 0.0028;
        const stress = cM[i][2];
        const amp = Math.max(0, stress-5) * 0.06 * motionScale;
        const phase = i*0.7;
        fx += Math.cos(t*0.025 + phase) * amp;
        fy += Math.sin(t*0.022 + phase*1.3) * amp;
        if (cS[i] < 0) {
          const ox = nodes[i].x - cx, oy = nodes[i].y - cy;
          const od = Math.hypot(ox,oy)+0.001;
          fx += (ox/od)*0.05; fy += (oy/od)*0.05;
        }
        nodes[i].vx = (nodes[i].vx + fx) * (hov!=null ? 0.55 : 0.72);
        nodes[i].vy = (nodes[i].vy + fy) * (hov!=null ? 0.55 : 0.72);
        const vmag = Math.hypot(nodes[i].vx, nodes[i].vy);
        if (vmag > 1.4) { nodes[i].vx = nodes[i].vx/vmag*1.4; nodes[i].vy = nodes[i].vy/vmag*1.4; }
      }
      for (let i=0;i<N;i++) {
        nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy;
        const m = 50;
        if (nodes[i].x<m){nodes[i].x=m;nodes[i].vx*=-0.4;}
        if (nodes[i].x>w-m){nodes[i].x=w-m;nodes[i].vx*=-0.4;}
        if (nodes[i].y<m){nodes[i].y=m;nodes[i].vy*=-0.4;}
        if (nodes[i].y>h-m){nodes[i].y=h-m;nodes[i].vy*=-0.4;}
      }
      for (let k=0;k<eds.length;k++) {
        const e = eds[k]; const el = edgeRefs.current[k]; if (!el) continue;
        const a = nodes[e.i], b = nodes[e.j];
        el.setAttribute('x1',a.x); el.setAttribute('y1',a.y);
        el.setAttribute('x2',b.x); el.setAttribute('y2',b.y);
      }
      for (let i=0;i<N;i++) {
        const g = nodeRefs.current[i];
        if (g) g.setAttribute('transform', `translate(${nodes[i].x}, ${nodes[i].y})`);
        const halo = haloRefs.current[i];
        if (halo) {
          const stress = cM[i][2];
          const pulse = 1 + Math.sin(t*0.08 + i) * (stress>6 ? 0.06 : 0.025);
          halo.setAttribute('r', (sz[i] + 7 + Math.min(18, Math.abs(cS[i])/30)) * pulse);
        }
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return React.createElement('svg', {
    width: '100%', height: '100%', viewBox: `0 0 ${width} ${height}`,
    style: { display:'block', background:'var(--scope-bg2)' },
  },
    React.createElement('g', { opacity: 0.16 },
      [0.20, 0.34, 0.46].map((f,i) => React.createElement('circle', {
        key:i, cx:width/2, cy:height/2, r: Math.min(width,height)*f,
        fill:'none', stroke:'var(--scope-border2)', strokeWidth:0.5, strokeDasharray:'2 4',
      })),
    ),
    React.createElement('g', null,
      edges.map((e,idx) => {
        const opacity = Math.max(0.10, e.w*0.85);
        const stroke = e.w > 0.6 ? 'var(--scope-amber)' : 'var(--scope-text3)';
        const sw = 0.5 + e.w*3.0;
        return React.createElement('line', {
          key: `${e.i}-${e.j}`,
          ref: el => (edgeRefs.current[idx] = el),
          stroke, strokeWidth: sw, opacity, strokeLinecap: 'round',
        });
      }),
    ),
    TS_NODES.map((n,i) => {
      const radius = sizeMap[i];
      const sig = sigma[i];
      const haloColor = sig >= 0 ? 'rgba(0,184,158,0.32)' : 'rgba(197,48,48,0.42)';
      return React.createElement('g', {
        key:n,
        ref: el => (nodeRefs.current[i] = el),
        style: { cursor: 'pointer' },
        onMouseEnter: () => setHovered(i),
        onMouseLeave: () => setHovered(null),
      },
        React.createElement('circle', { ref: el => (haloRefs.current[i] = el), r: radius+7, fill: haloColor }),
        React.createElement('circle', { r: radius, fill: TS_NODE_HEX[n], stroke: 'rgba(0,0,0,0.18)', strokeWidth: 1 }),
        React.createElement('circle', { r: radius*0.42, cx: -radius*0.28, cy: -radius*0.28, fill: 'rgba(255,255,255,0.22)' }),
        React.createElement('text', {
          textAnchor:'middle', dominantBaseline:'central',
          fontFamily:'var(--font-mono)', fontSize: Math.max(9, radius*0.45), fontWeight: 700,
          fill:'#fff', y: 1,
        }, `+${V[i].toFixed(1)}`),
        React.createElement('text', {
          textAnchor:'middle', fontFamily:'var(--font-mono)', fontSize: 10, fontWeight: 600,
          fill: 'var(--scope-text2)', y: -(radius + 8),
        }, n),
      );
    }),
  );
}

/* ── Dossier panels ── */
function DossierCard({ label, children, style }) {
  return React.createElement('div', {
    style: { background: 'var(--scope-bg2)', border: '1px solid var(--scope-border)', borderRadius: 4, padding: '12px 14px', ...style },
  },
    React.createElement('div', {
      style: { fontFamily:'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', marginBottom: 8 },
    }, label),
    children,
  );
}

function NodeBarTS({ name, value, max }) {
  const pct = Math.max(2, Math.min(100, (value / max) * 100));
  return React.createElement('div', { style: { display:'grid', gridTemplateColumns: '70px 1fr 40px', gap: 6, alignItems:'center', margin:'2px 0' } },
    React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text2)' } }, name),
    React.createElement('div', { style: { background: 'var(--scope-bg3)', height: 5, borderRadius: 99 } },
      React.createElement('div', { style: { height: '100%', width: `${pct}%`, background: 'var(--scope-teal)', borderRadius: 99 } }),
    ),
    React.createElement('span', { style: { fontFamily:'var(--font-mono)', fontSize: 10, color: 'var(--scope-text2)', textAlign:'right' } }, `+${value.toFixed(1)}`),
  );
}

/* ── Basin zeitgeist narratives ── */
function describeBasin(basin, r) {
  const m = {
    'Peak Expansion': [
      `Collective confidence is at full expression. Shared purpose requires no enforcement — it is experienced as natural and correct. Long-horizon thinking dominates public discourse. The dominant emotional register is expansive optimism with an undertow of self-congratulation. This is the phase in which the map is mistaken for the territory.`,
      'LOW',
    ],
    'Navigational': [
      `The mood is purposeful. A shared problem-solving orientation pervades institutional life. Pragmatism is valued over ideology. Trust in collective action is high; the social contract feels reciprocal. The dominant emotional register is forward-looking relief — crisis is recent memory, reconstruction is the work.`,
      'LOW',
    ],
    'Early Strain': [
      `Ambient unease without a clear object. Optimism has not inverted but nostalgia is becoming a political force. Institutions feel less competent than memory says they were. Competing narratives are fragmenting the shared discourse. The future is contested rather than assumed.`,
      'LOW',
    ],
    'Praetorian': [
      `Reactive, threat-saturated, rapid-cycling. No long-horizon strategy is coherent — only stimulus-response management. Emergency is the permanent condition. External enemies and internal traitors provide directionality to anxiety that has no structural resolution. Strongman behaviour functions as stress management, not governance.`,
      'CRITICAL',
    ],
    'Archipelago': [
      `Privatised coordination. Real organisation flows through corporate, factional, and network nodes below or beyond the state. Public institutions persist as theatre. Irony and detachment armour people against commitment. The dominant register is solipsistic intensity — deeply tribal within small groups, deeply indifferent outside them.`,
      'MEDIUM',
    ],
    'Mass Formation': [
      `A named anxiety object has crystallised. Approximately 20–30% of the population has entered a hypnosis-like state — capable of collective behaviour that no individual within it would endorse in isolation. Ethical self-awareness is suppressed in the hypnotised cohort. Dissent is experienced as existential threat, not disagreement.`,
      'HIGH',
    ],
    'Phantom I': [
      `Cultural and intellectual life persists — perhaps flourishes in isolated nodes — while the material substrate deteriorates. Art, law, and ritual continue as if the structure still holds. The map is mistaken for the territory, and this mistake is the proudest expression of civilisational identity. Decline is not legible from within the cultural forms that carry its memory.`,
      'MEDIUM',
    ],
    'Phantom II': [
      `Post-political fatigue has set in. Cynicism saturates public discourse; cultural forms persist but lack animating conviction — participation is performative rather than felt. The gap between institutional narrative and lived experience is total and no longer surprising.`,
      'MEDIUM',
    ],
  };
  return m[basin] || ['—', 'LOW'];
}

/* ── App ── */
function TelescopeApp() {
  const nations = React.useMemo(() => Object.keys(DATA), []);
  const [nation, setNation] = React.useState('USA');
  const [years, setYears] = React.useState([]);
  const [year, setYear] = React.useState(null);
  const [matrix, setMatrix] = React.useState(TS_NODES.map(() => [5,5,5,5]));
  const [playing, setPlaying] = React.useState(false);
  const [speedMode, setSpeedMode] = React.useState('Slow');
  const playRef = React.useRef(false);

  const stageRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 700, h: 460 });
  React.useEffect(() => {
    if (!stageRef.current) return;
    const ro = new ResizeObserver(es => {
      for (const e of es) setSize({ w: Math.max(400, e.contentRect.width), h: Math.max(360, e.contentRect.height) });
    });
    ro.observe(stageRef.current);
    return () => ro.disconnect();
  }, []);

  React.useEffect(() => {
    const yrs = Object.keys(DATA[nation] || {}).map(Number).sort((a,b)=>a-b);
    setYears(yrs);
    if (yrs.length) setYear(yrs[Math.floor(yrs.length/2)]);
    setPlaying(false);
  }, [nation]);

  React.useEffect(() => {
    if (!year || !DATA[nation] || !DATA[nation][year]) return;
    const rec = DATA[nation][year];
    setMatrix(TS_NODES.map(n => rec[n] ? [...rec[n]] : [5,5,5,5]));
  }, [nation, year]);

  React.useEffect(() => {
    playRef.current = playing;
    if (!playing) return;
    const ms = speedMode === 'Fast' ? 250 : speedMode === 'Normal' ? 500 : 900;
    let t;
    const tick = () => {
      if (!playRef.current) return;
      setYear(prev => {
        const yrs = Object.keys(DATA[nation] || {}).map(Number).sort((a,b)=>a-b);
        const idx = yrs.indexOf(prev);
        t = setTimeout(tick, ms);
        return idx < yrs.length - 1 ? yrs[idx+1] : yrs[0];
      });
    };
    t = setTimeout(tick, ms);
    return () => clearTimeout(t);
  }, [playing, nation, speedMode]);

  const r = React.useMemo(() => computeAll(matrix), [matrix]);
  const basin = React.useMemo(() => classifyBasin(r, matrix), [r, matrix]);
  const [zeitgeist, threatLevel] = describeBasin(basin, r);

  // Hero metrics
  const meanS = matrix.reduce((a,m) => a + m[2], 0) / matrix.length;
  const meanK = matrix.reduce((a,m) => a + m[1], 0) / matrix.length;
  const sk = meanS / Math.max(0.001, meanK);
  const praet = r.V[1] - r.V[0];
  const cogGap = r.alpha.reduce((a,b)=>a+b,0) / r.alpha.length / 10;

  // Sorted node bars (by V desc)
  const sorted = TS_NODES.map((n,i) => ({ n, v: r.V[i] })).sort((a,b) => b.v - a.v);
  const maxV = Math.max(...sorted.map(s => s.v), 1);

  // Power topology
  const sortedV = TS_NODES.map((n,i) => ({ n, v: r.V[i] })).sort((a,b) => b.v - a.v);
  const dominant = sortedV[0].n, weakest = sortedV[sortedV.length-1].n, shadow = sortedV[1].n;
  const sub = r.meanV > 14 ? 'Dynamic' : r.meanV > 11 ? 'Quiescent' : 'Strained';

  // Trajectory deltas — simple year-on-year
  const yIdx = years.indexOf(year);
  const prevR = yIdx > 0 && DATA[nation][years[yIdx-1]]
    ? computeAll(TS_NODES.map(n => DATA[nation][years[yIdx-1]][n] || [5,5,5,5]))
    : null;
  const skDelta = prevR ? (sk - (prevR.B.reduce((a,b)=>a+b,0)/8)).toFixed(2) : '—';

  const threatColor = threatLevel === 'CRITICAL' ? 'var(--scope-red)' : threatLevel === 'HIGH' ? 'var(--scope-red)' : threatLevel === 'MEDIUM' ? 'var(--scope-amber)' : 'var(--scope-teal2)';

  return React.createElement('div', { style: { background: 'var(--scope-bg)', minHeight: '100vh', color: 'var(--scope-text)', fontFamily: 'var(--font-sans)' } },

    /* Top header */
    React.createElement('header', { style: { padding: '18px 28px 12px', borderBottom: '1px solid var(--scope-border)', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' } },
      React.createElement('div', null,
        React.createElement('div', { style: { fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 200, letterSpacing: '-.01em' } },
          React.createElement('span', { style: { fontWeight: 600, color: 'var(--scope-amber)' } }, 'CAMS '),
          React.createElement('span', { style: { fontWeight: 500, color: 'var(--scope-text)' } }, 'Historical Telescope'),
        ),
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text3)', letterSpacing: '.1em', textTransform: 'uppercase', marginTop: 2 } },
          'Structural Coordination Physics · 9 Societies · 10 CE-2026 · grok v1.2'),
      ),
      React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text3)', letterSpacing: '.1em', textAlign: 'right' } },
        React.createElement('div', null, `SOC/${nation}`),
        React.createElement('div', null, `YR/${year || '—'}`),
      ),
    ),

    /* Society tabs */
    React.createElement('nav', { style: { padding: '0 28px', borderBottom: '1px solid var(--scope-border)', display: 'flex', gap: 0 } },
      nations.map(n => React.createElement('div', {
        key: n, onClick: () => setNation(n),
        style: {
          padding: '10px 16px', fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '.1em',
          textTransform: 'uppercase', cursor: 'pointer',
          color: n === nation ? 'var(--scope-amber)' : 'var(--scope-text3)',
          borderBottom: n === nation ? '2px solid var(--scope-amber)' : '2px solid transparent',
          marginBottom: -1, fontWeight: n === nation ? 700 : 500,
        },
      }, n)),
    ),

    /* Hero metrics row */
    React.createElement('div', { style: { padding: '20px 28px', borderBottom: '1px solid var(--scope-border)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0 } },
      [
        { lbl: 'S/K Ratio', val: sk.toFixed(2), sub: 'stress / capacity', color: sk > 1.4 ? 'var(--scope-red)' : 'var(--scope-text)' },
        { lbl: 'Praetorian', val: (praet >= 0 ? '+' : '') + praet.toFixed(2), sub: 'shield vs helm', color: praet > 0 ? 'var(--scope-red)' : 'var(--scope-text)' },
        { lbl: 'Cog. Gap', val: (cogGap >= 0 ? '+' : '') + cogGap.toFixed(2), sub: 'slow vs fast loop', color: 'var(--scope-text)' },
        { lbl: 'Bond Str.', val: r.Bagg.toFixed(3), sub: 'mean normalised', color: 'var(--scope-text)' },
      ].map((m,i) => React.createElement('div', { key: m.lbl, style: { textAlign: 'center', borderRight: i<3 ? '1px solid var(--scope-border)' : 'none' } },
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)' } }, m.lbl),
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 36, fontWeight: 500, color: m.color, marginTop: 6, fontVariantNumeric: 'tabular-nums' } }, m.val),
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text3)', marginTop: 2 } }, m.sub),
      )),
    ),

    /* Main grid */
    React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 } },

      /* LEFT column */
      React.createElement('section', { style: { padding: '20px 28px', borderRight: '1px solid var(--scope-border)' } },

        /* Year slider */
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 } },
          React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', color: 'var(--scope-text3)', textTransform: 'uppercase' } }, 'Year'),
          React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text3)' } }, years[0] || ''),
          React.createElement('input', {
            type: 'range', min: 0, max: Math.max(0, years.length - 1),
            value: years.indexOf(year), onChange: e => { setPlaying(false); setYear(years[+e.target.value]); },
            style: { flex: 1, accentColor: 'var(--scope-amber)' },
          }),
          React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--scope-text3)' } }, years[years.length-1] || ''),
          React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 600, color: 'var(--scope-amber)', minWidth: 60, textAlign: 'right' } }, year || '—'),
          React.createElement('select', {
            value: speedMode, onChange: e => setSpeedMode(e.target.value),
            style: { fontFamily: 'var(--font-mono)', fontSize: 11, padding: '4px 6px', border: '1px solid var(--scope-border)', borderRadius: 3, background: '#fff' },
          }, ['Slow', 'Normal', 'Fast'].map(s => React.createElement('option', { key: s, value: s }, s))),
          React.createElement('button', {
            onClick: () => setPlaying(p => !p),
            style: { fontFamily: 'var(--font-mono)', fontSize: 11, padding: '4px 12px', border: '1px solid var(--scope-border)', borderRadius: 3, background: '#fff', cursor: 'pointer', letterSpacing: '.1em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 4 },
          }, playing ? '■ Stop' : '▶ Roll'),
        ),

        /* Current attractor */
        React.createElement('div', { style: { background: 'var(--scope-bg2)', border: `1px solid ${threatColor === 'var(--scope-red)' ? 'var(--scope-red)' : 'var(--scope-border)'}`, borderRadius: 4, padding: '14px 18px', marginBottom: 16 } },
          React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 6 } },
            React.createElement('div', null,
              React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)' } }, 'Current attractor'),
              React.createElement('div', { style: { fontFamily: 'var(--font-display)', fontSize: 26, fontWeight: 600, letterSpacing: '-.01em', color: threatColor === 'var(--scope-red)' ? 'var(--scope-red)' : 'var(--scope-text)', textTransform: 'uppercase', marginTop: 2 } }, basin || '—'),
            ),
            React.createElement('div', {
              style: { fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '.1em', color: threatColor, padding: '4px 10px', border: `1px solid ${threatColor}`, borderRadius: 3 },
            }, `THREAT: ${threatLevel}`),
          ),
          React.createElement('p', { style: { fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--scope-text2)', lineHeight: 1.55, marginTop: 4 } }, zeitgeist),
        ),

        /* Coordination view stage */
        React.createElement('div', { style: { background: 'var(--scope-bg2)', border: '1px solid var(--scope-border)', borderRadius: 4, position: 'relative', overflow: 'hidden', height: 460, marginBottom: 12 } },
          React.createElement('div', { style: { position: 'absolute', top: 12, left: 14, fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', pointerEvents: 'none', zIndex: 2 } }, 'Live Coordination View'),
          React.createElement('div', { style: { position: 'absolute', bottom: 4, left: 14, fontFamily: 'var(--font-display)', fontSize: 64, color: 'var(--scope-bg3)', fontWeight: 200, pointerEvents: 'none', letterSpacing: '-.02em', lineHeight: 1 } }, year || ''),
          React.createElement('div', { style: { position: 'absolute', bottom: 4, right: 14, fontFamily: 'var(--font-display)', fontSize: 64, color: 'var(--scope-bg3)', fontWeight: 200, pointerEvents: 'none', letterSpacing: '-.02em', lineHeight: 1 } }, year || ''),
          React.createElement('div', { ref: stageRef, style: { position: 'absolute', inset: 0 } },
            React.createElement(CoordView, { matrix, r, width: size.w, height: size.h }),
          ),
        ),

        /* Attractor basin chip */
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 4px' } },
          React.createElement('div', null,
            React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)' } }, 'Attractor basin'),
            React.createElement('div', { style: { fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 600, letterSpacing: '-.01em', textTransform: 'uppercase', marginTop: 2 } }, basin || '—'),
          ),
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--scope-text2)', padding: '6px 12px', background: 'var(--scope-bg3)', borderRadius: 3 } }, `Transition / Tipping Basin`),
        ),

        /* Trajectory sparkline */
        React.createElement('div', { style: { marginTop: 18, fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', marginBottom: 6 } }, 'Trajectory — S/K ratio over time'),
        React.createElement(SparkSK, { nation, year, years }),

      ),

      /* RIGHT column — dossier */
      React.createElement('section', { style: { padding: '20px 28px' } },

        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', marginBottom: 6 } }, 'CAMS Intelligence Dossier'),
        React.createElement('div', { style: { fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 200, letterSpacing: '-.01em', marginBottom: 14 } },
          React.createElement('span', { style: { fontWeight: 500, color: 'var(--scope-text)' } }, `${nation}, ${year || '—'}`),
        ),

        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 } },
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, letterSpacing: '.1em', color: threatColor, padding: '5px 10px', border: `1px solid ${threatColor}`, borderRadius: 3, display: 'inline-flex', alignItems: 'center', gap: 6 } },
            React.createElement('span', { style: { width: 6, height: 6, borderRadius: 99, background: threatColor } }),
            `THREAT LEVEL: ${threatLevel}`,
          ),
          React.createElement('button', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', padding: '5px 12px', border: '1px solid var(--scope-border)', borderRadius: 3, background: '#fff', cursor: 'pointer' } }, 'Print'),
        ),

        /* 3-col panel grid */
        React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 } },

          /* Zeitgeist (full-height column 1, dark) */
          React.createElement('div', { style: { background: '#1a3340', color: '#e8eef0', padding: '16px 18px', borderRadius: 4, gridRow: 'span 2' } },
            React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-amber)', marginBottom: 10 } }, 'Zeitgeist — the feel of the era'),
            React.createElement('p', { style: { fontFamily: 'var(--font-typewriter)', fontSize: 14, lineHeight: 1.65, color: '#e8eef0' } }, zeitgeist),
          ),

          /* Power topology */
          DossierCard({
            label: 'Power topology',
            children: React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 } },
              [
                { l: 'Dominant Node', v: dominant },
                { l: 'Shadow Power', v: shadow },
                { l: 'Weakest Node', v: weakest },
                { l: 'MF Substrate', v: sub },
              ].map(c => React.createElement('div', { key: c.l, style: { padding: '6px 8px', background: 'var(--scope-bg3)', borderRadius: 3 } },
                React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)' } }, c.l),
                React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: 'var(--scope-amber2)', marginTop: 2 } }, c.v),
              )),
            ),
          }),

          /* Survival protocol */
          DossierCard({
            label: 'Survival protocol',
            children: React.createElement('p', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.6, color: 'var(--scope-text2)' } },
              basin === 'Praetorian'
                ? `Affiliate conspicuously with Shield actors — they hold operational coordination energy regardless of nominal titles. Never criticise the security apparatus in public. Keep Archive and Lore activity private: slow-loop thinking is dangerous here. Exit windows close at sk > 2.0.`
              : basin === 'Early Strain'
                ? `Cultivate ${dominant} actors but hedge. The basin is tipping — tomorrow's dominant node may not be today's. Avoid being publicly identified with any single faction. Build lateral bonds across nodes: that is what will survive the coming rearrangement.`
              : basin === 'Peak Expansion'
                ? `Align with the dominant node (${dominant}) — its coordination energy is real and reliable. Avoid direct criticism of imperial or expansionist narratives: the cost is social marginalisation with no countervailing reward. This is a good time to build durable Archive and Lore investments.`
              : basin === 'Navigational'
                ? `Work with the Helm — it has genuine authority and is using it for construction rather than defence. Invest in Archive and Lore nodes now, before the next transition. The window for institution-building is open.`
              : basin === 'Archipelago'
                ? `Find your enclave and operate within it. Cross-enclave coordination is possible but costly. The state is not your protector but it is not your enemy either — it simply lacks coherent agency. Avoid symbolic acts of loyalty or resistance: they cost more than they return.`
              : basin === 'Mass Formation'
                ? `Maximum caution. Do not attempt to reason with the hypnotised cohort — reason is not the medium of this state. Cultivate the non-hypnotised minority (~30% of population): they are not gone, merely quiet. Archive and Lore preservation is the strategic priority — institutions will be rebuilt from whatever survives this phase.`
              : (basin === 'Phantom I' || basin === 'Phantom II')
                ? `The cultural layer is alive and can be worked with. The material layer is unreliable — do not depend on infrastructure, supply chains, or official authority for anything critical. Build your own material networks. The intellectual wealth of this society is real and worth preserving; its political claims are not.`
              : `Maintain plural affiliations. No single node carries enough coordination energy to make exclusive loyalty rational.`,
            ),
          }),
        ),

        /* Nominal vs real */
        React.createElement('div', { style: { background: 'var(--scope-bg2)', border: '1px solid var(--scope-border)', borderRadius: 4, padding: '12px 14px', marginTop: 12 } },
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', marginBottom: 6 } }, 'Nominal vs real coordination'),
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6, color: 'var(--scope-text2)' } },
            React.createElement('div', null, 'Nominal: ',
              React.createElement('span', { style: { color: 'var(--scope-amber2)', fontWeight: 600 } }, `Head of state (${nation})`)),
            React.createElement('div', { style: { marginTop: 4 } }, 'Real: ',
              React.createElement('span', { style: { color: 'var(--scope-amber2)', fontWeight: 600 } }, dominant),
              ` — Real coordination has migrated to ${dominant}; nominal titles are secondary.`),
          ),
        ),

        React.createElement('p', { style: { fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6, color: 'var(--scope-text2)', marginTop: 12 } },
          `${dominant} overruns ${weakest} by ${Math.abs(Math.round((r.V[TS_NODES.indexOf(dominant)] - r.V[TS_NODES.indexOf(weakest)]) * 10))} index points — real coordination lives in the ${dominant.toLowerCase()} apparatus, not the executive chamber.`,
        ),

        /* Node bars */
        React.createElement('div', { style: { marginTop: 14 } },
          sorted.map(s => React.createElement(NodeBarTS, { key: s.n, name: s.n, value: s.v, max: maxV })),
        ),

        /* Trajectory note */
        React.createElement('div', { style: { background: 'var(--scope-bg2)', border: '1px solid var(--scope-border)', borderRadius: 4, padding: '12px 14px', marginTop: 14 } },
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--scope-text3)', marginBottom: 6 } }, 'Trajectory'),
          React.createElement('p', { style: { fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6, color: 'var(--scope-text2)' } },
            'The S/K trajectory is ',
            React.createElement('span', { style: { fontStyle: 'italic', color: 'var(--scope-red)' } }, sk > 1.4 ? 'deteriorating' : sk > 1.1 ? 'tipping' : 'stable'),
            `. Bond coherence is ${r.Bagg < 0.35 ? 'contracting' : 'holding'}.`,
          ),
        ),

        /* Closing italic */
        React.createElement('p', { style: { fontFamily: 'var(--font-typewriter)', fontSize: 13, lineHeight: 1.6, color: 'var(--scope-text2)', marginTop: 14, fontStyle: 'italic' } },
          basin === 'Praetorian'
            ? `The ${nation} of ${year} — a system in which the ostensible governors govern least. The titles remain; the coordination energy has migrated to the security apparatus. History will record the events; CAMS records the physics.`
          : (basin === 'Phantom I' || basin === 'Phantom II')
            ? `The ${nation} of ${year} — a civilisation that mistakes its cultural elaboration for structural health. The Archive glows while the foundations settle. This is the most poignant of basins: the most articulate moment is also the last.`
          : basin === 'Mass Formation'
            ? `The ${nation} of ${year} — a society in which the thermodynamic substrate for collective hypnosis is fully prepared. The named object has arrived or is imminent. Individual ethics remain intact in the non-hypnotised minority; collective ethics have been suspended.`
          : basin === 'Peak Expansion'
            ? `The ${nation} of ${year} — a coordination system operating at its designed apex. The eight nodes are coupled and purposeful. The trap is invisible from inside: the self-assurance of this moment is the seed of the next transition.`
          : basin === 'Archipelago'
            ? `The ${nation} of ${year} — a polity that has outsourced its coherence to private nodes. The formal architecture persists; the animating coordination has departed. This is not collapse — it is diffusion.`
          : basin === 'Navigational'
            ? `The ${nation} of ${year} — a society that has earned its optimism. The reconstruction instinct is strong; the social contract feels real. The hard work is ensuring the slow loop is rebuilt, not just the fast one.`
          : `The ${nation} of ${year} — a system balanced on the tipping edge. The metrics do not yet warrant alarm but the direction of travel is the question that matters now.`,
        ),

        /* Back link */
        React.createElement('div', { style: { marginTop: 24, paddingTop: 16, borderTop: '1px solid var(--scope-border)' } },
          React.createElement('a', { href: 'CAMS Network.html', style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--scope-teal2)', textDecoration: 'none', marginRight: 16 } }, '→ Network view'),
          React.createElement('a', { href: 'CAMS Interpreter.html', style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--scope-teal2)', textDecoration: 'none' } }, '→ Interpreter'),
        ),
      ),
    ),
  );
}

function SparkSK({ nation, year, years }) {
  // Compute a simple S/K series across all years
  const series = React.useMemo(() => {
    return years.map(y => {
      const rec = DATA[nation][y];
      if (!rec) return 0;
      let s = 0, k = 0;
      for (const n of TS_NODES) { const m = rec[n]; if (m) { s += m[2]; k += m[1]; } }
      return s / Math.max(0.001, k);
    });
  }, [nation, years]);
  if (!series.length) return null;
  const W = 700, H = 110, pad = 12;
  const max = Math.max(...series, 1.2);
  const min = Math.min(...series, 0);
  const xScale = i => pad + (i / Math.max(1, series.length-1)) * (W - pad*2);
  const yScale = v => pad + (1 - (v - min)/(max - min || 1)) * (H - pad*2);
  const path = series.map((v,i) => `${i===0?'M':'L'} ${xScale(i)} ${yScale(v)}`).join(' ');
  const idx = years.indexOf(year);
  return React.createElement('div', { style: { background: '#1a3340', borderRadius: 4, padding: 4 } },
    React.createElement('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, style: { display: 'block' } },
      React.createElement('path', { d: path, fill: 'none', stroke: 'var(--scope-amber)', strokeWidth: 1.4 }),
      idx >= 0 && React.createElement('line', {
        x1: xScale(idx), y1: pad, x2: xScale(idx), y2: H-pad,
        stroke: 'var(--scope-amber)', strokeWidth: 0.8, strokeDasharray: '2 3', opacity: 0.6,
      }),
      idx >= 0 && React.createElement('text', {
        x: xScale(idx), y: H-2, textAnchor: 'middle',
        fontFamily: 'var(--font-mono)', fontSize: 9, fill: 'var(--scope-amber)',
      }, year),
    ),
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(TelescopeApp));
