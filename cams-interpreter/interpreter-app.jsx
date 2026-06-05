/* ================================================================
   CAMS Interpreter + Epiphenomenon Detector — Main App
   ================================================================ */

const BASIN_COLORS = {
  'Peak Expansion': 'var(--basin-peak)',
  'Navigational': 'var(--basin-nav)',
  'Early Strain': 'var(--basin-strain)',
  'Praetorian': 'var(--basin-praetorian)',
  'Archipelago': 'var(--basin-archipelago)',
  'Mass Formation': 'var(--basin-mass)',
  'Phantom I': 'var(--basin-phantom)',
  'Phantom II': 'var(--basin-phantom)',
};

const NODE_COLORS = {
  Helm: 'var(--node-helm)', Shield: 'var(--node-shield)', Lore: 'var(--node-lore)',
  Stewards: 'var(--node-stewards)', Craft: 'var(--node-craft)', Hands: 'var(--node-hands)',
  Archive: 'var(--node-archive)', Flow: 'var(--node-flow)',
};

const MODE_COLORS = { C: 'var(--mode-coherence)', K: 'var(--mode-capacity)', S: 'var(--mode-stress)', A: 'var(--mode-abstraction)' };

/* ── Mapping tables (static reference) ── */
const TABLE_A = [
  { variable: 'τ (K/S)', direction: 'Falls < 1.2', genre: 'Decline-anxiety, apocalyptic, nostalgia, civilisational-twilight rhetoric' },
  { variable: 'τ', direction: 'Rises > 1.5', genre: 'Civic-progress, expansionist, futurist' },
  { variable: 'ε (overreach)', direction: 'Sustained > 1.3', genre: 'Ideological combat, conspiracy genres, "secret enemy"' },
  { variable: 'ε', direction: 'Near 1.0', genre: 'Pragmatist, technocratic register; "common sense"' },
  { variable: 'Λ (coordination)', direction: 'Falls < 0.45', genre: 'Gridlock / "broken system"; treason discourse' },
  { variable: 'Λ', direction: 'Rises, sustained', genre: 'Consensus politics, reformist energy' },
  { variable: 'σᵢ (activation)', direction: 'Strong +, slow-loop', genre: 'Missionary politics, ideological export' },
  { variable: 'σᵢ', direction: 'Strong − (inversion)', genre: 'Ressentiment, scapegoating, weaponised grievance' },
  { variable: 'σᵢ', direction: 'Near zero', genre: 'Post-political apathy; ironic detachment' },
  { variable: 'V_Helm', direction: 'Low', genre: '"Out of touch elite", anti-establishment populism' },
  { variable: 'V_Shield', direction: 'High rel. to Helm', genre: 'Security panic, securitisation of unrelated domains' },
  { variable: 'V_Lore', direction: 'High A, low C', genre: 'Ideological fragmentation, culture-war, identitarian combat' },
  { variable: 'V_Stewards', direction: 'Sustained high', genre: 'Property panic, NIMBY, "the boomers"' },
  { variable: 'V_Flow', direction: 'Disproportionate', genre: 'Financialisation discourse, "markets demand"' },
  { variable: 'V_Hands', direction: 'Depressed, high S', genre: '"Battler" populism, class-conflict, strike waves' },
  { variable: 'V_Craft', direction: 'Sustained low', genre: 'Manufacturing-nostalgia, "hollowing out"' },
  { variable: 'B(t)', direction: 'Falling', genre: 'Atomised identity; secession rhetoric' },
  { variable: 'Ω (shear)', direction: 'Persistent high', genre: 'Generational, regional, factional conflict simultaneously' },
];

const TABLE_B = [
  { basin: 'Peak Expansion', dominant: 'Triumphalist, civic-progress, expansive optimism', suppressed: 'Decline, apocalyptic, conspiracist', color: 'var(--basin-peak)' },
  { basin: 'Navigational', dominant: 'Reformist problem-solving, post-crisis reconstruction', suppressed: 'Triumphalism, nihilism', color: 'var(--basin-nav)' },
  { basin: 'Early Strain', dominant: 'Nostalgic, "decline begins", generational disquiet', suppressed: 'Triumphalism, deep cynicism', color: 'var(--basin-strain)' },
  { basin: 'Praetorian', dominant: 'Securitisation, foreign-threat panic, strongman appeals', suppressed: 'Reformist optimism', color: 'var(--basin-praetorian)' },
  { basin: 'Archipelago', dominant: 'Tribal-ironic, "system is rigged but I have my tribe"', suppressed: 'Civic-renewal, common-good rhetoric', color: 'var(--basin-archipelago)' },
  { basin: 'Mass Formation', dominant: 'Single named enemy, ritual denunciation, anti-dissent', suppressed: 'Genuine pluralist disagreement', color: 'var(--basin-mass)' },
  { basin: 'Phantom I', dominant: 'Heritage flourishing, cultural-pride, ceremonial', suppressed: 'Material-grievance, structural-critique', color: 'var(--basin-phantom)' },
  { basin: 'Phantom II', dominant: 'Cynicism, post-political fatigue, ironic-nihilist', suppressed: 'Heritage politics, civic optimism', color: 'var(--basin-phantom)' },
];

const TABLE_C = [
  { pathology: 'Executive Decoupling', condition: 'V_Helm < 6', fingerprint: 'Personality-fixated coverage; "leader unable to govern"; administrative paralysis' },
  { pathology: 'Praetorian Condition', condition: 'V_Shield > V_Helm sustained', fingerprint: '"National security" applied to non-security domains; militarised metaphor' },
  { pathology: 'Mythic-Material Decoupling', condition: 'High Lore/Archive, collapsing Hands/Craft', fingerprint: 'Culture-war saturation; ceremonial controversies dominating' },
  { pathology: 'Flow Collapse', condition: 'V_Flow drops sharply', fingerprint: '"Markets in panic"; debt-personification; credit-rating as moral verdict' },
  { pathology: 'Late Abstraction Collapse', condition: 'S leads A by 2–5 yrs', fingerprint: 'Genre exhaustion; nihilistic and absurdist registers; "satire is dead"' },
  { pathology: 'Thermodynamic Freeze', condition: 'System-wide low activation', fingerprint: 'Discourse thins; coverage volume drops; foreign press reports what domestic cannot' },
];

/* ── Styles ── */
const S = {
  shell: { display: 'grid', gridTemplateColumns: '260px 1fr', minHeight: '100vh' },
  sidebar: { background: 'var(--paper-0)', borderRight: 'var(--border-hair)', padding: '20px 16px', overflowY: 'auto', maxHeight: '100vh', position: 'sticky', top: 0 },
  main: { padding: '24px 28px 48px', minWidth: 0 },
  mark: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 },
  markW: { fontFamily: 'var(--font-display)', fontWeight: 900, fontSize: 26, color: 'var(--accent-ochre)', letterSpacing: '-.02em', lineHeight: 1 },
  markS: { fontSize: 9.5, letterSpacing: '.14em', textTransform: 'uppercase', color: 'var(--ink-500)', marginTop: 3, fontWeight: 600 },
  navH: { fontSize: 10, fontWeight: 700, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--ink-500)', margin: '16px 0 6px' },
  navA: (on) => ({ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 10px', fontSize: 13.5, color: 'var(--ink-900)', textDecoration: 'none', borderRadius: 4, cursor: 'pointer', background: on ? 'var(--paper-2)' : 'transparent', fontWeight: on ? 600 : 400 }),
  dot: (c) => ({ width: 8, height: 8, borderRadius: 99, background: c, flexShrink: 0 }),
  h1: { margin: '0 0 4px', fontSize: 28, color: 'var(--accent-teal)', letterSpacing: '-.01em', fontWeight: 700 },
  lede: { color: 'var(--ink-700)', fontSize: 13, marginBottom: 18, maxWidth: 720, fontFamily: 'var(--font-mono)', lineHeight: 1.5 },
  metrics: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 18 },
  metric: { background: 'var(--paper-0)', border: 'var(--border-hair)', borderRadius: 4, padding: '14px 16px' },
  metricL: { fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: 'var(--ink-500)' },
  metricN: { fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 500, letterSpacing: '-.02em', color: 'var(--ink-900)', marginTop: 6, fontVariantNumeric: 'tabular-nums' },
  metricS: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-500)', marginTop: 4 },
  sectionTitle: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: 'var(--ink-500)', marginBottom: 10, marginTop: 24 },
  card: { background: 'var(--paper-0)', border: 'var(--border-hair)', borderRadius: 4, marginBottom: 10 },
  badge: (clear) => ({ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 999, letterSpacing: '.03em', background: clear ? '#E6F2EC' : '#FBE9E6', color: clear ? '#1D5B45' : '#7A2A22' }),
};

/* ── Small components ── */

function Metric({ label, value, sub, color }) {
  return React.createElement('div', { style: S.metric },
    React.createElement('div', { style: S.metricL }, label),
    React.createElement('div', { style: { ...S.metricN, color: color || 'var(--ink-900)' } }, value),
    sub && React.createElement('div', { style: S.metricS }, sub),
  );
}

function NodeBar({ name, value, max, color }) {
  const pct = Math.max(0, Math.min(100, (Math.abs(value) / max) * 100));
  return React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '80px 1fr 50px', gap: 6, alignItems: 'center', margin: '3px 0', fontSize: 12.5 } },
    React.createElement('span', { style: { fontWeight: 500 } }, name),
    React.createElement('div', { style: { background: 'var(--paper-2)', height: 8, borderRadius: 99, overflow: 'hidden' } },
      React.createElement('div', { style: { height: '100%', width: `${pct}%`, background: color, borderRadius: 99 } }),
    ),
    React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-500)', textAlign: 'right' } }, value.toFixed(1)),
  );
}

function SignalItem({ signal, index }) {
  return React.createElement('div', { style: { padding: '10px 0', borderTop: index > 0 ? '1px solid var(--paper-3)' : 'none' } },
    React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 3 } },
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, color: 'var(--ink-300)' } }, index + 1),
      React.createElement('span', { style: { fontSize: 13, fontWeight: 600, color: 'var(--ink-900)' } }, signal.title),
    ),
    React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5, marginBottom: 4 } }, signal.body),
    signal.mapping && React.createElement('div', { style: { display: 'flex', gap: 6, alignItems: 'baseline' } },
      React.createElement('span', { style: { fontSize: 10, color: 'var(--accent-ochre)', fontWeight: 700 } }, `Table ${signal.table} →`),
      React.createElement('span', { style: { fontSize: 11.5, color: 'var(--ink-500)', fontFamily: 'var(--font-mono)', lineHeight: 1.4 } }, signal.mapping),
    ),
  );
}

function Category({ name, signals }) {
  const isClear = signals.length === 0;
  return React.createElement('div', { style: S.card },
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px' } },
      React.createElement('span', { style: { fontSize: 14, fontWeight: 700 } }, name),
      React.createElement('span', { style: S.badge(isClear) }, isClear ? '✓ clear' : `${signals.length} signal${signals.length > 1 ? 's' : ''}`),
    ),
    isClear
      ? React.createElement('div', { style: { padding: '4px 16px 14px', fontSize: 12.5, color: 'var(--ink-300)', fontStyle: 'italic' } }, 'No signals in this category.')
      : React.createElement('div', { style: { padding: '0 16px 14px' } }, signals.map((s, i) => React.createElement(SignalItem, { key: i, signal: s, index: i }))),
  );
}

function NetStats({ r }) {
  const meanB = r.B.reduce((a,b) => a+b, 0) / r.B.length;
  const stats = [
    { label: 'B(t)', value: meanB.toFixed(3), warn: meanB < 0.3 },
    { label: 'δ_B', value: r.sigmaV.toFixed(3), warn: false },
    { label: 'Ṽ', value: r.meanV.toFixed(3), warn: false },
    { label: 'D̃', value: r.meanS.toFixed(3), warn: false },
    { label: 'λ₂', value: r.Lambda.toFixed(3), warn: r.Lambda < 0.45 },
    { label: 'R(t)', value: r.Lambda < 0.001 ? '∞' : (1/r.Lambda).toFixed(1), warn: r.Lambda < 0.1, inf: r.Lambda < 0.001 },
  ];
  return React.createElement('div', { style: { ...S.card, padding: '14px 16px', marginTop: 0 } },
    React.createElement('div', { style: S.sectionTitle, key: 'h' }, 'Network Statistics'),
    React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8 } },
      stats.map(s => React.createElement('div', { key: s.label, style: { textAlign: 'center' } },
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-500)', marginBottom: 2 } }, s.label),
        React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: s.inf ? 16 : 20, fontWeight: 500, color: s.warn ? 'var(--mode-stress)' : 'var(--ink-900)', fontVariantNumeric: 'tabular-nums' } }, s.value),
      )),
    ),
  );
}

/* ── Collapsible reference tables ── */
function Collapsible({ title, subtitle, children }) {
  const [open, setOpen] = React.useState(false);
  return React.createElement('div', { style: { ...S.card, marginBottom: 8, overflow: 'hidden' } },
    React.createElement('div', { onClick: () => setOpen(!open), style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', cursor: 'pointer', userSelect: 'none' } },
      React.createElement('div', null,
        React.createElement('span', { style: { fontSize: 14, fontWeight: 700 } }, title),
        subtitle && React.createElement('span', { style: { fontSize: 11, color: 'var(--ink-500)', marginLeft: 8 } }, subtitle),
      ),
      React.createElement('span', { style: { fontSize: 12, color: 'var(--ink-300)', fontFamily: 'var(--font-mono)' } }, open ? '▾' : '▸'),
    ),
    open && React.createElement('div', { style: { padding: '0 16px 14px' } }, children),
  );
}

function RefTableA() {
  const th = { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', padding: '6px 6px', textAlign: 'left', borderBottom: '2px solid var(--paper-3)', whiteSpace: 'nowrap' };
  const td = { fontSize: 12, color: 'var(--ink-700)', padding: '6px', borderBottom: '1px solid var(--paper-3)', lineHeight: 1.4, verticalAlign: 'top' };
  const mono = { ...td, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-900)' };
  return React.createElement('table', { style: { width: '100%', borderCollapse: 'collapse' } },
    React.createElement('thead', null, React.createElement('tr', null,
      React.createElement('th', { style: th }, 'Variable'),
      React.createElement('th', { style: th }, 'Direction'),
      React.createElement('th', { style: th }, 'Predicted genre'),
    )),
    React.createElement('tbody', null, TABLE_A.map((r, i) => React.createElement('tr', { key: i },
      React.createElement('td', { style: mono }, r.variable),
      React.createElement('td', { style: mono }, r.direction),
      React.createElement('td', { style: td }, r.genre),
    ))),
  );
}

function RefTableB({ currentBasin }) {
  return React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: 0 } },
    TABLE_B.map((r, i) => {
      const active = r.basin === currentBasin;
      return React.createElement('div', { key: i, style: { display: 'grid', gridTemplateColumns: '130px 1fr 1fr', gap: 10, padding: '8px 0', borderBottom: i < TABLE_B.length - 1 ? '1px solid var(--paper-3)' : 'none', background: active ? 'var(--paper-2)' : 'transparent', margin: active ? '0 -8px' : 0, padding: active ? '8px' : '8px 0', borderRadius: active ? 4 : 0 } },
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6 } },
          React.createElement('span', { style: { width: 8, height: 8, borderRadius: 99, background: r.color, flexShrink: 0 } }),
          React.createElement('span', { style: { fontSize: 12.5, fontWeight: active ? 700 : 500, color: 'var(--ink-900)' } }, r.basin),
        ),
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', marginBottom: 2 } }, 'Dominant'),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--ink-700)', lineHeight: 1.4 } }, r.dominant),
        ),
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', marginBottom: 2 } }, 'Suppressed'),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--ink-300)', lineHeight: 1.4 } }, r.suppressed),
        ),
      );
    }),
  );
}

function RefTableC() {
  return React.createElement('div', null, TABLE_C.map((r, i) =>
    React.createElement('div', { key: i, style: { padding: '10px 0', borderBottom: i < TABLE_C.length - 1 ? '1px solid var(--paper-3)' : 'none' } },
      React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 3 } },
        React.createElement('span', { style: { fontSize: 13, fontWeight: 700, color: 'var(--mode-stress)' } }, r.pathology),
        React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-500)' } }, r.condition),
      ),
      React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5 } }, r.fingerprint),
    ),
  ));
}

/* ── Input matrix ── */
function InputMatrix({ matrix, setMatrix }) {
  const modes = ['C', 'K', 'S', 'A'];
  const handleChange = (nodeIdx, modeIdx, val) => {
    const next = matrix.map(r => [...r]);
    next[nodeIdx][modeIdx] = Math.max(1, Math.min(10, Number(val) || 0));
    setMatrix(next);
  };
  const inputS = { width: 42, padding: '3px 4px', fontFamily: 'var(--font-mono)', fontSize: 12, textAlign: 'right', border: 'var(--border-hair)', borderRadius: 2, background: 'var(--paper-0)', color: 'var(--ink-900)' };
  const thS = { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', padding: '6px 4px', textAlign: 'center', borderBottom: '2px solid var(--paper-3)' };

  return React.createElement('table', { style: { width: '100%', borderCollapse: 'collapse', fontSize: 12 } },
    React.createElement('thead', null, React.createElement('tr', null,
      React.createElement('th', { style: { ...thS, textAlign: 'left' } }, 'Node'),
      ...modes.map(m => React.createElement('th', { key: m, style: { ...thS, color: MODE_COLORS[m] } }, m)),
      React.createElement('th', { style: thS }, 'Vᵢ'),
    )),
    React.createElement('tbody', null, NODES.map((n, i) => {
      const v = matrix[i][0] + matrix[i][1] - matrix[i][2] + 0.5 * matrix[i][3];
      return React.createElement('tr', { key: n },
        React.createElement('td', { style: { padding: '4px 6px', fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6 } },
          React.createElement('span', { style: { ...S.dot(NODE_COLORS[n]), width: 6, height: 6 } }),
          n,
        ),
        ...modes.map((m, mi) => React.createElement('td', { key: m, style: { padding: '3px 2px', textAlign: 'center' } },
          React.createElement('input', { type: 'number', min: 1, max: 10, value: matrix[i][mi], onChange: e => handleChange(i, mi, e.target.value), style: inputS }),
        )),
        React.createElement('td', { style: { padding: '4px 6px', fontFamily: 'var(--font-mono)', fontSize: 12, textAlign: 'center', color: v < 6 ? 'var(--mode-stress)' : 'var(--ink-900)' } }, v.toFixed(1)),
      );
    })),
  );
}

/* ── Sidebar ── */
function Sidebar({ nations, selected, onSelect }) {
  const SOC_COLORS = {
    Australia: 'var(--soc-australia)', China: 'var(--soc-china)', Rome: 'var(--soc-rome)',
    USA: 'var(--soc-usa)', Germany: 'var(--soc-germany)', Italy: 'var(--soc-italy)',
    Spain: 'var(--soc-spain)', France: '#8B6FB0', UK: 'var(--soc-uk)',
    'South Africa': 'var(--soc-south-africa)', 'New Zealand': '#20a0b0',
    'Russia (1800-1830)': '#c04060', 'Russia (1992-2026)': '#c04060',
  };

  return React.createElement('aside', { style: S.sidebar },
    React.createElement('div', { style: S.mark },
      React.createElement('div', null,
        React.createElement('div', { style: S.markW }, 'CAMS'),
        React.createElement('div', { style: S.markS }, 'Interpreter'),
      ),
    ),
    React.createElement('h5', { style: S.navH }, 'Select Society'),
    ...nations.map(n =>
      React.createElement('div', { key: n, onClick: () => onSelect(n), style: S.navA(n === selected) },
        React.createElement('span', { style: S.dot(SOC_COLORS[n] || '#888') }),
        React.createElement('span', null, n),
      )
    ),
    React.createElement('div', { style: { marginTop: 12, fontSize: 11, color: 'var(--ink-300)', lineHeight: 1.5 } }, `${nations.length} societies · use year slider to navigate`),
  );
}

/* ── Year slider with transport controls ── */
function YearSlider({ years, year, setYear, playing, onPlay, onStop, onRoll, speed, setSpeed }) {
  const btnS = { padding: '5px 12px', fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-sans)', border: 'var(--border-hair)', borderRadius: 4, cursor: 'pointer', background: 'var(--paper-0)', color: 'var(--ink-900)', display: 'flex', alignItems: 'center', gap: 4 };
  const btnActive = { ...btnS, background: 'var(--ink-900)', color: 'var(--paper-0)' };
  return React.createElement('div', { style: { marginBottom: 16 } },
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 } },
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink-500)' } }, years[0]),
      React.createElement('input', {
        type: 'range', min: 0, max: years.length - 1,
        value: years.indexOf(year),
        onChange: e => { onStop(); setYear(years[+e.target.value]); },
        style: { flex: 1, accentColor: 'var(--accent-teal)' }
      }),
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink-500)' } }, years[years.length - 1]),
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 700, color: 'var(--ink-900)', minWidth: 50, textAlign: 'right' } }, year),
    ),
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 8 } },
      React.createElement('button', { onClick: onRoll, style: btnS, title: 'Roll — restart from first year and play' }, '⟳', ' Roll'),
      React.createElement('button', { onClick: playing ? onStop : onPlay, style: playing ? btnActive : btnS }, playing ? '■ Stop' : '▶ Start'),
      React.createElement('span', { style: { fontSize: 10, color: 'var(--ink-500)', textTransform: 'uppercase', letterSpacing: '.06em', marginLeft: 8 } }, 'Speed'),
      React.createElement('input', {
        type: 'range', min: 50, max: 1200, step: 50,
        value: 1250 - speed,
        onChange: e => setSpeed(1250 - +e.target.value),
        style: { width: 80, accentColor: 'var(--accent-teal)' }
      }),
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-500)', minWidth: 40 } }, `${speed}ms`),
      playing && React.createElement('span', { style: { fontSize: 11, color: 'var(--accent-teal)', fontWeight: 600, marginLeft: 8 } }, '● cycling'),
    ),
  );
}

/* ── Main App ── */
function App() {
  const nations = React.useMemo(() => Object.keys(DATA), []);
  const [nation, setNation] = React.useState(nations[0]);
  const [years, setYears] = React.useState([]);
  const [year, setYear] = React.useState(null);
  const [matrix, setMatrix] = React.useState(NODES.map(() => [5,5,5,5]));
  const [tab, setTab] = React.useState('detector');
  const [playing, setPlaying] = React.useState(false);
  const [speed, setSpeed] = React.useState(400); // ms per tick
  const playRef = React.useRef(false);

  // When nation changes, reset years
  React.useEffect(() => {
    const yrs = Object.keys(DATA[nation] || {}).map(Number).sort((a,b) => a-b);
    setYears(yrs);
    if (yrs.length) {
      setYear(yrs[yrs.length - 1]);
    }
    setPlaying(false);
    playRef.current = false;
  }, [nation]);

  // When year changes, load matrix
  React.useEffect(() => {
    if (!year || !DATA[nation] || !DATA[nation][year]) return;
    const rec = DATA[nation][year];
    setMatrix(NODES.map(n => rec[n] ? [...rec[n]] : [5,5,5,5]));
  }, [nation, year]);

  // Playback engine
  React.useEffect(() => {
    playRef.current = playing;
    if (!playing) return;
    let timer;
    const tick = () => {
      if (!playRef.current) return;
      setYear(prev => {
        const yrs = Object.keys(DATA[nation] || {}).map(Number).sort((a,b) => a-b);
        const idx = yrs.indexOf(prev);
        if (idx < yrs.length - 1) {
          timer = setTimeout(tick, speed);
          return yrs[idx + 1];
        } else {
          // cycle: wrap to start
          timer = setTimeout(tick, speed);
          return yrs[0];
        }
      });
    };
    timer = setTimeout(tick, speed);
    return () => clearTimeout(timer);
  }, [playing, nation, speed]);

  const handlePlay = () => { setPlaying(true); };
  const handleStop = () => { setPlaying(false); };
  const handleRoll = () => {
    // Start from beginning and play
    const yrs = Object.keys(DATA[nation] || {}).map(Number).sort((a,b) => a-b);
    if (yrs.length) setYear(yrs[0]);
    setPlaying(true);
  };

  // Compute
  const r = React.useMemo(() => computeAll(matrix), [matrix]);
  const signals = React.useMemo(() => generateSignals(r, matrix), [r, matrix]);
  const basin = React.useMemo(() => classifyBasin(r, matrix), [r, matrix]);
  const totalSignals = Object.values(signals).reduce((a, s) => a + s.length, 0);

  const tabs = [
    { id: 'detector', label: 'Epiphenomenon Detector' },
    { id: 'operators', label: 'Operators' },
    { id: 'criticality', label: 'Criticality' },
    { id: 'bond', label: 'Bond Network' },
    { id: 'reference', label: 'Reference Tables' },
  ];

  return React.createElement('div', { style: S.shell },
    React.createElement(Sidebar, { nations, selected: nation, onSelect: setNation }),
    React.createElement('main', { style: S.main },

      /* Title */
      React.createElement('h1', { style: S.h1 }, `${nation} — ${year || '…'}`),
      React.createElement('div', { style: S.lede }, `CAMS v3.2-R · ${basin} basin · ${r.attractor} attractor · ${r.kappaTier} · ${totalSignals} signal${totalSignals !== 1 ? 's' : ''}`),

      /* Year slider */
      years.length > 0 && React.createElement(YearSlider, { years, year, setYear, playing, onPlay: handlePlay, onStop: handleStop, onRoll: handleRoll, speed, setSpeed }),

      /* Metrics row — v3.2-R */
      React.createElement('div', { style: S.metrics },
        React.createElement(Metric, { label: 'V̄ · mean vitality', value: r.meanV.toFixed(1), sub: `σ_V = ${r.sigmaV.toFixed(2)}${r.disregard ? ' · DISREGARD' : ''}` }),
        React.createElement(Metric, { label: 'τ · affective tone', value: r.tau.toFixed(2), sub: r.tau < 1.0 ? 'stress-dominant' : r.tau < 1.2 ? 'marginal' : 'surplus', color: r.tau < 1.0 ? 'var(--mode-stress)' : undefined }),
        React.createElement(Metric, { label: 'κ · criticality', value: r.kappa === Infinity ? '∞' : r.kappa.toFixed(3), sub: r.kappaTier, color: r.kappaTier === 'EXTREME' || r.kappaTier === 'CRITICAL' ? 'var(--mode-stress)' : r.kappaTier === 'WARNING' ? 'var(--accent-ochre)' : undefined }),
        React.createElement(Metric, { label: 'Attractor', value: r.attractor, sub: `alarm: ${r.alarmConfidence}`, color: BASIN_COLORS[basin] || 'var(--ink-900)' }),
      ),
      React.createElement('div', { style: { ...S.metrics, marginTop: -6 } },
        React.createElement(Metric, { label: 'λ₂ · algebraic conn.', value: r.lambda2.toFixed(3), sub: 'Laplacian L = D−W' }),
        React.createElement(Metric, { label: 'ε · overreach', value: r.epsilon.toFixed(2), sub: 'A·S / K' }),
        React.createElement(Metric, { label: 'η_loop · Library', value: r.etaLoop === Infinity ? '∞' : r.etaLoop.toFixed(3), sub: r.etaLoop < 0.6 ? 'below poetry threshold' : 'active', color: r.etaLoop < 0.6 ? 'var(--mode-stress)' : undefined }),
        React.createElement(Metric, { label: 'x(t) · headroom', value: r.headroom.toFixed(3), sub: `x_min = ${r.headroomMin.toFixed(3)}${r.headroomMin <= 0 ? ' · COLLAPSE' : ''}`, color: r.headroomMin <= 0 ? 'var(--mode-stress)' : undefined }),
      ),

      /* Tab bar */
      React.createElement('div', { style: { display: 'flex', gap: 0, borderBottom: '2px solid var(--paper-3)', marginBottom: 16 } },
        tabs.map(t => React.createElement('div', {
          key: t.id, onClick: () => setTab(t.id),
          style: { padding: '8px 16px', fontSize: 12, fontWeight: tab === t.id ? 700 : 400, color: tab === t.id ? 'var(--accent-teal)' : 'var(--ink-500)', cursor: 'pointer', borderBottom: tab === t.id ? '2px solid var(--accent-teal)' : '2px solid transparent', marginBottom: -2, textTransform: 'uppercase', letterSpacing: '.06em' }
        }, t.label)),
      ),

      /* Detector tab */
      tab === 'detector' && React.createElement(React.Fragment, null,
        Object.entries(signals).map(([cat, sigs]) => React.createElement(Category, { key: cat, name: cat, signals: sigs })),
        React.createElement(NetStats, { r }),
        React.createElement('div', { style: { marginTop: 16, padding: '14px 16px', background: 'var(--paper-2)', borderRadius: 4, borderLeft: '3px solid ' + (BASIN_COLORS[basin] || 'var(--accent-ochre)') } },
          React.createElement('div', { style: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: BASIN_COLORS[basin] || 'var(--accent-ochre)', marginBottom: 4 } }, `${basin} basin — predicted genre mix`),
          React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5 } },
            `Dominant: ${(TABLE_B.find(b => b.basin === basin) || TABLE_B[1]).dominant}`),
          React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-300)', lineHeight: 1.5, marginTop: 2 } },
            `Suppressed: ${(TABLE_B.find(b => b.basin === basin) || TABLE_B[1]).suppressed}`),
        ),
      ),

      /* Operators tab */
      tab === 'operators' && React.createElement(React.Fragment, null,
        React.createElement('div', { style: S.sectionTitle }, 'Input matrix'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          React.createElement(InputMatrix, { matrix, setMatrix }),
        ),
        React.createElement('div', { style: S.sectionTitle }, 'Node vitality Vᵢ'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          NODES.map((n, i) => React.createElement(NodeBar, { key: n, name: n, value: r.V[i], max: Math.max(...r.V.map(Math.abs), 10), color: NODE_COLORS[n] })),
        ),
        React.createElement('div', { style: S.sectionTitle }, 'Per-node operators (v3.2-R)'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px', overflowX: 'auto' } },
          React.createElement('table', { style: { width: '100%', borderCollapse: 'collapse', fontSize: 12 } },
            React.createElement('thead', null, React.createElement('tr', null,
              ['Node', 'V', 'σᵢ', 'α', 'μ', 'γ', 'Γ', 'Σ', 'B'].map(h =>
                React.createElement('th', { key: h, style: { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', padding: '6px', textAlign: h === 'Node' ? 'left' : 'center', borderBottom: '2px solid var(--paper-3)' } }, h)
              ),
            )),
            React.createElement('tbody', null, NODES.map((n, i) =>
              React.createElement('tr', { key: n },
                React.createElement('td', { style: { padding: '5px 6px', fontWeight: 500 } }, n),
                [r.V[i], r.sigma[i], r.alpha[i], r.mu[i], r.gamma[i], r.Gamma[i], r.Sigma[i], r.B[i]].map((v, j) =>
                  React.createElement('td', { key: j, style: { padding: '5px 6px', fontFamily: 'var(--font-mono)', fontSize: 11, textAlign: 'center', borderBottom: '1px solid var(--paper-3)', color: (j === 0 && v < 6) || (j === 1 && v < 0) || (j === 5 && v > 3) || (j === 6 && v > 0) ? 'var(--mode-stress)' : 'var(--ink-700)' } }, v.toFixed(j === 7 ? 3 : j === 1 ? 0 : 2))
                ),
              )
            )),
          ),
        ),
      ),

      /* Criticality tab — v3.2-R */
      tab === 'criticality' && React.createElement(React.Fragment, null,
        React.createElement('div', { style: S.sectionTitle }, 'Cognitive-Plane Criticality κ(t)'),
        React.createElement('div', { style: { ...S.card, padding: '16px' } },
          React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 } },
            React.createElement(Metric, { label: 'κ(t)', value: r.kappa === Infinity ? '∞' : r.kappa.toFixed(4), sub: r.kappaTier, color: r.kappaTier === 'EXTREME' || r.kappaTier === 'CRITICAL' ? 'var(--mode-stress)' : r.kappaTier === 'WARNING' ? 'var(--accent-ochre)' : undefined }),
            React.createElement(Metric, { label: 'χ_K · synchronisability', value: r.chiK === Infinity ? '∞' : r.chiK.toFixed(4) }),
            React.createElement(Metric, { label: 'ω · rate dispersion', value: r.omega.toFixed(4) }),
            React.createElement(Metric, { label: 'R · master stability', value: r.R === Infinity ? '∞' : r.R.toFixed(3) }),
          ),
          React.createElement('div', { style: { fontSize: 11, color: 'var(--ink-500)', marginBottom: 12 } }, 'κ threshold tiers (v3.2-R recalibrated):'),
          React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 } },
            [
              { tier: 'WATCH', thresh: '≥ 0.30', det: '100%', fp: '89%' },
              { tier: 'WARNING', thresh: '≥ 0.35', det: '67%', fp: '75%' },
              { tier: 'CRITICAL', thresh: '≥ 0.42', det: '33%', fp: '43%' },
              { tier: 'EXTREME', thresh: '≥ 0.57', det: 'catastrophic', fp: '—' },
            ].map(t => React.createElement('div', { key: t.tier, style: { padding: '8px 10px', background: r.kappaTier === t.tier ? 'var(--paper-2)' : 'transparent', border: r.kappaTier === t.tier ? '2px solid var(--accent-ochre)' : 'var(--border-hair)', borderRadius: 4 } },
              React.createElement('div', { style: { fontSize: 11, fontWeight: 700, color: r.kappaTier === t.tier ? 'var(--accent-ochre)' : 'var(--ink-500)' } }, t.tier),
              React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink-900)', marginTop: 2 } }, t.thresh),
              React.createElement('div', { style: { fontSize: 10, color: 'var(--ink-300)', marginTop: 2 } }, `det: ${t.det} · FP: ${t.fp}`),
            )),
          ),
        ),

        React.createElement('div', { style: S.sectionTitle }, 'Headroom & Library Attractor'),
        React.createElement('div', { style: { ...S.card, padding: '16px' } },
          React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 } },
            React.createElement(Metric, { label: 'x(t) · headroom', value: r.headroom.toFixed(4), color: r.headroom < 0 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'x_min · weakest-link', value: r.headroomMin.toFixed(4), sub: r.headroomMin <= 0 ? 'COLLAPSE CONDITION' : 'nominal', color: r.headroomMin <= 0 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'η_loop · Library', value: r.etaLoop === Infinity ? '∞' : r.etaLoop.toFixed(4), sub: r.etaLoop < 0.6 ? 'below poetry threshold' : 'active', color: r.etaLoop < 0.6 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'B(t) · aggregate', value: r.Bagg.toFixed(4) }),
          ),
        ),

        React.createElement('div', { style: S.sectionTitle }, 'Phase Space Φ(t)'),
        React.createElement('div', { style: { ...S.card, padding: '16px' } },
          React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 } },
            React.createElement(Metric, { label: 'V̄', value: r.meanV.toFixed(2), sub: r.meanV < 12 ? 'CRISIS (< 12)' : 'nominal', color: r.meanV < 12 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'σ_V', value: r.sigmaV.toFixed(2), sub: r.sigmaV > 3.5 ? 'CRISIS (> 3.5)' : 'nominal', color: r.sigmaV > 3.5 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'Attractor', value: r.attractor }),
            React.createElement(Metric, { label: 'Composite Alarm', value: r.alarmConfidence, color: r.alarmConfidence === 'HIGH' ? 'var(--mode-stress)' : r.alarmConfidence === 'MEDIUM' ? 'var(--accent-ochre)' : undefined }),
          ),
          r.disregard && React.createElement('div', { style: { marginTop: 10, padding: '8px 12px', background: '#E6F2EC', borderRadius: 4, fontSize: 12, color: '#1D5B45', fontWeight: 600 } },
            'DISREGARD trigger active — V̄ > 15 ∧ σ_V < 2.0. False crisis signatures suppressed.'
          ),
        ),

        React.createElement('div', { style: S.sectionTitle }, 'ESCH Activation Index σᵢ'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          NODES.map((n, i) => React.createElement(NodeBar, {
            key: n, name: `${n} (${r.sigma[i] >= 0 ? '+' : ''}${r.sigma[i].toFixed(0)})`,
            value: Math.abs(r.sigma[i]), max: Math.max(...r.sigma.map(Math.abs), 1),
            color: r.sigma[i] < 0 ? 'var(--mode-stress)' : 'var(--mode-capacity)'
          })),
          React.createElement('div', { style: { fontSize: 11, color: 'var(--ink-300)', marginTop: 8 } }, 'σᵢ = (Aᵢ · Cᵢ) · (Kᵢ − Sᵢ). Negative = destructive activation. Green = constructive.'),
        ),
      ),

      /* Bond tab */
      tab === 'bond' && React.createElement(React.Fragment, null,
        React.createElement('div', { style: S.sectionTitle }, 'Bond strength per node Bᵢ'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          NODES.map((n, i) => React.createElement(NodeBar, { key: n, name: n, value: r.B[i], max: Math.max(...r.B), color: '#4090d0' })),
        ),
        React.createElement('div', { style: S.sectionTitle }, 'Top bond pairs'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          React.createElement('table', { style: { borderCollapse: 'collapse', fontSize: 12 } },
            React.createElement('thead', null, React.createElement('tr', null,
              React.createElement('th', { style: { textAlign: 'left', padding: '6px', fontSize: 10, fontWeight: 700, color: 'var(--ink-500)', borderBottom: '2px solid var(--paper-3)' } }, 'Pair'),
              React.createElement('th', { style: { padding: '6px', fontSize: 10, fontWeight: 700, color: 'var(--ink-500)', borderBottom: '2px solid var(--paper-3)' } }, 'w'),
            )),
            React.createElement('tbody', null, r.pairs.slice(0, 10).map((p, i) =>
              React.createElement('tr', { key: i },
                React.createElement('td', { style: { padding: '5px 6px' } }, `${NODES[p.i]} ↔ ${NODES[p.j]}`),
                React.createElement('td', { style: { padding: '5px 6px', fontFamily: 'var(--font-mono)', fontSize: 11, textAlign: 'center' } }, p.w.toFixed(3)),
              )
            )),
          ),
        ),
        React.createElement('div', { style: S.sectionTitle }, 'Spectral decomposition'),
        React.createElement('div', { style: { ...S.card, padding: '12px 16px' } },
          React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 } },
            React.createElement(Metric, { label: 'λ₁ (Perron)', value: r.eigs[0].toFixed(4) }),
            React.createElement(Metric, { label: 'Λ = λ₂', value: r.eigs[1].toFixed(4), color: r.eigs[1] < 0.45 ? 'var(--mode-stress)' : undefined }),
            React.createElement(Metric, { label: 'Spectral gap', value: (r.eigs[0] - r.eigs[1]).toFixed(4) }),
            React.createElement(Metric, { label: 'trace(W)', value: r.eigs.reduce((a,b) => a+b, 0).toFixed(3) }),
          ),
          r.eigs.map((e, i) => React.createElement(NodeBar, { key: i, name: `λ${i+1}${i===1?' = Λ':''}`, value: Math.abs(e), max: Math.max(...r.eigs.map(Math.abs)), color: i === 1 ? 'var(--accent-teal)' : 'var(--ink-300)' })),
        ),
      ),

      /* Reference tables tab */
      tab === 'reference' && React.createElement(React.Fragment, null,
        React.createElement(Collapsible, { title: 'Table A — Operator → Epiphenomenon', subtitle: `${TABLE_A.length} predictions` },
          React.createElement(RefTableA)),
        React.createElement(Collapsible, { title: 'Table B — Basin → Genre Mix', subtitle: `${TABLE_B.length} basins` },
          React.createElement(RefTableB, { currentBasin: basin })),
        React.createElement(Collapsible, { title: 'Table C — Pathology → Discourse Fingerprint', subtitle: `${TABLE_C.length} signatures` },
          React.createElement(RefTableC)),
        React.createElement('div', { style: { marginTop: 16, padding: '14px 16px', background: 'var(--paper-2)', borderRadius: 4, borderLeft: '3px solid var(--accent-ochre)' } },
          React.createElement('div', { style: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--accent-ochre)', marginBottom: 4 } }, 'Falsification path'),
          React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5 } },
            'Trove (Australian newspapers 1803–present) at article-level granularity, regressed against the Australian CAMS trajectory. If genre-frequency vectors are independent of CAMS state, the framework is falsified at the surface-prediction tier.'),
        ),
      ),

    ),
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
