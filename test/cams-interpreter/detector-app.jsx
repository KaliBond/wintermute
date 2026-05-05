
/* ================================================================
   CAMS Epiphenomenon Detector v2
   Integrates signal detection with formal mapping tables
   (Operator→Epiphenomenon, Basin→Genre, Pathology→Discourse)
   ================================================================ */

const detectorStyles = {
  shell: { maxWidth: 820, margin: '0 auto', padding: '28px 20px 48px' },
  header: { marginBottom: 24 },
  h1: { fontFamily: 'var(--font-sans)', fontSize: 26, fontWeight: 700, color: 'var(--accent-teal)', letterSpacing: '-.01em', marginBottom: 2 },
  sub: { fontSize: 12, color: 'var(--ink-500)', fontFamily: 'var(--font-mono)', lineHeight: 1.5, maxWidth: 640 },
  sectionTitle: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: 'var(--ink-500)', marginBottom: 10, marginTop: 28 },
};

/* ── Signal data (current snapshot) ── */
const SIGNALS = [
  {
    category: 'Social Conditions', status: 'clear', signals: []
  },
  {
    category: 'Power Dynamics', status: 'alert', signals: [
      {
        title: 'Near-disconnection — λ₂=0.000',
        body: 'Fiedler value near zero. Network partition risk is acute — integration can be severed by small perturbations.',
        mappings: ['Λ falls < 0.45 → Gridlock / "broken system" framings; treason discourse', 'Bond Strength B(t) falling → Atomised political identity; secession rhetoric']
      }
    ]
  },
  {
    category: 'Elite Behaviour', status: 'clear', signals: []
  },
  {
    category: 'Social Mood', status: 'alert', signals: [
      {
        title: 'Hands affect is neutral (K=5.0, S=5.0)',
        body: 'The mass population is managing stress within capacity. No acute demoralisation signal — equilibrium with underlying tension.',
        mappings: ['σᵢ near zero (K ≈ S) → Post-political apathy; ironic-detachment register; "vibes" politics']
      },
      {
        title: 'Moderate Hands coherence (C=5.0)',
        body: 'Civic participation is present but contingent — compliance culture rather than genuine civic investment. The population cooperates instrumentally, not from conviction.',
        mappings: ['V_Hands depressed with high S → "Battler" populism, class-conflict framing (not yet triggered — threshold not met)']
      }
    ]
  },
  {
    category: 'Secondary Phenomena', status: 'clear', signals: []
  }
];

const NET_STATS = [
  { label: 'B(t)', value: '0.000', warn: false },
  { label: 'δ_B', value: '0.000', warn: false },
  { label: 'Ṽ', value: '7.500', warn: false },
  { label: 'D̃', value: '5.000', warn: false },
  { label: 'λ₂', value: '0.000', warn: true },
  { label: 'R(t)', value: '∞', warn: true, inf: true },
];

/* ── Table A: Operator → Epiphenomenon ── */
const TABLE_A = [
  { variable: 'τ (K/S)', direction: 'Falls < 1.2', genre: 'Decline-anxiety, apocalyptic, nostalgia, civilisational-twilight rhetoric', mechanism: 'Capacity perceived as inadequate to felt load; symbolic system narrates deficit' },
  { variable: 'τ', direction: 'Rises > 1.5', genre: 'Civic-progress, expansionist, futurist', mechanism: 'Surplus capacity narrated as virtue' },
  { variable: 'ε (overreach)', direction: 'Sustained > 1.3', genre: 'Ideological combat, conspiracy genres, "secret enemy"', mechanism: 'Abstraction runs ahead of material substrate' },
  { variable: 'ε', direction: 'Near 1.0', genre: 'Pragmatist, technocratic register; "common sense"', mechanism: 'Abstraction matched to capacity' },
  { variable: 'Λ (coordination)', direction: 'Falls < 0.45', genre: 'Gridlock / "broken system"; treason discourse', mechanism: 'Coordination failure felt as structural betrayal' },
  { variable: 'Λ', direction: 'Rises, sustained', genre: 'Consensus politics, "adults in the room", reformist energy', mechanism: 'Coordination success legible afterwards' },
  { variable: 'σᵢ (activation)', direction: 'Strong +, slow-loop', genre: 'Missionary politics, ideological export', mechanism: 'C × A amplified by surplus K−S' },
  { variable: 'σᵢ', direction: 'Strong − (inversion)', genre: 'Ressentiment, scapegoating, weaponised grievance', mechanism: 'High A·C amplifying K<S deficit' },
  { variable: 'σᵢ', direction: 'Near zero', genre: 'Post-political apathy; ironic detachment; "vibes"', mechanism: 'Activation indeterminate; affect uncoupled' },
  { variable: 'V_Helm', direction: 'Low', genre: '"Out of touch elite", anti-establishment populism', mechanism: 'Executive not seen as authoritative coordinator' },
  { variable: 'V_Shield', direction: 'High rel. to Helm', genre: 'Security panic, securitisation of unrelated domains', mechanism: 'Praetorian configuration' },
  { variable: 'V_Lore', direction: 'High A, low C', genre: 'Ideological fragmentation, culture-war, identitarian combat', mechanism: 'Competing narratives without alignment' },
  { variable: 'V_Archive vs V_Hands', direction: 'Archive high, material low', genre: 'Heritage politics, monument controversies', mechanism: 'Phantom Type I — cultural compensation' },
  { variable: 'V_Stewards', direction: 'Sustained high', genre: 'Property panic, NIMBY, "the boomers"', mechanism: 'Property as political fact dominates' },
  { variable: 'V_Flow', direction: 'Disproportionate', genre: 'Financialisation discourse, "markets demand"', mechanism: 'Coordination migrated to commerce node' },
  { variable: 'V_Hands', direction: 'Depressed, high S', genre: '"Battler" populism, class-conflict, strike waves', mechanism: 'Labour node experienced as squeezed' },
  { variable: 'V_Craft', direction: 'Sustained low', genre: 'Manufacturing-nostalgia, "hollowing out", regional grievance', mechanism: 'Displaced producer identity' },
  { variable: 'B(t)', direction: 'Falling', genre: 'Atomised identity; secession / breakaway rhetoric', mechanism: 'Coordination conductance below cohesion threshold' },
  { variable: 'Ω (shear)', direction: 'Persistent high', genre: 'Generational, regional, factional conflict genres simultaneously', mechanism: 'Different subsystems decompensating at different rates' },
];

/* ── Table B: Basin → Genre Mix ── */
const TABLE_B = [
  { basin: 'Peak Expansion', dominant: 'Triumphalist, civic-progress, expansive optimism', suppressed: 'Decline, apocalyptic, conspiracist', color: 'var(--basin-peak)' },
  { basin: 'Navigational', dominant: 'Reformist problem-solving, post-crisis reconstruction', suppressed: 'Triumphalism, nihilism', color: 'var(--basin-nav)' },
  { basin: 'Early Strain', dominant: 'Nostalgic, "decline begins", generational disquiet', suppressed: 'Triumphalism, deep cynicism', color: 'var(--basin-strain)' },
  { basin: 'Praetorian', dominant: 'Securitisation, foreign-threat panic, strongman appeals', suppressed: 'Reformist optimism, technocratic register', color: 'var(--basin-praetorian)' },
  { basin: 'Archipelago', dominant: 'Tribal-ironic, "system is rigged but I have my tribe"', suppressed: 'Civic-renewal, common-good rhetoric', color: 'var(--basin-archipelago)' },
  { basin: 'Mass Formation', dominant: 'Single named enemy, ritual denunciation, anti-dissent panic', suppressed: 'Genuine pluralist disagreement', color: 'var(--basin-mass)' },
  { basin: 'Phantom I', dominant: 'Heritage flourishing, cultural-pride, ceremonial elaboration', suppressed: 'Material-grievance, structural-critique', color: 'var(--basin-phantom)' },
  { basin: 'Phantom II', dominant: 'Cynicism, post-political fatigue, ironic-nihilist', suppressed: 'Heritage politics, civic optimism', color: 'var(--basin-phantom)' },
];

/* ── Table C: Pathology → Discourse ── */
const TABLE_C = [
  { pathology: 'Executive Decoupling', condition: 'V_Helm < 6; weak Helm↔Archive, Helm↔Hands bonds', fingerprint: 'Personality-fixated coverage; "leader unable to govern"; cabinet-reshuffle obsession; vocabulary of administrative paralysis' },
  { pathology: 'Praetorian Condition', condition: 'V_Shield > V_Helm sustained', fingerprint: 'Security-bracket expansion; "national security" applied to non-security domains; militarised metaphor in domestic policy' },
  { pathology: 'Mythic-Material Decoupling', condition: 'High Lore/Archive, collapsing Hands/Craft', fingerprint: 'Culture-war saturation; "they hate us for our values" displacing material grievance; ceremonial controversies dominating' },
  { pathology: 'Flow Collapse', condition: 'V_Flow drops sharply', fingerprint: '"Markets in panic"; daily-cycle financial coverage; debt-personification; credit-rating as moral verdict' },
  { pathology: 'Late Abstraction Collapse', condition: 'S leads A by 2–5 yrs', fingerprint: 'Genre exhaustion — even ideology becomes risible; nihilistic and absurdist registers; "satire is dead"' },
  { pathology: 'Thermodynamic Freeze', condition: 'System-wide low activation', fingerprint: 'Discourse thins; coverage volume drops; foreign press carries domestic reporting that domestic press cannot' },
];

/* ── Components ── */

function Badge({ status, count }) {
  const isClear = status === 'clear';
  const s = {
    fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600,
    padding: '2px 8px', borderRadius: 999, letterSpacing: '.03em',
    background: isClear ? '#E6F2EC' : '#FBE9E6',
    color: isClear ? '#1D5B45' : '#7A2A22',
  };
  return React.createElement('span', { style: s }, isClear ? '✓ clear' : `${count} signal${count > 1 ? 's' : ''}`);
}

function SignalCard({ signal, index }) {
  const [open, setOpen] = React.useState(true);
  return React.createElement('div', { style: { padding: '10px 0', borderTop: index > 0 ? '1px solid var(--paper-3)' : 'none' } },
    React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 3 } },
      React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, color: 'var(--ink-300)' } }, index + 1),
      React.createElement('span', { style: { fontSize: 13, fontWeight: 600, color: 'var(--ink-900)' }, dangerouslySetInnerHTML: { __html: signal.title.replace(/`([^`]+)`/g, '<code style="font-family:var(--font-mono);font-size:11.5px;background:var(--paper-2);padding:1px 4px;border-radius:2px">$1</code>') } }),
    ),
    React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5, marginBottom: signal.mappings?.length ? 6 : 0 } }, signal.body),
    signal.mappings?.length > 0 && React.createElement('div', { style: { marginTop: 4 } },
      signal.mappings.map((m, i) =>
        React.createElement('div', { key: i, style: { display: 'flex', gap: 6, alignItems: 'baseline', marginTop: 3 } },
          React.createElement('span', { style: { fontSize: 10, color: 'var(--accent-ochre)', fontWeight: 700, flexShrink: 0 } }, '→'),
          React.createElement('span', { style: { fontSize: 11.5, color: 'var(--ink-500)', fontFamily: 'var(--font-mono)', lineHeight: 1.4 } }, m),
        )
      )
    )
  );
}

function Category({ cat }) {
  return React.createElement('div', { style: { background: 'var(--paper-0)', border: 'var(--border-hair)', borderRadius: 4, marginBottom: 10 } },
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px' } },
      React.createElement('span', { style: { fontSize: 14, fontWeight: 700, color: 'var(--ink-900)' } }, cat.category),
      React.createElement(Badge, { status: cat.status, count: cat.signals.length }),
    ),
    cat.signals.length === 0
      ? React.createElement('div', { style: { padding: '4px 16px 14px', fontSize: 12.5, color: 'var(--ink-300)', fontStyle: 'italic' } }, 'No signals in this category.')
      : React.createElement('div', { style: { padding: '0 16px 14px' } },
          cat.signals.map((s, i) => React.createElement(SignalCard, { key: i, signal: s, index: i }))
        )
  );
}

function NetStats() {
  return React.createElement('div', { style: { marginTop: 16, background: 'var(--paper-0)', border: 'var(--border-hair)', borderRadius: 4, padding: '14px 16px' } },
    React.createElement('h4', { style: detectorStyles.sectionTitle, key: 'h' }, 'Network Statistics'),
    React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8 } },
      NET_STATS.map(s =>
        React.createElement('div', { key: s.label, style: { textAlign: 'center' } },
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-500)', marginBottom: 2 } }, s.label),
          React.createElement('div', { style: { fontFamily: 'var(--font-mono)', fontSize: s.inf ? 16 : 20, fontWeight: 500, color: s.warn ? 'var(--mode-stress)' : 'var(--ink-900)', fontVariantNumeric: 'tabular-nums', letterSpacing: '-.02em' } }, s.value),
        )
      )
    )
  );
}

/* ── Mapping Tables ── */

function TableSection({ title, subtitle, children }) {
  const [expanded, setExpanded] = React.useState(false);
  return React.createElement('div', { style: { marginTop: 14, background: 'var(--paper-0)', border: 'var(--border-hair)', borderRadius: 4, overflow: 'hidden' } },
    React.createElement('div', {
      style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', cursor: 'pointer', userSelect: 'none' },
      onClick: () => setExpanded(!expanded)
    },
      React.createElement('div', null,
        React.createElement('span', { style: { fontSize: 14, fontWeight: 700, color: 'var(--ink-900)' } }, title),
        subtitle && React.createElement('span', { style: { fontSize: 11, color: 'var(--ink-500)', marginLeft: 8 } }, subtitle),
      ),
      React.createElement('span', { style: { fontSize: 12, color: 'var(--ink-300)', fontFamily: 'var(--font-mono)', transition: 'transform .2s' } }, expanded ? '▾' : '▸'),
    ),
    expanded && React.createElement('div', { style: { padding: '0 16px 14px' } }, children)
  );
}

function OperatorTable() {
  const thS = { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid var(--paper-3)', whiteSpace: 'nowrap' };
  const tdS = { fontSize: 12, color: 'var(--ink-700)', padding: '7px 8px', borderBottom: '1px solid var(--paper-3)', lineHeight: 1.4, verticalAlign: 'top' };
  const monoTd = { ...tdS, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-900)' };

  return React.createElement('table', { style: { width: '100%', borderCollapse: 'collapse' } },
    React.createElement('thead', null,
      React.createElement('tr', null,
        React.createElement('th', { style: thS }, 'Variable'),
        React.createElement('th', { style: thS }, 'Direction'),
        React.createElement('th', { style: thS }, 'Predicted genre'),
        React.createElement('th', { style: { ...thS, minWidth: 160 } }, 'Mechanism'),
      )
    ),
    React.createElement('tbody', null,
      TABLE_A.map((r, i) =>
        React.createElement('tr', { key: i },
          React.createElement('td', { style: monoTd }, r.variable),
          React.createElement('td', { style: monoTd }, r.direction),
          React.createElement('td', { style: tdS }, r.genre),
          React.createElement('td', { style: { ...tdS, color: 'var(--ink-500)', fontSize: 11.5 } }, r.mechanism),
        )
      )
    )
  );
}

function BasinTable() {
  return React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
    TABLE_B.map((r, i) =>
      React.createElement('div', { key: i, style: { display: 'grid', gridTemplateColumns: '140px 1fr 1fr', gap: 10, padding: '8px 0', borderBottom: i < TABLE_B.length - 1 ? '1px solid var(--paper-3)' : 'none', alignItems: 'start' } },
        React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6 } },
          React.createElement('span', { style: { width: 8, height: 8, borderRadius: 99, background: r.color, flexShrink: 0 } }),
          React.createElement('span', { style: { fontSize: 12.5, fontWeight: 600, color: 'var(--ink-900)' } }, r.basin),
        ),
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', marginBottom: 2 } }, 'Dominant'),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--ink-700)', lineHeight: 1.4 } }, r.dominant),
        ),
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--ink-500)', marginBottom: 2 } }, 'Suppressed'),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--ink-300)', lineHeight: 1.4 } }, r.suppressed),
        ),
      )
    )
  );
}

function PathologyTable() {
  return React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: 0 } },
    TABLE_C.map((r, i) =>
      React.createElement('div', { key: i, style: { padding: '10px 0', borderBottom: i < TABLE_C.length - 1 ? '1px solid var(--paper-3)' : 'none' } },
        React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 3 } },
          React.createElement('span', { style: { fontSize: 13, fontWeight: 700, color: 'var(--mode-stress)' } }, r.pathology),
          React.createElement('span', { style: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-500)' } }, r.condition),
        ),
        React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5 } }, r.fingerprint),
      )
    )
  );
}

/* ── Trove validation note ── */
function TroveNote() {
  return React.createElement('div', { style: { marginTop: 20, padding: '14px 16px', background: 'var(--paper-2)', borderRadius: 4, borderLeft: '3px solid var(--accent-ochre)' } },
    React.createElement('div', { style: { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em', color: 'var(--accent-ochre)', marginBottom: 4 } }, 'Falsification path'),
    React.createElement('p', { style: { fontSize: 12.5, color: 'var(--ink-700)', lineHeight: 1.5 } },
      'Trove (Australian newspapers 1803–present) at article-level granularity, regressed against the Australian CAMS trajectory. If genre-frequency vectors are independent of CAMS state, the framework is falsified at the surface-prediction tier.'
    ),
  );
}

/* ── App ── */
function App() {
  return React.createElement('div', { style: detectorStyles.shell },
    React.createElement('div', { style: detectorStyles.header },
      React.createElement('h1', { style: detectorStyles.h1 }, 'Epiphenomenon Detector'),
      React.createElement('div', { style: detectorStyles.sub }, 'Secondary signal analysis · surface-level phenomena decomposed into latent drivers via CAMS phase-space coordinates'),
    ),

    React.createElement('div', { style: detectorStyles.sectionTitle }, 'Signal categories'),
    SIGNALS.map((cat, i) => React.createElement(Category, { key: i, cat })),

    React.createElement(NetStats),

    React.createElement('div', { style: detectorStyles.sectionTitle, key: 'mt' }, 'Formal mappings'),
    React.createElement(TableSection, { title: 'Table A — Operator → Epiphenomenon', subtitle: '19 directional predictions' },
      React.createElement(OperatorTable)
    ),
    React.createElement(TableSection, { title: 'Table B — Basin → Genre Mix Signature', subtitle: '8 attractor basins' },
      React.createElement(BasinTable)
    ),
    React.createElement(TableSection, { title: 'Table C — Pathology → Discourse Fingerprint', subtitle: '6 pathology signatures' },
      React.createElement(PathologyTable)
    ),

    React.createElement(TroveNote),
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
