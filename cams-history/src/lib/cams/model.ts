export const NODES = [
  { id: "Helm", gloss: "Leadership", family: "executive" },
  { id: "Shield", gloss: "Security", family: "executive" },
  { id: "Lore", gloss: "Narrative", family: "cognitive" },
  { id: "Stewards", gloss: "Stewardship", family: "cognitive" },
  { id: "Craft", gloss: "Skill", family: "material" },
  { id: "Hands", gloss: "Labour", family: "material" },
  { id: "Archive", gloss: "Memory", family: "cognitive" },
  { id: "Flow", gloss: "Exchange", family: "material" },
] as const;

export type Family = (typeof NODES)[number]["family"];

export const FAMILY_LABEL: Record<Family, string> = {
  executive: "Executive",
  cognitive: "Cognitive",
  material: "Material",
};

const LORE = 2;
const HANDS = 5;
const ARCHIVE = 6;

export type NodeVec = [number, number, number, number];

export type YearMetrics = {
  V: number[];
  vMean: number;
  vMin: number;
  maxS: number;
  B: number[];
  bEff: number;
  BS: number[];
  lambda2: number;
  etaFloor: number;
  dVMean: number | null;
  omega: number | null;
  crisis: boolean;
  flags: { dV: boolean; vMin: boolean; severe: boolean; maxS: boolean };
  pathologies: string[];
};

export type Sample = { y: number; nodes: NodeVec[] };

export type PanelSociety = { name: string; years: number[][] };

export type ArchetypeId = "freeze" | "chronic" | "coherent" | "buffer";

export type CaseYear = { y: number; n: number[][]; reg: string };

export type CaseRound = {
  id: string;
  truth: string;
  archetype: ArchetypeId;
  blurb: string;
  distractors: string[];
  onsets: number[];
  years: CaseYear[];
};

export const ARCHETYPES: { id: ArchetypeId; name: string; blurb: string }[] = [
  {
    id: "freeze",
    name: "Freeze / fracture collapse",
    blurb: "Coordination value falls to zero or below. The library attractor folds.",
  },
  {
    id: "chronic",
    name: "Chronic cyclic instability",
    blurb: "Systemic crises repeat. Recovery never regains a high ceiling.",
  },
  {
    id: "coherent",
    name: "High-coherence mobilisation",
    blurb: "Strong bonds take the shock. A sharp fall, then fast reconstitution.",
  },
  {
    id: "buffer",
    name: "Absorptive buffer",
    blurb: "Shocks are damped. No systemic onset inside the window.",
  },
];

export const CALLS_PER_ROUND = 2;
export const LEAD_WINDOW = 5;
export const SESSION_ROUNDS = 3;
export const GLYPHS = ["Γ", "Δ", "Θ", "Λ", "Ξ", "Σ", "Φ", "Ψ", "Ω"];

export type CallKind = "hit" | "late" | "false";

export type CrisisCall = {
  index: number;
  kind: CallKind;
  points: number;
  onsetIndex: number | null;
  lead: number;
};

export function nodeValue(n: NodeVec): number {
  return n[0] + n[1] - n[2] + 0.5 * n[3];
}

export function formatSigned(n: number, dp = 1): string {
  const text = n.toFixed(dp);
  return n > 0 ? `+${text}` : text;
}

export function asNodes(rows: number[][]): NodeVec[] {
  return rows.map((row) => [row[0] ?? 0, row[1] ?? 0, row[2] ?? 0, row[3] ?? 0]);
}

export function unpackRow(row: number[]): Sample {
  const nodes: NodeVec[] = [];
  for (let i = 0; i < 8; i++) {
    const o = 1 + i * 4;
    nodes.push([row[o] ?? 0, row[o + 1] ?? 0, row[o + 2] ?? 0, row[o + 3] ?? 0]);
  }
  return { y: row[0] ?? 0, nodes };
}

export function rowVMean(row: number[]): number {
  const sample = unpackRow(row);
  return sample.nodes.reduce((sum, node) => sum + nodeValue(node), 0) / 8;
}

function jacobiEigenvalues(matrix: Float64Array, n: number): Float64Array {
  const a = Float64Array.from(matrix);
  for (let sweep = 0; sweep < 64; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    }
    if (off < 1e-14) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        for (let k = 0; k < n; k++) {
          const ikp = a[k * n + p];
          const ikq = a[k * n + q];
          a[k * n + p] = c * ikp - s * ikq;
          a[k * n + q] = s * ikp + c * ikq;
        }
        for (let k = 0; k < n; k++) {
          const ipk = a[p * n + k];
          const iqk = a[q * n + k];
          a[p * n + k] = c * ipk - s * iqk;
          a[q * n + k] = s * ipk + c * iqk;
        }
      }
    }
  }
  const ev = new Float64Array(n);
  for (let i = 0; i < n; i++) ev[i] = a[i * n + i];
  return ev.sort();
}

export function computeYear(nodes: NodeVec[], prev: NodeVec[] | null, dt: number): YearMetrics {
  const V = new Array<number>(8);
  const h = new Array<number>(8);
  let vMean = 0;
  let vMin = Infinity;
  let maxS = -Infinity;
  for (let i = 0; i < 8; i++) {
    const node = nodes[i] ?? [0, 0, 0, 0];
    V[i] = nodeValue(node);
    h[i] = (0.6 * node[0] + 0.4 * node[3]) / 10;
    vMean += V[i];
    if (V[i] < vMin) vMin = V[i];
    if (node[2] > maxS) maxS = node[2];
  }
  vMean /= 8;

  const B = new Array<number>(64).fill(0);
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      const si = nodes[i]?.[2] ?? 0;
      const sj = nodes[j]?.[2] ?? 0;
      B[i * 8 + j] =
        i === j ? 1 : Math.sqrt((h[i] ?? 0) * (h[j] ?? 0)) * Math.pow(2, -(si + sj) / 10);
    }
  }

  let pairSum = 0;
  for (let i = 0; i < 7; i++) {
    for (let j = i + 1; j < 8; j++) pairSum += B[i * 8 + j] ?? 0;
  }
  const bEff = pairSum / 28;

  const BS = new Array<number>(8);
  for (let i = 0; i < 8; i++) {
    let row = 0;
    for (let j = 0; j < 8; j++) row += B[i * 8 + j] ?? 0;
    BS[i] = row / 8;
  }

  const laplacian = new Float64Array(64);
  for (let i = 0; i < 8; i++) {
    let row = 0;
    for (let j = 0; j < 8; j++) row += B[i * 8 + j] ?? 0;
    for (let j = 0; j < 8; j++) {
      laplacian[i * 8 + j] = (i === j ? row : 0) - (B[i * 8 + j] ?? 0);
    }
  }
  const lambda2 = jacobiEigenvalues(laplacian, 8)[1] ?? 0;
  const handsStress = nodes[HANDS]?.[2] ?? 0;
  const etaFloor = Math.min(BS[LORE] ?? 0, BS[ARCHIVE] ?? 0) / Math.max(handsStress, 2);

  let dVMean: number | null = null;
  let omega: number | null = null;
  if (prev) {
    const rates = new Array<number>(8);
    let rateMean = 0;
    let delta = 0;
    const step = Math.max(dt, 1e-9);
    for (let i = 0; i < 8; i++) {
      const previous = nodeValue(prev[i] ?? [0, 0, 0, 0]);
      rates[i] = ((V[i] ?? 0) - previous) / step;
      rateMean += rates[i] ?? 0;
      delta += (V[i] ?? 0) - previous;
    }
    rateMean /= 8;
    dVMean = delta / 8;
    let variance = 0;
    for (let i = 0; i < 8; i++) variance += ((rates[i] ?? 0) - rateMean) ** 2;
    omega = Math.sqrt(variance / 8);
  }

  const flags = {
    dV: dVMean !== null && dVMean < -1,
    vMin: vMin < 4,
    severe: vMin < 0,
    maxS: maxS > 8,
  };
  const crisis = flags.dV || flags.vMin || flags.maxS;
  const pathologies: string[] = [];
  const helm = V[0] ?? 0;
  const shield = V[1] ?? 0;
  const stewards = V[3] ?? 0;
  const lore = V[LORE] ?? 0;
  const hands = V[HANDS] ?? 0;
  if (helm < 6 && (B[0 * 8 + ARCHIVE] ?? 1) < 0.4) pathologies.push("Executive Decoupling");
  if (shield > helm && shield > stewards) pathologies.push("Praetorian Condition");
  if (lore > 8 && hands < 4) pathologies.push("Mythic-Material Decoupling");
  if (vMean <= 0 && etaFloor < 0.03) pathologies.push("Thermodynamic Freeze");

  return { V, vMean, vMin, maxS, B, bEff, BS, lambda2, etaFloor, dVMean, omega, crisis, flags, pathologies };
}

export function computeSeries(samples: Sample[]): YearMetrics[] {
  return samples.map((sample, index) => {
    const prev = index > 0 ? (samples[index - 1]?.nodes ?? null) : null;
    const dt = index > 0 ? sample.y - (samples[index - 1]?.y ?? sample.y) : 1;
    return computeYear(sample.nodes, prev, dt || 1);
  });
}

const seriesCache = new Map<string, { samples: Sample[]; metrics: YearMetrics[] }>();

export function societySeries(society: PanelSociety): { samples: Sample[]; metrics: YearMetrics[] } {
  const hit = seriesCache.get(society.name);
  if (hit) return hit;
  const samples = society.years.map(unpackRow);
  const metrics = computeSeries(samples);
  const packed = { samples, metrics };
  seriesCache.set(society.name, packed);
  return packed;
}

export function caseSeries(round: CaseRound): { samples: Sample[]; metrics: YearMetrics[] } {
  const samples = round.years.map((year) => ({ y: year.y, nodes: asNodes(year.n) }));
  return { samples, metrics: computeSeries(samples) };
}

export function topBonds(B: number[], node: number, count = 3): { index: number; bond: number }[] {
  const bonds: { index: number; bond: number }[] = [];
  for (let j = 0; j < 8; j++) {
    if (j !== node) bonds.push({ index: j, bond: B[node * 8 + j] ?? 0 });
  }
  bonds.sort((a, b) => b.bond - a.bond);
  return bonds.slice(0, count);
}

export function bandLabel(m: YearMetrics): string {
  if (m.pathologies.includes("Thermodynamic Freeze") || m.vMean <= 0) return "Freeze";
  if (m.crisis) return "Systemic stress";
  if (m.vMean >= 8) return "High coherence";
  return "Adaptive";
}

export function reading(m: YearMetrics): string {
  if (m.pathologies.includes("Thermodynamic Freeze")) {
    return "System value is at or below zero, and the memory floor has collapsed.";
  }
  if (m.pathologies.includes("Praetorian Condition")) {
    return "Shield is outvaluing both Helm and the Stewards.";
  }
  if (m.pathologies.includes("Executive Decoupling")) {
    return "Helm is weak, and its bond to Archive has thinned.";
  }
  if (m.pathologies.includes("Mythic-Material Decoupling")) {
    return "Lore is carrying the system while Hands have failed.";
  }
  if (m.flags.severe) return "At least one node has fallen through zero.";
  if (m.flags.dV) return "System value stepped down hard on this sample.";
  if (m.flags.maxS) return "A node is under extreme stress.";
  if (m.crisis) return "Inside the strained band. The weakest node is under 4.";
  if (m.vMean >= 8) return "High-coherence band. The bonds are earning their keep.";
  return "Adaptive band. No systemic flag on this step.";
}

export function nearestIndex(samples: Sample[], year: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < samples.length; i++) {
    const dist = Math.abs((samples[i]?.y ?? 0) - year);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

export function sharpestDrop(metrics: YearMetrics[]): number {
  let index = 0;
  let worst = 0;
  for (let i = 1; i < metrics.length; i++) {
    const drop = metrics[i]?.dVMean;
    if (drop !== null && drop !== undefined && drop < worst) {
      worst = drop;
      index = i;
    }
  }
  return index;
}

export type Notable = { index: number; year: number; value: number };

export function notables(samples: Sample[], metrics: YearMetrics[]): {
  drops: Notable[];
  omega: Notable[];
  eta: Notable[];
} {
  const pack = (pick: (m: YearMetrics) => number | null, count: number, high: boolean): Notable[] => {
    const rows: Notable[] = [];
    for (let i = 1; i < metrics.length; i++) {
      const metric = metrics[i];
      const sample = samples[i];
      if (!metric || !sample) continue;
      const value = pick(metric);
      if (value === null || !Number.isFinite(value)) continue;
      rows.push({ index: i, year: sample.y, value });
    }
    rows.sort((a, b) => (high ? b.value - a.value : a.value - b.value));
    return rows.slice(0, count);
  };
  return {
    drops: pack((m) => m.dVMean, 3, false),
    omega: pack((m) => m.omega, 3, true),
    eta: pack((m) => m.etaFloor, 3, false),
  };
}

let panelPromise: Promise<PanelSociety[]> | null = null;
let casesPromise: Promise<CaseRound[]> | null = null;

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} failed (${response.status})`);
  return response.json() as Promise<T>;
}

export function loadPanel(): Promise<PanelSociety[]> {
  panelPromise ??= loadJson<PanelSociety[]>(`${import.meta.env.BASE_URL}data/panel.json`).catch((error: unknown) => {
    panelPromise = null;
    throw error;
  });
  return panelPromise;
}

export function loadCases(): Promise<CaseRound[]> {
  casesPromise ??= loadJson<CaseRound[]>(`${import.meta.env.BASE_URL}data/cases.json`).catch((error: unknown) => {
    casesPromise = null;
    throw error;
  });
  return casesPromise;
}

export function judgeCall(round: CaseRound, cursorIndex: number, resolved: readonly number[]): CrisisCall {
  const at = (year: number) => round.years.findIndex((sample) => sample.y === year);
  let best: { oi: number; lead: number } | null = null;
  for (let oi = 0; oi < round.onsets.length; oi++) {
    if (resolved.includes(oi)) continue;
    const index = at(round.onsets[oi] ?? -1);
    if (index < 0) continue;
    const lead = index - cursorIndex;
    if (lead > 0 && lead <= LEAD_WINDOW && (!best || lead < best.lead)) best = { oi, lead };
  }
  if (best) {
    const hit = best;
    return {
      index: cursorIndex,
      kind: "hit",
      points: Math.min(60 + 25 * hit.lead, 160),
      onsetIndex: hit.oi,
      lead: hit.lead,
    };
  }
  let late: number | null = null;
  for (let oi = 0; oi < round.onsets.length; oi++) {
    if (resolved.includes(oi)) continue;
    const index = at(round.onsets[oi] ?? -1);
    if (index < 0) continue;
    const ago = cursorIndex - index;
    if (ago >= 0 && ago <= LEAD_WINDOW) late = oi;
  }
  if (late !== null) {
    return { index: cursorIndex, kind: "late", points: 40, onsetIndex: late, lead: 0 };
  }
  return { index: cursorIndex, kind: "false", points: -50, onsetIndex: null, lead: 0 };
}

export function passOnsets(
  round: CaseRound,
  from: number,
  to: number,
  resolved: readonly number[],
  passed: readonly number[],
): number[] {
  const next = [...passed];
  round.onsets.forEach((year, oi) => {
    const index = round.years.findIndex((sample) => sample.y === year);
    if (index > from && index <= to && !resolved.includes(oi) && !next.includes(oi)) next.push(oi);
  });
  return next;
}

export type RoundScore = {
  round: CaseRound;
  calls: CrisisCall[];
  identityGuess: string | null;
  archetypeGuess: ArchetypeId | null;
  identityPoints: number;
  archetypePoints: number;
  callPoints: number;
  discipline: number;
  total: number;
  missed: number;
};

export function scoreRound(
  round: CaseRound,
  calls: CrisisCall[],
  resolved: readonly number[],
  identityGuess: string | null,
  archetypeGuess: ArchetypeId | null,
): RoundScore {
  const callPoints = calls.reduce((sum, call) => sum + call.points, 0);
  const identityPoints = identityGuess === round.truth ? 100 : 0;
  const archetypePoints = archetypeGuess === round.archetype ? 100 : 0;
  const discipline = round.onsets.length === 0 && calls.length === 0 ? 120 : 0;
  const missed = round.onsets.filter((_, index) => !resolved.includes(index)).length;
  return {
    round,
    calls,
    identityGuess,
    archetypeGuess,
    identityPoints,
    archetypePoints,
    callPoints,
    discipline,
    total: callPoints + identityPoints + archetypePoints + discipline,
    missed,
  };
}

export function pickSession(cases: CaseRound[], count = SESSION_ROUNDS): CaseRound[] {
  const bag = [...cases];
  const out: CaseRound[] = [];
  while (out.length < count && bag.length) {
    out.push(bag.splice(Math.floor(Math.random() * bag.length), 1)[0] as CaseRound);
  }
  return out;
}

export function identityOptions(round: CaseRound, session: CaseRound[]): string[] {
  const banned = new Set(session.filter((item) => item !== round).map((item) => item.truth));
  const bag = round.distractors.filter((name) => name !== round.truth && !banned.has(name));
  const picks = [round.truth];
  while (picks.length < 4 && bag.length) {
    picks.push(bag.splice(Math.floor(Math.random() * bag.length), 1)[0] as string);
  }
  for (let i = picks.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const swap = picks[i] as string;
    picks[i] = picks[j] as string;
    picks[j] = swap;
  }
  return picks;
}

export function findCase(cases: CaseRound[], society: string): CaseRound | undefined {
  return cases.find((round) => round.truth === society);
}

export const BEST_KEY = "cams-history.prophet-best";

export const INSTRUMENTS: {
  key: "vMean" | "vMin" | "bEff" | "lambda2" | "etaFloor" | "omega";
  label: string;
  hint: string;
  lo: number;
  hi: number;
  dp: number;
  neutral?: boolean;
  good: (value: number) => boolean;
}[] = [
  { key: "vMean", label: "V̄", hint: "Mean node value", lo: -6, hi: 16, dp: 1, good: (v) => v > 4 },
  { key: "vMin", label: "Vmin", hint: "Weakest node", lo: -8, hi: 12, dp: 1, good: (v) => v >= 4 },
  { key: "bEff", label: "B̄", hint: "Mean pairwise bond", lo: 0, hi: 0.5, dp: 2, good: (v) => v > 0.18 },
  { key: "lambda2", label: "λ₂", hint: "Algebraic connectivity", lo: 0, hi: 3.6, dp: 2, good: (v) => v > 1 },
  { key: "etaFloor", label: "η", hint: "Memory floor", lo: 0, hi: 0.2, dp: 2, good: (v) => v > 0.04 },
  {
    key: "omega",
    label: "ω",
    hint: "Dispersion of the step",
    lo: 0,
    hi: 3.2,
    dp: 2,
    neutral: true,
    good: () => true,
  },
];
