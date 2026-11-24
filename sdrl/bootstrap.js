/* SDRL annual-data bootstrap.
   Mount real repository rows before app.js computes chart-wide bounds and panels.
   No missing year is interpolated. Sparse unified-panel rows remain the fallback. */
(() => {
  "use strict";

  const DATA = CAMS_DATA;
  const NODES = DATA.nodes;
  const NN = NODES.length;
  const RAW_BASE = "https://raw.githubusercontent.com/KaliBond/wintermute/main/data/nations/";
  const LOCAL_BASE = "../data/nations/";
  const ENS_BASES = [
    "Argentina", "Australia", "Brazil", "Cambodia", "Canada", "Chile", "China",
    "Colombia", "Denmark", "France", "Germany", "Greece", "Hong_Kong", "India",
    "Indonesia", "Iran", "Iraq", "Israel", "Italy", "Japan", "Laos", "Latvia",
    "Lebanon", "Mongolia", "Netherlands", "NewZealand", "Norway", "Pakistan",
    "Philippines", "Poland", "Russia", "Saudi_Arabia", "Singapore", "South_Africa",
    "Sweden", "Syria", "Thailand", "Turkiye", "UAE", "UK", "Ukraine", "Venezuela"
  ];
  const ALIASES = {
    "United States": "USA", "United States of America": "USA",
    "United Kingdom": "UK", "Great Britain": "UK",
    "United Arab Emirates": "UAE", "Turkey": "Turkiye", "Türkiye": "Turkiye",
    "New Zealand": "NewZealand", "Hong Kong": "Hong_Kong"
  };
  const report = {
    mounted: [], fallback: [], errors: [], started: new Date().toISOString(),
    policy: "Real source rows only; no annual interpolation. Mounted rows use JUNO v1.2-Final; fallback rows retain their panel labels."
  };
  window.SDRL_MOUNT = report;

  const norm = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
  const baseByNorm = Object.fromEntries(ENS_BASES.concat(["USA"]).map(b => [norm(b), b]));
  const avg = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
  const std = a => { if (!a.length) return null; const m = avg(a); return Math.sqrt(avg(a.map(x => (x - m) ** 2))); };
  const round6 = x => Math.round(x * 1e6) / 1e6;
  const median = a => {
    if (!a.length) return null;
    const b = [...a].sort((x, y) => x - y), m = Math.floor(b.length / 2);
    return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
  };

  function eigenvalues(M) {
    const n = NN, A = M.map(row => Float64Array.from(row));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-18) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const phi = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
        const c = Math.cos(phi), s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
      }
    }
    return Array.from({ length: n }, (_, i) => A[i][i]).sort((a, b) => a - b);
  }

  function lambda2(B) {
    // Standard graph Laplacian: the self-loop is excluded from degree.
    const deg = Array.from({ length: NN }, (_, i) => {
      let sum = 0;
      for (let j = 0; j < NN; j++) if (j !== i) sum += B[i][j];
      return sum;
    });
    const L = B.map((row, i) => Float64Array.from(row, (v, j) => i === j ? deg[i] : -v));
    return eigenvalues(L)[1];
  }

  function system(C, K, S, A) {
    const q = C.map((c, i) => (0.6 * c + 0.4 * A[i]) / 10);
    const V = C.map((c, i) => c + K[i] - S[i] + 0.5 * A[i]);
    const B = Array.from({ length: NN }, () => new Float64Array(NN));
    for (let i = 0; i < NN; i++) {
      B[i][i] = 1;
      for (let j = i + 1; j < NN; j++) {
        B[i][j] = B[j][i] = Math.sqrt(q[i] * q[j]) * Math.pow(2, -(S[i] + S[j]) / 10);
      }
    }
    const upper = [];
    for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) upper.push(B[i][j]);
    const BS = NODES.map((_, i) => avg(Array.from(B[i])));
    const iL = NODES.indexOf("Lore"), iA = NODES.indexOf("Archive"), iH = NODES.indexOf("Hands");
    return {
      V, B, B_eff: avg(upper), lambda2: lambda2(B),
      eta_floor: Math.min(BS[iL], BS[iA]) / Math.max(S[iH], 2),
      V_mean: avg(V), V_std: std(V), V_min: Math.min(...V), max_S: Math.max(...S)
    };
  }
  window.SDRL_MATH = { system, lambda2 };

  function regimeIndex(label) {
    const target = norm(label);
    let i = DATA.regimes.findIndex(r => norm(r.label) === target);
    if (i >= 0) return i;
    if (label === "Unclassified") {
      i = DATA.regimes.findIndex(r => /unclassified|intermediate|uncertain/i.test(r.label));
      if (i >= 0) return i;
    }
    throw new Error(`Regime label missing from dashboard payload: ${label}`);
  }

  function classify(C, K, S, A, sys) {
    const sigma = C.map((c, i) => {
      const delta = K[i] - S[i];
      return (A[i] * c / 100) * Math.sign(delta) * Math.max(Math.abs(delta), 0.1);
    });
    const V = round6(sys.V_mean), N = round6(sys.V_min), B = round6(sys.B_eff);
    const smin = round6(Math.min(...sigma));
    let label;
    if (V < 0 && N < -3 && B < 0.15) label = "Freeze/Collapse";
    else if (V < 6 && N < 0 && B < 0.20) label = "Systemic crisis";
    else if (smin < -0.65) label = "Phantom Type II";
    else if (N < 4.0) label = "Local Node Failure";
    else if (V > 10 && N > 5.0 && B > 0.28 && smin > -0.25) label = "Stable adaptive";
    else if ((V >= 4 && V <= 10) || V > 10) label = "Strained";
    else label = "Unclassified";
    return regimeIndex(label);
  }

  function deriveSeries(years, C, K, S, A, withRegimes) {
    const out = { years: [...years], C, K, S_v: S, A, nodeV: [], V: [], S: [], N: [], B: [], L: [], E: [], W: [], R: [] };
    for (let k = 0; k < years.length; k++) {
      const sys = system(C[k], K[k], S[k], A[k]);
      out.nodeV.push(sys.V);
      out.V.push(sys.V_mean); out.S.push(sys.V_std); out.N.push(sys.V_min);
      out.B.push(sys.B_eff); out.L.push(sys.lambda2); out.E.push(sys.eta_floor);
      out.W.push(k ? std(sys.V.map((v, i) => v - out.nodeV[k - 1][i])) : null);
      out.R.push(withRegimes ? classify(C[k], K[k], S[k], A[k], sys) : null);
    }
    return out;
  }

  function parseCsvLine(line) {
    const cells = []; let cell = "", quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') { cell += '"'; i++; }
        else quoted = !quoted;
      } else if (ch === "," && !quoted) { cells.push(cell); cell = ""; }
      else cell += ch;
    }
    cells.push(cell);
    return cells;
  }

  function csvRows(text) {
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(line => line.trim());
    if (!lines.length) return [];
    const head = parseCsvLine(lines[0]).map(s => s.trim());
    const col = name => head.indexOf(name);
    const iy = col("Year"), inode = col("Node");
    const cols = ["Coherence", "Capacity", "Stress", "Abstraction"].map(col);
    if (iy < 0 || inode < 0 || cols.some(i => i < 0)) throw new Error("Unexpected CAMS CSV header");
    const rows = [];
    for (let n = 1; n < lines.length; n++) {
      const cells = parseCsvLine(lines[n]);
      const year = Number(cells[iy]), node = cells[inode] && cells[inode].trim();
      if (!Number.isFinite(year) || !NODES.includes(node)) continue;
      const values = cols.map(i => {
        const raw = cells[i] == null ? "" : cells[i].trim();
        if (!raw || /^NA$/i.test(raw)) return null;
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      });
      rows.push({ year, node, values });
    }
    return rows;
  }

  function seriesFromRows(rows, minScorers, roundMean) {
    const grouped = new Map();
    for (const row of rows) {
      if (!grouped.has(row.year)) grouped.set(row.year, new Map());
      const byNode = grouped.get(row.year);
      if (!byNode.has(row.node)) byNode.set(row.node, []);
      byNode.get(row.node).push(row.values);
    }
    const years = [], C = [], K = [], S = [], A = [];
    for (const year of [...grouped.keys()].sort((a, b) => a - b)) {
      const byNode = grouped.get(year), vitals = [[], [], [], []];
      let complete = true;
      for (const node of NODES) {
        const entries = byNode.get(node) || [];
        for (let v = 0; v < 4; v++) {
          const values = entries.map(e => e[v]).filter(Number.isFinite);
          if (values.length < minScorers) { complete = false; break; }
          const mean = avg(values);
          vitals[v].push(roundMean ? Math.round(mean * 10) / 10 : mean);
        }
        if (!complete) break;
      }
      // Incomplete node-years remain absent. They are never filled from neighbours.
      if (complete && vitals.every(v => v.length === NN)) {
        years.push(year); C.push(vitals[0]); K.push(vitals[1]); S.push(vitals[2]); A.push(vitals[3]);
      }
    }
    return years.length ? deriveSeries(years, C, K, S, A, true) : null;
  }

  function repeatedStates(s) {
    let n = 0;
    for (let k = 1; k < s.years.length; k++) {
      let same = true;
      for (const key of ["C", "K", "S_v", "A"]) {
        for (let i = 0; i < NN; i++) if (s[key][k][i] !== s[key][k - 1][i]) { same = false; break; }
        if (!same) break;
      }
      if (same) n++;
    }
    return n;
  }

  function sampling(s) {
    const gaps = s.years.slice(1).map((y, i) => y - s.years[i]);
    return { gap: median(gaps), annual: median(gaps) === 1 };
  }

  async function fetchText(file) {
    let lastError;
    for (const base of [LOCAL_BASE, RAW_BASE]) {
      try {
        const response = await fetch(base + file, { cache: "default" });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return await response.text();
      } catch (error) { lastError = error; }
    }
    throw lastError || new Error(`Unable to fetch ${file}`);
  }

  function keepUnified(name, reason) {
    const old = DATA.societies[name];
    const rebuilt = deriveSeries(old.years, old.C, old.K, old.S_v, old.A, false);
    // Recompute canonical diagnostics, including corrected lambda2, but retain panel regimes.
    for (const key of ["nodeV", "V", "S", "N", "B", "L", "E", "W"]) old[key] = rebuilt[key];
    old.source = {
      kind: "unified", label: "JUNO unified panel", reason,
      annual: sampling(old).annual, repeated: repeatedStates(old), classifier: "original panel labels"
    };
    report.fallback.push({ society: name, rows: old.years.length, reason });
  }

  function mountSeries(name, series, source) {
    series.source = Object.assign({
      annual: sampling(series).annual, repeated: repeatedStates(series), classifier: "JUNO v1.2-Final"
    }, source);
    DATA.societies[name] = series;
    report.mounted.push(Object.assign({ society: name, rows: series.years.length,
      first: series.years[0], last: series.years[series.years.length - 1] }, series.source));
  }

  function fileBaseFor(name) {
    if (ALIASES[name]) return ALIASES[name];
    return baseByNorm[norm(name)] || null;
  }

  async function mountOne(name) {
    const base = fileBaseFor(name);
    if (!base) { keepUnified(name, "no annual repository file mapped"); return; }
    const old = DATA.societies[name];
    try {
      if (base === "USA") {
        const texts = await Promise.all(Array.from({ length: 5 }, (_, i) => fetchText(`USA_v12opt_pass${i + 1}.csv`)));
        const rows = texts.flatMap(csvRows);
        const series = seriesFromRows(rows, 3, true); // CAMNATIONSM5N: mean when at least three scorers remain.
        if (series && series.years.length > old.years.length && sampling(series).annual) {
          mountSeries(name, series, {
            kind: "annual-ensemble", label: "USA v12opt five-pass annual ensemble",
            files: "USA_v12opt_pass1.csv … pass5.csv", lineage: "CAMNATIONSM5N; distinct from the unified panel"
          });
        } else keepUnified(name, "annual five-pass series did not improve coverage");
      } else {
        const file = `${base}_ENS.csv`;
        const series = seriesFromRows(csvRows(await fetchText(file)), 1, false);
        if (series && series.years.length > old.years.length && sampling(series).annual) {
          mountSeries(name, series, { kind: "annual-ens", label: `${base}_ENS.csv annual source rows`, files: file,
            lineage: "Repository ENS source; repeated states retained" });
        } else keepUnified(name, "ENS file is not denser annual coverage");
      }
    } catch (error) {
      report.errors.push({ society: name, message: String(error && error.message || error) });
      keepUnified(name, "annual source unavailable; unified fallback retained");
    } finally {
      const status = document.getElementById("bootStatus");
      const done = report.mounted.length + report.fallback.length;
      if (status) status.textContent = `Checking source rows: ${done} of ${Object.keys(DATA.societies).length} societies`;
    }
  }

  function quantile(sorted, q) {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function rebuildCorridor() {
    const points = [];
    for (const [name, s] of Object.entries(DATA.societies)) {
      for (let k = 0; k < s.years.length; k++) points.push({ name, B: s.B[k], V: s.V[k] });
    }
    if (points.length < 20) return;
    const bmin = Math.min(...points.map(p => p.B)), bmax = Math.max(...points.map(p => p.B));
    const bins = 32, rows = [];
    for (let i = 0; i < bins; i++) {
      const lo = bmin + (bmax - bmin) * i / bins, hi = bmin + (bmax - bmin) * (i + 1) / bins;
      const values = points.filter(p => p.B >= lo && (p.B < hi || (i === bins - 1 && p.B <= hi))).map(p => p.V).sort((a, b) => a - b);
      if (values.length < 8) continue;
      rows.push([(lo + hi) / 2, quantile(values, 0.5), quantile(values, 0.16), quantile(values, 0.84)]);
    }
    // The corridor is an empirical p16/p50/p84 band, not a fitted law.
    if (rows.length >= 2) DATA.mainSeq = rows;
    report.observations = points.length;
    report.societies = Object.keys(DATA.societies).length;
    report.yearMin = Math.min(...Object.values(DATA.societies).map(s => s.years[0]));
    report.yearMax = Math.max(...Object.values(DATA.societies).map(s => s.years[s.years.length - 1]));
    DATA.meta = Object.assign(DATA.meta || {}, { observations: points.length, societies: report.societies,
      yearMin: report.yearMin, yearMax: report.yearMax, mountedAnnual: report.mounted.length });
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src; script.async = false;
      script.onload = resolve; script.onerror = () => reject(new Error(`Unable to load ${src}`));
      document.body.appendChild(script);
    });
  }

  async function start() {
    if (document.readyState === "loading") {
      await new Promise(resolve => document.addEventListener("DOMContentLoaded", resolve, { once: true }));
    }
    const names = Object.keys(DATA.societies);
    // Modest concurrency avoids hammering the raw-data host on first load.
    let cursor = 0;
    await Promise.all(Array.from({ length: 6 }, async () => {
      while (cursor < names.length) {
        const name = names[cursor++];
        await mountOne(name);
      }
    }));
    rebuildCorridor();
    await loadScript("app.js");
    try { await loadScript("repair.js"); }
    catch (error) { console.warn("SDRL interface repair layer unavailable; using base interface.", error); }
    if (typeof init === "function") init();
    const status = document.getElementById("bootStatus");
    if (status) status.hidden = true;
  }

  start().catch(error => {
    console.error("SDRL startup failed", error);
    const status = document.getElementById("bootStatus");
    if (status) {
      status.hidden = false;
      status.classList.add("error");
      status.textContent = "The dashboard could not finish mounting its source data. Reload to try again.";
    }
  });
})();
