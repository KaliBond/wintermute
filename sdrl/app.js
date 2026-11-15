/* The Social Dynamics Reference Library — app.js
   Canonical operators (v1.0-Final / v1.1) re-implemented in JS:
   V_i = C+K-S+0.5A ; q_i=(0.6C+0.4A)/10 ; B_ij = sqrt(q_i q_j) * 2^(-(S_i+S_j)/10) ;
   B_eff = mean upper triangle ; lambda2 = 2nd eigenvalue of bond Laplacian ;
   eta_floor = min(BS_Lore, BS_Archive) / max(S_Hands, 2).
   Data: CAMS_DATA (data/data.js), canonical drift verified 3.6e-15. */
"use strict";

const D = CAMS_DATA;
const NODES = D.nodes;                       // 8, canonical order
const NN = NODES.length;
const LAYER_COLOR = { cognitive: "#7fb8d9", executive: "#2f6f7d", material: "#46c08a" };
const NODE_LAYER = {};
D.layers.cognitive.forEach(n => NODE_LAYER[n] = "cognitive");
D.layers.executive.forEach(n => NODE_LAYER[n] = "executive");
D.layers.material.forEach(n => NODE_LAYER[n] = "material");
const REGIME = D.regimes.map((r, i) => ({ i, label: r.label,
  color: ["#46c08a", "#e0a83e", "#8b98a5", "#9d7bd0", "#e05d44", "#7fc4e8", "#d063a8"][i] }));

/* ---------------- canonical math ---------------- */
const canon = (() => {
  const nodeValue = (C, K, S, A) => C + K - S + 0.5 * A;
  const quality = (C, A) => (0.6 * C + 0.4 * A) / 10;
  function bondMatrix(C, K, S, A) {
    const q = C.map((c, i) => quality(c, A[i]));
    const B = Array.from({ length: NN }, () => new Float64Array(NN));
    for (let i = 0; i < NN; i++) { B[i][i] = 1;
      for (let j = i + 1; j < NN; j++)
        B[i][j] = B[j][i] = Math.sqrt(q[i] * q[j]) * Math.pow(2, -(S[i] + S[j]) / 10);
    } return B;
  }
  const bEff = B => { let s = 0, n = 0;
    for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) { s += B[i][j]; n++; } return s / n; };
  // cyclic Jacobi eigenvalues for a small symmetric matrix
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
    const deg = Array.from({ length: NN }, (_, i) => { let s = 0; for (let j = 0; j < NN; j++) s += B[i][j]; return s; });
    const L = B.map((row, i) => Float64Array.from(row, (v, j) => (i === j ? deg[i] : -v)));
    return eigenvalues(L)[1];
  }
  function system(C, K, S, A) {
    const V = C.map((c, i) => nodeValue(c, K[i], S[i], A[i]));
    const B = bondMatrix(C, K, S, A);
    const BS = NODES.map((_, i) => { let s = 0; for (let j = 0; j < NN; j++) s += B[i][j]; return s / NN; });
    const iL = NODES.indexOf("Lore"), iA = NODES.indexOf("Archive"), iH = NODES.indexOf("Hands");
    return {
      V, B, B_eff: bEff(B), lambda2: lambda2(B),
      eta_floor: Math.min(BS[iL], BS[iA]) / Math.max(S[iH], 2),
      V_mean: V.reduce((a, b) => a + b, 0) / NN,
      V_std: Math.sqrt(V.reduce((a, b) => a + (b - V.reduce((x, y) => x + y, 0) / NN) ** 2, 0) / NN),
      V_min: Math.min(...V), max_S: Math.max(...S),
    };
  }
  return { nodeValue, bondMatrix, bEff, lambda2, system };
})();

/* ---------------- state ---------------- */
const SOC_NAMES = Object.keys(D.societies).sort();
const state = {
  society: SOC_NAMES.includes("Australia") ? "Australia" : SOC_NAMES[0],
  lo: 0, hi: 0, year: 0, playing: false,
  hidden: new Set(), probe: null, probeShow: true, tab: "dynamics",
};
const soc = () => D.societies[state.society];
const yidx = () => { const s = soc(); return s.years.indexOf(state.year); };
const YEARS_ALL = (() => { const set = new Set(); SOC_NAMES.forEach(n => D.societies[n].years.forEach(y => set.add(y))); return [...set].sort((a, b) => a - b); })();

/* ---------------- canvas helpers ---------------- */
function ctx2d(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}
const REGIME_OF = r => REGIME[r] || REGIME[2];

/* flat array of every polity-year for dust + kNN */
const DUST = (() => { const out = [];
  for (const name of SOC_NAMES) { const s = D.societies[name];
    for (let k = 0; k < s.years.length; k++)
      out.push({ name, year: s.years[k], V: s.V[k], B: s.B[k], R: s.R[k] });
  } return out; })();
const BX = DUST.map(d => d.B), VY = DUST.map(d => d.V);
const BMIN = Math.min(...BX), BMAX = Math.max(...BX), VMIN = Math.min(...VY), VMAX = Math.max(...VY);

/* ---------------- star chart ---------------- */
function drawChart(hover) {
  const c = document.getElementById("starChart");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 52, r: 14, t: 12, b: 34 };
  const X = b => pad.l + (b - BMIN) / (BMAX - BMIN) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v - VMIN) / (VMAX - VMIN) * (h - pad.t - pad.b);

  // grid
  ctx.strokeStyle = "rgba(27,42,51,0.10)"; ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(27,42,51,0.75)"; ctx.font = "10px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  for (let i = 0; i <= 4; i++) {
    const v = VMIN + (VMAX - VMIN) * i / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(1), 8, y + 3);
    const b = BMIN + (BMAX - BMIN) * i / 4, x = X(b);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.fillText(b.toFixed(2), x - 10, h - 12);
  }
  ctx.fillStyle = "rgba(27,42,51,0.9)";
  ctx.fillText("V̄ mean node value", pad.l, pad.t + 10);
  if (w > 620) {
    ctx.textAlign = "right";
    ctx.fillText("B_eff aggregate bond strength", w - pad.r, pad.t + 10);
    ctx.textAlign = "left";
  }

  // slope of adaptation: band = p16..p84 surface, line = median rest line
  const ms = D.mainSeq;
  ctx.beginPath();
  ms.forEach((r, i) => { const x = X(r[0]), y = Y(r[2]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  for (let i = ms.length - 1; i >= 0; i--) ctx.lineTo(X(ms[i][0]), Y(ms[i][3]));
  const slopeGrad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
  slopeGrad.addColorStop(0, "rgba(47,111,125,0.20)");
  slopeGrad.addColorStop(1, "rgba(47,111,125,0.03)");
  ctx.closePath(); ctx.fillStyle = slopeGrad; ctx.fill();
  ctx.beginPath();
  ms.forEach((r, i) => { const x = X(r[0]), y = Y(r[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = "rgba(47,111,125,0.9)"; ctx.lineWidth = 2.2; ctx.stroke();

  // dust
  for (const p of DUST) {
    if (state.hidden.has(p.R)) continue;
    ctx.fillStyle = REGIME_OF(p.R).color + "55";
    ctx.beginPath(); ctx.arc(X(p.B), Y(p.V), 1.7, 0, 7); ctx.fill();
  }

  // selected trajectory
  const s = soc();
  const kLo = s.years.findIndex(y => y >= state.lo), kHi = s.years.findIndex(y => y > state.hi);
  const hi = kHi === -1 ? s.years.length - 1 : kHi - 1;
  ctx.lineWidth = 2.2; ctx.lineJoin = "round";
  for (let k = Math.max(0, kLo); k < hi; k++) {
    ctx.strokeStyle = REGIME_OF(s.R[k]).color;
    ctx.beginPath(); ctx.moveTo(X(s.B[k]), Y(s.V[k])); ctx.lineTo(X(s.B[k + 1]), Y(s.V[k + 1])); ctx.stroke();
  }
  // year ticks
  const span = Math.max(1, hi - Math.max(0, kLo));
  const step = Math.max(1, Math.ceil(span / 10));
  ctx.fillStyle = "rgba(27,42,51,0.85)"; ctx.font = "9.5px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  for (let k = Math.max(0, kLo); k <= hi; k += step) {
    ctx.beginPath(); ctx.arc(X(s.B[k]), Y(s.V[k]), 2.6, 0, 7);
    ctx.fillStyle = "#ffffff"; ctx.fill();
    ctx.strokeStyle = REGIME_OF(s.R[k]).color; ctx.lineWidth = 1.4; ctx.stroke();
    if ((k - kLo) % (step * 2) === 0) ctx.fillText(s.years[k], X(s.B[k]) + 5, Y(s.V[k]) - 5);
  }
  // direction of travel: the Holling trip, up and back down the slope
  const astep = Math.max(1, Math.round(span / 9));
  ctx.fillStyle = "rgba(27,42,51,0.85)";
  for (let k = Math.max(0, kLo); k < hi; k += astep) {
    const x1 = X(s.B[k]), y1 = Y(s.V[k]), x2 = X(s.B[k + 1]), y2 = Y(s.V[k + 1]);
    const ang = Math.atan2(y2 - y1, x2 - x1), mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    ctx.beginPath();
    ctx.moveTo(mx + 8 * Math.cos(ang), my + 8 * Math.sin(ang));
    ctx.lineTo(mx + 3.2 * Math.cos(ang + 2.5), my + 3.2 * Math.sin(ang + 2.5));
    ctx.lineTo(mx + 3.2 * Math.cos(ang - 2.5), my + 3.2 * Math.sin(ang - 2.5));
    ctx.closePath(); ctx.fill();
  }
  // crosshair at current year
  const ki = yidx();
  if (ki >= 0) {
    const x = X(s.B[ki]), y = Y(s.V[ki]);
    ctx.strokeStyle = "rgba(27,42,51,0.22)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, 5.5, 0, 7);
    ctx.fillStyle = REGIME_OF(s.R[ki]).color; ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
  }
  // probe
  if (state.probeShow && state.probe) {
    const p = probeSystem();
    const x = X(p.B_eff), y = Y(p.V_mean);
    ctx.strokeStyle = "#2f6f7d"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let a = 0; a < 8; a++) { const ang = a * Math.PI / 4 - Math.PI / 2;
      const px = x + 9 * Math.cos(ang), py = y + 9 * Math.sin(ang);
      a ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.fillStyle = "#2f6f7d"; ctx.fill();
    ctx.fillStyle = "#2f6f7d"; ctx.font = "10px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.fillText("probe", x + 12, y + 18);
  }
  // hover tooltip
  const tip = document.getElementById("chartTip");
  if (hover) {
    let best = null, bd = 12;
    for (const p of DUST) {
      const d = Math.hypot(X(p.B) - hover.x, Y(p.V) - hover.y);
      if (d < bd) { bd = d; best = p; }
    }
    if (best) {
      tip.hidden = false;
      let nodesLine = "";
      const so = D.societies[best.name], kk = so.years.indexOf(best.year);
      if (kk >= 0) {
        const V = so.nodeV[kk];
        const iMx = V.indexOf(Math.max(...V)), iMn = V.indexOf(Math.min(...V));
        const inv = NODES.filter((n, i) => so.A[kk][i] * so.C[kk][i] * (so.K[kk][i] - so.S_v[kk][i]) < 0);
        nodesLine = `<br>nodes: max ${NODES[iMx]} ${V[iMx].toFixed(1)} · min ${NODES[iMn]} ${V[iMn].toFixed(1)}` +
          (inv.length ? `<br>inverting: ${inv.join(", ")}` : "");
      }
      tip.innerHTML = `${best.name} · ${best.year}<br>V̄ ${best.V.toFixed(2)} · B ${best.B.toFixed(3)}<br><span class="tt-regime" style="color:${REGIME_OF(best.R).color}">${REGIME_OF(best.R).label}</span>${nodesLine}`;
      tip.style.left = Math.min(w - 170, hover.x + 14) + "px";
      tip.style.top = Math.max(4, hover.y - 46) + "px";
    } else tip.hidden = true;
  } else tip.hidden = true;
}

/* the eight places on the slope, at the selected year — the nodes stay in view */
function renderNodeChips() {
  const el = document.getElementById("nodechips");
  const k = yidx();
  if (!el) return;
  if (k < 0) { el.innerHTML = ""; return; }
  const s = soc();
  el.innerHTML = `<span class="nc-year">${state.society} · ${state.year} — the eight places</span>` +
    NODES.map((n, i) => {
      const V = s.nodeV[k][i];
      const sig = s.A[k][i] * s.C[k][i] * (s.K[k][i] - s.S_v[k][i]);
      return `<span class="nc${sig < 0 ? " inv" : ""}" title="${NODE_DESC[n].role} — ${NODE_DESC[n].desc}">` +
        `<i style="background:${LAYER_COLOR[NODE_LAYER[n]]}"></i>${n} <b>${V >= 0 ? "+" : ""}${V.toFixed(1)}</b></span>`;
    }).join("");
}

/* ---------------- spectral strip ---------------- */
function drawStrip() {
  const c = document.getElementById("spectralStrip");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const NB = 26, bw = (BMAX - BMIN) / NB;
  const counts = Array.from({ length: NB }, () => REGIME.map(() => 0));
  for (const p of DUST) {
    const b = Math.min(NB - 1, Math.floor((p.B - BMIN) / bw));
    counts[b][p.R]++;
  }
  counts.forEach((col, i) => {
    const total = col.reduce((a, b) => a + b, 0);
    if (!total) return;
    let y = h;
    col.forEach((n, r) => {
      const hh = n / total * h;
      ctx.fillStyle = REGIME[r].color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(i / NB * w + 0.5, y - hh, w / NB - 1, hh);
      y -= hh;
    });
  });
  ctx.globalAlpha = 1;
}

/* ---------------- regime scope ---------------- */
function drawScope() {
  const c = document.getElementById("scopeStrip");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const s = soc();
  const y0 = s.years[0], y1 = s.years[s.years.length - 1];
  const pad = { l: 40, r: 8, t: 8, b: 18 };
  const vmin = Math.min(...s.V) - 0.4, vmax = Math.max(...s.V) + 0.4;
  const X = yr => pad.l + (yr - y0) / Math.max(1, y1 - y0) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v - vmin) / (vmax - vmin) * (h - pad.t - pad.b);
  ctx.strokeStyle = "rgba(27,42,51,0.12)";
  ctx.fillStyle = "rgba(27,42,51,0.75)"; ctx.font = "9.5px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  for (let i = 0; i <= 3; i++) {
    const v = vmin + (vmax - vmin) * i / 3, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(1), 4, y + 3);
  }
  // regime segments
  ctx.lineWidth = 2;
  for (let k = 0; k < s.years.length - 1; k++) {
    ctx.strokeStyle = REGIME_OF(s.R[k]).color;
    ctx.beginPath(); ctx.moveTo(X(s.years[k]), Y(s.V[k])); ctx.lineTo(X(s.years[k + 1]), Y(s.V[k + 1])); ctx.stroke();
  }
  // current year
  const ki = yidx();
  if (ki >= 0) {
    const x = X(s.years[ki]), y = Y(s.V[ki]);
    ctx.strokeStyle = "rgba(27,42,51,0.65)"; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fillStyle = REGIME_OF(s.R[ki]).color; ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6; ctx.stroke();
  }
  ctx.fillStyle = "rgba(27,42,51,0.75)";
  ctx.fillText(String(y0), pad.l, h - 5);
  ctx.fillText(String(y1), w - pad.r - 30, h - 5);
  ctx.fillStyle = "rgba(47,111,125,0.9)";
  ctx.fillText(`${state.society} · V̄(t)`, pad.l + 4, pad.t + 9);
}

function scopeReadout() {
  const s = soc(), k = yidx();
  const el = document.getElementById("scopeReadout");
  if (k < 0) { el.innerHTML = ""; return; }
  const flags = [];
  if (k > 0 && s.V[k] - s.V[k - 1] < -1.0) flags.push(["ΔV̄ < −1", "sharp mean decline"]);
  if (s.N[k] < 4) flags.push(["V_min < 4", "node in critical zone"]);
  if (s.N[k] < 0) flags.push(["V_min < 0", "severe"]);
  const maxS = Math.max(...s.S_v[k]);
  if (maxS > 8) flags.push(["max S > 8", "node under severe stress"]);
  const m = [["V̄", s.V[k].toFixed(2)], ["sigma V", s.S[k].toFixed(2)],
    ["V_min", s.N[k].toFixed(2)], ["B_eff", s.B[k].toFixed(3)],
    ["lambda2", s.L[k].toFixed(3)], ["eta floor", s.E[k].toFixed(3)]];
  el.innerHTML = m.map(([kk, vv]) =>
    `<div class="metric"><div class="k">${kk}</div><div class="v">${vv}</div></div>`).join("") +
    flags.map(([kk, vv]) => `<div class="metric flag"><div class="k">${kk}</div><div class="v">${vv}</div></div>`).join("");
  const chip = document.getElementById("regimeChip");
  const rg = REGIME_OF(s.R[k]);
  chip.textContent = rg.label; chip.style.color = rg.color;
}

function scopeMix() {
  const s = soc();
  const kLo = Math.max(0, s.years.findIndex(y => y >= state.lo));
  const kHi = s.years.findIndex(y => y > state.hi);
  const hi = (kHi === -1 ? s.years.length : kHi);
  const counts = REGIME.map(() => 0);
  for (let k = kLo; k < hi; k++) counts[s.R[k]]++;
  const total = Math.max(1, hi - kLo);
  document.getElementById("scopeMix").innerHTML =
    `<div class="mix-row"><span class="lbl" style="color:var(--cyan)">series mix, yrs</span><span class="bar"></span><span class="pct">${hi - kLo}</span></div>` +
    REGIME.map(r => { const n = counts[r.i];
      return `<div class="mix-row"><span class="lbl">${r.label}</span><span class="bar"><i style="width:${(n / total * 100).toFixed(1)}%;background:${r.color}"></i></span><span class="pct">${(n / total * 100).toFixed(0)}%</span></div>`;
    }).join("");
}

/* ---------------- attractor wheel ---------------- */
function drawWheel() {
  const c = document.getElementById("wheel");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const s = soc(), k = yidx();
  if (w < 80 || h < 80) return;   // degenerate canvas mid-layout; skip the frame
  const cx = w / 2, cy = h / 2, R = Math.max(20, Math.min(w, h) / 2 - 46);
  // attractor ring
  ctx.strokeStyle = "rgba(27,42,51,0.18)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
  const ki = k < 0 ? 0 : k;
  const V = s.nodeV[ki];
  for (let i = 0; i < NN; i++) {
    const a0 = i * 2 * Math.PI / NN - Math.PI / 2;
    const node = NODES[i];
    const r = R * Math.max(0.04, Math.min(1, V[i] / 12));
    const col = LAYER_COLOR[NODE_LAYER[node]];
    // sector glyph: arclength wedge
    ctx.beginPath();
    ctx.arc(cx, cy, r, a0 - 0.42, a0 + 0.42);
    ctx.arc(cx, cy, 6, a0 + 0.42, a0 - 0.42, true);
    ctx.closePath();
    ctx.fillStyle = col + "55"; ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke();
    // attractor point on the ring
    const ax = cx + R * Math.cos(a0), ay = cy + R * Math.sin(a0);
    ctx.beginPath(); ctx.arc(ax, ay, 3, 0, 7); ctx.fillStyle = col; ctx.fill();
    // labels (clamped to the canvas so narrow screens don't clip names)
    ctx.font = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.fillStyle = col;
    const ly = cy + (R + 20) * Math.sin(a0);
    const align = Math.abs(Math.cos(a0)) < 0.3 ? "center" : (Math.cos(a0) > 0 ? "left" : "right");
    let lx = cx + (R + 20) * Math.cos(a0);
    const nameW = ctx.measureText(node).width;
    if (align === "left") lx = Math.min(lx, w - 4 - nameW);
    else if (align === "right") lx = Math.max(lx, 4 + nameW);
    else lx = Math.max(4 + nameW / 2, Math.min(lx, w - 4 - nameW / 2));
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(node, lx, ly);
    ctx.fillStyle = "rgba(27,42,51,0.8)";
    ctx.fillText(V[i].toFixed(1), lx, ly + 13);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  ctx.font = "12px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  ctx.fillStyle = "rgba(27,42,51,0.9)";
  ctx.fillText(`${state.year}`, cx, cy - 4);
  const rg = REGIME_OF(s.R[ki]);
  ctx.fillStyle = rg.color;
  ctx.font = "10px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  ctx.fillText(rg.label, cx, cy + 12);
}

/* ---------------- node vitals sparklines ---------------- */
const VITAL_COLOR = { C: "#46c08a", K: "#2f6f7d", S: "#e05d44", A: "#9d7bd0" };
function drawVitals() {
  const c = document.getElementById("vitals");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const s = soc();
  const kLo = Math.max(0, s.years.findIndex(y => y >= state.lo));
  const kHi = s.years.findIndex(y => y > state.hi);
  const hi = (kHi === -1 ? s.years.length : kHi);
  const padL = 74, padR = 10;
  const rowH = h / NN;
  const vital = (node, key) => s[key].slice(kLo, hi).map(row => row[node]);
  const ki = yidx();
  for (let i = 0; i < NN; i++) {
    const y0 = i * rowH;
    ctx.fillStyle = i % 2 ? "rgba(27,42,51,0.03)" : "transparent";
    ctx.fillRect(0, y0, w, rowH);
    ctx.font = "10.5px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.fillStyle = LAYER_COLOR[NODE_LAYER[NODES[i]]];
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(NODES[i], 6, y0 + rowH / 2);
    ["C", "K", "S", "A"].forEach(key => {
      const ser = vital(i, key === "S" ? "S_v" : key);
      if (ser.length < 2) return;
      ctx.strokeStyle = VITAL_COLOR[key]; ctx.lineWidth = 1.1; ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ser.forEach((v, t) => {
        const x = padL + t / (ser.length - 1) * (w - padL - padR);
        const y = y0 + rowH - 5 - v / 10 * (rowH - 10);
        t ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke(); ctx.globalAlpha = 1;
    });
    // current year marker
    if (ki >= kLo && ki < hi) {
      const x = padL + (ki - kLo) / Math.max(1, hi - kLo - 1) * (w - padL - padR);
      ctx.strokeStyle = "rgba(27,42,51,0.4)"; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, y0 + 3); ctx.lineTo(x, y0 + rowH - 3); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  ctx.textBaseline = "alphabetic";
}

/* ---------------- data scorer ---------------- */
const FADERS = [["C", "coherence"], ["K", "capacity"], ["S", "stress"], ["A", "abstraction"]];
function buildScorer() {
  const wrap = document.getElementById("scorerCards");
  wrap.innerHTML = "";
  NODES.forEach((node, i) => {
    const card = document.createElement("div");
    card.className = "node-card";
    card.style.borderTop = `2px solid ${LAYER_COLOR[NODE_LAYER[node]]}`;
    card.innerHTML = `<h4>${node}<span class="nv" id="nv-${i}">—</span></h4>` +
      FADERS.map(([key, name]) => `
        <div class="fader f-${key}">
          <label for="f-${i}-${key}">${key}</label>
          <input type="range" id="f-${i}-${key}" min="0" max="10" step="0.5" value="5" data-node="${i}" data-vital="${key}" aria-label="${node} ${name}">
          <output id="o-${i}-${key}">5</output>
        </div>`).join("");
    wrap.appendChild(card);
  });
  wrap.addEventListener("input", e => {
    if (!state.probe) return;
    const t = e.target;
    if (t.matches("input[type=range]")) {
      const i = +t.dataset.node, key = t.dataset.vital;
      state.probe[key][i] = +t.value;
      document.getElementById(`o-${i}-${key}`).textContent = (+t.value).toString();
      updateDerived(); drawChart();
    }
  });
  document.getElementById("probeReset").addEventListener("click", resetProbe);
  document.getElementById("probeShow").addEventListener("change", e => {
    state.probeShow = e.target.checked; drawChart();
  });
}
function resetProbe() {
  const s = soc(), k = Math.max(0, yidx());
  state.probe = { C: [...s.C[k]], K: [...s.K[k]], S: [...s.S_v[k]], A: [...s.A[k]] };
  NODES.forEach((_, i) => FADERS.forEach(([key]) => {
    const el = document.getElementById(`f-${i}-${key}`);
    if (el) { el.value = state.probe[key][i];
      document.getElementById(`o-${i}-${key}`).textContent = state.probe[key][i]; }
  }));
  updateDerived(); drawChart();
}
function probeSystem() {
  const p = state.probe;
  return canon.system(p.C, p.K, p.S, p.A);
}
function probeRegimeNeighbour(sys) {
  // 15-NN vote in standardised (B_eff, V_mean) space, descriptive only
  const bM = (BMIN + BMAX) / 2, bS = (BMAX - BMIN) / 2 || 1;
  const vM = (VMIN + VMAX) / 2, vS = (VMAX - VMIN) / 2 || 1;
  const dist = DUST.map(p => ({ d: ((p.B - sys.B_eff) / bS) ** 2 + ((p.V - sys.V_mean) / vS) ** 2, r: p.R }));
  dist.sort((a, b) => a.d - b.d);
  const votes = REGIME.map(() => 0);
  dist.slice(0, 15).forEach(p => votes[p.r]++);
  let best = 0; votes.forEach((v, i) => { if (v > votes[best]) best = i; });
  return { regime: REGIME[best], share: votes[best] / 15 };
}
function updateDerived() {
  if (!state.probe) return;
  const sys = probeSystem();
  NODES.forEach((_, i) => { const el = document.getElementById(`nv-${i}`); if (el) el.textContent = "V " + sys.V[i].toFixed(1); });
  const flags = [];
  if (sys.V_min < 4) flags.push("V_min < 4");
  if (sys.V_min < 0) flags.push("V_min < 0 severe");
  if (sys.max_S > 8) flags.push("max S > 8");
  const nb = probeRegimeNeighbour(sys);
  document.getElementById("derivedReadout").innerHTML = [
    ["V̄", sys.V_mean.toFixed(2)], ["sigma V", sys.V_std.toFixed(2)],
    ["V_min", sys.V_min.toFixed(2)], ["B_eff", sys.B_eff.toFixed(3)],
    ["lambda2", sys.lambda2.toFixed(3)], ["eta floor", sys.eta_floor.toFixed(3)],
  ].map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("") +
  `<div class="metric"><div class="k">regime neighbourhood (k-NN)</div><div class="v" style="color:${nb.regime.color}">${nb.regime.label} <small>${Math.round(nb.share * 100)}%</small></div></div>` +
  (flags.length ? `<div class="metric flag"><div class="k">crisis flags</div><div class="v">${flags.join(" · ")}</div></div>` : "");
}

/* ---------------- events ---------------- */
// the window is gone: the sweep always covers the society's whole sampled series
function fullRange() {
  const s = soc();
  state.lo = s.years[0];
  state.hi = s.years[s.years.length - 1];
}
function markActiveLink() {
  document.querySelectorAll("#socLinks a").forEach(a =>
    a.classList.toggle("on", a.dataset.soc === state.society));
}
// the year slider mirrors state: it steps the society's sampled years and
// travels with the sweep, so the indicator and the instruments never disagree
function syncYearSlider() {
  const s = soc(), ysl = document.getElementById("yearSlider");
  if (!ysl) return;
  ysl.max = String(s.years.length - 1);
  ysl.value = String(Math.max(0, yidx()));
}
function selectSociety(n) {
  state.society = n; togglePlay(false);
  markActiveLink();
  state.probe = null; resetProbe();
  fullRange();
  setYear(soc().years[soc().years.length - 1]);
}
function setYear(y) {
  const s = soc();
  // nearest sampled year
  let best = s.years[0], bd = 1e9;
  for (const yy of s.years) { const d = Math.abs(yy - y); if (d < bd) { bd = d; best = yy; } }
  state.year = best;
  document.getElementById("yearNow").textContent = best;
  ["yrBig", "yrBigChart"].forEach(id => { const yb = document.getElementById(id); if (yb) yb.textContent = best; });
  syncYearSlider();
  updateURL();
  renderAll();
}
function renderAll() {
  // split page: the machine and the slope sit side by side, both always drawn
  drawChart(); drawScope(); scopeReadout(); scopeMix(); renderNodeChips();
  renderDossier(); drawSkTimeline(); renderSysVit();
  drawWheel();  // the eight nodes live on the page proper, always drawn
  const benchOpen = !document.getElementById("panel-bench").hidden;
  if (benchOpen && state.tab === "dynamics") drawVitals();
  if (benchOpen && state.tab === "scorer") updateDerived();
}
let raf = null, lastT = 0;
function tick(t) {
  if (!state.playing) return;
  const s = soc();
  const interval = Math.max(250, 60000 / s.years.length);  // full sweep runs no faster than one minute
  if (t - lastT > interval) {
    lastT = t;
    const k = yidx();
    const next = s.years[k + 1] !== undefined ? s.years[k + 1] : state.lo;
    setYear(next);
  }
  raf = requestAnimationFrame(tick);
}
function togglePlay(force) {
  state.playing = force !== undefined ? force : !state.playing;
  document.querySelector(".ic-play").style.display = state.playing ? "none" : "";
  document.querySelector(".ic-pause").style.display = state.playing ? "" : "none";
  if (state.playing) { lastT = 0; raf = requestAnimationFrame(tick); }
  else if (raf) cancelAnimationFrame(raf);
}

function init() {
  // URL state: ?soc=USA&yr=1933&view=constellation
  const params = new URLSearchParams(location.search);
  const psoc = params.get("soc");
  if (psoc && SOC_NAMES.includes(psoc)) state.society = psoc;
  // society links (no dropdown: the names themselves are the navigation)
  const links = document.getElementById("socLinks");
  SOC_NAMES.forEach(n => {
    const a = document.createElement("a");
    a.href = "#"; a.textContent = n; a.dataset.soc = n;
    a.addEventListener("click", e => { e.preventDefault(); selectSociety(n); });
    links.appendChild(a);
  });
  markActiveLink();

  // view switch
  initConstSVG();
  initSkScrub();

  document.getElementById("playBtn").addEventListener("click", () => togglePlay());

  // year slider: manual year selection; snaps to the nearest sampled year
  document.getElementById("yearSlider").addEventListener("input", e =>
    setYear(soc().years[+e.target.value]));

  // legend with toggles
  const leg = document.getElementById("chartLegend");
  REGIME.forEach(r => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="swatch" style="background:${r.color}"></span>${r.label}`;
    b.title = "toggle " + r.label + " points";
    b.addEventListener("click", () => {
      state.hidden.has(r.i) ? state.hidden.delete(r.i) : state.hidden.add(r.i);
      b.classList.toggle("off"); drawChart();
    });
    leg.appendChild(b);
  });

  // wheel legend
  document.getElementById("wheelLegend").innerHTML = Object.entries(LAYER_COLOR)
    .map(([k, v]) => `<span><b style="color:${v}">${k[0].toUpperCase()}</b> ${k} loop</span>`).join("");

  // chart hover
  const chart = document.getElementById("starChart");
  chart.addEventListener("mousemove", e => {
    const r = chart.getBoundingClientRect();
    drawChart({ x: e.clientX - r.left, y: e.clientY - r.top });
  });
  chart.addEventListener("mouseleave", () => drawChart());
  chart.addEventListener("click", e => {
    const r = chart.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    // nearest own-society year within 14px -> select year
    const s = soc(), pad = { l: 52, r: 14, t: 12, b: 34 };
    const X = b => pad.l + (b - BMIN) / (BMAX - BMIN) * (r.width - pad.l - pad.r);
    const Y = v => r.height - pad.b - (v - VMIN) / (VMAX - VMIN) * (r.height - pad.t - pad.b);
    let best = null, bd = 14;
    s.years.forEach((yr, k) => { const d = Math.hypot(X(s.B[k]) - x, Y(s.V[k]) - y); if (d < bd) { bd = d; best = yr; } });
    if (best !== null) setYear(best);
  });

  // scope strip scrub
  const scope = document.getElementById("scopeStrip");
  const scrub = e => {
    const r = scope.getBoundingClientRect();
    const s = soc(), pad = { l: 40, r: 8 };
    const frac = (e.clientX - r.left - pad.l) / (r.width - pad.l - pad.r);
    const y0 = s.years[0], y1 = s.years[s.years.length - 1];
    const yr = Math.round(y0 + Math.max(0, Math.min(1, frac)) * (y1 - y0));
    setYear(yr);
  };
  let scrubbing = false;
  scope.addEventListener("pointerdown", e => { scrubbing = true; scope.setPointerCapture(e.pointerId); scrub(e); });
  scope.addEventListener("pointermove", e => { if (scrubbing) scrub(e); });
  scope.addEventListener("pointerup", () => scrubbing = false);

  // tabs
  const tabs = { "tab-dynamics": "dynamics", "tab-scorer": "scorer" };
  Object.entries(tabs).forEach(([id, key]) => {
    document.getElementById(id).addEventListener("click", () => {
      state.tab = key;
      Object.entries(tabs).forEach(([id2, k2]) => {
        document.getElementById(id2).setAttribute("aria-selected", String(k2 === key));
        document.getElementById("bench-" + k2).hidden = k2 !== key;
      });
      renderAll();
    });
  });

  // the bench is tucked away: it distracts from the simple dynamics unless asked for
  const benchPanel = document.getElementById("panel-bench");
  const benchToggle = document.getElementById("benchToggle");
  function setBench(open, scroll) {
    benchPanel.hidden = !open;
    benchToggle.setAttribute("aria-expanded", String(open));
    benchToggle.innerHTML = (open ? "Hide instrument bench " : "Instrument bench ") + "<span aria-hidden='true'>" + (open ? "\u25BE" : "\u25B8") + "</span>";
    if (open) { if (state.tab === "dynamics") drawVitals(); else updateDerived(); if (scroll) benchPanel.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
    updateURL();
  }
  benchToggle.addEventListener("click", () => setBench(benchPanel.hidden, true));
  window.__setBench = setBench;

  buildScorer();
  window.addEventListener("resize", renderAll);

  // init range + year + probe
  const s = soc();
  fullRange();
  const pyr = parseInt(params.get("yr"), 10);
  setYear(Number.isFinite(pyr) ? pyr : s.years[s.years.length - 1]);
  markActiveLink();
  resetProbe();
  drawStrip();
  if (params.get("bench") === "1") setBench(true, false);
  startConst();  // the spring machine never stops
}

document.addEventListener("DOMContentLoaded", init);

/* ---------------- constellation view (port of neuralnations.org/test/telescope_2) ----------------
   Spring weights use the canonical bond B_ij (the prototype's Gaussian-kernel W is replaced).
   Ring colour = sign of sigma_i = (A*C)(K-S): teal when the node activates, red when it inverts.
   Basin indices: sk = S-bar/K-bar and praetorian verified exactly against the prototype data;
   b = canonical B_eff; cog_gap = (V cognitive-bar - V material-bar)/10. Disclosed in the panel. */
"use strict";

const TS_NODE_HEX = { Lore: "#2E8B6B", Archive: "#2F7A5C", Helm: "#C84C3E", Stewards: "#D98B2B",
  Shield: "#B0392E", Craft: "#3A6FA8", Hands: "#4B7CAE", Flow: "#6A8FB8" };
const NODE_DESC = {
  Helm: { role: "Executive / Governance", desc: "Who steers collective decisions" },
  Shield: { role: "Military / Security", desc: "Who holds organised force" },
  Lore: { role: "Cultural Wisdom", desc: "Shared meaning & moral memory" },
  Stewards: { role: "Rule of Law / Property", desc: "Who arbitrates rights & contracts" },
  Archive: { role: "Institutional Memory", desc: "Knowledge systems & records" },
  Craft: { role: "Technology / Industry", desc: "Material production capacity" },
  Hands: { role: "Labour", desc: "Working population vitality" },
  Flow: { role: "Commerce / Markets", desc: "Exchange & circulation of value" },
};
const BASINS = {
  PEAK_EXPANSION: { name: "PEAK EXPANSION", sub: "High-Coherence Imperial Attractor", col: "#f0a020",
    meaning: "System at designed apex. Collective confidence is total. Build durable Archive & Lore positions now — the trap is invisible from inside." },
  NAVIGATIONAL: { name: "NAVIGATIONAL", sub: "Reconstruction / Confident Basin", col: "#40b0e0",
    meaning: "Purposeful forward motion. Crisis is recent memory. The social contract feels real. Invest in slow-loop institutions while the window is open." },
  EARLY_STRAIN: { name: "EARLY STRAIN", sub: "Transition / Tipping Basin", col: "#d09020",
    meaning: "Ambient unease without clear object. Nostalgia is becoming political. Institutions feel less competent than memory claims. The future is contested." },
  PRAETORIAN: { name: "PRAETORIAN", sub: "Military-Executive Crisis Basin", col: "#d04030",
    meaning: "Reactive, threat-saturated, rapid-cycling. No long-horizon strategy survives. Emergency is permanent. Affiliate with Shield — titles are theatre." },
  ARCHIPELAGO: { name: "ARCHIPELAGO", sub: "Post-Coherence Fragmentation", col: "#7070a0",
    meaning: "Real coordination has left the building. Public institutions are theatre. Find your enclave and operate within it. Cross-enclave work is costly." },
  MASS_FORMATION: { name: "MASS FORMATION", sub: "Hypnotic Totalist Attractor", col: "#900020",
    meaning: "A named anxiety object has crystallised. 20–30% of the population is in hypnosis-like state. Reason is not the medium. Preserve Archive & Lore for the rebuild." },
  PHANTOM: { name: "PHANTOM", sub: "Cultural Persistence / Material Collapse", col: "#8050b0",
    meaning: "Cultural & intellectual life flourishes while the material substrate rots. The map is mistaken for the territory — and this mistake is the proudest expression of identity." },
};
const ZEIT = {
  PEAK_EXPANSION: "Collective confidence is at full expression. Shared purpose requires no enforcement — it is experienced as natural and correct. Long-horizon thinking dominates public discourse. The dominant emotional register is expansive optimism with an undertow of self-congratulation. This is the phase in which the map is mistaken for the territory.",
  NAVIGATIONAL: "The mood is purposeful. A shared problem-solving orientation pervades institutional life. Pragmatism is valued over ideology. Trust in collective action is high; the social contract feels reciprocal. The dominant emotional register is forward-looking relief — crisis is recent memory, reconstruction is the work.",
  EARLY_STRAIN: "Ambient unease without a clear object. Optimism has not inverted but nostalgia is becoming a political force. Institutions feel less competent than memory says they were. Competing narratives are fragmenting the shared discourse. The future is contested rather than assumed.",
  PRAETORIAN: "Reactive, threat-saturated, rapid-cycling. No long-horizon strategy is coherent — only stimulus-response management. Emergency is the permanent condition. External enemies and internal traitors provide directionality to anxiety that has no structural resolution. Strongman behaviour functions as stress management, not governance.",
  ARCHIPELAGO: "Privatised coordination. Real organisation flows through corporate, factional, and network nodes below or beyond the state. Public institutions persist as theatre. Irony and detachment armour people against commitment. The dominant register is solipsistic intensity — deeply tribal within small groups, deeply indifferent outside them.",
  MASS_FORMATION: "A named anxiety object has crystallised. Approximately 20–30% of the population has entered a hypnosis-like state — capable of collective behaviour that no individual within it would endorse in isolation. Ethical self-awareness is suppressed in the hypnotised cohort. Dissent is experienced as existential threat, not disagreement.",
  PHANTOM: "Cultural and intellectual life persists — perhaps flourishes in isolated nodes — while the material substrate deteriorates. Art, law, and ritual continue as if the structure still holds. The map is mistaken for the territory, and this mistake is the proudest expression of civilisational identity. Decline is not legible from within the cultural forms that carry its memory.",
};
const SURV = {
  PEAK_EXPANSION: "Align with the dominant node ({top}) — its coordination energy is real and reliable. Avoid direct criticism of imperial or expansionist narratives: the cost is social marginalisation with no countervailing reward. This is a good time to build durable Archive and Lore investments.",
  NAVIGATIONAL: "Work with the Helm — it has genuine authority and is using it for construction rather than defence. Invest in Archive and Lore nodes now, before the next transition. The window for institution-building is open.",
  EARLY_STRAIN: "Cultivate {top} actors but hedge. The basin is tipping — tomorrow's dominant node may not be today's. Avoid being publicly identified with any single faction. Build lateral bonds across nodes: that is what will survive the coming rearrangement.",
  PRAETORIAN: "Affiliate conspicuously with Shield actors — they hold operational coordination energy regardless of nominal titles. Never criticise the security apparatus in public. Keep Archive and Lore activity private: slow-loop thinking is dangerous here. Exit windows close at sk > 2.0.",
  ARCHIPELAGO: "Find your enclave and operate within it. Cross-enclave coordination is possible but costly. The state is not your protector but it is not your enemy either — it simply lacks coherent agency. Avoid symbolic acts of loyalty or resistance: they cost more than they return.",
  MASS_FORMATION: "Maximum caution. Do not attempt to reason with the hypnotised cohort — reason is not the medium of this state. Cultivate the non-hypnotised minority (~30% of population): they are not gone, merely quiet. Archive and Lore preservation is the strategic priority — institutions will be rebuilt from whatever survives this phase.",
  PHANTOM: "The cultural layer is alive and can be worked with. The material layer is unreliable — do not depend on infrastructure, supply chains, or official authority for anything critical. Build your own material networks. The intellectual wealth of this society is real and worth preserving; its political claims are not.",
};
/* downstream of the web: what the meta-system's epiphenomena do to deliberation.
   Interpretation layer, keyed to the same basin heuristic as ZEIT/SURV. */
const DELIB = {
  PEAK_EXPANSION: "Deliberation rides on confidence. Long-horizon thinking is cheap, consensus feels natural — and the failure modes of overconfidence go unexamined.",
  NAVIGATIONAL: "Deliberation works. Pragmatism outranks ideology and trust in collective problem-solving is high — the capacity to think together is the web's great downstream gift.",
  EARLY_STRAIN: "Deliberation strains. Nostalgia competes with evidence, competence is doubted, the shared discourse fragments — sense-making turns tribal and the future becomes contested.",
  PRAETORIAN: "Deliberation gives way to reaction. Threat framing crowds out reflection; stimulus-response replaces the long view, and the emergency tempo permits no second thoughts.",
  ARCHIPELAGO: "Deliberation privatises. Sense-making retreats into enclaves, the public square persists as theatre, and cross-enclave reasoning grows costly.",
  MASS_FORMATION: "Deliberation is suspended for the hypnotised cohort. Dissent reads as existential threat; reason is not the medium of this state.",
  PHANTOM: "Deliberation flourishes in isolated nodes while the substrate rots — cultural brilliance mistaken for systemic health.",
};
function nominalHolder(society, yr) {
  const T = {
    USA: yr > 1933 ? { role: "Federal Executive", power: "United States Government" } : { role: "Federal Government", power: "United States" },
    Germany: yr >= 1933 && yr <= 1945 ? { role: "Reichskanzler", power: "Nazi Reich" } :
      yr < 1918 ? { role: "Kaiser", power: "German Empire" } :
      yr < 1933 ? { role: "Reichspräsident", power: "Weimar Republic" } :
      { role: "Federal Chancellor", power: "Federal Republic" },
    China: yr < 1912 ? { role: "Qing Emperor", power: "Qing Dynasty" } :
      yr < 1949 ? { role: "KMT / Warlords", power: "Republic of China" } :
      yr < 1976 ? { role: "Chairman Mao", power: "PRC — Maoist" } :
      { role: "General Secretary", power: "PRC — Reform era" },
    UK: yr < 1918 ? { role: "Prime Minister", power: "British Empire" } : { role: "Prime Minister", power: "United Kingdom" },
    Spain: yr >= 1936 && yr <= 1939 ? { role: "Nationalist / Republican", power: "Spanish Civil War" } :
      yr >= 1939 && yr <= 1975 ? { role: "Generalísimo Franco", power: "Francoist Spain" } :
      { role: "Prime Minister", power: "Kingdom of Spain" },
    Italy: yr >= 1922 && yr <= 1943 ? { role: "Il Duce", power: "Fascist Italy" } : { role: "Prime Minister", power: "Republic of Italy" },
    Australia: { role: "Prime Minister", power: "Commonwealth of Australia" },
    "South Africa": yr < 1994 ? { role: "State President", power: "Apartheid South Africa" } : { role: "President", power: "Republic of South Africa" },
  };
  return T[society] || { role: "Head of State", power: society };
}
const avg = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);

function basinInputs(k) {
  const s = soc();
  const C = s.C[k], K = s.K[k], S = s.S_v[k], A = s.A[k], V = s.nodeV[k];
  const iH = NODES.indexOf("Helm"), iS = NODES.indexOf("Shield");
  const cog = ["Archive", "Lore", "Stewards"].map(n => NODES.indexOf(n));
  const mat = ["Craft", "Flow", "Hands"].map(n => NODES.indexOf(n));
  const sys = canon.system(C, K, S, A);
  return {
    sk: avg(S) / avg(K), b: sys.B_eff,
    praetorian: (V[iS] - V[iH]) / (V[iS] + V[iH]),
    cog_gap: (avg(cog.map(i => V[i])) - avg(mat.map(i => V[i]))) / 10,
    sys, V,
  };
}
function classifyBasin(d) {
  const { sk, praetorian, cog_gap, b } = d;
  if (b > 0.7 && sk < 0.75 && praetorian < 0.2) return "PEAK_EXPANSION";
  if (sk < 0.95 && cog_gap > 0.1 && b > 0.4) return "NAVIGATIONAL";
  if (cog_gap < -0.18 && b < 0.3 && sk > 1.4) return "MASS_FORMATION";
  if (praetorian > 0.35 && sk > 1.3) return "PRAETORIAN";
  if (b < 0.25 && sk > 1.1 && cog_gap > 0.15) return "PHANTOM";
  if (b < 0.3 && sk > 1.15) return "ARCHIPELAGO";
  return "EARLY_STRAIN";
}
function threatLevel(d) {
  const { sk, praetorian, b } = d;
  if (sk > 1.8 || praetorian > 0.5 || b < 0.1) return { level: "CRITICAL", col: "#d03030" };
  if (sk > 1.4 || praetorian > 0.3 || b < 0.25) return { level: "HIGH", col: "#d07020" };
  if (sk > 1.1 || praetorian > 0.15) return { level: "MODERATE", col: "#c0a020" };
  return { level: "LOW", col: "#30b090" };
}
/* per-society cached sk series for the timeline */
const skCache = {};
function skSeries() {
  const name = state.society;
  if (!skCache[name]) {
    const s = D.societies[name];
    skCache[name] = s.years.map((_, k) => basinInputs(k).sk);
  }
  return skCache[name];
}

/* ---------------- constellation physics + rendering ---------------- */
const NS = "http://www.w3.org/2000/svg";
const constState = { nodes: null, animT: 0, raf: null, els: null, society: null, hover: null };
const SVG_W = 900, SVG_H = 460;

function initConstSVG() {
  const svg = document.getElementById("const-svg");
  svg.setAttribute("viewBox", `0 0 ${SVG_W} ${SVG_H}`);
  svg.innerHTML = "";
  const ringsG = document.createElementNS(NS, "g");
  [0.20, 0.34, 0.46].forEach(f => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", SVG_W / 2); c.setAttribute("cy", SVG_H / 2);
    c.setAttribute("r", Math.min(SVG_W, SVG_H) * f);
    c.setAttribute("fill", "none"); c.setAttribute("stroke", "#dad2b8");
    c.setAttribute("stroke-width", "0.6"); c.setAttribute("stroke-dasharray", "2 4");
    c.setAttribute("opacity", "0.5");
    ringsG.appendChild(c);
  });
  svg.appendChild(ringsG);
  const edgesG = document.createElementNS(NS, "g");
  const edgeEls = [];
  for (let k = 0; k < 12; k++) {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", "0");
    edgesG.appendChild(line); edgeEls.push(line);
  }
  svg.appendChild(edgesG);
  const nodesG = document.createElementNS(NS, "g");
  const nodeEls = [];
  NODES.forEach(n => {
    const g = document.createElementNS(NS, "g");
    g.style.cursor = "pointer";
    const halo = document.createElementNS(NS, "circle");
    const body = document.createElementNS(NS, "circle");
    body.setAttribute("fill", TS_NODE_HEX[n] || "#888");
    body.setAttribute("stroke", "rgba(0,0,0,0.25)");
    body.setAttribute("stroke-width", "1");
    const spec = document.createElementNS(NS, "circle");
    spec.setAttribute("fill", "rgba(255,255,255,0.22)");
    const valTxt = document.createElementNS(NS, "text");
    valTxt.setAttribute("text-anchor", "middle");
    valTxt.setAttribute("dominant-baseline", "central");
    valTxt.setAttribute("font-family", "ui-monospace, 'SF Mono', Menlo, Consolas, monospace");
    valTxt.setAttribute("font-weight", "700");
    valTxt.setAttribute("fill", "#fff");
    valTxt.setAttribute("y", "1");
    const nameTxt = document.createElementNS(NS, "text");
    nameTxt.setAttribute("text-anchor", "middle");
    nameTxt.setAttribute("font-family", "ui-monospace, 'SF Mono', Menlo, Consolas, monospace");
    nameTxt.setAttribute("font-size", "10");
    nameTxt.setAttribute("font-weight", "600");
    nameTxt.setAttribute("fill", "#8ba0ae");
    nameTxt.textContent = n;
    g.append(halo, body, spec, valTxt, nameTxt);
    g.addEventListener("pointerenter", e => { constState.hover = n; showConstTip(n, e); });
    g.addEventListener("pointermove", e => { if (constState.hover === n) moveConstTip(e); });
    g.addEventListener("pointerleave", () => { constState.hover = null; document.getElementById("constTip").hidden = true; });
    nodesG.appendChild(g);
    nodeEls.push({ g, halo, body, spec, valTxt, nameTxt });
  });
  svg.appendChild(nodesG);
  constState.els = { edgeEls, nodeEls };
  resetConstNodes();
}
function resetConstNodes() {
  constState.nodes = NODES.map((_, i) => {
    const a = i * 2 * Math.PI / NN - Math.PI / 2;
    return { x: SVG_W / 2 + 150 * Math.cos(a), y: SVG_H / 2 + 150 * Math.sin(a), vx: 0, vy: 0 };
  });
  constState.animT = 0;
}
function showConstTip(n, e) {
  const s = soc(), k = Math.max(0, yidx());
  const i = NODES.indexOf(n);
  const sig = s.A[k][i] * s.C[k][i] * (s.K[k][i] - s.S_v[k][i]);
  const B = canon.bondMatrix(s.C[k], s.K[k], s.S_v[k], s.A[k]);
  const weak = NODES.map((nm, j) => ({ nm, j, b: B[i][j] })).filter(p => p.j !== i)
    .sort((a, b) => a.b - b.b).slice(0, 2);
  const tip = document.getElementById("constTip");
  tip.innerHTML = `<b style="color:${TS_NODE_HEX[n]}">${n}</b> — ${NODE_DESC[n].role}<br>` +
    `<span style="color:var(--faint)">${NODE_DESC[n].desc} · place ${i + 1} of 8 in the wheel.</span><br>` +
    `C ${s.C[k][i]} · K ${s.K[k][i]} · S ${s.S_v[k][i]} · A ${s.A[k][i]}<br>` +
    `V ${s.nodeV[k][i].toFixed(1)} · σ ${sig >= 0 ? "+" : ""}${sig.toFixed(0)} ${sig < 0 ? "(inverted)" : "(active)"}<br>` +
    `weakest couplings: ${weak.map(w => `${w.nm} (B ${w.b.toFixed(2)})`).join(" · ")}<br>` +
    `<span style="color:var(--faint)">B degrades with stress on either end — ${n} does not control its own bonds.</span>`;
  tip.hidden = false; moveConstTip(e);
}
function moveConstTip(e) {
  const tip = document.getElementById("constTip");
  const r = document.getElementById("const-svg").getBoundingClientRect();
  tip.style.left = Math.min(r.width - 230, e.clientX - r.left + 14) + "px";
  tip.style.top = Math.max(4, e.clientY - r.top - 60) + "px";
}

function constFrame() {
  const k = yidx();
  if (k < 0) { constState.raf = requestAnimationFrame(constFrame); return; }
  const s = soc();
  if (constState.society !== state.society) { constState.society = state.society; resetConstNodes(); }
  const C = s.C[k], K = s.K[k], S = s.S_v[k], A = s.A[k], V = s.nodeV[k];
  const B = canon.bondMatrix(C, K, S, A);
  const W = NODES.map((_, i) => NODES.map((__, j) => B[i][j]));
  const sigma = NODES.map((_, i) => (A[i] * C[i]) * (K[i] - S[i]));

  const Vabs = V.map(Math.abs);
  const vMin = Math.min(...Vabs), vMax = Math.max(...Vabs, 1);
  const sizeMap = Vabs.map(v => 18 + (vMax === vMin ? 0.5 : (v - vMin) / (vMax - vMin)) * 22);

  const allPairs = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) allPairs.push({ i, j, w: W[i][j] });
  allPairs.sort((a, b) => b.w - a.w);
  const edges = allPairs.slice(0, 12);
  const wThresh = (allPairs.map(p => p.w).sort((a, b) => b - a)[Math.min(11, allPairs.length - 1)]) || 0;

  const cx = SVG_W / 2, cy = SVG_H / 2;
  const Lmin = Math.min(SVG_W, SVG_H) * 0.18;
  const Lmax = Math.min(SVG_W, SVG_H) * 0.50;
  const nodes = constState.nodes;
  const t = constState.animT;

  for (let i = 0; i < 8; i++) {
    let fx = 0, fy = 0;
    for (let j = 0; j < 8; j++) {
      if (i === j) continue;
      const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
      const dist = Math.hypot(dx, dy) + 0.001;
      const ww = W[i][j];
      if (ww >= wThresh) {
        const L = Lmin + (Lmax - Lmin) * (1 - ww);
        const kk = 0.012 * (0.25 + ww);
        const f = kk * (dist - L);
        fx += (dx / dist) * f; fy += (dy / dist) * f;
      }
      const rep = 6500 / (dist * dist);
      fx -= (dx / dist) * rep; fy -= (dy / dist) * rep;
    }
    fx += (cx - nodes[i].x) * 0.0028;
    fy += (cy - nodes[i].y) * 0.0028;
    const stress = S[i];
    const amp = Math.max(0, stress - 5) * 0.06;
    const phase = i * 0.7;
    fx += Math.cos(t * 0.025 + phase) * amp;
    fy += Math.sin(t * 0.022 + phase * 1.3) * amp;
    if (sigma[i] < 0) {
      const ox = nodes[i].x - cx, oy = nodes[i].y - cy;
      const od = Math.hypot(ox, oy) + 0.001;
      fx += (ox / od) * 0.05; fy += (oy / od) * 0.05;
    }
    nodes[i].vx = (nodes[i].vx + fx) * 0.72;
    nodes[i].vy = (nodes[i].vy + fy) * 0.72;
    const vmag = Math.hypot(nodes[i].vx, nodes[i].vy);
    if (vmag > 1.4) { nodes[i].vx = nodes[i].vx / vmag * 1.4; nodes[i].vy = nodes[i].vy / vmag * 1.4; }
  }
  for (let i = 0; i < 8; i++) {
    nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy;
    const m = 50;
    if (nodes[i].x < m) { nodes[i].x = m; nodes[i].vx *= -0.4; }
    if (nodes[i].x > SVG_W - m) { nodes[i].x = SVG_W - m; nodes[i].vx *= -0.4; }
    if (nodes[i].y < m) { nodes[i].y = m; nodes[i].vy *= -0.4; }
    if (nodes[i].y > SVG_H - m) { nodes[i].y = SVG_H - m; nodes[i].vy *= -0.4; }
  }

  const { edgeEls, nodeEls } = constState.els;
  edges.forEach((e, k2) => {
    const el = edgeEls[k2]; if (!el) return;
    const na = nodes[e.i], nb = nodes[e.j];
    el.setAttribute("x1", na.x.toFixed(1)); el.setAttribute("y1", na.y.toFixed(1));
    el.setAttribute("x2", nb.x.toFixed(1)); el.setAttribute("y2", nb.y.toFixed(1));
    el.setAttribute("opacity", Math.max(0.10, e.w * 2.2).toFixed(2));
    el.setAttribute("stroke", k2 < 4 ? "#2f6f7d" : "#b8ac98");
    el.setAttribute("stroke-width", (0.5 + e.w * 4.0).toFixed(2));
  });
  for (let k2 = edges.length; k2 < 12; k2++) { const el = edgeEls[k2]; if (el) el.setAttribute("opacity", "0"); }

  nodeEls.forEach(({ g, halo, body, spec, valTxt, nameTxt }, i) => {
    g.setAttribute("transform", `translate(${nodes[i].x.toFixed(1)},${nodes[i].y.toFixed(1)})`);
    const sz = sizeMap[i];
    const sig = sigma[i];
    const stress = S[i];
    const pulse = 1 + Math.sin(t * 0.08 + i) * (stress > 6 ? 0.06 : 0.025);
    halo.setAttribute("r", ((sz + 7 + Math.min(18, Math.abs(sig) / 30)) * pulse).toFixed(1));
    halo.setAttribute("fill", sig >= 0 ? "rgba(0,184,158,0.32)" : "rgba(197,48,48,0.42)");
    body.setAttribute("r", sz.toFixed(1));
    spec.setAttribute("r", (sz * 0.42).toFixed(1));
    spec.setAttribute("cx", (-sz * 0.28).toFixed(1));
    spec.setAttribute("cy", (-sz * 0.28).toFixed(1));
    valTxt.textContent = (V[i] >= 0 ? "+" : "") + V[i].toFixed(1);
    valTxt.setAttribute("font-size", Math.max(9, sz * 0.45).toFixed(1));
    nameTxt.setAttribute("y", (-(sz + 8)).toFixed(1));
  });
  constState.animT++;
  constState.raf = requestAnimationFrame(constFrame);
}
function startConst() { if (!constState.raf) constState.raf = requestAnimationFrame(constFrame); }

/* ---------------- S/K timeline ---------------- */
function drawSkTimeline() {
  const c = document.getElementById("skTimeline");
  const { ctx, w, h } = ctx2d(c);
  ctx.clearRect(0, 0, w, h);
  const s = soc(), sks = skSeries(), yrs = s.years;
  const vmin = 0.3, vmax = 2.5;
  const X = i => 6 + (w - 12) * i / Math.max(1, yrs.length - 1);
  const Y = v => h - 6 - (Math.max(vmin, Math.min(vmax, v)) - vmin) / (vmax - vmin) * (h - 14);
  // bands
  ctx.fillStyle = "rgba(48,176,144,0.05)"; ctx.fillRect(0, Y(1), w, h);
  ctx.fillStyle = "rgba(208,48,48,0.06)"; ctx.fillRect(0, 0, w, Y(1));
  ctx.strokeStyle = "rgba(47,111,125,0.35)"; ctx.setLineDash([3, 5]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, Y(1)); ctx.lineTo(w, Y(1)); ctx.stroke(); ctx.setLineDash([]);
  // area
  ctx.beginPath();
  sks.forEach((v, i) => { i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); });
  ctx.lineTo(X(yrs.length - 1), h); ctx.lineTo(X(0), h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(210,60,50,0.30)");
  grad.addColorStop(0.5, "rgba(240,160,32,0.15)");
  grad.addColorStop(1, "rgba(0,180,160,0.05)");
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  sks.forEach((v, i) => { i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); });
  ctx.strokeStyle = "rgba(240,160,32,0.8)"; ctx.lineWidth = 1.5; ctx.stroke();
  const ki = yidx();
  if (ki >= 0) {
    ctx.beginPath(); ctx.arc(X(ki), Y(sks[ki]), 4, 0, 7);
    ctx.fillStyle = "#2f6f7d"; ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.65)"; ctx.font = "9px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.textAlign = "center"; ctx.fillText(yrs[ki], X(ki), h - 1);
  }
  ctx.fillStyle = "rgba(100,140,160,0.6)"; ctx.font = "8.5px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
  const step = Math.max(1, Math.floor(yrs.length / 8));
  for (let i = 0; i < yrs.length; i += step) ctx.fillText(yrs[i], X(i), h - 1);
}
function initSkScrub() {
  const c = document.getElementById("skTimeline");
  const scrub = e => {
    const s = soc(), r = c.getBoundingClientRect();
    const frac = (e.clientX - r.left - 6) / (r.width - 12);
    const i = Math.round(Math.max(0, Math.min(1, frac)) * (s.years.length - 1));
    setYear(s.years[i]);
  };
  let down = false;
  c.addEventListener("pointerdown", e => { down = true; c.setPointerCapture(e.pointerId); scrub(e); });
  c.addEventListener("pointermove", e => { if (down) scrub(e); });
  c.addEventListener("pointerup", () => down = false);
}

/* ---------------- system worries: the meta-system's own vitals ----------------
   Conditions of the whole web, attributable to no single node. Canonical operators
   only; lambda2/eta_floor references are percentiles of the panel's own values. */
const PANEL = (() => {
  const L = [], E = [];
  for (const n of SOC_NAMES) { const s = D.societies[n];
    for (const v of s.L) L.push(v);
    for (const v of s.E) E.push(v); }
  const pct = (arr, x) => arr.length ? arr.filter(v => v < x).length / arr.length : 0;
  return { pctL: v => pct(L, v), pctE: v => pct(E, v) };
})();
function renderSysVit() {
  const el = document.getElementById("sysvit");
  const k = yidx();
  if (!el) return;
  if (k < 0) { el.innerHTML = ""; return; }
  const s = soc();
  const sys = canon.system(s.C[k], s.K[k], s.S_v[k], s.A[k]);
  const d = basinInputs(k);
  const iMin = sys.V.indexOf(sys.V_min);
  const iMax = s.S_v[k].indexOf(Math.max(...s.S_v[k]));
  const sig = NODES.map((_, i) => s.A[k][i] * s.C[k][i] * (s.K[k][i] - s.S_v[k][i]));
  const inv = NODES.filter((_, i) => sig[i] < 0);

  const worries = [];
  if (d.sk > 1) worries.push(`aggregate stress exceeds aggregate capacity (sk ${d.sk.toFixed(2)} &gt; 1) — the web is consuming its reserves${d.sk > 1.4 ? ", already past the Praetorian threshold" : ""}.`);
  if (sys.V_min < 0) worries.push(`${NODES[iMin]} has fallen below zero (V ${sys.V_min.toFixed(1)}) — the lowest place in the web, and its deficit drains every bond it touches.`);
  else if (sys.V_min < 4) worries.push(`${NODES[iMin]} is in the critical zone (V ${sys.V_min.toFixed(1)}) — the web carries its weakest place at everyone else's expense.`);
  if (s.S_v[k][iMax] > 8) worries.push(`${NODES[iMax]} carries severe stress (S ${s.S_v[k][iMax].toFixed(1)}); stress enters the bond sum on both ends (B &prop; 2<sup>&minus;(S<sub>i</sub>+S<sub>j</sub>)/10</sup>), so its load is a deficit outside any one node's control.`);
  if (inv.length) worries.push(`${inv.join(", ")} ${inv.length > 1 ? "invert" : "inverts"} (&sigma; &lt; 0): activation fails wherever capacity no longer covers stress — a local shortfall that recolours the whole halo.`);
  if (PANEL.pctL(sys.lambda2) < 0.10) worries.push(`connectivity &lambda;&#8322; = ${sys.lambda2.toFixed(3)} sits in the lowest decile of the panel — the web hangs on a few strands.`);
  if (PANEL.pctE(sys.eta_floor) < 0.10) worries.push(`the memory floor &eta;<sub>floor</sub> = ${sys.eta_floor.toFixed(3)} is in the lowest decile of the panel — Lore and Archive coupling is thin against Hands stress.`);
  if (!worries.length) worries.push("no system-level flags this year — the web's worries are its ordinary ones.");
  const basin = classifyBasin(d);

  el.innerHTML = `
    <div class="sysvit-head"><h4>System worries</h4>
      <span class="sub">system-level epiphenomena of the eight nodes — owned by no single node; read downstream as the weather of deliberation</span></div>
    <div class="sysvit-metrics">
      ${[["V&#772;", sys.V_mean.toFixed(2)], ["sigma V", sys.V_std.toFixed(2)], ["B_eff", sys.B_eff.toFixed(3)],
         ["lambda2", sys.lambda2.toFixed(3)], ["eta floor", sys.eta_floor.toFixed(3)], ["sk", d.sk.toFixed(2)]]
        .map(([kk, vv]) => `<div class="metric"><div class="k">${kk}</div><div class="v">${vv}</div></div>`).join("")}
    </div>
    <ul class="worry-list">${worries.map(w => `<li>${w}</li>`).join("")}</ul>
    <div class="downstream"><h5>Downstream — deliberation or reaction</h5><p>${DELIB[basin]}</p></div>`;
}

/* ---------------- dossier ---------------- */
function renderDossier() {
  const k = yidx();
  const body = document.getElementById("dossierBody");
  if (k < 0) { body.innerHTML = ""; return; }
  const s = soc();
  const d = basinInputs(k);
  const basin = classifyBasin(d);
  const bInfo = BASINS[basin];
  const threat = threatLevel(d);
  const V = s.nodeV[k];
  const sorted = NODES.slice().sort((a, b) => V[NODES.indexOf(b)] - V[NODES.indexOf(a)]);
  const topNode = sorted[0];
  const hollow = NODES.filter((n, i) => s.A[k][i] > s.K[k][i] * 1.3);
  const shadow = sorted[1];
  const nh = nominalHolder(state.society, state.year);
  const praetorianGap = d.praetorian > 0.25
    ? `Shield overruns Helm by ${(d.praetorian * 100).toFixed(0)} index points — real coordination lives in the security apparatus, not the executive chamber.`
    : d.praetorian < -0.15
    ? "Helm is significantly stronger than Shield — executive authority is genuine, not performed."
    : "Helm and Shield are near-balanced — the executive retains meaningful authority alongside the security apparatus.";
  // trajectory text (prior two sampled periods)
  let traj = "Insufficient prior data for trajectory analysis.";
  if (k >= 2) {
    const sk2 = basinInputs(k - 2).sk, b2 = basinInputs(k - 2).b;
    const skDelta = d.sk - sk2, bDelta = d.b - b2;
    const dir = skDelta > 0.1 ? "deteriorating" : skDelta < -0.1 ? "improving" : "stable";
    const bDir = bDelta < -0.03 ? "contracting" : bDelta > 0.03 ? "expanding" : "stable";
    traj = `The S/K trajectory is <span class="em">${dir}</span> (Δ${skDelta >= 0 ? "+" : ""}${skDelta.toFixed(2)} over prior two periods). Bond coherence is <span class="em">${bDir}</span>. `;
    if (dir === "deteriorating" && d.sk > 1.2) traj += `At current rate, the Praetorian threshold (sk > 1.4) is ${d.sk > 1.25 ? "imminent" : "approaching"}. `;
    if (dir === "improving" && d.sk < 1.0) traj += "Recovery trajectory is consistent with a Navigational basin within 1–2 periods if bond coherence holds.";
    if (basin === "PRAETORIAN") traj += `The Shield node is at ${V[NODES.indexOf("Shield")].toFixed(1)} against Helm at ${V[NODES.indexOf("Helm")].toFixed(1)}. The praetorian configuration is self-reinforcing absent external shock or deliberate slow-loop investment.`;
    if (basin === "PHANTOM") traj += "Archive and Lore node values remain high despite system-wide deterioration — the Phantom signature. Cultural elaboration is compensating for material decline.";
  }
  document.getElementById("dossierSub").textContent =
    `${state.society} · ${state.year} · basin indices: sk ${d.sk.toFixed(2)} · b ${d.b.toFixed(3)} · praetorian ${d.praetorian.toFixed(2)} · cog_gap ${d.cog_gap.toFixed(2)}`;
  body.innerHTML = `
    <div class="basin-row" style="margin:0">
      <span class="basin-chip" style="color:${bInfo.col};border-color:${bInfo.col}55">${bInfo.name}</span>
      <span class="threat-chip" style="color:${threat.col};border-color:${threat.col}55">THREAT ${threat.level}</span>
      <span class="basin-meta">${bInfo.sub}</span>
    </div>
    <div class="block"><h4>Basin reading</h4><p>${bInfo.meaning}</p></div>
    <div class="block"><h4>Power topology</h4>
      <p class="powerline">Nominal: ${nh.role} · ${nh.power}</p>
      <p class="powerline">Real coordination: <span class="em">${topNode}</span> (shadow: ${shadow})</p>
      <p class="powerline">${praetorianGap}</p>
      ${hollow.length ? `<p class="hline">Hollow nodes (A &gt; 1.3×K): <span class="hollow">${hollow.join(", ")}</span> — abstraction outruns capacity.</p>` : ""}
    </div>
    <div class="block"><h4>Zeitgeist — the surface feature of the eight attractors</h4><p>${ZEIT[basin]}</p></div>
    <div class="block"><h4>Survival note</h4><p>${SURV[basin].replace("{top}", topNode)}</p></div>
    <div class="block"><h4>Trajectory</h4><p>${traj}</p></div>`;
}

/* ---------------- URL state ---------------- */
function updateURL() {
  const p = new URLSearchParams();
  p.set("soc", state.society); p.set("yr", state.year);
  if (!document.getElementById("panel-bench").hidden) p.set("bench", "1");
  history.replaceState(null, "", "?" + p.toString());
}
