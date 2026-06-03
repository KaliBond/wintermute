/* ──────────────────────────────────────────────────────────────────────────
   Social Cognition in Real Time — eight sub-brains
   Each CAMS node is an exteriorised cognitive function rendered as its own
   sub-brain (a small neural cluster). It fires on its live ESCH activation
   σ_i(t) = A_i · C_i · (K_i − S_i).  The eight sub-brains are wired into one
   distributed mind; inter-brain links brighten where two nodes co-fire.
   Play the timeline and watch the society think, desynchronise, go dark.
   Aggregate of this field is the Zeitgeist Index Z(t)=Ā·C̄·(K̄−S̄).
   ────────────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  const SOCIETIES = {
    Germany:   { csv: "data/germany_cams5_ensemble_mean.csv" },
    Argentina: { csv: "data/argentina_cams5_ensemble_1950_2026.csv" },
    Australia: { csv: "data/nations/Australia_ENS.csv" },
    Canada:    { csv: "data/nations/Canada_ENS.csv" },
    Chile:     { csv: "data/nations/Chile_ENS.csv" },
    China:     { csv: "data/nations/China_ENS.csv" },
    Colombia:  { csv: "data/nations/Colombia_ENS.csv" },
    France:    { csv: "data/nations/France_ENS.csv" },
    India:     { csv: "data/nations/India_ENS.csv" },
    Iran:      { csv: "data/nations/Iran_ENS.csv" },
    Japan:     { csv: "data/nations/Japan_ENS.csv" },
    Norway:    { csv: "data/nations/Norway_ENS.csv" },
    Poland:    { csv: "data/nations/Poland_ENS.csv" },
    Russia:    { csv: "data/nations/Russia_ENS.csv" },
    Sweden:    { csv: "data/nations/Sweden_ENS.csv" },
    USA:       { csv: "data/nations/USA_ENS.csv" },
  };
  const ORDER = ["Helm", "Shield", "Flow", "Hands", "Craft", "Stewards", "Lore", "Archive"];
  const IDENT = {
    Helm: "#e05a4d", Shield: "#e8806f", Flow: "#e6b422", Hands: "#9aa3a6", Craft: "#3fc7a8",
    Stewards: "#e8923c", Lore: "#b07fce", Archive: "#5aa6e0",
  };
  const FUNC = { Helm: "executive attention", Shield: "threat monitoring", Flow: "circulation", Hands: "execution", Craft: "skilled production", Stewards: "resource governance", Lore: "meaning-making", Archive: "memory" };

  let data = {}, society = "Germany", years = [], idx = 0, playing = false, dispYear = null, lastT = 0;
  const STEP_MS = 640;
  const brains = {};   // node -> {pts:[{x,y}], edges:[[i,j,len]]}

  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const sigmaOf = (m) => m.A * m.C * (m.K - m.S);
  const Vof = (m) => m.C + m.K - m.S + m.A / 2;

  // deterministic RNG so each sub-brain has a stable shape
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function buildBrain(seed) {
    const r = rng(seed), N = 7, pts = [];
    for (let i = 0; i < N; i++) {
      // two-lobe blob in unit coords
      const lobe = i % 2 ? 0.42 : -0.42;
      pts.push({ x: lobe + (r() - 0.5) * 0.7, y: (r() - 0.5) * 1.3, ph: r() * 6.28 });
    }
    const edges = [];
    for (let i = 0; i < N; i++) {
      const d = pts.map((p, j) => ({ j, l: Math.hypot(p.x - pts[i].x, p.y - pts[i].y) })).filter((o) => o.j !== i).sort((a, b) => a.l - b.l);
      for (let k = 0; k < 2; k++) { const j = d[k].j; if (!edges.some((e) => (e[0] === j && e[1] === i))) edges.push([i, j, d[k].l]); }
    }
    return { pts, edges };
  }
  ORDER.forEach((n, k) => { brains[n] = buildBrain(k * 1013 + 7); });

  function parse(text) {
    const L = text.trim().split(/\r?\n/), h = L[0].split(",");
    const iY = h.indexOf("Year"), iN = h.indexOf("Node"),
      iC = h.indexOf("Coherence"), iK = h.indexOf("Capacity"), iS = h.indexOf("Stress"), iA = h.indexOf("Abstraction");
    const byYear = {};
    for (let r = 1; r < L.length; r++) {
      const c = L[r].split(","); if (c.length < 5) continue;
      const y = +c[iY], n = c[iN].trim();
      (byYear[y] = byYear[y] || {})[n] = { C: +c[iC], K: +c[iK], S: +c[iS], A: +c[iA] };
    }
    const yrs = Object.keys(byYear).map(Number).sort((a, b) => a - b).filter((y) => ORDER.every((n) => byYear[y][n]));
    const allAbs = []; yrs.forEach((y) => ORDER.forEach((n) => allAbs.push(Math.abs(sigmaOf(byYear[y][n])))));
    allAbs.sort((a, b) => a - b);
    const maxAbs = Math.max(allAbs[Math.floor(allAbs.length * 0.80)] || 1, 1); // robust reference scale (80th pct)
    return { years: yrs, byYear, maxAbs };
  }
  function loadSociety(name) {
    if (data[name]) return afterLoad(name);
    fetch(SOCIETIES[name].csv).then((r) => r.text()).then((t) => { data[name] = parse(t); afterLoad(name); })
      .catch((e) => { const el = document.getElementById("sc-fallback"); if (el) { el.style.display = "block"; el.textContent = "Could not load " + name + ": " + e.message; } });
  }
  function afterLoad(name) {
    society = name;                       // only switch once data is in hand
    years = data[society].years;
    const saved = +localStorage.getItem("sc-year-" + society);
    idx = years.indexOf(saved); if (idx < 0) idx = years.indexOf(SOCIETIES[society].open); if (idx < 0) idx = 0;
    dispYear = years[idx];
    const sl = document.getElementById("sc-slider"); sl.min = 0; sl.max = years.length - 1; sl.step = 1; sl.value = idx;
    document.querySelectorAll("[data-sc-soc]").forEach((b) => b.classList.toggle("on", b.dataset.scSoc === society));
    const fb = document.getElementById("sc-fallback"); if (fb) { fb.style.display = "none"; fb.textContent = ""; }
    renderReadout();
    try { draw(performance.now()); } catch (e) { surfaceErr(e); }
  }
  function surfaceErr(e) {
    const el = document.getElementById("sc-fallback");
    if (el) { el.style.display = "block"; el.textContent = "draw error: " + (e && e.message ? e.message : e) + " @ " + (e && e.stack ? e.stack.split('\n')[1] : ''); }
    console.error("sc draw error", e);
  }

  function nodeStateAt(yf) {
    const i0 = Math.max(0, years.findIndex((y) => y >= Math.floor(yf)));
    const a = years[i0], b = years[Math.min(years.length - 1, i0 + 1)];
    const f = b === a ? 0 : Math.max(0, Math.min(1, (yf - a) / (b - a)));
    const out = {};
    ORDER.forEach((n) => { const ma = data[society].byYear[a][n], mb = data[society].byYear[b][n];
      out[n] = { C: ma.C + (mb.C - ma.C) * f, K: ma.K + (mb.K - ma.K) * f, S: ma.S + (mb.S - ma.S) * f, A: ma.A + (mb.A - ma.A) * f }; });
    return out;
  }
  function sigColor(t, alpha) {
    t = Math.max(-1, Math.min(1, t)); let r, g, b;
    if (t >= 0) { const u = Math.pow(t, 0.7); r = 19 + 131 * u; g = 32 + 188 * u; b = 42 + 173 * u; }
    else { const u = Math.pow(-t, 0.7); r = 19 + 205 * u; g = 32 + 38 * u; b = 42 + 10 * u; }
    return `rgba(${r | 0},${g | 0},${b | 0},${alpha})`;
  }

  // ── render ────────────────────────────────────────────────────────────────
  function draw(now) {
    const cv = document.getElementById("sc-field");
    if (!cv || !years.length || !data[society]) return;
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.345;
    const st = nodeStateAt(dispYear), maxAbs = data[society].maxAbs;
    const pos = {}, sig = {}, tt = {};
    ORDER.forEach((n, k) => { const ang = (-90 + k * 45) * Math.PI / 180; pos[n] = { x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) }; sig[n] = sigmaOf(st[n]); tt[n] = sig[n] / maxAbs; });

    // loop-side wash: cool slow-loop (left), warm fast-loop (right)
    const lg = g.createLinearGradient(cx - R, 0, cx + R, 0);
    lg.addColorStop(0, "rgba(90,166,224,0.045)"); lg.addColorStop(0.5, "rgba(0,0,0,0)"); lg.addColorStop(1, "rgba(224,90,77,0.045)");
    g.fillStyle = lg; g.beginPath(); g.arc(cx, cy, R + 70, 0, 7); g.fill();

    // inter-brain co-firing links
    for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
      const a = ORDER[i], b = ORDER[j];
      if (tt[a] > 0.08 && tt[b] > 0.08) {
        const w = Math.min(tt[a], tt[b]);
        const pulse = 0.5 + 0.5 * Math.sin(now / 600 + i + j);
        g.strokeStyle = `rgba(111,179,192,${(w * 0.42 * (0.6 + 0.4 * pulse)).toFixed(3)})`;
        g.lineWidth = 0.5 + w * 2.6;
        g.beginPath(); g.moveTo(pos[a].x, pos[a].y); g.lineTo(pos[b].x, pos[b].y); g.stroke();
      }
    }

    // sub-brains
    ORDER.forEach((n) => {
      const t = tt[n], mag = Math.min(1, Math.abs(t)), p = pos[n];
      const br = 20 + mag * 22;            // sub-brain radius
      const B = brains[n];
      // halo
      const grd = g.createRadialGradient(p.x, p.y, 2, p.x, p.y, br * 2.5);
      grd.addColorStop(0, sigColor(t, 0.45 + 0.4 * mag)); grd.addColorStop(0.45, sigColor(t, 0.16)); grd.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = grd; g.beginPath(); g.arc(p.x, p.y, br * 2.5, 0, 7); g.fill();
      // membrane: dark base + σ-tint so the body glows by activation
      g.fillStyle = "rgba(10,16,22,0.6)"; g.beginPath(); g.ellipse(p.x, p.y, br * 1.18, br * 1.32, 0, 0, 7); g.fill();
      g.fillStyle = sigColor(t, 0.12 + 0.5 * mag); g.beginPath(); g.ellipse(p.x, p.y, br * 1.18, br * 1.32, 0, 0, 7); g.fill();
      g.strokeStyle = IDENT[n]; g.globalAlpha = 0.85; g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(p.x, p.y, br * 1.18, br * 1.32, 0, 0, 7); g.stroke(); g.globalAlpha = 1;
      // internal edges
      const P = B.pts.map((q) => ({ x: p.x + q.x * br, y: p.y + q.y * br, ph: q.ph }));
      g.lineWidth = 1;
      B.edges.forEach(([i, j]) => {
        const a = P[i], b = P[j];
        const fire = mag * (0.5 + 0.5 * Math.sin(now / 320 + a.ph + b.ph));
        g.strokeStyle = sigColor(t, 0.18 + 0.5 * fire * mag);
        g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
      });
      // internal neurons
      P.forEach((q) => {
        const flick = 0.55 + 0.45 * Math.sin(now / 300 + q.ph);
        const a = mag < 0.1 ? 0.28 : 0.45 + 0.55 * mag * flick;
        const rr = 1.6 + mag * 2.4 * (0.7 + 0.3 * flick);
        g.fillStyle = sigColor(t, a); g.beginPath(); g.arc(q.x, q.y, rr, 0, 7); g.fill();
      });
      // "dark — alive but not thinking"
      if (mag < 0.1) {
        g.strokeStyle = "rgba(218,210,184,0.4)"; g.setLineDash([2, 3]); g.lineWidth = 1;
        g.beginPath(); g.ellipse(p.x, p.y, br * 1.4, br * 1.55, 0, 0, 7); g.stroke(); g.setLineDash([]);
      }
      // label
      const out = p.x < cx - 4 ? "right" : (p.x > cx + 4 ? "left" : "center");
      const ly = p.y < cy - 4 ? p.y - br * 1.5 - 8 : (p.y > cy + 4 ? p.y + br * 1.5 + 14 : p.y);
      const lx = out === "center" ? p.x : (out === "right" ? p.x - br * 1.3 - 8 : p.x + br * 1.3 + 8);
      g.fillStyle = "#EDE7D4"; g.font = "600 12px -apple-system, sans-serif"; g.textAlign = out; g.textBaseline = "middle";
      g.fillText(n, p.x, ly);
    });

    // unified-mind centre
    const sbar = mean(ORDER.map((n) => sig[n])), tb = sbar / maxAbs;
    const cpulse = 0.85 + 0.15 * Math.sin(now / 500);
    g.fillStyle = sigColor(tb, 0.10); g.beginPath(); g.arc(cx, cy, (24 + Math.abs(tb) * 20) * cpulse, 0, 7); g.fill();
    g.fillStyle = "#EDE7D4"; g.font = "700 13px ui-monospace, monospace"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(Math.round(dispYear), cx, cy - 6);
    g.fillStyle = "#6FB3C0"; g.font = "600 9px ui-monospace, monospace"; g.fillText("σ̄ " + sbar.toFixed(0), cx, cy + 9);
  }

  function renderReadout() {
    const y = years[idx]; localStorage.setItem("sc-year-" + society, y);
    const st = data[society].byYear[y], maxAbs = data[society].maxAbs;
    const sigs = ORDER.map((n) => ({ n, s: sigmaOf(st[n]), v: Vof(st[n]) }));
    const sbar = mean(sigs.map((x) => x.s)), dark = sigs.filter((x) => Math.abs(x.s) / maxAbs < 0.1).length, vbar = mean(sigs.map((x) => x.v));
    document.getElementById("sc-year-label").textContent = y;
    document.getElementById("sc-sigbar").textContent = sbar.toFixed(0);
    document.getElementById("sc-vbar").textContent = vbar.toFixed(1);
    document.getElementById("sc-dark").textContent = dark + " / 8";
    const norm = Math.max(...sigs.map((x) => Math.abs(x.s)), 1);
    document.getElementById("sc-bars").innerHTML = [...sigs].sort((a, b) => b.s - a.s).map((x) => {
      const pct = Math.abs(x.s) / norm * 100, pos = x.s >= 0;
      return `<div class="sc-bar-row"><span class="sc-bar-name"><i style="background:${IDENT[x.n]}"></i>${x.n}<em>${FUNC[x.n]}</em></span>
        <span class="sc-bar-track"><span class="sc-bar-fill ${pos ? 'pos' : 'neg'}" style="width:${pct.toFixed(0)}%"></span></span>
        <span class="sc-bar-val ${pos ? '' : 'neg'}">${x.s >= 0 ? '+' : ''}${x.s.toFixed(0)}</span></div>`;
    }).join("");
    const AC = mean(ORDER.map((n) => st[n].A * st[n].C)), KS = mean(ORDER.map((n) => st[n].K - st[n].S));
    let state, note, col;
    if (KS >= 0 && sbar > 0.35 * maxAbs) { state = "Vital thought"; note = "coherent ideas backed by free energy — the mind is working"; col = "#2F6F7D"; }
    else if (KS < 0 && AC > 30) { state = "Brilliant but fragile"; note = "high abstraction over an eroding substrate — peak culture, no headroom"; col = "#d97706"; }
    else if (KS < 0) { state = "Decoupled & anxious"; note = "stress exceeds capacity — thrashing, entropy rising"; col = "#c0392b"; }
    else { state = "Nascent / dim"; note = "capacity present, coherence thin — alive but not yet thinking"; col = "#6B7480"; }
    const b = document.getElementById("sc-state"); b.textContent = state; b.style.background = col;
    document.getElementById("sc-state-note").textContent = note;
  }

  function frame(now) {
    if (playing) {
      if (!lastT) lastT = now;
      dispYear += ((now - lastT) / STEP_MS) * (years[1] - years[0]); lastT = now;
      let ni = years.findIndex((y) => y >= dispYear); if (ni < 0) ni = years.length - 1; if (years[ni] > dispYear && ni > 0) ni--;
      if (ni !== idx) { idx = ni; document.getElementById("sc-slider").value = idx; renderReadout(); }
      if (dispYear >= years[years.length - 1]) { dispYear = years[years.length - 1]; setPlaying(false); }
    }
    try { draw(now || 0); } catch (e) { surfaceErr(e); }
    requestAnimationFrame(frame);
  }
  function setPlaying(p) {
    playing = p; lastT = 0;
    const btn = document.getElementById("sc-play"); if (btn) btn.innerHTML = p ? "❚❚&nbsp; Pause" : "▶&nbsp; Play timeline";
    if (p && idx >= years.length - 1) { idx = 0; dispYear = years[0]; }
  }

  function boot() {
    document.getElementById("sc-slider").addEventListener("input", (e) => { setPlaying(false); idx = +e.target.value; dispYear = years[idx]; renderReadout(); try { draw(performance.now()); } catch (err) { surfaceErr(err); } });
    document.getElementById("sc-play").addEventListener("click", () => setPlaying(!playing));
    document.querySelectorAll("[data-sc-soc]").forEach((b) => b.addEventListener("click", () => { setPlaying(false); loadSociety(b.dataset.scSoc); }));
    loadSociety("Germany");
    requestAnimationFrame(frame);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
