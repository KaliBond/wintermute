/* CAMS Nations Lens v1.0 — single-year fortune instrument.
   Window 1900–2010. Eight paladins × four vitals (C, K, A, S) = one block of
   8 rows × 4 numbers. Z = (A·C)·(K−S) is computed here, never by the model. */
"use strict";

const NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"];
const VITALS = ["C", "K", "A", "S"];
const YEAR_MIN = 1900, YEAR_MAX = 2010;

const NATIONS = [
  "Afghanistan","Algeria","Argentina","Australia","Austria-Hungary / Austria","Belgium","Brazil",
  "Canada","Chile","China","Colombia","Congo (DRC)","Cuba","Czechoslovakia / Czechia","Denmark",
  "Egypt","Ethiopia","Finland","France","Germany","Ghana","Greece","Hungary","Iceland","India",
  "Indonesia","Iran (Persia)","Iraq","Ireland","Israel","Italy","Japan","Kenya","Korea / South Korea",
  "Lebanon","Malaysia","Mexico","Mongolia","Morocco","Myanmar","Nepal","Netherlands","New Zealand",
  "Nigeria","North Korea","Norway","Ottoman Empire / Türkiye","Pakistan","Peru","Philippines","Poland",
  "Portugal","Russia / USSR","Saudi Arabia","Senegal","South Africa","Spain","Sri Lanka","Sweden",
  "Switzerland","Syria","Tanzania","Thailand","Ukraine","United Kingdom","United States","Venezuela",
  "Vietnam","Yugoslavia / Serbia","Zambia","Zimbabwe"
];

const NODE_LINES = {
  Helm:     "the executive core — direction, decision, command",
  Shield:   "protection — defence, policing, courts, borders",
  Lore:     "the knowledge keepers — schools, universities, science, press",
  Stewards: "administration — the civil machinery: tax, census, registry",
  Craft:    "making — industry, infrastructure, technique",
  Hands:    "the working population — labour, skill, everyday wellbeing",
  Archive:  "memory and record — statistics, heritage, collective self-knowledge",
  Flow:     "exchange — trade, money, movement of goods, people, ideas"
};

function buildPrompt(nation, year, session) {
  return `CAMS NATIONS LENS v1.0  |  NATION: ${nation}  |  YEAR: ${year}  |  SESSION: ${session}

STEP 0 - SESSION CHECK
If this conversation contains ANY earlier messages from me or from you, reply with
the single word CONTAMINATED and nothing else. Otherwise continue.

You are reading one nation in one year under the CAMS lens. Score the eight
paladins below on the four vitals, for ${nation} in the year ${year}.

RULES
1. Score ${year} as it was, relative to other nations of the SAME DECADE.
   1 = the lowest level of any nation that decade. 10 = the highest.
2. Use only what was true up to the end of ${year}. Ignore everything after it.
   The READ must speak of ${year}, never beyond it. No prophecy.
3. The four vitals, scored for EACH paladin separately:
   C Coherence  - how far the node's parts pull together
   K Capacity   - the means available to it
   A Abstraction - its reach of abstract machinery (writing, law, money, standard measures)
   S Stress     - pressure on it this year. HIGH = MORE PRESSURE, not more failure.
4. EVIDENCE GATE - the most important rule.
   Every row carries one note of 15 words or fewer naming a specific DATED fact
   the row rests on (an event, law, census, work, figure - with its year), and
   one tier: A = direct record of the period, B = established scholarship,
   C = inference from the named fact. If you cannot name a specific fact, every
   score in the row is NA and the tier is NA. A vague impression is NOT evidence.
5. Score each paladin independently. Judge structure and function, not virtue.
   A nation's fame or present reputation is not evidence.
6. If the nation's present borders did not hold in ${year}, score the polity that
   held the territory, and say so in the note.

THE EIGHT PALADINS
${NODES.map((n) => ` ${n}: ${NODE_LINES[n]}`).join("\n")}

OUTPUT
First, one CSV code block and nothing before it, with this header once and rows
in exactly this order:
node,C,K,A,S,tier,note
${NODES.join("\n")}
(scores are integers 1-10 or NA; tier is A, B, C, or NA; notes have no commas)

Then on their own lines:
MOTIFS
3 to 5 lines, each: motif name - one dated fact of ${year} behind it
READ
120 words or fewer. The spirit of ${year}: declarative, concrete, no hedging,
no prophecy, tying the motifs to the numbers.
Last line, alone: SESSION ${session}`;
}

/* ---------- RFC-ish CSV (mirrors the observatory parser) ---------- */
function parseCsvGrid(input) {
  const s = input.replace(/^﻿/, "");
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let i = 0; i < s.length; i += 1) {
    const c = s[i];
    if (inQuotes) {
      if (c === '"') {
        if (s[i + 1] === '"') { field += '"'; i += 1; continue; }
        inQuotes = false; continue;
      }
      field += c; continue;
    }
    if (c === '"') { inQuotes = true; continue; }
    if (c === ",") { row.push(field); field = ""; continue; }
    if (c === "\n" || c === "\r") {
      if (c === "\r" && s[i + 1] === "\n") i += 1;
      if (row.length > 0 || field.length > 0) { row.push(field); rows.push(row); }
      field = ""; row = []; continue;
    }
    field += c;
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

const wordCount = (t) => (t.trim() ? t.trim().split(/\s+/).length : 0);
const HEADER = ["node", "C", "K", "A", "S", "tier", "note"];

function validateReply(raw, session) {
  const problems = [], warnings = [];
  if (raw.trim().toUpperCase() === "CONTAMINATED")
    return { ok: false, problems: ["Model reported CONTAMINATED; start a fresh chat"], warnings };

  const fence = /```(?:csv)?\s*\n([\s\S]*?)\n```/i.exec(raw);
  if (!fence) return { ok: false, problems: ["No CSV code block found"], warnings };

  const grid = parseCsvGrid((fence[1] || "").trim());
  const head = grid[0] || [];
  if (head.length !== HEADER.length || !HEADER.every((h, i) => head[i] === h))
    problems.push("CSV header must be exactly: node,C,K,A,S,tier,note");
  const data = grid.slice(1).filter((r) => r.length > 0);
  if (data.length !== 8) problems.push(`Expected 8 rows (one per paladin), received ${data.length}`);

  const rows = [];
  data.forEach((cells, i) => {
    const line = i + 1;
    if (cells.length !== 7) { problems.push(`Row ${line}: expected 7 fields`); return; }
    const [node, C, K, A, S, tier, note] = cells.map((x) => (x ?? "").trim());
    if (node !== NODES[i]) problems.push(`Row ${line}: expected node ${NODES[i]}, received "${node}"`);
    const scores = { C, K, A, S };
    const allNA = VITALS.every((v) => scores[v] === "NA");
    const anyNA = VITALS.some((v) => scores[v] === "NA");
    if (anyNA && !allNA) problems.push(`Row ${line} (${node}): NA must cover all four vitals or none`);
    if (!allNA) {
      for (const v of VITALS) {
        const s = scores[v];
        if (!/^\d+$/.test(s) || +s < 1 || +s > 10) problems.push(`Row ${line} (${node}): ${v} must be 1-10 or NA`);
      }
      if (!["A", "B", "C"].includes(tier)) problems.push(`Row ${line} (${node}): tier must be A, B or C`);
      if (!note) problems.push(`Row ${line} (${node}): evidence note missing`);
    } else if (tier !== "NA") {
      problems.push(`Row ${line} (${node}): NA scores require tier NA`);
    }
    if (wordCount(note) > 15) problems.push(`Row ${line} (${node}): note longer than 15 words`);
    if (note && !/(1[0-9]{3}|20[0-1][0-9])/.test(note))
      warnings.push(`Row ${line} (${node}): note names no year - the gate asks for dated facts`);
    rows.push({ node, scores, tier, note });
  });

  if (!new RegExp(`^SESSION\\s+${session}\\s*$`, "m").test(raw))
    problems.push(`Expected SESSION ${session} on its own line`);

  const mMot = /(?:^|\n)MOTIFS\s*\n([\s\S]*?)(?=\nREAD\s*\n|$)/i.exec(raw);
  const motifLines = mMot ? mMot[1].split("\n").map((l) => l.trim()).filter(Boolean) : [];
  if (motifLines.length < 3 || motifLines.length > 5)
    problems.push(`Expected 3-5 motif lines, found ${motifLines.length}`);
  const motifs = motifLines.map((l) => {
    const parts = l.split(/\s+[-–—]\s+/);
    return { name: (parts[0] || l).trim(), fact: (parts.slice(1).join(" — ") || "").trim() };
  });
  motifs.forEach((m, i) => {
    if (!m.fact) warnings.push(`Motif ${i + 1}: no dated fact after the dash`);
    else if (!/(1[0-9]{3}|20[0-1][0-9])/.test(m.fact)) warnings.push(`Motif ${i + 1}: fact carries no year`);
  });

  const mRead = /(?:^|\n)READ\s*\n([\s\S]*?)(?=\n\s*SESSION|$)/i.exec(raw);
  const read = mRead ? mRead[1].trim() : "";
  if (!read) problems.push("READ section missing");
  else if (wordCount(read) > 120) problems.push(`READ is ${wordCount(read)} words; the cap is 120`);

  return { ok: problems.length === 0, rows, motifs, read, problems, warnings };
}

/* Z per paladin: (A·C)·(K−S). Void if any vital is NA. Range ±900. */
function zOf(scores) {
  if (VITALS.some((v) => scores[v] === "NA")) return null;
  const [c, k, a, s] = VITALS.map((v) => +scores[v]);
  return a * c * (k - s);
}

/* ---------- radar: four vitals as concentric octagons ---------- */
function radarSVG(rows) {
  const size = 300, cx = size / 2, cy = size / 2 - 6, R = 100;
  const pt = (i, v) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 8;
    const r = (v / 10) * R;
    return [cx + r * Math.cos(ang), cy + r * Math.sin(ang)];
  };
  const poly = (vals) =>
    vals.map((v, i) => (v == null ? [cx, cy] : pt(i, v)).map((n) => n.toFixed(1)).join(",")).join(" ");
  let rings = "";
  for (let g = 2; g <= 10; g += 2) {
    rings += `<polygon points="${NODES.map((_, i) => pt(i, g).map((n) => n.toFixed(1)).join(",")).join(" ")}"
      fill="none" stroke="rgba(23,28,22,.14)" stroke-width="1" ${g === 10 ? 'stroke-dasharray="none"' : 'stroke-dasharray="3 4"'}/>`;
  }
  const spokes = NODES.map((_, i) => {
    const [x, y] = pt(i, 10);
    return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(23,28,22,.14)"/>`;
  }).join("");
  const labels = NODES.map((n, i) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 8;
    const x = cx + (R + 22) * Math.cos(ang), y = cy + (R + 22) * Math.sin(ang);
    const anchor = Math.abs(Math.cos(ang)) < 0.4 ? "middle" : Math.cos(ang) > 0 ? "start" : "end";
    return `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="${anchor}"
      font-family="IBM Plex Mono,monospace" font-size="10.5" fill="#2c332b" letter-spacing=".04em">${n}</text>`;
  }).join("");
  const fills = { C: "rgba(23,28,22,.16)", K: "none", A: "none", S: "rgba(201,44,80,.10)" };
  const strokes = { C: "#171c16", K: "#55604f", A: "#8a7a4a", S: "#c92c50" };
  const dash = { C: "none", K: "6 3", A: "2 3", S: "none" };
  const polys = VITALS.map((v) => {
    const vals = rows.map((r) => (r.scores[v] === "NA" ? null : +r.scores[v]));
    return `<polygon points="${poly(vals)}" fill="${fills[v]}" stroke="${strokes[v]}"
      stroke-width="${v === "C" || v === "S" ? 2 : 1.4}" stroke-dasharray="${dash[v]}" stroke-linejoin="round"/>`;
  }).join("");
  const legend = VITALS.map((v, i) => `
    <line x1="${40 + i * 60}" y1="318" x2="${54 + i * 60}" y2="318"
      stroke="${strokes[v]}" stroke-width="2" stroke-dasharray="${dash[v]}"/>
    <text x="${58 + i * 60}" y="322" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="#2c332b">${v}</text>`).join("");
  return `<svg viewBox="-46 -6 392 332" width="100%" role="img" aria-label="Eight paladins, four vitals">${rings}${spokes}${polys}${labels}${legend}</svg>`;
}

/* ---------- jar (this device only) ---------- */
const JAR_KEY = "cams-fortune-jar-v1";
const loadJar = () => {
  try { return JSON.parse(localStorage.getItem(JAR_KEY)) || []; } catch { return []; }
};
const saveJar = (j) => localStorage.setItem(JAR_KEY, JSON.stringify(j.slice(0, 60)));

function meanZ(rows) {
  const zs = rows.map((r) => zOf(r.scores)).filter((z) => z != null);
  return zs.length ? zs.reduce((a, b) => a + b, 0) / zs.length : null;
}
const fmtZ = (z) => (z == null ? "void" : (z > 0 ? "+" : "") + Math.round(z));

function ribbonSVG(entries) {
  const w = 640, h = 120, pad = 30;
  const pts = entries.filter((e) => e.zMean != null).sort((a, b) => a.year - b.year);
  if (pts.length < 2) return "";
  const xs = pts.map((p) => p.year), zs = pts.map((p) => p.zMean);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const zMax = Math.max(...zs.map(Math.abs), 100);
  const X = (yr) => pad + ((yr - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const Y = (z) => h / 2 - (z / zMax) * (h / 2 - 14);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${X(p.year).toFixed(1)},${Y(p.zMean).toFixed(1)}`).join(" ");
  const dots = pts.map((p) =>
    `<circle cx="${X(p.year).toFixed(1)}" cy="${Y(p.zMean).toFixed(1)}" r="3.5"
      fill="${p.zMean < 0 ? "#c92c50" : "#171c16"}"><title>${p.year}: ${fmtZ(p.zMean)}</title></circle>`).join("");
  const labels = pts.map((p) =>
    `<text x="${X(p.year).toFixed(1)}" y="${h - 4}" text-anchor="middle" font-size="9.5"
      font-family="IBM Plex Mono,monospace" fill="#55604f">${p.year}</text>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="Zeitgeist ribbon">
    <line x1="${pad}" y1="${h / 2}" x2="${w - pad}" y2="${h / 2}" stroke="rgba(23,28,22,.25)" stroke-dasharray="3 4"/>
    <path d="${path}" fill="none" stroke="#171c16" stroke-width="1.8" stroke-linejoin="round"/>${dots}${labels}</svg>`;
}

function renderJar() {
  const zone = document.getElementById("jar-zone");
  const jar = loadJar();
  if (!jar.length) {
    zone.innerHTML = `<div class="empty">The jar is empty. Crack a cookie above and it will keep itself here.</div>`;
    return;
  }
  const items = jar.map((e, i) => `
    <div class="jar-item">
      <span class="ji-ny">${e.nation} · <em>${e.year}</em></span>
      <span class="ji-z ${e.zMean != null && e.zMean < 0 ? "neg" : ""}">Z̄ ${fmtZ(e.zMean)}</span>
      <button type="button" class="btn ghost" style="min-height:38px;padding:6px 14px;font-size:13px"
        data-dl="${i}">Numbers CSV</button>
    </div>`).join("");
  const byNation = {};
  jar.forEach((e) => (byNation[e.nation] ||= []).push(e));
  const ribbons = Object.entries(byNation)
    .filter(([, es]) => es.filter((x) => x.zMean != null).length >= 2)
    .map(([n, es]) => `<div class="ribbon"><h4>${n} — the ribbon so far</h4>${ribbonSVG(es)}</div>`).join("");
  zone.innerHTML = `<div class="jar-list">${items}</div>${ribbons}`;
  zone.querySelectorAll("[data-dl]").forEach((b) =>
    b.addEventListener("click", () => downloadCsv(jar[+b.dataset.dl])));
}

/* ---------- downloads ---------- */
function downloadText(name, text, type = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function longCsv(e) {
  const head = "nation,year,session,receipt,model,model_version,node,vital,score,tier,note";
  const lines = [];
  e.rows.forEach((r) => VITALS.forEach((v) => {
    lines.push([e.nation, e.year, e.session, e.receipt, e.model, e.version, r.node, v,
      r.scores[v], r.tier, `"${r.note.replace(/"/g, '""')}"`].join(","));
  }));
  return head + "\n" + lines.join("\n");
}
const downloadCsv = (e) =>
  downloadText(`${e.nation.replace(/\W+/g, "_")}_${e.year}_${e.session}.csv`, longCsv(e), "text/csv");

/* ---------- the card ---------- */
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderCard(e) {
  const z = meanZ(e.rows);
  const motifHtml = e.motifs.map((m) => `
    <div class="motif"><span class="dot"></span><span><span class="m-name">${esc(m.name)}</span>
      ${m.fact ? `<span class="m-fact"> — ${esc(m.fact)}</span>` : ""}</span></div>`).join("");
  const tableRows = e.rows.map((r) => {
    const zr = zOf(r.scores);
    return `<tr><td style="font-weight:600">${r.node}</td>
      ${VITALS.map((v) => `<td class="s">${r.scores[v]}</td>`).join("")}
      <td class="t">${r.tier}</td>
      <td class="s" style="${zr != null && zr < 0 ? "color:#c92c50" : ""}">${fmtZ(zr)}</td>
      <td class="n">${esc(r.note)}</td></tr>`;
  }).join("");
  const sealZ = z == null
    ? `<div class="zseal void"><span class="z">NA</span><span class="zlab">undeclared</span></div>`
    : `<div class="zseal"><span class="z">${fmtZ(z)}</span><span class="zlab">mean Z</span></div>`;
  document.getElementById("card-zone").innerHTML = `
  <article class="cookie" id="cookie">
    <div class="cookie-head">
      <p class="ny">${esc(e.nation)} · <span class="yr">${e.year}</span></p>
      <p class="meta">${esc(e.model)} ${esc(e.version)}<br>${e.utc} UTC<br>receipt ${e.receipt}</p>
    </div>
    <hr class="tear">
    <div class="slip">
      <p class="fortune">“${esc(e.read)}”</p>
      <div class="motifs">${motifHtml}</div>
      <div class="slip-foot">
        <p class="sig">SESSION ${e.session} · sha256 ${e.hash.slice(0, 16)}…<br>
          What one model said under one instrument on one day. Check the footnotes; argue.</p>
        ${sealZ}
      </div>
    </div>
    <hr class="tear">
    <div class="ledger">
      <h3>The ledger — we get the numbers</h3>
      <div class="ledger-grid">
        <div class="radar-box">${radarSVG(e.rows)}
          <div class="zline"><span>Z̄ = mean (A·C)·(K−S)</span>
            <span class="zv ${z != null && z < 0 ? "neg" : ""}">${z == null ? "void" : fmtZ(z)}</span></div>
        </div>
        <div style="overflow-x:auto">
          <table class="scores">
            <thead><tr><th>Paladin</th><th>C</th><th>K</th><th>A</th><th>S</th><th>tier</th><th>Z</th><th>evidence</th></tr></thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="cookie-actions">
      <button type="button" class="btn" id="dl-csv">Download the numbers (CSV)</button>
      <button type="button" class="btn ghost" id="dl-json">Download the cookie (JSON)</button>
    </div>
  </article>`;
  document.getElementById("dl-csv").addEventListener("click", () => downloadCsv(e));
  document.getElementById("dl-json").addEventListener("click", () =>
    downloadText(`${e.receipt}.json`, JSON.stringify(e, null, 2), "application/json"));
  document.getElementById("cookie").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- wiring ---------- */
const $ = (id) => document.getElementById(id);
const state = { session: null, nation: null, year: null, prompt: null };

const newSession = () => {
  const abc = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from(crypto.getRandomValues(new Uint8Array(6))).map((b) => abc[b % abc.length]).join("");
};

function fillNations() {
  const sel = $("nation");
  sel.innerHTML = `<option value="" selected disabled>Choose…</option>` +
    NATIONS.map((n) => `<option>${n}</option>`).join("");
}

function draw(random) {
  let nation = $("nation").value, year = $("year").value.trim();
  if (random || !nation || !/^\d{4}$/.test(year) || +year < YEAR_MIN || +year > YEAR_MAX) {
    nation = NATIONS[Math.floor(Math.random() * NATIONS.length)];
    year = String(YEAR_MIN + Math.floor(Math.random() * (YEAR_MAX - YEAR_MIN + 1)));
    $("nation").value = nation; $("year").value = year;
  }
  state.session = newSession();
  state.nation = nation; state.year = +year;
  state.prompt = buildPrompt(nation, year, state.session);
  $("session-code").textContent = state.session;
  $("session-chip").hidden = false;
  $("prompt-view").textContent = state.prompt;
  $("prompt-details").hidden = false;
  $("btn-copy").disabled = false;
  $("btn-check").disabled = false;
  $("btn-file").disabled = false;
  $("card-zone").innerHTML = "";
  $("verdict").innerHTML = "";
  $("raw").value = "";
}

async function copyPrompt() {
  if (!state.prompt) return;
  try {
    await navigator.clipboard.writeText(state.prompt);
    $("btn-copy").textContent = "Copied";
    setTimeout(() => ($("btn-copy").textContent = "Copy prompt"), 1500);
  } catch {
    $("prompt-details").open = true;
  }
}

function verdictHtml(problems, warnings) {
  let h = "";
  if (problems.length)
    h += `<div class="problems" role="alert"><h3>Not yet — ${problems.length} structural ${problems.length === 1 ? "problem" : "problems"}</h3><ul>${problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>`;
  if (warnings.length)
    h += `<div class="warnings"><h3>Review notes</h3><ul>${warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
  return h;
}

async function file(checkOnly) {
  const v = $("verdict");
  const fieldProblems = [];
  if (!state.prompt) fieldProblems.push("Draw a prompt first.");
  if ($("model").value.trim().length < 2 || !$("version").value.trim())
    fieldProblems.push("Name the actual model and its version.");
  if (!$("fresh").checked) fieldProblems.push("Fresh, memory-free chat attestation is required.");
  const raw = $("raw").value;
  if (raw.trim().length < 50) fieldProblems.push("Paste the model's whole reply.");
  if (fieldProblems.length) { v.innerHTML = verdictHtml(fieldProblems, []); return; }
  const res = validateReply(raw, state.session);
  if (checkOnly || !res.ok) { v.innerHTML = verdictHtml(res.problems, res.warnings); return; }
  const hashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw));
  const hash = Array.from(new Uint8Array(hashBuf)).map((b) => b.toString(16).padStart(2, "0")).join("");
  const e = {
    receipt: "CF-" + hash.slice(0, 8).toUpperCase(),
    utc: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
    nation: state.nation, year: state.year, session: state.session,
    model: $("model").value.trim(), version: $("version").value.trim(),
    hash, rows: res.rows, motifs: res.motifs, read: res.read, zMean: meanZ(res.rows),
    warnings: res.warnings
  };
  const jar = loadJar();
  if (!jar.some((x) => x.hash === hash)) { jar.unshift(e); saveJar(jar); }
  v.innerHTML = verdictHtml([], res.warnings);
  renderCard(e);
  renderJar();
}

fillNations();
renderJar();
$("btn-random").addEventListener("click", () => draw(true));
$("btn-draw").addEventListener("click", () => draw(false));
$("btn-copy").addEventListener("click", copyPrompt);
$("btn-check").addEventListener("click", () => file(true));
$("btn-file").addEventListener("click", () => file(false));
