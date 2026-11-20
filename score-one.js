/* Score One: browser-only CAMS scoring for one society and one year.
   No server, no API key. The visitor runs the prompt in any model and pastes the reply back.

   Reused code (kept in step with the originals; test/score-one.test.js checks the first two):
     - parseCsv(): copied unchanged from workers/citizen-observatory/src/validate.js (Observatory)
     - SESSION line regex: the Observatory's pattern from the same file
     - nodeValues(): the JUNO Calculator's formulas from juno_calculator.html calc()
     - renderTable / download helpers: after cams-scorer.js (renderTable, csvToDownloadUrl)

   PROTOCOL below is CAMS_SCORING_PROTOCOL_V2_1.md quoted verbatim (line endings normalised to LF).
   test/score-one.test.js fails if this string drifts from the Markdown file. */
(function (root) {
  'use strict';

  var PROTOCOL = "# CAMS Scoring Protocol v2.1\n\n## Civilizational Analysis and Measurement System\n\n### Scoring Layer Only\n\n---\n\n## Purpose\n\nThis protocol is for the **scoring stage only** of CAMS analysis.\n\nIts purpose is to guide an evaluator — human or LLM — in assigning evidence-based scores to the **eight CAMS nodes** on four dimensions:\n\n- **Coherence**\n- **Capacity**\n- **Stress**\n- **Abstraction**\n\nThis protocol does **not** include downstream calculation.\n\nDo **not** calculate:\n\n- Node Value\n- Bond Strength\n- system-level averages\n- any derived metric\n\nYour task ends when you have produced a complete **8-node scoring snippet** in the required format.\n\n---\n\n## Philosophy & Approach\n\n### Core Principle\n\nYou are analysing civilizational cognition through **institutional coordination capacity**.\n\nThink thermodynamically: societies are metabolic systems that acquire, process, store, coordinate, and act on information. Your task is to assess how well each core function is operating under real historical conditions.\n\n### Validator Stance: Evidence-Based Realism\n\nWhen scoring, adopt the following position:\n\n- **Capacity**: score based on **demonstrated performance**, not theoretical potential\n- **Abstraction**: score based on **operational sophistication**, not aspirational complexity\n- **Coherence**: score based on **actual coordination**, not formal structure\n- **Stress**: score based on **entropy production** — that is, observed breakdown rate — not subjective discomfort\n\n### Key Distinctions\n\n- A node can be **structurally sophisticated** but **functionally impaired**\n- A node can be under **heavy pressure** without being highly stressed, if it continues to function coherently\n- A node can face **limited pressure** but still show high stress if it is fragmenting or failing\n\n### Scoring Rule\n\n**Score what is operationally true, not what ought to be true, and not what might be possible under better conditions.**\n\n### Independence Rule\n\nScore each node on the evidence for that node.\n\nDo **not** adjust scores to make the total system appear balanced, elegant, symmetrical, or mathematically plausible.\nDo **not** anticipate downstream formulas.\n\n---\n\n## Step 1: Define the Eight Nodes\n\n| Node | Function | What to Assess |\n|---|---|---|\n| **Helm** | Strategic coordination and executive decision-making | Can leadership formulate direction, align institutions, and execute decisions? |\n| **Shield** | Security provision and force projection | Can the system maintain order, defend itself, and project coercive force when required? |\n| **Lore** | Knowledge synthesis and cultural legitimation | Can knowledge institutions generate, preserve, interpret, and legitimate shared understanding? |\n| **Stewards** | Resource ownership and capital allocation | Can wealth-holding or asset-controlling actors allocate resources productively to system needs? |\n| **Craft** | Skilled production and professional coordination | Can specialised workers and professionals deliver sophisticated functions reliably? |\n| **Hands** | Labour execution and basic production | Can the labouring base mobilise and perform large-scale practical work? |\n| **Archive** | Institutional memory and information storage | Can critical information be preserved, retrieved, and transmitted across time? |\n| **Flow** | Commerce, trade, and circulation | Can goods, services, money, and logistical throughput move effectively through the system? |\n\n---\n\n## Step 2: Assign Scores on the Four Dimensions\n\nAll scores are integers from **1 to 10**.\n\n---\n\n## Step 3: Output Rules\n\nReturn exactly **8 rows**, one for each node:\n\n- Helm\n- Shield\n- Lore\n- Stewards\n- Craft\n- Hands\n- Archive\n- Flow\n\n### Required fields\n\n- Society\n- Year\n- Node\n- Coherence\n- Capacity\n- Stress\n- Abstraction\n\n### Rules\n\n- Scores must be integers from 1 to 10\n- Each node must appear exactly once\n- Return raw scores only\n- Do not include notes\n- Do not include explanations\n- Do not calculate Node Value\n- Do not calculate Bond Strength\n- Do not calculate averages, totals, or derived metrics\n- Do not add narrative commentary unless explicitly requested\n\n### Preferred Output Format\n\n```csv\nSociety,Year,Node,Coherence,Capacity,Stress,Abstraction\nMarkerx,1897,Helm,7,6,4,7\nMarkerx,1897,Shield,4,5,3,5\nMarkerx,1897,Lore,8,7,4,8\nMarkerx,1897,Stewards,5,4,7,6\nMarkerx,1897,Craft,6,6,4,7\nMarkerx,1897,Hands,4,4,8,5\nMarkerx,1897,Archive,8,7,3,7\nMarkerx,1897,Flow,4,6,6,6\n```\n";

  var NODES = ['Helm', 'Shield', 'Lore', 'Stewards', 'Craft', 'Hands', 'Archive', 'Flow'];
  var HEADER = ['Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'];
  var DIMS = [['Coherence', 'C'], ['Capacity', 'K'], ['Stress', 'S'], ['Abstraction', 'A']];
  var LONG_HEADER = 'Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength';

  /** Parse CSV text into arrays of fields (RFC 4180: quotes, doubled quotes, CRLF). */
  function parseCsv(text) {
    const rows = [];
    let row = [], field = '', inQuotes = false, i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
          inQuotes = false; i++; continue;
        }
        field += ch; i++; continue;
      }
      if (ch === '"' && field === '') { inQuotes = true; i++; continue; }
      if (ch === ',') { row.push(field); field = ''; i++; continue; }
      if (ch === '\r' && text[i + 1] === '\n') { i++; continue; }
      if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; i++; continue; }
      field += ch; i++;
    }
    if (inQuotes) throw new Error('unterminated quoted field');
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    return rows.filter((r) => !(r.length === 1 && r[0].trim() === ''));
  }

  function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  /** Random SESSION code: 6 characters, no look-alikes (0/O, 1/I/L). */
  function newSession(rand) {
    var abc = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789', out = '';
    var bytes = new Uint8Array(6);
    if (rand) { for (var i = 0; i < 6; i++) bytes[i] = rand(); }
    else (root.crypto || require('crypto').webcrypto).getRandomValues(bytes);
    for (var j = 0; j < 6; j++) out += abc[bytes[j] % abc.length];
    return out;
  }

  function cleanSociety(s) { return String(s || '').replace(/\s+/g, ' ').trim(); }
  function csvField(s) { return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }

  /** Validate the visitor's request. Returns a list of problems (empty = fine). */
  function checkRequest(society, year) {
    var p = [];
    if (!cleanSociety(society)) p.push('Enter a society.');
    if (!/^-?\d{1,4}$/.test(String(year).trim())) p.push('Enter a single year as a whole number (for example 1937, or -480 for 480 BCE).');
    return p;
  }

  function buildPrompt(society, year, session) {
    society = cleanSociety(society); year = String(year).trim();
    var example = NODES.map(function (n) { return csvField(society) + ',' + year + ',' + n + ',<C>,<K>,<S>,<A>'; }).join('\n');
    return PROTOCOL.trim() + '\n\n---\n\n' +
      '## Assignment\n\n' +
      'Society: ' + society + '\n' +
      'Year: ' + year + '\n\n' +
      'CONTAMINATION CHECK: This must be a fresh chat. If this conversation already contains CAMS scores, ' +
      'Neural Nations material, or earlier scoring of this society, reply with the single word CONTAMINATED and nothing else.\n\n' +
      'Otherwise score the eight nodes for ' + society + ' in ' + year + ' following the protocol above. ' +
      'Return exactly 8 rows inside one ```csv code block, with this header and the society and year written exactly as below on every row:\n\n' +
      '```csv\n' + HEADER.join(',') + '\n' + example + '\n```\n\n' +
      'Replace each <C>, <K>, <S>, <A> with an integer from 1 to 10. ' +
      'After the code block, write this line on its own, exactly:\n\n' +
      'SESSION ' + session + '\n';
  }

  /* Validate a pasted reply. Returns
       { ok, problems: [..global..], rowErrors: [{row, node, messages:[..]}], rows: [{Node,C,K,S,A}] }
     rows are in canonical node order when ok. */
  function validateReply(raw, req) {
    var problems = [], rowErrors = [];
    raw = String(raw || '');
    var society = cleanSociety(req.society), year = String(req.year).trim();
    if (!raw.trim()) return { ok: false, problems: ['The reply is empty. Paste the model\'s reply or upload a .txt file.'], rowErrors: rowErrors, rows: [] };
    if (raw.trim().toUpperCase() === 'CONTAMINATED') {
      return { ok: false, problems: ['The model reported CONTAMINATED. Start a fresh chat and run the prompt again.'], rowErrors: rowErrors, rows: [] };
    }

    var sessionLine = new RegExp('^[ \\t>*_`#-]*SESSION[ \\t*_`:-]*' + escapeRegex(req.session) + '[*_`]*[ \\t]*$', 'mi');
    if (!sessionLine.test(raw)) {
      var other = /^[ \t>*_`#-]*SESSION[ \t*_`:-]*([A-Za-z0-9]+)/mi.exec(raw);
      problems.push(other ? 'SESSION code does not match: expected ' + req.session + ', found ' + other[1] + '. The reply may come from a different prompt.'
                          : 'SESSION line missing: the reply must end with "SESSION ' + req.session + '".');
    }

    var block = /```(?:csv)?[ \t]*\r?\n([\s\S]*?)\r?\n?```/i.exec(raw), text;
    if (block) text = block[1];
    else {
      var at = raw.search(/^\s*Society\s*,\s*Year\s*,/mi);
      if (at < 0) { problems.push('No CSV found: expected a ```csv code block starting with the header ' + HEADER.join(',') + '.'); return { ok: false, problems: problems, rowErrors: rowErrors, rows: [] }; }
      text = raw.slice(at).split(/\r?\n\s*\r?\n/)[0];
    }

    var table;
    try { table = parseCsv(text.trim()); }
    catch (e) { problems.push('CSV parsing error: ' + e.message); return { ok: false, problems: problems, rowErrors: rowErrors, rows: [] }; }
    var header = (table[0] || []).map(function (h) { return h.trim().toLowerCase(); });
    if (header.join(',') !== HEADER.join(',').toLowerCase()) problems.push('CSV header must be exactly: ' + HEADER.join(','));
    var body = table.slice(1);
    if (body.length !== 8) problems.push('Expected exactly 8 data rows, found ' + body.length + '.');

    var seen = {}, rows = [];
    body.forEach(function (f, i) {
      var msgs = [], node = (f[2] || '').trim();
      var canon = NODES.filter(function (n) { return n.toLowerCase() === node.toLowerCase(); })[0];
      if (f.length !== 7) msgs.push('has ' + f.length + ' fields; expected 7');
      if (!canon) msgs.push(node ? 'unknown node "' + node + '"' : 'node name missing');
      else if (seen[canon]) msgs.push(canon + ' appears more than once');
      if (cleanSociety(f[0]).toLowerCase() !== society.toLowerCase()) msgs.push('society is "' + cleanSociety(f[0]) + '"; expected "' + society + '"');
      if (String(f[1] || '').trim() !== year) msgs.push('year is "' + String(f[1] || '').trim() + '"; expected ' + year);
      var vals = {};
      DIMS.forEach(function (d, k) {
        var v = String(f[3 + k] == null ? '' : f[3 + k]).trim();
        if (!/^\d+$/.test(v) || +v < 1 || +v > 10) msgs.push(d[0] + ' is "' + v + '"; must be a whole number from 1 to 10');
        vals[d[1]] = +v;
      });
      if (canon) seen[canon] = true;
      if (msgs.length) rowErrors.push({ row: i + 1, node: canon || node || '?', messages: msgs });
      rows.push({ Node: canon, C: vals.C, K: vals.K, S: vals.S, A: vals.A });
    });
    var missing = NODES.filter(function (n) { return !seen[n]; });
    if (missing.length) problems.push('Missing node' + (missing.length > 1 ? 's' : '') + ': ' + missing.join(', ') + '.');

    var ok = !problems.length && !rowErrors.length;
    if (ok) rows = NODES.map(function (n) { return rows.filter(function (r) { return r.Node === n; })[0]; });
    return { ok: ok, problems: problems, rowErrors: rowErrors, rows: ok ? rows : [] };
  }

  /** JUNO v1.2-Final, as in juno_calculator.html calc(). */
  function nodeValues(rows) {
    var sc = {}, q = {}, V = {}, BS = {};
    rows.forEach(function (r) { sc[r.Node] = r; });
    NODES.forEach(function (n) {
      q[n] = (0.6 * sc[n].C + 0.4 * sc[n].A) / 10;
      V[n] = sc[n].C + sc[n].K - sc[n].S + 0.5 * sc[n].A;
    });
    NODES.forEach(function (n, i) {
      var bonds = NODES.map(function (m, j) {
        if (i === j) return null;
        return Math.sqrt(q[n] * q[m]) * Math.pow(2, -(sc[n].S + sc[m].S) / 10);
      }).filter(function (x) { return x !== null; });
      BS[n] = bonds.reduce(function (a, b) { return a + b; }, 0) / bonds.length;
    });
    return { V: V, BS: BS };
  }

  /** Long CSV, 8 rows, JUNO Calculator column order and rounding. */
  function longCsv(rows, req) {
    var soc = csvField(cleanSociety(req.society)), yr = String(req.year).trim(), nv = nodeValues(rows);
    return LONG_HEADER + '\n' + rows.map(function (r) {
      return [soc, yr, r.Node, r.C, r.K, r.S, r.A, nv.V[r.Node].toFixed(3), nv.BS[r.Node].toFixed(5)].join(',');
    }).join('\n') + '\n';
  }

  /** Wide CSV: one line, society, year, 32 values, model, session, date. */
  function wideCsv(rows, req, model, date) {
    var head = ['Society', 'Year'], vals = [csvField(cleanSociety(req.society)), String(req.year).trim()];
    rows.forEach(function (r) {
      DIMS.forEach(function (d) { head.push(r.Node + '_' + d[1]); vals.push(r[d[1]]); });
    });
    head.push('Model', 'Session', 'Date');
    vals.push(csvField(String(model || '').trim()), req.session, date);
    return head.join(',') + '\n' + vals.join(',') + '\n';
  }

  var api = { PROTOCOL: PROTOCOL, NODES: NODES, HEADER: HEADER, LONG_HEADER: LONG_HEADER, parseCsv: parseCsv,
              newSession: newSession, checkRequest: checkRequest, buildPrompt: buildPrompt, validateReply: validateReply,
              nodeValues: nodeValues, longCsv: longCsv, wideCsv: wideCsv };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ScoreOne = api;
})(typeof window !== 'undefined' ? window : globalThis);
