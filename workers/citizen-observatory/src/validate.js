/* Structural validation of one pasted model reply against its assignment.
   Port of parse_and_validate() in the CAMS_Citizen_Observatory_Prototype
   server.py, with three changes:
     - the SESSION line may carry a colon or Markdown emphasis
       ("SESSION: X7Y2", "**SESSION X7Y2**"); the code itself must still match;
     - an all-NA reply is flagged for review (still structurally valid);
     - parsing is a small RFC 4180 reader, so quoted notes are handled. */

export const METRICS = ['C', 'K', 'A', 'S'].flatMap((g) => [1, 2, 3, 4].map((n) => g + n));
export const HEADER = ['case_id', 'arm', 'rater_id', 'metric', 'score', 'tier', 'note'];

/** Parse CSV text into arrays of fields (RFC 4180: quotes, doubled quotes, CRLF). */
export function parseCsv(text) {
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

/** Returns { rows, problems, warnings }. Any problem means reject. */
export function validateReply(raw, assignment) {
  const problems = [], warnings = [];
  if (raw.trim().toUpperCase() === 'CONTAMINATED') {
    return { rows: [], problems: ['Model reported CONTAMINATED; start a fresh chat'], warnings };
  }
  const block = /```(?:csv)?[ \t]*\r?\n([\s\S]*?)\r?\n```/i.exec(raw);
  if (!block) return { rows: [], problems: ['No CSV code block found'], warnings };

  let table;
  try { table = parseCsv(block[1].trim()); }
  catch (e) { return { rows: [], problems: [`CSV parsing error: ${e.message}`], warnings }; }
  const [header = [], ...body] = table;
  if (header.join(',') !== HEADER.join(',')) problems.push('CSV header differs from required seven columns');

  const rows = [];
  body.forEach((fields, i) => {
    if (fields.length !== HEADER.length) { rows.push(null); problems.push(`Row ${i + 1}: malformed CSV field count`); return; }
    rows.push(Object.fromEntries(HEADER.map((h, k) => [h, fields[k]])));
  });

  if (rows.length !== 128) problems.push(`Expected 128 data rows, received ${rows.length}`);

  const code = escapeRegex(assignment.session);
  const sessionLine = new RegExp(`^[ \\t>*_\`#-]*SESSION[ \\t*_\`:-]*${code}[*_\`]*[ \\t]*$`, 'mi');
  if (!sessionLine.test(raw)) problems.push(`Expected SESSION verification code (${assignment.session}) is missing`);

  if (rows.length === 128 && rows.every(Boolean)) {
    const expected = assignment.cases.flatMap((c) => METRICS.map((m) => `${c}|${m}`));
    const got = rows.map((r) => `${r.case_id}|${r.metric}`);
    if (got.join(',') !== expected.join(',')) problems.push('Cases or metrics do not match the assigned order');
  }

  let naCount = 0;
  rows.forEach((row, i) => {
    if (!row) return;
    if (row.arm !== 'named' || row.rater_id !== assignment.rater) problems.push(`Row ${i + 1}: incorrect arm or rater ID`);
    const score = row.score.trim(), tier = row.tier.trim(), note = row.note.trim();
    if (score === 'NA') {
      naCount++;
      if (tier !== 'NA') problems.push(`Row ${i + 1}: NA score requires NA tier`);
      if (!note) warnings.push(`Row ${i + 1}: empty NA explanation`);
    } else {
      const n = /^\d+$/.test(score) ? Number(score) : NaN;
      if (!(n >= 1 && n <= 10) || !['A', 'B', 'C'].includes(tier)) problems.push(`Row ${i + 1}: score/tier invalid`);
      if (!note) problems.push(`Row ${i + 1}: evidence note missing`);
    }
    if (note.split(/\s+/).filter(Boolean).length > 15) problems.push(`Row ${i + 1}: note longer than 15 words`);
  });

  const valid = rows.filter(Boolean);
  if (valid.length && naCount === 0) warnings.push('No NA values: evidence coverage may merit human review');
  if (valid.length && naCount === valid.length) warnings.push('Every score is NA: flagged for human review');

  return { rows: valid, problems: [...new Set(problems)].slice(0, 20), warnings: [...new Set(warnings)].slice(0, 20) };
}
