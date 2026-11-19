/* ───────────────────────────────────────────────────────────────
   citizen-observatory — Cloudflare Worker + D1

   Server side of the CAMS Citizen Scoring Observatory
   (neuralnations.org/observatory.html). Ported from the local
   FastAPI prototype (CAMS_Citizen_Observatory_Prototype, 19 Nov 2026).

   It hands out one of the 105 unchanged CAMS v1.2 scoring prompts,
   accepts a visitor-pasted model reply, checks its structure and
   stores it in D1 for later research review. It never calls a model.

   Endpoints
     GET  /api/stats
     GET  /api/assignment?family=any|claude|deepseek|gemini|gpt|grok|kimi|qwen
     POST /api/submit

   Privacy: no names, emails, IP addresses or device location are
   stored. The rate limiter keys on the connecting IP in memory only.

   Deploy: see README.md in this folder.
   ─────────────────────────────────────────────────────────────── */

import PROMPTS from './prompts.json';
import { validateReply } from './validate.js';

const ALLOWED_ORIGINS = ['https://neuralnations.org', 'https://www.neuralnations.org'];
const COHORT = 'public_v1';           // never pool with the controlled 105-session cohort
const MAX_BODY_BYTES = 200_000;
const FAMILIES = [...new Set(Object.values(PROMPTS).map((p) => p.family))].sort();

function isAllowedOrigin(origin) {
  if (!origin) return false;
  if (ALLOWED_ORIGINS.includes(origin)) return true;
  if (/^https:\/\/[a-z0-9-]+\.wintermute-8fc\.pages\.dev$/.test(origin)) return true; // Cloudflare previews
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': isAllowedOrigin(origin) ? origin : ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

function json(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...cors(origin) },
  });
}

async function sha256Hex(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function receiptId() {
  const b = crypto.getRandomValues(new Uint8Array(6));
  return 'CAMS-' + [...b].map((x) => x.toString(16).padStart(2, '0')).join('').toUpperCase();
}

function str(v, min, max) {
  if (v === null || v === undefined) return min === 0 ? null : undefined;
  if (typeof v !== 'string') return undefined;
  const t = v.trim();
  if (t.length < min || t.length > max) return undefined;
  return t === '' ? null : t;
}

async function stats(env, origin) {
  const { results } = await env.DB.prepare(
    'SELECT family, COUNT(*) AS n FROM submissions GROUP BY family').all();
  const total = results.reduce((s, r) => s + r.n, 0);
  return json({
    received_submissions: total,
    received_scoring_rows: total * 128,
    families: Object.fromEntries(results.map((r) => [r.family, r.n])),
    available_assignments: Object.keys(PROMPTS).length,
  }, 200, origin);
}

async function assignment(url, env, origin) {
  const family = url.searchParams.get('family') || 'any';
  const options = Object.values(PROMPTS).filter((p) => family === 'any' || p.family === family);
  if (!options.length) return json({ errors: [`Unknown family. Use any or one of: ${FAMILIES.join(', ')}`] }, 400, origin);
  const { results } = await env.DB.prepare(
    'SELECT assignment_id, COUNT(*) AS n FROM submissions GROUP BY assignment_id').all();
  const counts = Object.fromEntries(results.map((r) => [r.assignment_id, r.n]));
  const min = Math.min(...options.map((p) => counts[p.id] || 0));
  const pool = options.filter((p) => (counts[p.id] || 0) === min);
  const pick = pool[crypto.getRandomValues(new Uint32Array(1))[0] % pool.length];
  const { sha256, ...publicFields } = pick;
  return json(publicFields, 200, origin);
}

async function submit(request, env, origin) {
  if (!isAllowedOrigin(origin)) return json({ errors: ['Submissions are accepted from neuralnations.org only'] }, 403, origin);
  if (env.SUBMIT_LIMITER) {
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    const { success } = await env.SUBMIT_LIMITER.limit({ key: ip });
    if (!success) return json({ errors: ['Too many submissions in a short time. Please wait a minute and try again.'] }, 429, origin);
  }
  const len = Number(request.headers.get('Content-Length') || 0);
  if (len > MAX_BODY_BYTES) return json({ errors: ['Submission too large'] }, 413, origin);

  let body;
  try { body = await request.json(); } catch { return json({ errors: ['Request body must be JSON'] }, 400, origin); }

  const p = PROMPTS[body.assignment_id];
  if (!p) return json({ errors: ['Unknown assignment'] }, 400, origin);
  if (typeof body.raw_response !== 'string' || body.raw_response.length < 50 || body.raw_response.length > 100_000) {
    return json({ errors: ['Paste the complete model reply (50 to 100,000 characters)'] }, 422, origin);
  }
  const actualModel = str(body.actual_model, 2, 100);
  const modelVersion = str(body.model_version, 1, 100);
  const localTime = str(body.visitor_local_time, 0, 100);
  const place = str(body.optional_place, 0, 100);
  if (!actualModel || !modelVersion) return json({ errors: ['Model name and model version are required'] }, 422, origin);
  if (localTime === undefined || place === undefined) return json({ errors: ['Optional fields must be 100 characters or fewer'] }, 422, origin);
  if (body.attested_fresh !== true) return json({ errors: ['Fresh, memory-free chat attestation is required'] }, 422, origin);
  if (body.consent !== true) return json({ errors: ['Consent to store the submission is required'] }, 422, origin);

  const { rows, problems, warnings } = validateReply(body.raw_response, p);
  if (problems.length) return json({ errors: problems, warnings }, 422, origin);

  const rawHash = await sha256Hex(body.raw_response);
  const receipt = receiptId();
  const now = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  try {
    await env.DB.prepare(
      `INSERT INTO submissions (submission_id, assignment_id, family, actual_model, model_version,
         utc_timestamp, visitor_local_time, optional_place, attested_fresh, consent, raw_hash,
         prompt_sha256, cohort, raw_response, rows_json, status, warnings_json, review_status)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, 1, ?9, ?10, ?11, ?12, ?13, 'structurally_valid', ?14, 'pending')`
    ).bind(receipt, p.id, p.family, actualModel, modelVersion, now, localTime, place,
      rawHash, p.sha256, COHORT, body.raw_response, JSON.stringify(rows), JSON.stringify(warnings)).run();
  } catch (e) {
    if (String(e.message || e).includes('UNIQUE')) {
      return json({ errors: ['Identical response already submitted'] }, 409, origin);
    }
    throw e;
  }
  return json({
    receipt_id: receipt, timestamp_utc: now, place, model: actualModel, model_version: modelVersion,
    rows_accepted: rows.length,
    status: 'Structurally valid; awaiting provenance and research-quality review',
    warnings,
  }, 200, origin);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const url = new URL(request.url);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors(origin) });
    try {
      if (request.method === 'GET' && url.pathname === '/api/stats') return await stats(env, origin);
      if (request.method === 'GET' && url.pathname === '/api/assignment') return await assignment(url, env, origin);
      if (request.method === 'POST' && url.pathname === '/api/submit') return await submit(request, env, origin);
      return json({ errors: ['Not found'] }, 404, origin);
    } catch (e) {
      console.error(e);
      return json({ errors: ['Server error; your submission was not stored. Please try again.'] }, 500, origin);
    }
  },
};
