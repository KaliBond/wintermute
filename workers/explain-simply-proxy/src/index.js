/* ───────────────────────────────────────────────────────────────
   explain-simply-proxy — Cloudflare Worker

   Holds the Kimi/Moonshot API key server-side and forwards chat
   completion requests from neuralnations.org's "Explain Simply"
   feature (explain-simply.js). The browser never sees the key.

   Deploy:
     1. wrangler login
     2. wrangler secret put KIMI_API_KEY      (paste the rotated key)
     3. wrangler deploy
     4. Point window.EXPLAIN_SIMPLY_CONFIG.endpoint at the deployed
        Worker URL (printed by `wrangler deploy`), with no apiKey.
   ─────────────────────────────────────────────────────────────── */

const ALLOWED_ORIGINS = [
  'https://neuralnations.org',
  'https://www.neuralnations.org',
];

const MODEL = 'kimi-k2-0905-preview';
const MAX_TOKENS_CAP = 700;
const MAX_CONTENT_CHARS = 20000; // guards against runaway PDF/page text in the prompt
const UPSTREAM_URL = 'https://api.moonshot.ai/v1/chat/completions';

function isAllowedOrigin(origin) {
  if (!origin) return false;
  if (ALLOWED_ORIGINS.includes(origin)) return true;
  // Local dev convenience — any localhost/127.0.0.1 port.
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

function corsHeaders(origin) {
  const allowOrigin = isAllowedOrigin(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

function jsonResponse(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders(origin),
    },
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    if (!isAllowedOrigin(origin)) {
      return jsonResponse({ error: 'origin not allowed' }, 403, origin);
    }

    if (request.method !== 'POST') {
      return jsonResponse({ error: 'method not allowed' }, 405, origin);
    }

    if (!env.KIMI_API_KEY) {
      return jsonResponse({ error: 'proxy not configured' }, 500, origin);
    }

    let payload;
    try {
      payload = await request.json();
    } catch (e) {
      return jsonResponse({ error: 'invalid JSON body' }, 400, origin);
    }

    const messages = Array.isArray(payload.messages) ? payload.messages : null;
    if (!messages || !messages.length) {
      return jsonResponse({ error: 'messages array required' }, 400, origin);
    }

    // Trim/validate — only forward role + content, cap content length.
    const cleanMessages = messages.slice(0, 4).map((m) => ({
      role: typeof m.role === 'string' ? m.role : 'user',
      content: String(m.content || '').slice(0, MAX_CONTENT_CHARS),
    }));

    const temperature = typeof payload.temperature === 'number'
      ? Math.min(Math.max(payload.temperature, 0), 1)
      : 0.85;
    const maxTokens = typeof payload.max_tokens === 'number'
      ? Math.min(payload.max_tokens, MAX_TOKENS_CAP)
      : MAX_TOKENS_CAP;

    let upstreamRes;
    try {
      upstreamRes = await fetch(UPSTREAM_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${env.KIMI_API_KEY}`,
        },
        body: JSON.stringify({
          model: MODEL, // server-controlled — client can't pick a pricier model
          temperature,
          max_tokens: maxTokens,
          messages: cleanMessages,
        }),
      });
    } catch (e) {
      return jsonResponse({ error: 'upstream request failed' }, 502, origin);
    }

    const upstreamBody = await upstreamRes.text();
    return new Response(upstreamBody, {
      status: upstreamRes.status,
      headers: {
        'Content-Type': 'application/json',
        ...corsHeaders(origin),
      },
    });
  },
};
