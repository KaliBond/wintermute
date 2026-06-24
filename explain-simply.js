/* ───────────────────────────────────────────────────────────────
   Explain Simply — behaviour
   • Builds one modal, wires every [data-explain-simply] trigger.
   • Default text is the canonical plain-English explanation.
   • "Regenerate with AI" generates a fresh explanation on the fly,
     tailored to the current page. Falls back to pre-written variants
     when no AI helper is available.
   ─────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  /* ───────────────────────────────────────────────────────────
     PROVIDER CONFIG  — set one of these to enable live AI on the
     real site. All three are OpenAI-compatible chat APIs, so only
     the endpoint, model, and key differ. Define window.EXPLAIN_SIMPLY_CONFIG
     BEFORE this script (e.g. in a small inline <script> in <head>).

       window.EXPLAIN_SIMPLY_CONFIG = {
         // Kimi / Moonshot
         endpoint: 'https://api.moonshot.ai/v1/chat/completions',
         model:    'kimi-k2-0905-preview',
         // Grok / xAI
         // endpoint: 'https://api.x.ai/v1/chat/completions',
         // model:    'grok-4',
         // GPT / OpenAI
         // endpoint: 'https://api.openai.com/v1/chat/completions',
         // model:    'gpt-4o-mini',
         apiKey: 'sk-...'          //  ⚠ see security note below
       };

     ⚠ SECURITY: a key placed in client-side JS is public to anyone
     who views source. For production, point `endpoint` at a tiny
     serverless proxy you control (Cloudflare Worker / Netlify fn)
     that injects the key server-side and forwards to the provider —
     then leave `apiKey` blank here. The request shape is identical.

     If no config and no window.claude helper are present, the modal
     still works: it rotates through hand-written fallback explanations.
     ─────────────────────────────────────────────────────────────── */
  var CONFIG = (typeof window !== 'undefined' && window.EXPLAIN_SIMPLY_CONFIG) || {};

  // Canonical default explanation (shown first, no AI needed).
  var DEFAULT_TEXT = [
    "The foundational insight is that human societies are Complex Adaptive Systems, and we can recover how well they coordinate by studying historical records. CAMS turns that idea into a practical measurement tool. It looks at eight core functions every society must perform \u2014 leadership, security, knowledge, memory, production, labour, resource flow, and stewardship \u2014 and tracks how healthy and connected they are.",
    "When these functions work well together, societies tend to be stable and adaptable. When they fall out of sync, stress builds up and coordination breaks down, often long before it becomes obvious. CAMS makes those hidden patterns visible.",
    "This site is an open research project. Everything \u2014 the data, the methods, the tools \u2014 is published so anyone can examine it, test it, or build on it. No grand theories, just honest measurement."
  ];

  // Pre-written fallback variants (used when the AI helper is unavailable).
  var FALLBACKS = [
    [
      "Think of any society \u2014 a country, an empire, even a big organisation \u2014 as a living system that has to keep eight basic jobs running at once: leading, protecting, knowing, remembering, making things, doing the work, moving resources around, and looking after the future. CAMS is a way of measuring how well those eight jobs are doing, and how well they're working together.",
      "Healthy societies keep those parts in balance. When a few of them start pulling apart, pressure quietly builds \u2014 usually well before anyone notices a crisis. CAMS reads historical records to spot that strain early, and to compare one place or era against another on the same honest scale.",
      "It isn't a grand theory of why nations rise or fall. It's a measurement instrument \u2014 and all of its data, formulas, and tools are published in the open so anyone can check the work."
    ],
    [
      "Neural Nations is a research project built around a simple question: can you actually measure how well a society is holding together? Its framework, CAMS, treats every society as eight working parts \u2014 leadership, security, knowledge, memory, production, labour, resource flow, and stewardship \u2014 and scores how strong each one is and how tightly they're linked.",
      "When those parts stay connected, a society absorbs shocks and adapts. When they drift apart, stress accumulates faster than the society can cope, and coordination starts to fail \u2014 often years before it shows up in the headlines. CAMS is designed to make that slow, hidden drift visible.",
      "Everything here is open. The datasets, the scoring methods, and the tools are all published so you can examine them, test them, or argue with them. The aim is careful measurement, not prophecy."
    ]
  ];

  var modal = null;
  var lastFocused = null;
  var aiCache = [];   // AI variants gathered this session
  var fbIndex = 0;    // fallback rotation pointer
  var currentItem = null; // active per-item context, or null for site-level

  var ICON_X = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  var ICON_SPIN = '<svg class="es-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-3-6.7" opacity="0.95"/></svg>';

  function buildModal() {
    var o = document.createElement('div');
    o.className = 'es-overlay';
    o.setAttribute('role', 'dialog');
    o.setAttribute('aria-modal', 'true');
    o.setAttribute('aria-labelledby', 'es-title');
    o.innerHTML =
      '<div class="es-dialog">' +
        '<div class="es-head">' +
          '<div>' +
            '<p class="es-eyebrow">In plain English</p>' +
            '<h2 class="es-title" id="es-title">What is Neural Nations?</h2>' +
          '</div>' +
          '<button class="es-close" type="button" aria-label="Close">' + ICON_X + '</button>' +
        '</div>' +
        '<div class="es-body">' +
          '<div class="es-prose"></div>' +
          '<div class="es-skeleton" aria-hidden="true">' +
            '<div class="es-skeleton-line"></div><div class="es-skeleton-line"></div>' +
            '<div class="es-skeleton-line"></div><div class="es-skeleton-line"></div>' +
            '<div class="es-skeleton-line"></div><div class="es-skeleton-line"></div>' +
            '<div class="es-skeleton-line"></div>' +
          '</div>' +
        '</div>' +
        '<div class="es-foot">' +
          '<span class="es-note">A friendly, jargon-free summary. Regenerate for a fresh take.</span>' +
          '<button class="es-regen" type="button">' + ICON_SPIN +
            '<span class="es-regen-label"></span></button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(o);

    o.querySelector('.es-close').addEventListener('click', close);
    o.querySelector('.es-regen').addEventListener('click', regenerate);
    o.addEventListener('mousedown', function (e) { if (e.target === o) close(); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && o.classList.contains('es-open')) close();
    });
    return o;
  }

  function setProse(paras) {
    var prose = modal.querySelector('.es-prose');
    prose.innerHTML = '';
    paras.forEach(function (t) {
      var p = document.createElement('p');
      p.textContent = t;
      prose.appendChild(p);
    });
  }

  // open()                    → site-level "What is Neural Nations?"
  // open({ title, item })     → item-level explainer for one tool/paper.
  //   item = { name, kind, desc }
  function open(opts) {
    opts = opts || {};
    if (!modal) modal = buildModal();
    lastFocused = document.activeElement;
    currentItem = opts.item || null;

    var titleEl = modal.querySelector('.es-title');
    var noteEl = modal.querySelector('.es-note');

    if (currentItem) {
      // Per-item: tailored title, no static seed — generate on the fly.
      titleEl.textContent = opts.title || currentItem.name || 'In plain English';
      noteEl.textContent = 'A plain-English take on “' + (currentItem.name || 'this') + '”. Regenerate for another.';
      modal.classList.add('es-open');
      document.body.style.overflow = 'hidden';
      modal.querySelector('.es-close').focus();
      regenerate();   // immediately generate for this item
    } else {
      // Site-level: canonical default text, regenerate optional.
      titleEl.textContent = opts.title || 'What is Neural Nations?';
      noteEl.textContent = 'A friendly, jargon-free summary. Regenerate for a fresh take.';
      setProse(DEFAULT_TEXT);
      modal.classList.add('es-open');
      document.body.style.overflow = 'hidden';
      modal.querySelector('.es-close').focus();
    }
  }

  function close() {
    if (!modal) return;
    modal.classList.remove('es-open');
    document.body.style.overflow = '';
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  function pageContext() {
    var title = (document.title || '').replace(/\s*[\u2014\-|].*$/, '').trim();
    var h1 = document.querySelector('h1');
    var override = document.body.getAttribute('data-explain-context');
    return {
      page: title || 'Neural Nations',
      heading: h1 ? h1.textContent.trim() : '',
      hint: override || ''
    };
  }

  function regenerate() {
    var btn = modal.querySelector('.es-regen');
    var body = modal.querySelector('.es-body');
    if (btn.classList.contains('es-working')) return;
    btn.classList.add('es-working');
    btn.disabled = true;
    body.classList.add('es-busy');

    generate().then(function (paras) {
      setProse(paras);
    }).catch(function () {
      // graceful fallback to a pre-written variant
      setProse(FALLBACKS[fbIndex % FALLBACKS.length]);
      fbIndex++;
    }).then(function () {
      body.classList.remove('es-busy');
      btn.classList.remove('es-working');
      btn.disabled = false;
    });
  }

  // Build the instruction prompt (shared by all providers).
  function buildPrompt() {
    return currentItem ? buildItemPrompt(currentItem) : buildSitePrompt();
  }

  var CAMS_PRIMER = "Neural Nations is an open research site about CAMS — a framework that treats human societies as Complex Adaptive Systems and measures how well they coordinate, using eight core functions (leadership, security, knowledge, memory, production, labour, resource flow, and stewardship). When those functions stay connected a society is resilient; when they decouple, stress builds and coordination fails, often long before it is obvious. All data, methods and tools are published openly — honest measurement, not a grand theory.";

  // Per-item: explain one specific tool or paper from the Explore page.
  function buildItemPrompt(item) {
    var label = item.kind ? (item.kind.toLowerCase()) : 'item';
    return "Context — " + CAMS_PRIMER + "\n\n" +
      "On the site's Explore page there is a " + label + " titled:\n" +
      "\u201c" + item.name + "\u201d\n" +
      (item.desc ? ("Its catalogue blurb reads: \u201c" + item.desc + "\u201d\n") : "") +
      "\nWrite a warm, plain-English explanation of what THIS specific " + label + " is and why someone might open it. Audience: a curious non-expert. Rules:\n" +
      "- 90 to 140 words, 2 short paragraphs.\n" +
      "- No jargon, no equations, no Greek letters, no buzzwords. Decode any technical terms from the blurb into everyday language.\n" +
      "- Be concrete about what the reader will see, learn, or be able to do.\n" +
      "- Warm and clear, not promotional. Do not oversell.\n" +
      "- No greeting, no title, no markdown, no bullet points. Plain prose only.\n\n" +
      "Return only the explanation text.";
  }

  // Site-level: explain Neural Nations / CAMS in general.
  function buildSitePrompt() {
    var ctx = pageContext();
    return "You are writing a warm, accessible explainer for visitors to Neural Nations, an open research site about CAMS \u2014 a framework that treats human societies as Complex Adaptive Systems and measures how well they coordinate, using eight core functions (leadership, security, knowledge, memory, production, labour, resource flow, and stewardship).\n\n" +
      "Write a fresh, friendly, plain-English explanation of what this is and why it matters. Audience: a smart but non-expert newcomer. Rules:\n" +
      "- 150 to 200 words, in 3 short paragraphs.\n" +
      "- No jargon, no equations, no buzzwords. Warm and clear, not promotional.\n" +
      "- Make the eight functions feel concrete.\n" +
      "- Mention that the data, methods and tools are open and published \u2014 it is honest measurement, not a grand theory.\n" +
      "- Do not use a greeting, a title, markdown, or bullet points. Plain prose only.\n" +
      "- Vary the wording and angle from a textbook definition; give it a slightly fresh take each time.\n\n" +
      "Current page: \"" + ctx.page + "\"" +
      (ctx.heading ? " (heading: \"" + ctx.heading + "\")" : "") +
      (ctx.hint ? ". Extra context: " + ctx.hint : "") +
      ". Where natural, make the explanation feel relevant to this page.\n\n" +
      "Return only the explanation text.";
  }

  function toParas(text) {
    var paras = String(text || '').trim().split(/\n\s*\n+/)
      .map(function (s) { return s.replace(/\s+/g, ' ').trim(); })
      .filter(Boolean);
    if (!paras.length) throw new Error('empty');
    aiCache.push(paras);
    return paras;
  }

  // OpenAI-compatible chat completion — works for Kimi/Moonshot, Grok/xAI,
  // GPT/OpenAI, or any proxy that mirrors that shape.
  function fetchOpenAICompatible(prompt) {
    var headers = { 'Content-Type': 'application/json' };
    if (CONFIG.apiKey) headers['Authorization'] = 'Bearer ' + CONFIG.apiKey;
    return fetch(CONFIG.endpoint, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        model: CONFIG.model || 'gpt-4o-mini',
        temperature: typeof CONFIG.temperature === 'number' ? CONFIG.temperature : 0.85,
        max_tokens: 600,
        messages: [{ role: 'user', content: prompt }]
      })
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (data) {
      var msg = data && data.choices && data.choices[0] && data.choices[0].message;
      return toParas(msg && msg.content);
    });
  }

  // Returns a Promise<string[]> of paragraphs.
  // Provider order: configured endpoint (Kimi/Grok/GPT) → window.claude
  // (prototype only) → hand-written fallbacks.
  function generate() {
    var prompt = buildPrompt();

    if (CONFIG && CONFIG.endpoint) {
      return fetchOpenAICompatible(prompt);
    }

    var hasClaude = typeof window !== 'undefined' && window.claude &&
                    typeof window.claude.complete === 'function';
    if (hasClaude) {
      return window.claude.complete(prompt).then(toParas);
    }

    // No provider available — graceful, honest fallback.
    return new Promise(function (resolve) {
      setTimeout(function () {
        if (currentItem) {
          resolve(itemFallback(currentItem));
        } else {
          resolve(FALLBACKS[fbIndex % FALLBACKS.length]);
          fbIndex++;
        }
      }, 550);
    });
  }

  // When no AI provider is wired up, build a readable explanation for an
  // item from the catalogue text itself.
  function itemFallback(item) {
    var kind = (item.kind || 'item').toLowerCase();
    var p1 = '“' + item.name + '” is one of the ' + kind + 's published on Neural Nations’ Explore page' +
      (item.desc ? ', described there as: “' + item.desc + '”' : '') + '.';
    var p2 = 'In short: it applies the CAMS framework — which reads a society as eight working parts that have to stay in sync — to help you see how well those parts are holding together. Open it to explore the data and patterns for yourself; everything behind it is published in the open. (Connect a Kimi, Grok, or GPT key to generate a freshly tailored plain-English summary here.)';
    return [p1, p2];
  }

  // ── Per-item enhancement (Explore page) ─────────────────────
  // Wrap every catalogue card and attach a subtle "explain" trigger that
  // opens the modal with that item's own title + blurb as context.
  function badgeKind(a) {
    var b = a.querySelector('.badge');
    return b ? b.textContent.trim() : 'item';
  }
  function cardName(a) {
    // Title = the anchor's direct text, minus badges and the description span.
    var clone = a.cloneNode(true);
    clone.querySelectorAll('.badge, span[style], code, br').forEach(function (n) { n.remove(); });
    return (clone.textContent || '').replace(/\s+/g, ' ').trim();
  }
  function cardDesc(a) {
    var span = a.querySelector('span[style*="display:block"], span[style*="display: block"]');
    if (!span) return '';
    var c = span.cloneNode(true);
    c.querySelectorAll('a, code').forEach(function (n) {
      n.replaceWith(document.createTextNode(' ' + (n.textContent || '') + ' '));
    });
    return (c.textContent || '').replace(/\s+/g, ' ').trim();
  }

  function enhanceArtifacts() {
    var cards = document.querySelectorAll('.artifact-link');
    if (!cards.length) return;
    var EXPLAIN_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4z"/></svg>';
    cards.forEach(function (a) {
      if (a.getAttribute('data-es-enhanced')) return;
      a.setAttribute('data-es-enhanced', '1');

      var wrap = document.createElement('div');
      wrap.className = 'es-card';
      a.parentNode.insertBefore(wrap, a);
      wrap.appendChild(a);

      var item = { name: cardName(a), kind: badgeKind(a), desc: cardDesc(a) };

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'es-card-btn';
      btn.setAttribute('aria-label', 'Explain “' + item.name + '” in plain English');
      btn.innerHTML = EXPLAIN_ICON + '<span>Explain</span>';
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        open({ item: item });
      });
      wrap.appendChild(btn);
    });
  }

  function wire() {
    var triggers = document.querySelectorAll('[data-explain-simply]');
    for (var i = 0; i < triggers.length; i++) {
      triggers[i].addEventListener('click', function (e) {
        e.preventDefault();
        open();
      });
    }
    // Auto-enhance Explore-style catalogue cards, unless opted out.
    if (!(document.body && document.body.getAttribute('data-es-no-cards'))) {
      enhanceArtifacts();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
