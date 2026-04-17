#!/usr/bin/env python3
"""
CAMS National Mood, Vision-Affect, and Mythic Attractor Report Pipeline
Reads a CAMS CSV, computes all metrics, calls Claude API for narrative, outputs HTML.

Usage:
    python generate_report.py cleaned_datasets/France_1900_2026_cleaned.csv
    python generate_report.py data.csv --country "Germany" --snapshot 5 --output report.html

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set in the environment.
"""

import argparse
import csv
import statistics
import datetime
import json
import sys
from pathlib import Path
import anthropic

# ── CAMS constants ────────────────────────────────────────────────────────────

STANDARD_NODES = ['Helm', 'Shield', 'Archive', 'Lore', 'Stewards', 'Craft', 'Hands', 'Flow']

MYTHIC_MAP = {
    'Helm': 'King', 'Shield': 'Warrior', 'Archive': 'Library',
    'Lore': 'Temple', 'Stewards': 'Manor', 'Craft': 'Workshop',
    'Hands': 'Harvesters', 'Flow': 'Agora',
}

DELIBERATIVE_NODES = {'Archive', 'Lore', 'Stewards'}
REACTIVE_NODES     = {'Helm', 'Shield', 'Hands', 'Flow'}

# ── CSV loading ───────────────────────────────────────────────────────────────

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def load_dataset(path: str) -> tuple[str, list[dict]]:
    """Load CSV → (society_name, parsed_rows)."""
    raw = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            raw.append({k.strip(): v.strip() for k, v in row.items()})

    if not raw:
        print(f"ERROR: {path} is empty.")
        sys.exit(1)

    # Detect column names (case-insensitive)
    keys = list(raw[0].keys())
    def col(*candidates):
        for c in candidates:
            for k in keys:
                if k.lower() == c.lower():
                    return k
        return None

    c_society  = col('Society', 'Nation', 'Country')
    c_year     = col('Year', 'year')
    c_node     = col('Node', 'node')
    c_coh      = col('Coherence')
    c_cap      = col('Capacity')
    c_str      = col('Stress')
    c_abs      = col('Abstraction')
    c_nv       = col('Node Value', 'NodeValue')
    c_bs       = col('Bond Strength', 'BondStrength')

    parsed = []
    for r in raw:
        yr = safe_float(r.get(c_year or 'Year'))
        if yr is None:
            continue
        node = r.get(c_node or 'Node', '').strip()
        if not node:
            continue
        parsed.append({
            'society': r.get(c_society or 'Society', Path(path).stem),
            'year':    int(yr),
            'node':    node,
            'C':       safe_float(r.get(c_coh)),
            'K':       safe_float(r.get(c_cap)),
            'S':       safe_float(r.get(c_str)),
            'A':       safe_float(r.get(c_abs)),
            'NV':      safe_float(r.get(c_nv)),
            'B':       safe_float(r.get(c_bs)),
        })

    society = parsed[0]['society'] if parsed else Path(path).stem
    return society, parsed

# ── Metric computation ────────────────────────────────────────────────────────

def _avg(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return statistics.mean(vals) if vals else None

def node_averages(rows: list[dict], nodes: list[str]) -> dict:
    """Average all CAMS fields per node across given rows."""
    result = {}
    for node in nodes:
        nr = [r for r in rows if r['node'] == node]
        if not nr:
            continue
        result[node] = {f: _avg(nr, f) for f in ('C', 'K', 'S', 'A', 'NV', 'B')}
    return result

def add_node_operators(nd: dict) -> dict:
    """Add Vision, Affect, sigma to each node dict."""
    out = {}
    for node, v in nd.items():
        C, K, S, A = v.get('C'), v.get('K'), v.get('S'), v.get('A')
        V = (A * C) if A is not None and C is not None else None
        F = (K - S) if K is not None and S is not None else None
        sigma = (V * F) if V is not None and F is not None else None
        out[node] = {**v, 'V': V, 'F': F, 'sigma': sigma}
    return out

def system_scalars(nodes_ops: dict) -> dict:
    """Compute system-level scalars from node operator dict."""
    def mf(field, subset=None):
        vals = []
        for n, nd in nodes_ops.items():
            if subset and n not in subset:
                continue
            v = nd.get(field)
            if v is not None:
                vals.append(v)
        return statistics.mean(vals) if vals else None

    C_mean = mf('C'); K_mean = mf('K'); S_mean = mf('S'); A_mean = mf('A')
    NV_mean = mf('NV'); B_mean = mf('B')
    headroom  = mf('F')               # mean K-S across all nodes
    sys_vision = mf('V')              # mean A×C  ≈ Y(t)
    sys_sigma  = mf('sigma')
    M   = S_mean                      # Metabolic load = mean stress
    Y   = sys_vision                  # Mythic integration = mean vision
    psi = mf('F', DELIBERATIVE_NODES) # deliberative affect
    phi = mf('S', REACTIVE_NODES)     # reactive stress
    chi = (psi - phi) if psi is not None and phi is not None else None
    theta = (phi / psi) if psi and phi is not None else None
    lam = B_mean                      # coupling proxy = mean bond strength
    K_minus_C = (K_mean - C_mean) if K_mean and C_mean else None
    SAI = (A_mean / C_mean) if A_mean and C_mean else None

    hr = headroom or 0
    if   hr > 3:   classification = 'Resilient'
    elif hr > 1:   classification = 'Stable'
    elif hr > 0:   classification = 'Functional'
    elif hr > -1:  classification = 'Strained'
    elif hr > -2:  classification = 'Critical'
    else:          classification = 'Collapse Risk'

    return dict(
        C_mean=C_mean, K_mean=K_mean, S_mean=S_mean, A_mean=A_mean,
        NV_mean=NV_mean, B_mean=B_mean,
        M=M, Y=Y, headroom=headroom, sys_vision=sys_vision, sys_sigma=sys_sigma,
        psi=psi, phi=phi, chi=chi, theta=theta, lam=lam,
        K_minus_C=K_minus_C, SAI=SAI, SR=headroom, H=NV_mean,
        classification=classification,
    )

def historical_portrait(all_rows: list[dict], nodes: list[str], n_eras: int = 5) -> list[dict]:
    """Divide dataset into eras and compute key scalars per era."""
    years = sorted(set(r['year'] for r in all_rows))
    if len(years) < 2:
        return []
    span = years[-1] - years[0]
    era_size = max(10, span // n_eras)
    eras = []
    start = years[0]
    while start <= years[-1]:
        end = min(start + era_size - 1, years[-1])
        era_rows = [r for r in all_rows if start <= r['year'] <= end and r['node'] in nodes]
        if era_rows:
            nd = add_node_operators(node_averages(era_rows, nodes))
            sc = system_scalars(nd)
            eras.append({
                'era': f"{start}–{end}",
                'M': sc['M'], 'Y': sc['Y'], 'headroom': sc['headroom'],
                'lam': sc['lam'], 'theta': sc['theta'],
            })
        start = end + 1
    return eras

def data_summary(country: str, snapshot_years: list[int],
                 nodes_ops: dict, sc: dict, hist: list[dict]) -> str:
    """Build text block passed to Claude."""
    f = lambda v, d=2: f"{v:.{d}f}" if v is not None else "N/A"

    lines = [
        f"COUNTRY: {country}",
        f"SNAPSHOT PERIOD: {min(snapshot_years)}–{max(snapshot_years)}",
        "",
        "=== SYSTEM SNAPSHOT ===",
        f"Mean Coherence: {f(sc['C_mean'])}",
        f"Mean Capacity: {f(sc['K_mean'])}",
        f"Mean Stress: {f(sc['S_mean'])}",
        f"Mean Abstraction: {f(sc['A_mean'])}",
        f"Mean Node Value: {f(sc['NV_mean'])}",
        f"Mean Bond Strength: {f(sc['B_mean'])}",
        f"Stress Resilience (K-S): {f(sc['SR'])}",
        f"System Vision (mean A×C): {f(sc['sys_vision'])}",
        f"System Sigma (mean σ): {f(sc['sys_sigma'])}",
        f"Classification: {sc['classification']}",
        "",
        "=== SCALAR OPERATORS ===",
        f"M(t) Metabolic Load: {f(sc['M'])}",
        f"Y(t) Mythic Integration: {f(sc['Y'])}",
        f"Headroom (mean K-S): {f(sc['headroom'])}",
        f"Ψ Deliberative field (Archive+Lore+Stewards affect): {f(sc['psi'])}",
        f"Φ Reactive field (Helm+Shield+Hands+Flow stress): {f(sc['phi'])}",
        f"χ = Ψ - Φ: {f(sc['chi'])}",
        f"Θ = Φ/Ψ: {f(sc['theta'])}",
        f"Λ coupling proxy (mean Bond Strength): {f(sc['lam'])}",
        f"K-C delta: {f(sc['K_minus_C'])}",
        f"SAI (A/C): {f(sc['SAI'])}",
        "",
        "=== NODE TABLE (Vision = A×C, Affect = K-S, σ = Vision × Affect) ===",
    ]
    for node in STANDARD_NODES:
        if node not in nodes_ops:
            continue
        nd = nodes_ops[node]
        attractor = MYTHIC_MAP.get(node, node)
        lines.append(
            f"{node} ({attractor}): "
            f"C={f(nd.get('C'))} K={f(nd.get('K'))} S={f(nd.get('S'))} A={f(nd.get('A'))} "
            f"NV={f(nd.get('NV'))} B={f(nd.get('B'))} "
            f"Vision={f(nd.get('V'))} Affect={f(nd.get('F'))} σ={f(nd.get('sigma'))}"
        )

    sigmas = [(n, nodes_ops[n]['sigma']) for n in STANDARD_NODES
              if n in nodes_ops and nodes_ops[n].get('sigma') is not None]
    sigmas.sort(key=lambda x: x[1], reverse=True)
    lines += ["", "=== SIGMA RANKING (highest to lowest) ==="]
    lines += [f"  {n} ({MYTHIC_MAP.get(n,n)}): σ={f(s)}" for n, s in sigmas]
    sv = ", ".join(f(nodes_ops[n].get('sigma')) for n in STANDARD_NODES if n in nodes_ops)
    lines += ["", f"System Vector P({max(snapshot_years)}) = ({sv})"]

    if hist:
        lines += ["", "=== HISTORICAL PORTRAIT ==="]
        lines += [f"  {e['era']}: M={f(e['M'])} Y={f(e['Y'])} "
                  f"Headroom={f(e['headroom'])} Θ={f(e['theta'])}" for e in hist]

    return "\n".join(lines)

# ── Claude API narrative generation ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You generate the narrative and interpretive sections of a CAMS National Mood, \
Vision-Affect, and Mythic Attractor Report.

CAMS treats civilisations as eight coupled institutional nodes:
  Helm (King) — executive coordination, strategic decision-making
  Shield (Warrior) — defence, coercive capacity, security apparatus
  Archive (Library) — institutional memory, records, regulatory continuity
  Lore (Temple) — shared meaning, legitimacy narratives, cultural identity
  Stewards (Manor) — asset control, resource stewardship, capital
  Craft (Workshop) — technical expertise, specialised knowledge, innovation
  Hands (Harvesters) — labour base, demographic health, motivational engagement
  Flow (Agora) — circulatory exchange, logistics, commerce, information

All metrics are on a 1-10 scale (or 1-100 for Bond Strength):
  C = Coherence  K = Capacity  S = Stress  A = Abstraction
  Vision = A × C  |  Affect = K − S  |  σ = Vision × Affect

System scalars:
  M(t) = mean Stress (metabolic load)
  Y(t) = mean Vision (mythic integration)
  Headroom = mean(K−S)
  Ψ = mean Affect of deliberative nodes (Archive, Lore, Stewards)
  Φ = mean Stress of reactive nodes (Helm, Shield, Hands, Flow)
  χ = Ψ − Φ   Θ = Φ/Ψ   Λ = mean Bond Strength

Tone reference: The France 2026 report opened with "The Library remembers \
perfectly; the Temple has lost its congregation; the King cannot be heard." \
Match that quality — analytical and precise, but willing to be genuinely poetic \
and specific. Ground every interpretive claim in the actual numbers.

Return ONLY a JSON object with these exact keys:

exec_phrase: string — the one-sentence structural image (poetic opener)
exec_summary: {
  classification_label, strongest_nodes, weakest_nodes,
  key_anomaly, system_condition,
  structural_reading, mood_reading, mythic_pole_a, mythic_pole_b
}
vision_affect_reading: string — section 6 narrative (3-5 paragraphs)
structural_dynamics: {
  primary_anomaly, dominant_anchor, main_weakness, archive_lore, fast_slow
}
mythopoetic: {
  mood_texture, mood_expanded, governing_metaphor, metaphor_explanation,
  persona, persona_why, gift, wound, gift_wound_reading,
  mythic_tension_a, mythic_tension_b, mythic_tension_interp,
  mythic_close
}
mythic_alignment: array of 8 objects, one per node in order \
[Helm, Shield, Archive, Lore, Stewards, Craft, Hands, Flow]:
  { node, attractor, state, reading }
  state must be one of: integrated | thinning | overloaded | hollowed out |
  captured | decoupled | weaponised | latent | renewing
implications: {
  strength,
  risk_1_title, risk_1,
  risk_2_title, risk_2,
  risk_3_title, risk_3,
  opportunity
}
final_assessment: {
  classification_label, headline_strength, headline_vulnerability,
  dominant_attractor, endangered_attractor,
  decisive_relation, threatening_process, final_line
}

Return the JSON object only — no preamble, no code fences.
"""

def generate_narrative(client: anthropic.Anthropic, summary: str, country: str) -> dict:
    """Stream Claude API call; return parsed JSON narrative dict."""
    user_msg = (
        f"Here is the fully computed CAMS data for {country}:\n\n"
        f"{summary}\n\n"
        "Generate all narrative sections. Be specific and evocative. "
        "Every claim must be grounded in the numbers above. "
        "Return only the JSON object."
    )

    text = ""
    print("\n── Claude narrative (streaming) ──────────────────────────────")
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk
            print(chunk, end="", flush=True)
    print("\n────────────────────────────────────────────────────────────\n")

    # Extract JSON
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON found in Claude response. Raw output:\n" + text[:500])
    return json.loads(text[start:end])

# ── HTML rendering ────────────────────────────────────────────────────────────

def fv(v, d=2):
    """Format a float value or return em-dash."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)

def render_html(country: str, period: str, snapshot_years: list[int],
                nodes_ops: dict, sc: dict, hist: list[dict],
                narr: dict, date_str: str, csv_name: str) -> str:

    ex  = narr.get('exec_summary', {})
    vis = narr.get('vision_affect_reading', '')
    st  = narr.get('structural_dynamics', {})
    my  = narr.get('mythopoetic', {})
    mal = narr.get('mythic_alignment', [])
    imp = narr.get('implications', {})
    fin = narr.get('final_assessment', {})

    # Node table rows
    node_rows = ""
    for node in STANDARD_NODES:
        if node not in nodes_ops:
            continue
        nd  = nodes_ops[node]
        sig = nd.get('sigma')
        sc_class = ("positive" if sig is not None and sig > 0
                    else "negative" if sig is not None and sig < 0 else "")
        node_rows += (
            f"<tr><td><strong>{node}</strong></td>"
            f"<td>{MYTHIC_MAP.get(node, node)}</td>"
            f"<td>{fv(nd.get('C'))}</td><td>{fv(nd.get('K'))}</td>"
            f"<td>{fv(nd.get('S'))}</td><td>{fv(nd.get('A'))}</td>"
            f"<td>{fv(nd.get('NV'))}</td><td>{fv(nd.get('B'))}</td>"
            f"<td>{fv(nd.get('V'))}</td><td>{fv(nd.get('F'))}</td>"
            f"<td class='{sc_class}'><strong>{fv(sig)}</strong></td></tr>\n"
        )

    # Mythic alignment rows
    mal_rows = ""
    for item in mal:
        mal_rows += (
            f"<tr><td><strong>{item.get('node','')}</strong></td>"
            f"<td>{item.get('attractor','')}</td>"
            f"<td><em>{item.get('state','')}</em></td>"
            f"<td>{item.get('reading','')}</td></tr>\n"
        )

    # Historical rows
    hist_rows = ""
    for e in hist:
        hist_rows += (
            f"<tr><td>{e['era']}</td><td>{fv(e.get('M'))}</td>"
            f"<td>{fv(e.get('Y'))}</td><td>{fv(e.get('headroom'))}</td>"
            f"<td>{fv(e.get('lam'))}</td><td>{fv(e.get('theta'))}</td></tr>\n"
        )

    sv = ", ".join(fv(nodes_ops[n].get('sigma')) for n in STANDARD_NODES if n in nodes_ops)
    system_vector = f"P({max(snapshot_years)}) ≈ ({sv})"

    exec_phrase = narr.get('exec_phrase', '')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{country} — CAMS Mood Report {period}</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
body{{font-family:'Cormorant Garamond',serif;line-height:1.9;color:#2c2c2c;background:#f8f7f4;margin:0;padding:40px 20px}}
.wrap{{max-width:880px;margin:0 auto;background:#fff;padding:64px 72px;border-radius:4px;box-shadow:0 4px 24px rgba(0,0,0,.07)}}
h1{{font-size:2.7em;color:#1a1a2e;margin:0 0 8px;font-weight:600;letter-spacing:-.02em}}
.series{{font-family:'Inter',sans-serif;font-size:.78em;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#999;margin-bottom:36px}}
.exec-phrase{{font-size:1.35em;font-style:italic;color:#1a3a52;border-left:4px solid #1a3a52;padding:14px 22px;margin:28px 0 38px;background:#f0f5fa;border-radius:0 6px 6px 0}}
h2{{font-family:'Inter',sans-serif;font-size:.73em;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#999;margin:48px 0 14px;border-top:1px solid #eee;padding-top:26px}}
h3{{font-family:'Inter',sans-serif;font-size:.82em;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#555;margin:26px 0 8px}}
p{{margin:1.3em 0;font-size:1.04em}}
table{{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:.8em;margin:16px 0 24px}}
th{{background:#1a1a2e;color:#fff;padding:8px 10px;text-align:left;font-weight:600;font-size:.76em;letter-spacing:.04em}}
td{{padding:7px 10px;border-bottom:1px solid #eee;vertical-align:top}}
tr:nth-child(even) td{{background:#fafafa}}
td.positive{{color:#059669;font-weight:700}}
td.negative{{color:#dc2626;font-weight:700}}
.divider{{text-align:center;color:#bbb;font-size:1.3em;margin:34px 0}}
.myth-close{{font-size:1.22em;font-style:italic;color:#1a3a52;text-align:center;padding:18px;margin:28px 0;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.final-line{{font-size:1.28em;font-style:italic;color:#1a1a2e;text-align:center;padding:20px;margin:28px 0;background:#f0f5fa;border-radius:6px}}
.vector{{font-family:'Inter',sans-serif;font-size:.76em;color:#666;background:#f8f8f8;padding:9px 13px;border-radius:4px;margin:8px 0 18px;word-break:break-all}}
.footer{{font-family:'Inter',sans-serif;font-size:.73em;color:#bbb;margin-top:48px;padding-top:18px;border-top:1px solid #eee;line-height:1.9}}
strong{{color:#1a1a2e}}
@media print{{body{{background:#fff;padding:0}}.wrap{{box-shadow:none;padding:40px}}}}
</style>
</head>
<body>
<div class="wrap">

<p class="series">Neural Nations CAMS Ensemble &middot; v3.2-R.2-VA &middot; {date_str}</p>
<h1>{country}</h1>
<p class="series">National Mood, Vision&#8209;Affect &amp; Mythic Attractor Report &mdash; {period}</p>

<div class="exec-phrase">&ldquo;{exec_phrase}&rdquo;</div>

<h2>1. Executive Summary</h2>
<p>{country} in {period} presents as a <strong>{ex.get('classification_label', sc['classification'])}</strong> system.</p>
<p>Its strongest nodes are <strong>{ex.get('strongest_nodes','')}</strong>.</p>
<p>Its weakest or most burdened nodes are <strong>{ex.get('weakest_nodes','')}</strong>.</p>
<p>The principal anomaly is <strong>{ex.get('key_anomaly','')}</strong>.</p>
<p>The system is best described as <strong>{ex.get('system_condition','')}</strong>.</p>
<p>At the level of structure, the society is {ex.get('structural_reading','')}.</p>
<p>At the level of mood, it feels like {ex.get('mood_reading','')}.</p>
<p>At the level of myth, it is a struggle between <strong>{ex.get('mythic_pole_a','')}</strong> and <strong>{ex.get('mythic_pole_b','')}.</strong></p>

<h2>2. Data Source &amp; Method Notes</h2>
<p><strong>Dataset:</strong> {csv_name}<br>
<strong>Snapshot rule:</strong> Average of last {len(snapshot_years)} years ({min(snapshot_years)}&ndash;{max(snapshot_years)})<br>
<strong>Vision&ndash;Affect:</strong> Vision = A&times;C &nbsp;&middot;&nbsp; Affect = K&minus;S &nbsp;&middot;&nbsp; &sigma; = Vision &times; Affect<br>
<strong>Stress convention:</strong> Positive = strain &nbsp;&middot;&nbsp;
<strong>Node mapping:</strong> Standard CAMS (Helm, Shield, Archive, Lore, Stewards, Craft, Hands, Flow)</p>

<h2>3. System Snapshot</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean Coherence</td><td>{fv(sc['C_mean'])}</td></tr>
<tr><td>Mean Capacity</td><td>{fv(sc['K_mean'])}</td></tr>
<tr><td>Mean Stress</td><td>{fv(sc['S_mean'])}</td></tr>
<tr><td>Mean Abstraction</td><td>{fv(sc['A_mean'])}</td></tr>
<tr><td>Mean Node Value</td><td>{fv(sc['NV_mean'])}</td></tr>
<tr><td>Mean Bond Strength</td><td>{fv(sc['B_mean'])}</td></tr>
<tr><td>Stress Resilience (K&minus;S)</td><td>{fv(sc['SR'])}</td></tr>
<tr><td>System Vision (mean A&times;C)</td><td>{fv(sc['sys_vision'])}</td></tr>
<tr><td>System Sigma</td><td>{fv(sc['sys_sigma'])}</td></tr>
<tr><td>Adjusted System Health (mean NV)</td><td>{fv(sc['H'])}</td></tr>
<tr><td>Classification</td><td><strong>{sc['classification']}</strong></td></tr>
</table>

<h2>4. System Scalar Operators</h2>
<table>
<tr><th>Operator</th><th>Value</th></tr>
<tr><td>M(t) &mdash; Metabolic Load (mean S)</td><td>{fv(sc['M'])}</td></tr>
<tr><td>Y(t) &mdash; Mythic Integration (mean A&times;C)</td><td>{fv(sc['Y'])}</td></tr>
<tr><td>Mean Headroom (K&minus;S)</td><td>{fv(sc['headroom'])}</td></tr>
<tr><td>&Psi; &mdash; Deliberative field (Archive+Lore+Stewards affect)</td><td>{fv(sc['psi'])}</td></tr>
<tr><td>&Phi; &mdash; Reactive field (Helm+Shield+Hands+Flow stress)</td><td>{fv(sc['phi'])}</td></tr>
<tr><td>&chi; = &Psi; &minus; &Phi;</td><td>{fv(sc['chi'])}</td></tr>
<tr><td>&Theta; = &Phi;/&Psi; (reactive/deliberative ratio)</td><td>{fv(sc['theta'])}</td></tr>
<tr><td>&Lambda; &mdash; Coupling proxy (mean Bond Strength)</td><td>{fv(sc['lam'])}</td></tr>
<tr><td>K&minus;C Delta</td><td>{fv(sc['K_minus_C'])}</td></tr>
<tr><td>SAI (A/C ratio)</td><td>{fv(sc['SAI'])}</td></tr>
</table>

<h2>5. Current Node Table</h2>
<table>
<tr><th>Node</th><th>Attractor</th><th>C</th><th>K</th><th>S</th><th>A</th>
<th>NV</th><th>B</th><th>Vision (A&times;C)</th><th>Affect (K&minus;S)</th><th>&sigma;</th></tr>
{node_rows}
</table>
<div class="vector">{system_vector}</div>

<div class="divider">&starf;</div>

<h2>6. Vision&ndash;Affect Reading</h2>
<p>{vis}</p>

<h2>7. Structural Dynamics</h2>
<h3>7.1 Primary Anomaly</h3><p>{st.get('primary_anomaly','')}</p>
<h3>7.2 Dominant Anchor</h3><p>{st.get('dominant_anchor','')}</p>
<h3>7.3 Main Weakness</h3><p>{st.get('main_weakness','')}</p>
<h3>7.4 Archive&ndash;Lore Relation</h3><p>{st.get('archive_lore','')}</p>
<h3>7.5 Fast-Loop / Slow-Loop Relation</h3><p>{st.get('fast_slow','')}</p>

<h2>8. Mythopoetic Layer</h2>
<h3>8.1 National Mood Texture</h3>
<p>{country} currently feels like <strong>{my.get('mood_texture','')}</strong>.</p>
<p>{my.get('mood_expanded','')}</p>
<h3>8.2 Governing Metaphor</h3>
<p><strong>{country} is {my.get('governing_metaphor','')}.</strong></p>
<p>{my.get('metaphor_explanation','')}</p>
<h3>8.3 Civilisational Persona</h3>
<p>If this system were a character, it would be <strong>{my.get('persona','')}</strong>.</p>
<p>{my.get('persona_why','')}</p>
<h3>8.4 Gift and Wound</h3>
<p><strong>Civilisational gift:</strong> {my.get('gift','')}<br>
<strong>Civilisational wound:</strong> {my.get('wound','')}</p>
<p>{my.get('gift_wound_reading','')}</p>
<h3>8.5 Mythic Tension</h3>
<p>This period is a struggle between <strong>{my.get('mythic_tension_a','')}</strong>
and <strong>{my.get('mythic_tension_b','')}.</strong></p>
<p>{my.get('mythic_tension_interp','')}</p>
<h3>8.6 Mythic Alignment by Node</h3>
<table>
<tr><th>Node</th><th>Attractor</th><th>State</th><th>Reading</th></tr>
{mal_rows}
</table>
<div class="myth-close">&ldquo;{my.get('mythic_close','')}&rdquo;</div>

<h2>9. Historical Phase Portrait</h2>
<table>
<tr><th>Era</th><th>M(t)</th><th>Y(t)</th><th>Headroom</th><th>&Lambda;</th><th>&Theta;</th></tr>
{hist_rows}
</table>

<h2>11. Implications, Risk &amp; Opportunity</h2>
<h3>Strength</h3><p>{imp.get('strength','')}</p>
<h3>Risk 1 &mdash; {imp.get('risk_1_title','')}</h3><p>{imp.get('risk_1','')}</p>
<h3>Risk 2 &mdash; {imp.get('risk_2_title','')}</h3><p>{imp.get('risk_2','')}</p>
<h3>Risk 3 &mdash; {imp.get('risk_3_title','')}</h3><p>{imp.get('risk_3','')}</p>
<h3>Opportunity</h3><p>{imp.get('opportunity','')}</p>

<h2>12. Final Assessment</h2>
<p>{country} is a <strong>{fin.get('classification_label', sc['classification'])}</strong>
system with <strong>{fin.get('headline_strength','')}</strong>
but also <strong>{fin.get('headline_vulnerability','')}.</strong></p>
<p>Its strongest civilisational image is <strong>{fin.get('dominant_attractor','')}.</strong><br>
Its most endangered image is <strong>{fin.get('endangered_attractor','')}.</strong><br>
The decisive question is whether <strong>{fin.get('decisive_relation','')}</strong>
can be re-coupled before <strong>{fin.get('threatening_process','')}</strong> hardens into structure.</p>
<div class="final-line">&ldquo;{fin.get('final_line','')}&rdquo;</div>

<div class="footer">
Report generated: {date_str}<br>
Model: claude-opus-4-6 (Neural Nations CAMS Ensemble v3.2-R.2-VA)<br>
Dataset: {csv_name} &nbsp;&middot;&nbsp; Snapshot: {min(snapshot_years)}&ndash;{max(snapshot_years)} ({len(snapshot_years)} years)<br>
Neural Nations CAMS Research &middot; Open Science (Common Property) &middot; Co-authored with Claude Sonnet&nbsp;4.6 (Anthropic)
</div>

</div>
</body>
</html>"""

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Generate a CAMS National Mood Report from a CSV.')
    ap.add_argument('csv', help='Path to CAMS CSV file')
    ap.add_argument('--country',  help='Country name override')
    ap.add_argument('--snapshot', type=int, default=3,
                    help='Number of most-recent years to average (default: 3)')
    ap.add_argument('--output',   help='Output HTML path (default: auto-named in script dir)')
    args = ap.parse_args()

    # Load
    print(f"Loading {args.csv} ...")
    society, rows = load_dataset(args.csv)
    country = args.country or society

    # Determine which standard nodes are present
    present = set(r['node'] for r in rows)
    nodes   = [n for n in STANDARD_NODES if n in present]
    if not nodes:
        print(f"ERROR: None of the standard CAMS nodes found.\nPresent: {present}")
        sys.exit(1)

    all_years = sorted(set(r['year'] for r in rows))
    print(f"Country : {country}")
    print(f"Nodes   : {nodes}")
    print(f"Years   : {all_years[0]}–{all_years[-1]} ({len(all_years)} years)")
    print(f"Snapshot: last {args.snapshot} year(s)")

    # Compute
    snap_years = all_years[-args.snapshot:]
    snap_rows  = [r for r in rows if r['year'] in snap_years]
    nd_avg     = node_averages(snap_rows, nodes)
    nd_ops     = add_node_operators(nd_avg)
    sc         = system_scalars(nd_ops)
    hist       = historical_portrait(rows, nodes)
    period     = f"{min(snap_years)}–{max(snap_years)}"

    print(f"\nClassification : {sc['classification']}")
    print(f"Headroom       : {fv(sc['headroom'])}")
    print(f"System Sigma   : {fv(sc['sys_sigma'])}")

    # Claude API
    summary = data_summary(country, snap_years, nd_ops, sc, hist)
    client  = anthropic.Anthropic()
    narr    = generate_narrative(client, summary, country)

    # Render HTML
    date_str = datetime.date.today().isoformat()
    html     = render_html(country, period, snap_years, nd_ops, sc, hist,
                           narr, date_str, Path(args.csv).name)

    # Save
    if args.output:
        out_path = args.output
    else:
        stem     = f"{country.replace(' ','_')}_{max(snap_years)}_CAMS_Report"
        out_path = str(Path(__file__).parent / f"{stem}.html")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Report written to: {out_path}")

if __name__ == '__main__':
    main()
