#!/usr/bin/env python3
"""
CAMS National Mood, Vision-Affect, and Mythic Attractor Report Pipeline
Reads a CAMS CSV, computes all metrics, calls Claude API for narrative, outputs HTML.

Usage:
    python generate_report.py cleaned_datasets/France_1900_2026_cleaned.csv
    python generate_report.py data.csv --country "Germany" --snapshot 5
    python generate_report.py data.csv --manifest mindscapes/reports-manifest.json

Flags:
    --country   Override country name detected from CSV
    --snapshot  Years to average for snapshot (default: 3)
    --output    Output HTML path (default: auto-named in script dir)
    --manifest  Path to reports-manifest.json (enables section 10 + auto-update)
    --assessor  Who scored the data (default: "CAMS Ensemble")
    --flag      Stress polarity flag (default: "positive = strain")

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

NODE_GRAMMAR = {
    'Helm':     {'organic': 'Central nervous system',   'function': 'Executive coordination',    'failure': 'Paralysis / capture',              'timescale': 'Years–decades'},
    'Shield':   {'organic': 'Immune + musculoskeletal', 'function': 'Defence / order',           'failure': 'Atrophy or hyperactivation',        'timescale': 'Years–decades'},
    'Archive':  {'organic': 'Long-term memory',         'function': 'Institutional record',      'failure': 'Amnesia or rigidity',              'timescale': 'Decades–centuries'},
    'Lore':     {'organic': 'Proprioception',           'function': 'Meaning / legitimacy',      'failure': 'Normative collapse / capture',     'timescale': 'Centuries'},
    'Stewards': {'organic': 'Connective tissue',        'function': 'Asset stewardship',         'failure': 'Rent-seeking / decay',             'timescale': 'Decades'},
    'Craft':    {'organic': 'Specialised organs',       'function': 'Technical expertise',       'failure': 'Skill erosion',                    'timescale': 'Decades'},
    'Hands':    {'organic': 'Cellular mass',            'function': 'Labour base',               'failure': 'Demographic / motivational failure','timescale': 'Years'},
    'Flow':     {'organic': 'Circulatory system',       'function': 'Exchange / logistics',      'failure': 'Mythic-material decoupling',        'timescale': 'Months–decades'},
}

DELIBERATIVE_NODES = {'Archive', 'Lore', 'Stewards'}
REACTIVE_NODES     = {'Helm', 'Shield', 'Hands', 'Flow'}

TIER_MAP = {
    'Resilient':     'resilient',
    'Stable':        'stable',
    'Functional':    'stable',
    'Strained':      'strained',
    'Critical':      'critical',
    'Collapse Risk': 'critical',
}

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

    keys = list(raw[0].keys())
    def col(*candidates):
        for c in candidates:
            for k in keys:
                if k.lower() == c.lower():
                    return k
        return None

    c_society = col('Society', 'Nation', 'Country')
    c_year    = col('Year', 'year')
    c_node    = col('Node', 'node')
    c_coh     = col('Coherence')
    c_cap     = col('Capacity')
    c_str     = col('Stress')
    c_abs     = col('Abstraction')
    c_nv      = col('Node Value', 'NodeValue')
    c_bs      = col('Bond Strength', 'BondStrength')

    parsed = []
    for r in raw:
        yr   = safe_float(r.get(c_year or 'Year'))
        node = r.get(c_node or 'Node', '').strip()
        if yr is None or not node:
            continue
        parsed.append({
            'society': r.get(c_society or 'Society', Path(path).stem),
            'year': int(yr), 'node': node,
            'C': safe_float(r.get(c_coh)), 'K': safe_float(r.get(c_cap)),
            'S': safe_float(r.get(c_str)), 'A': safe_float(r.get(c_abs)),
            'NV': safe_float(r.get(c_nv)), 'B': safe_float(r.get(c_bs)),
        })

    society = parsed[0]['society'] if parsed else Path(path).stem
    return society, parsed

# ── Metric computation ────────────────────────────────────────────────────────

def _avg(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return statistics.mean(vals) if vals else None

def node_averages(rows, nodes):
    result = {}
    for node in nodes:
        nr = [r for r in rows if r['node'] == node]
        if nr:
            result[node] = {f: _avg(nr, f) for f in ('C','K','S','A','NV','B')}
    return result

def add_node_operators(nd):
    out = {}
    for node, v in nd.items():
        C, K, S, A = v.get('C'), v.get('K'), v.get('S'), v.get('A')
        V     = (A * C) if A is not None and C is not None else None
        F     = (K - S) if K is not None and S is not None else None
        sigma = (V * F) if V is not None and F is not None else None
        out[node] = {**v, 'V': V, 'F': F, 'sigma': sigma}
    return out

def system_scalars(nodes_ops):
    def mf(field, subset=None):
        vals = [nd[field] for n, nd in nodes_ops.items()
                if (not subset or n in subset) and nd.get(field) is not None]
        return statistics.mean(vals) if vals else None

    C_mean = mf('C'); K_mean = mf('K'); S_mean = mf('S'); A_mean = mf('A')
    NV_mean = mf('NV'); B_mean = mf('B')
    headroom   = mf('F')
    sys_vision = mf('V')
    sys_sigma  = mf('sigma')
    M     = S_mean
    Y     = sys_vision
    psi   = mf('F', DELIBERATIVE_NODES)
    phi   = mf('S', REACTIVE_NODES)
    chi   = (psi - phi) if psi is not None and phi is not None else None
    theta = (phi / psi)  if psi and phi is not None and psi != 0 else None
    lam   = B_mean
    K_minus_C = (K_mean - C_mean) if K_mean and C_mean else None
    SAI       = (A_mean / C_mean) if A_mean and C_mean else None

    hr = headroom or 0
    if   hr > 3:  classification = 'Resilient'
    elif hr > 1:  classification = 'Stable'
    elif hr > 0:  classification = 'Functional'
    elif hr > -1: classification = 'Strained'
    elif hr > -2: classification = 'Critical'
    else:         classification = 'Collapse Risk'

    return dict(
        C_mean=C_mean, K_mean=K_mean, S_mean=S_mean, A_mean=A_mean,
        NV_mean=NV_mean, B_mean=B_mean,
        M=M, Y=Y, headroom=headroom, sys_vision=sys_vision, sys_sigma=sys_sigma,
        psi=psi, phi=phi, chi=chi, theta=theta, lam=lam,
        K_minus_C=K_minus_C, SAI=SAI, SR=headroom, H=NV_mean,
        classification=classification,
    )

def historical_portrait(all_rows, nodes, n_eras=5):
    years = sorted(set(r['year'] for r in all_rows))
    if len(years) < 2:
        return []
    span     = years[-1] - years[0]
    era_size = max(10, span // n_eras)
    eras, start = [], years[0]
    while start <= years[-1]:
        end      = min(start + era_size - 1, years[-1])
        era_rows = [r for r in all_rows if start <= r['year'] <= end and r['node'] in nodes]
        if era_rows:
            nd = add_node_operators(node_averages(era_rows, nodes))
            sc = system_scalars(nd)
            # Posture from classification
            hr = sc['headroom'] or 0
            sig = sc['sys_sigma'] or 0
            if   hr > 3 and sig > 100:  posture = 'Generative'
            elif hr > 1:                posture = 'Stable'
            elif hr > 0:                posture = 'Functional'
            elif hr > -1:               posture = 'Strained'
            else:                       posture = 'Critical'
            eras.append({'era': f"{start}–{end}", 'M': sc['M'], 'Y': sc['Y'],
                         'headroom': sc['headroom'], 'lam': sc['lam'],
                         'theta': sc['theta'], 'posture': posture})
        start = end + 1
    return eras

# ── Data summary for Claude ───────────────────────────────────────────────────

def data_summary(country, snapshot_years, nodes_ops, sc, hist, peers=None):
    f = lambda v, d=2: f"{v:.{d}f}" if v is not None else "N/A"

    lines = [
        f"COUNTRY: {country}",
        f"SNAPSHOT PERIOD: {min(snapshot_years)}–{max(snapshot_years)}",
        "",
        "=== CANONICAL NODE GRAMMAR ===",
        "Node | Organic analogue | Primary function | Mythic attractor | Failure mode | Timescale",
    ]
    for n in STANDARD_NODES:
        g = NODE_GRAMMAR[n]
        lines.append(f"  {n} ({MYTHIC_MAP[n]}) | {g['organic']} | {g['function']} | {g['failure']} | {g['timescale']}")

    lines += [
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
        f"K-C Delta: {f(sc['K_minus_C'])}",
        f"SAI (A/C): {f(sc['SAI'])}",
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
        "",
        "=== NODE TABLE (Vision=A×C  Affect=K-S  σ=Vision×Affect) ===",
    ]
    for node in STANDARD_NODES:
        if node not in nodes_ops:
            continue
        nd = nodes_ops[node]
        g  = NODE_GRAMMAR[node]
        # Infer likely failure mode from metrics
        sig = nd.get('sigma') or 0
        aff = nd.get('F') or 0
        vis = nd.get('V') or 0
        if   sig > 100:               mode_hint = 'integrated'
        elif sig > 0:                 mode_hint = 'latent'
        elif aff < -2:                mode_hint = 'hollowed out'
        elif aff < 0 and vis < 40:    mode_hint = 'decoupled'
        elif aff < 0:                 mode_hint = 'thinning'
        else:                         mode_hint = 'overloaded'
        lines.append(
            f"  {node} ({MYTHIC_MAP[node]}) organic={g['organic']} failure-mode={g['failure']} "
            f"timescale={g['timescale']} hint={mode_hint} | "
            f"C={f(nd.get('C'))} K={f(nd.get('K'))} S={f(nd.get('S'))} A={f(nd.get('A'))} "
            f"NV={f(nd.get('NV'))} B={f(nd.get('B'))} "
            f"Vision={f(nd.get('V'))} Affect={f(nd.get('F'))} σ={f(nd.get('sigma'))}"
        )

    sigmas = [(n, nodes_ops[n]['sigma']) for n in STANDARD_NODES
              if n in nodes_ops and nodes_ops[n].get('sigma') is not None]
    sigmas.sort(key=lambda x: x[1], reverse=True)
    lines += ["", "=== SIGMA RANKING ==="]
    lines += [f"  {n} ({MYTHIC_MAP.get(n,n)}): σ={f(s)}" for n, s in sigmas]
    sv = ", ".join(f(nodes_ops[n].get('sigma')) for n in STANDARD_NODES if n in nodes_ops)
    lines += ["", f"System Vector P({max(snapshot_years)}) = ({sv})"]

    if hist:
        lines += ["", "=== HISTORICAL PORTRAIT ==="]
        lines += [f"  {e['era']}: M={f(e['M'])} Y={f(e['Y'])} "
                  f"Headroom={f(e['headroom'])} Θ={f(e['theta'])} Posture={e['posture']}"
                  for e in hist]

    if peers:
        lines += ["", "=== PEER SOCIETIES (from manifest) ==="]
        for p in peers:
            m = p.get('metrics', {})
            lines.append(
                f"  {p['country']} ({p['period']}): "
                f"classification={p.get('classification','?')} "
                f"M={f(m.get('M'))} Y={f(m.get('Y'))} "
                f"headroom={f(m.get('headroom'))} sigma={f(m.get('sys_sigma'))} "
                f"theta={f(m.get('theta'))} "
                f"strongest={p.get('strongest',[])} weakest={p.get('weakest',[])}"
            )

    return "\n".join(lines)

# ── Claude API ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You generate the narrative and interpretive sections of a CAMS National Mood, \
Vision-Affect, and Mythic Attractor Report.

## Framework

CAMS treats civilisations as eight coupled institutional nodes. Each node has \
an organic analogue (what it is in the body of society), a primary function, \
a mythic attractor (how the civilisation imagines it), a canonical failure mode, \
and a timescale:

  Helm     (King)       | Central nervous system   | Executive coordination  | Paralysis / capture              | Years–decades
  Shield   (Warrior)    | Immune + musculoskeletal  | Defence / order         | Atrophy or hyperactivation       | Years–decades
  Archive  (Library)    | Long-term memory          | Institutional record    | Amnesia or rigidity              | Decades–centuries
  Lore     (Temple)     | Proprioception            | Meaning / legitimacy    | Normative collapse / capture     | Centuries
  Stewards (Manor)      | Connective tissue         | Asset stewardship       | Rent-seeking / decay             | Decades
  Craft    (Workshop)   | Specialised organs        | Technical expertise     | Skill erosion                    | Decades
  Hands    (Harvesters) | Cellular mass             | Labour base             | Demographic / motivational fail  | Years
  Flow     (Agora)      | Circulatory system        | Exchange / logistics    | Mythic-material decoupling       | Months–decades

Metrics (1–10 scale):
  C = Coherence   K = Capacity   S = Stress   A = Abstraction
  Vision = A × C  |  Affect = K − S  |  σ (sigma) = Vision × Affect

System scalars:
  M(t) = mean Stress (metabolic load)
  Y(t) = mean Vision (mythic integration)
  Headroom = mean(K−S)
  Ψ = mean Affect of deliberative nodes (Archive, Lore, Stewards)
  Φ = mean Stress of reactive nodes (Helm, Shield, Hands, Flow)
  χ = Ψ − Φ   Θ = Φ/Ψ   Λ = mean Bond Strength

## Generation rules

1. Write the analytical layer first. Then derive the mood and mythic layer from \
the actual node pattern. Never do it the other way round.

2. A metaphor must map to a real structural relationship. If Helm is decoupled, \
the King cannot be heard. If Archive is strong and Lore is weak, the Library is \
full but the Temple is thinning. If Hands are depleted while Stewards remain \
strong, the Manor stands while the Harvesters tire.

3. Treat V = A×C as vision, F = K−S as affect, σ = VF as vision-under-feeling. \
That keeps the report psychologically legible without becoming vague.

4. Use the mythic attractors as interpretive faces of the nodes, not replacements \
for them. The function tells you what the node does. The attractor tells you how \
the civilisation imagines, legitimises, or emotionally experiences that function.

5. Avoid empty flourish. The poetry must be earned by the metrics.

6. Tone reference: "The Library remembers perfectly; the Temple has lost its \
congregation; the King cannot be heard." Match that quality — analytically \
precise, genuinely poetic, specific to the numbers.

7. For each node's Short Reading in node_short_readings: write one clinically \
precise sentence that captures what the metrics mean for that node's function. \
Start with the mythic attractor name in bold-style (e.g. "King deposed:" or \
"Library intact:"), then the structural fact.

8. For operator_reading: one short paragraph (2–4 sentences) explaining the \
decisive scalar relationship — the Θ ratio, χ direction, and what regime this \
puts the system in. Cite the actual numbers.

9. For historical_reading: one paragraph explaining the trajectory — what has \
changed, what has persisted, what phase transition matters most.

10. For comparative_notes: if peer data is provided, write one paragraph placing \
this society in relation to peers. If no peers, write an empty string.

11. Mythic state for each node must be exactly one of:
    integrated | thinning | overloaded | hollowed out | captured | \
    decoupled | weaponised | latent | renewing

## Output

Return ONLY a JSON object with these exact keys — no preamble, no code fences:

exec_phrase: string
exec_summary: {
  classification_label, strongest_nodes, weakest_nodes,
  key_anomaly, system_condition,
  structural_reading, mood_reading, mythic_pole_a, mythic_pole_b
}
node_short_readings: {
  Helm, Shield, Archive, Lore, Stewards, Craft, Hands, Flow
}
operator_reading: string
vision_affect_reading: string
structural_dynamics: {
  primary_anomaly, dominant_anchor, main_weakness, archive_lore, fast_slow
}
mythopoetic: {
  mood_texture, mood_expanded,
  governing_metaphor, metaphor_explanation,
  persona, persona_why,
  gift, wound, gift_wound_reading,
  mythic_tension_a, mythic_tension_b, mythic_tension_interp,
  mythic_close
}
mythic_alignment: array of 8 objects [{node, attractor, state, reading}] in \
order [Helm, Shield, Archive, Lore, Stewards, Craft, Hands, Flow]
historical_reading: string
comparative_notes: string
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
"""

def generate_narrative(client, summary, country):
    user_msg = (
        f"Here is the fully computed CAMS data for {country}:\n\n"
        f"{summary}\n\n"
        "Generate all narrative sections following the generation rules exactly. "
        "Every claim must be grounded in the numbers. "
        "Return only the JSON object."
    )
    text = ""
    print("\n── Claude narrative (streaming) ──────────────────────────────")
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 2000},
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk
            print(chunk, end="", flush=True)
    print("\n────────────────────────────────────────────────────────────\n")

    # Strip code fences if present
    clean = text.replace('```json', '').replace('```', '')
    start = clean.find('{'); end = clean.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON in Claude response:\n" + text[:600])
    return json.loads(clean[start:end])

# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"reports": []}

def update_manifest(manifest_path, country, period, generated_date,
                    csv_name, sc, nodes_ops, narr, out_path, manifest):
    """Add or update this report's entry in the manifest JSON."""
    reports = manifest.get('reports', [])
    report_id = f"{country.lower().replace(' ','-')}-{max(period.split('–'))}"
    try:
        rel_path = str(Path(out_path).relative_to(Path(manifest_path).parent.parent)).replace('\\','/')
    except ValueError:
        rel_path = str(Path(out_path).resolve().relative_to(Path(manifest_path).resolve().parent.parent)).replace('\\','/')

    sigmas = {n: round(nodes_ops[n]['sigma'], 2) for n in STANDARD_NODES
              if n in nodes_ops and nodes_ops[n].get('sigma') is not None}
    strongest = sorted(sigmas, key=sigmas.get, reverse=True)[:3]
    weakest   = sorted(sigmas, key=sigmas.get)[:2]

    tier = TIER_MAP.get(sc['classification'], 'strained')

    country_flags = {
        'france':'🇫🇷','germany':'🇩🇪','norway':'🇳🇴','sweden':'🇸🇪','uk':'🇬🇧',
        'england':'🏴󠁧󠁢󠁥󠁮󠁧󠁿','usa':'🇺🇸','australia':'🇦🇺','canada':'🇨🇦','japan':'🇯🇵',
        'china':'🇨🇳','india':'🇮🇳','russia':'🇷🇺','italy':'🇮🇹','spain':'🇪🇸',
    }
    flag = country_flags.get(country.lower(), '🌐')

    entry = {
        "id": report_id, "country": country, "flag": flag,
        "period": period, "generated": generated_date,
        "dataset": csv_name,
        "classification": narr.get('final_assessment',{}).get('classification_label', sc['classification']),
        "classification_tier": tier,
        "exec_phrase": narr.get('exec_phrase',''),
        "file": rel_path,
        "metrics": {
            "M": round(sc['M'], 2) if sc['M'] else None,
            "Y": round(sc['Y'], 2) if sc['Y'] else None,
            "headroom": round(sc['headroom'], 2) if sc['headroom'] else None,
            "sys_sigma": round(sc['sys_sigma'], 2) if sc['sys_sigma'] else None,
            "theta": round(sc['theta'], 2) if sc['theta'] else None,
            "lambda": round(sc['lam'], 2) if sc['lam'] else None,
        },
        "node_sigmas": sigmas,
        "strongest": strongest,
        "weakest": weakest,
    }

    # Replace existing or append
    idx = next((i for i, r in enumerate(reports) if r.get('id') == report_id), None)
    if idx is not None:
        reports[idx] = entry
    else:
        reports.append(entry)

    manifest['reports'] = reports
    manifest['generated'] = generated_date

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✓ Manifest updated: {manifest_path}")

# ── HTML rendering ────────────────────────────────────────────────────────────

def fv(v, d=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)

def render_html(country, period, snapshot_years, nodes_ops, sc, hist,
                narr, date_str, csv_name, assessor, flag_note, peers=None):

    ex    = narr.get('exec_summary', {})
    nsr   = narr.get('node_short_readings', {})
    opr   = narr.get('operator_reading', '')
    vis   = narr.get('vision_affect_reading', '')
    st    = narr.get('structural_dynamics', {})
    my    = narr.get('mythopoetic', {})
    mal   = narr.get('mythic_alignment', [])
    hr    = narr.get('historical_reading', '')
    cn    = narr.get('comparative_notes', '')
    imp   = narr.get('implications', {})
    fin   = narr.get('final_assessment', {})

    # Node table rows (with Short Reading column)
    node_rows = ""
    for node in STANDARD_NODES:
        if node not in nodes_ops:
            continue
        nd  = nodes_ops[node]
        sig = nd.get('sigma')
        sc_class = ("positive" if sig is not None and sig > 0
                    else "negative" if sig is not None and sig < 0 else "")
        short = nsr.get(node, '')
        node_rows += (
            f"<tr>"
            f"<td><strong>{node}</strong></td>"
            f"<td>{MYTHIC_MAP.get(node, node)}</td>"
            f"<td>{fv(nd.get('C'))}</td><td>{fv(nd.get('K'))}</td>"
            f"<td>{fv(nd.get('S'))}</td><td>{fv(nd.get('A'))}</td>"
            f"<td>{fv(nd.get('NV'))}</td><td>{fv(nd.get('B'))}</td>"
            f"<td>{fv(nd.get('V'))}</td><td>{fv(nd.get('F'))}</td>"
            f"<td class='{sc_class}'><strong>{fv(sig)}</strong></td>"
            f"<td class='short-read'>{short}</td>"
            f"</tr>\n"
        )

    # Mythic alignment rows
    mal_rows = "".join(
        f"<tr><td><strong>{i.get('node','')}</strong></td>"
        f"<td>{i.get('attractor','')}</td>"
        f"<td><em>{i.get('state','')}</em></td>"
        f"<td>{i.get('reading','')}</td></tr>\n"
        for i in mal
    )

    # Historical rows
    hist_rows = "".join(
        f"<tr><td>{e['era']}</td><td>{fv(e.get('M'))}</td>"
        f"<td>{fv(e.get('Y'))}</td><td>{fv(e.get('headroom'))}</td>"
        f"<td>{fv(e.get('lam'))}</td><td>{fv(e.get('theta'))}</td>"
        f"<td>{e.get('posture','')}</td></tr>\n"
        for e in hist
    )

    # Section 10 peers table
    peer_rows = ""
    if peers:
        for p in peers:
            if p['country'] == country:
                continue
            m = p.get('metrics', {})
            is_self = p['country'] == country
            row_class = ' class="self-row"' if is_self else ''
            peer_rows += (
                f"<tr{row_class}><td><strong>{p['country']}</strong></td>"
                f"<td>{p.get('period','')}</td>"
                f"<td>{fv(m.get('theta'))}</td>"
                f"<td>{fv(m.get('headroom'))}</td>"
                f"<td>{fv(m.get('sys_sigma'))}</td>"
                f"<td>{p.get('classification','')}</td></tr>\n"
            )

    # Current report row for comparison table
    self_row = (
        f"<tr class='self-row'><td><strong>→ {country}</strong></td>"
        f"<td>{period}</td>"
        f"<td>{fv(sc['theta'])}</td>"
        f"<td>{fv(sc['headroom'])}</td>"
        f"<td>{fv(sc['sys_sigma'])}</td>"
        f"<td>{sc['classification']}</td></tr>\n"
    )

    sv = ", ".join(fv(nodes_ops[n].get('sigma')) for n in STANDARD_NODES if n in nodes_ops)
    system_vector = f"P({max(snapshot_years)}) ≈ ({sv})"
    exec_phrase   = narr.get('exec_phrase', '')

    section10 = ""
    if peers and len([p for p in peers if p['country'] != country]) > 0:
        section10 = f"""
<h2>10. Comparative Position</h2>
<table>
<tr><th>Society</th><th>Period</th><th>&Theta;</th><th>Headroom</th><th>System &sigma;</th><th>Classification</th></tr>
{self_row}{peer_rows}
</table>
{"<p>" + cn + "</p>" if cn else ""}
"""

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
.back{{font-family:'Inter',sans-serif;font-size:.8em;color:#764ba2;text-decoration:none;display:inline-flex;align-items:center;gap:5px;margin-bottom:28px}}
.back:hover{{color:#667eea}}
.wrap{{max-width:940px;margin:0 auto;background:#fff;padding:64px 72px;border-radius:4px;box-shadow:0 4px 24px rgba(0,0,0,.07)}}
h1{{font-size:2.7em;color:#1a1a2e;margin:0 0 8px;font-weight:600;letter-spacing:-.02em}}
.series{{font-family:'Inter',sans-serif;font-size:.78em;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#999;margin-bottom:36px}}
.exec-phrase{{font-size:1.35em;font-style:italic;color:#1a3a52;border-left:4px solid #1a3a52;padding:14px 22px;margin:28px 0 38px;background:#f0f5fa;border-radius:0 6px 6px 0}}
h2{{font-family:'Inter',sans-serif;font-size:.73em;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#999;margin:48px 0 14px;border-top:1px solid #eee;padding-top:26px}}
h3{{font-family:'Inter',sans-serif;font-size:.82em;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#555;margin:26px 0 8px}}
p{{margin:1.3em 0;font-size:1.04em}}
table{{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:.78em;margin:16px 0 24px}}
th{{background:#1a1a2e;color:#fff;padding:8px 10px;text-align:left;font-weight:600;font-size:.74em;letter-spacing:.04em}}
td{{padding:7px 10px;border-bottom:1px solid #eee;vertical-align:top}}
tr:nth-child(even) td{{background:#fafafa}}
td.positive{{color:#059669;font-weight:700}} td.negative{{color:#dc2626;font-weight:700}}
td.short-read{{font-size:.82em;color:#555;font-style:italic;min-width:180px}}
tr.self-row td{{background:#f0f5fa;font-weight:600}}
.operator-reading{{background:#f8f5ff;border-left:3px solid #764ba2;padding:12px 18px;margin:12px 0 20px;font-size:.96em;border-radius:0 6px 6px 0}}
.divider{{text-align:center;color:#bbb;font-size:1.3em;margin:34px 0}}
.myth-close{{font-size:1.22em;font-style:italic;color:#1a3a52;text-align:center;padding:18px;margin:28px 0;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.final-line{{font-size:1.28em;font-style:italic;color:#1a1a2e;text-align:center;padding:20px;margin:28px 0;background:#f0f5fa;border-radius:6px}}
.vector{{font-family:'Inter',sans-serif;font-size:.74em;color:#666;background:#f8f8f8;padding:9px 13px;border-radius:4px;margin:8px 0 18px;word-break:break-all}}
.footer{{font-family:'Inter',sans-serif;font-size:.73em;color:#bbb;margin-top:48px;padding-top:18px;border-top:1px solid #eee;line-height:1.9}}
strong{{color:#1a1a2e}}
@media(max-width:700px){{.wrap{{padding:32px 20px}}td.short-read{{display:none}}}}
@media print{{body{{background:#fff;padding:0}}.wrap{{box-shadow:none;padding:40px}}.back{{display:none}}}}
</style>
</head>
<body>
<div style="max-width:940px;margin:0 auto"><a class="back" href="../mindscapes.html">&larr; Mindscapes</a></div>
<div class="wrap">

<p class="series">Neural Nations CAMS Ensemble &middot; v3.2-R.3-VA &middot; {date_str}</p>
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
<strong>Assessor:</strong> {assessor}<br>
<strong>Snapshot rule:</strong> Average of last {len(snapshot_years)} years ({min(snapshot_years)}&ndash;{max(snapshot_years)})<br>
<strong>Vision&ndash;Affect:</strong> Vision = A&times;C &nbsp;&middot;&nbsp; Affect = K&minus;S &nbsp;&middot;&nbsp; &sigma; = Vision &times; Affect<br>
<strong>Stress convention:</strong> {flag_note}<br>
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
<tr><td>K&minus;C Delta</td><td>{fv(sc['K_minus_C'])}</td></tr>
<tr><td>SAI (A/C)</td><td>{fv(sc['SAI'])}</td></tr>
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
{"<div class='operator-reading'>" + opr + "</div>" if opr else ""}

<h2>5. Current Node Table</h2>
<table>
<tr><th>Node</th><th>Attractor</th><th>C</th><th>K</th><th>S</th><th>A</th>
<th>NV</th><th>B</th><th>Vision (A&times;C)</th><th>Affect (K&minus;S)</th><th>&sigma;</th>
<th>Short Reading</th></tr>
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
<tr><th>Era</th><th>M(t)</th><th>Y(t)</th><th>Headroom</th><th>&Lambda;</th><th>&Theta;</th><th>Posture</th></tr>
{hist_rows}
</table>
{"<p>" + hr + "</p>" if hr else ""}

{section10}

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
Model: claude-opus-4-6 (Neural Nations CAMS Ensemble v3.2-R.3-VA)<br>
Assessor: {assessor} &nbsp;&middot;&nbsp; Dataset: {csv_name} &nbsp;&middot;&nbsp;
Snapshot: {min(snapshot_years)}&ndash;{max(snapshot_years)} ({len(snapshot_years)} years)<br>
Stress polarity: {flag_note}<br>
Neural Nations CAMS Research &middot; Open Science (Common Property) &middot; Co-authored with Claude Opus&nbsp;4.6 (Anthropic)
</div>

</div>
</body>
</html>"""

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Generate a CAMS National Mood Report from a CSV.')
    ap.add_argument('csv',          help='Path to CAMS CSV file')
    ap.add_argument('--country',    help='Country name override')
    ap.add_argument('--snapshot',   type=int, default=3, help='Years to average (default: 3)')
    ap.add_argument('--output',     help='Output HTML path')
    ap.add_argument('--manifest',   help='Path to reports-manifest.json (enables §10 + auto-update)',
                    default=str(Path(__file__).parent / 'reports-manifest.json'))
    ap.add_argument('--assessor',   default='CAMS Ensemble', help='Who scored the data')
    ap.add_argument('--flag',       default='positive = strain', help='Stress polarity convention')
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    society, rows = load_dataset(args.csv)
    country = args.country or society

    present = set(r['node'] for r in rows)
    nodes   = [n for n in STANDARD_NODES if n in present]
    if not nodes:
        print(f"ERROR: No standard CAMS nodes found. Present: {present}")
        sys.exit(1)

    all_years = sorted(set(r['year'] for r in rows))
    print(f"Country : {country}")
    print(f"Nodes   : {nodes}")
    print(f"Years   : {all_years[0]}–{all_years[-1]} ({len(all_years)} years)")
    print(f"Snapshot: last {args.snapshot} year(s)")

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

    # Load manifest for peer comparison
    manifest = load_manifest(args.manifest)
    peers    = manifest.get('reports', []) if manifest else []

    summary = data_summary(country, snap_years, nd_ops, sc, hist, peers or None)
    client  = anthropic.Anthropic()
    narr    = generate_narrative(client, summary, country)

    date_str = datetime.date.today().isoformat()

    if args.output:
        out_path = args.output
    else:
        stem     = f"{country.replace(' ','_')}_{max(snap_years)}_CAMS_Report"
        out_path = str(Path(__file__).parent / f"{stem}.html")

    html = render_html(country, period, snap_years, nd_ops, sc, hist,
                       narr, date_str, Path(args.csv).name,
                       args.assessor, args.flag, peers or None)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ Report written to: {out_path}")

    # Update manifest
    if args.manifest and Path(args.manifest).exists():
        update_manifest(args.manifest, country, period, date_str,
                        Path(args.csv).name, sc, nd_ops, narr, out_path, manifest)

if __name__ == '__main__':
    main()
