"""Build sitemap.xml and the page list inside site-index.html from one source.

Usage (from repo root):  python scripts/build_site_index.py

Every public page must appear in GROUPS below. The script fails if a listed
page doesn't exist, is listed twice, or if a tracked page is neither listed
nor in EXCLUDE -- so new pages can't silently fall out of the index.
"""
import html, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BASE = 'https://neuralnations.org/'

# Pages deliberately left out of both the sitemap and the index.
EXCLUDE_PREFIXES = ('test/', 'templates/', 'app/')
EXCLUDE = {
    'auth-callback',                            # OAuth plumbing
    'migration-plan', 'cams-network',           # noindex
    'test', 'research',                         # 301 redirects (_redirects)
    'cams-reconstructor',                       # meta-refresh stub
    'mindscapes/Russia_2024-2026_CAMS_Report',  # duplicate of Russia_2026
}

# (group, default type, changefreq, priority, entries). An entry is a slug or
# (slug, type). Slugs are extensionless URLs; folders end in '/', home is ''.
GROUPS = [
    ('Start Here', 'Guide', 'monthly', '0.9', [
        '', 'start-here', 'what-is-cams', 'why-cams', 'model', 'framework/', 'faq',
        'a-short-history-of-cams', 'cams-project-history', 'cams-foundational-conversations',
    ]),
    ('Results & Predictions', 'Findings', 'monthly', '0.8', [
        'results', 'validation', 'predictions', 'status_report', 'escalation-archetypes',
        'cams-failure-modes', 'failure-modes-story', 'great-power-attractors', 'applications', 'reports',
    ]),
    ('Datasets', 'Dataset', 'weekly', '0.8', [
        'datasets', 'cams/', 'epiphenomenon/data', 'repository-access', 'request-access',
    ]),
    ('Tools & Explorers', 'Tool', 'monthly', '0.7', [
        'explore', 'cams', 'cams-scorer', 'cams-scorer-local', 'cams-explorer', 'juno-36-explorer', 'juno_calculator',
        'cams-interpreter', 'cams-interpreter/', 'cams-compare/', 'cams-zeitgeist', 'cams-advanced-analysis',
        'cams-integrated-dashboard', 'cams-network-explorer', 'attractors', 'cams-3d-attractor',
        'cams-coordination', 'seshat-explorer', 'granger-validator', 'cams-telescope/',
        ('cams-telescope/guide', 'Guide'), 'cams-instrument', 'cams-morphospace-atlas', 'mind-reader',
        ('cams-diy-kit', 'Guide'),
    ]),
    ('Papers & Validation', 'Paper', 'monthly', '0.7', [
        'research/', 'research/cams-validation-2026', 'research/pr-gap2-findings', 'pr-gap2-full-paper',
        'pr-gap2-rupture-study', 'analysis/pr_gap2_report', 'analysis/pr_gap2_synthesis',
        'analysis/pr_gap2_formalism_audit', 'analysis/pr_gap2_nct_loso_report', 'analysis/pr_gap2_robson_plane',
        'analysis/pr_gap2_general_audience', 'analysis/inelasticity_results', 'juno-v1-2', 'bond-strength-spec',
        'blind-protocol-v111', 'blind-test', 'cams-blind-test-assessment', 'cams-testing-report-june2026',
        'cams5-thesis-validation-report', 'ensemble-validation', 'paladin-validation',
        'operator-portability-synthesis', 'peer-review', 'response-to-critics', 'functional_taxonomy', 'figures',
        'metabolism-meaning', 'scale-invariance', 'thermodynamic-jungle', 'war', 'social-cognition',
        'coupled-cattle', 'meta-animal-under-the-microscope',
    ]),
    ('Country & Case Studies', 'Study', 'monthly', '0.6', [
        'the-empty-pulpit', ('the-empty-pulpit-map', 'Map'), 'usa-empty-pulpit-perspectives',
        'usa-empty-pulpit-detective-panel', 'usa-outlook-set', 'us-legitimacy-cams-study',
        'analysis/us_legitimacy_cams_study', 'analysis/us_interest_burden_cams_study',
        'uk-arc-story', ('uk-arc-map', 'Map'), 'uk-detective-panel', 'uk-outlook-set',
        'norway-arc-story', ('norway-arc-map', 'Map'), 'norway-detective-panel', 'norway-outlook-set',
        'nz-arc-story', ('nz-arc-map', 'Map'), 'germany-arc-story', ('germany-arc-map', 'Map'),
        'canada-arc-story', ('canada-arc-map', 'Map'), 'argentina-cams-story', ('argentina-cams-map', 'Map'),
        'burning-house-story', ('burning-house-map', 'Map'), 'burning-house-detective-panel',
        'netherlands-cams-v1-2', 'slavery-path-dependency-2023', 'kimi-coordination-analysis',
    ]),
    ('Aha! Maps', 'Map', 'monthly', '0.6', [
        'aha-maps', 'bond-wreck-map', 'mirror-escalation-map', 'us-covid-cams-map',
    ]),
    ('Mindscapes', 'Report', 'monthly', '0.6', [
        'mindscapes', 'mindscapes/Argentina_2026_CAMS_Report', 'mindscapes/France_2026_CAMS_Report',
        'mindscapes/New_Zealand_2026_CAMS_Report', 'mindscapes/Norway_2026_CAMS_Report',
        'mindscapes/Russia_2026_CAMS_Report', 'mindscapes/UK_2024-2026_CAMS_Report',
    ]),
    ('Series & Essays', 'Series', 'yearly', '0.5', [
        'newsletter',
        *[f'complex-adaptive-humans-{i}' for i in list(range(1, 26)) + list(range(28, 33))],
        'the-borrowed-nation', *[f'the-borrowed-nation-{i}' for i in range(1, 6)],
        'epiphenomenon/', 'epiphenomenon/australia-story', 'epiphenomenon/without-hindcasting', 'anxiety-machine',
    ]),
    ('Diary & About', 'Site', 'monthly', '0.5', [
        'research-diary', 'site-index', 'explore-index', 'grok', 'about', 'contact',
    ]),
]

PRIORITY_OVERRIDE = {'': '1.0'}
FREQ_OVERRIDE = {'': 'weekly', 'research-diary': 'weekly', 'explore': 'weekly', 'site-index': 'weekly'}

# For pages whose <title> is missing or unhelpful.
TITLE_OVERRIDE = {
    '': 'Neural Nations — Home',
    'framework/': 'CAMS v3.2 Framework',
    'cams-instrument': 'CAMS Instrument',
    'cams-morphospace-atlas': 'CAMS Morphospace Atlas',
    'burning-house-map': 'The Burning House Economy — Aha! Map',
    'canada-arc-map': 'Canada — Arc Map',
    'mirror-escalation-map': 'Mirror Escalation — Aha! Map',
    'us-covid-cams-map': 'US COVID — Aha! Map',
}


def slug_of(path):
    s = path[:-5]
    if s == 'index':
        return ''
    return s[:-5] if s.endswith('/index') else s


def file_of(slug):
    return 'index.html' if slug == '' else (slug + 'index.html' if slug.endswith('/') else slug + '.html')


def href_of(slug):
    return 'index.html' if slug == '' else (slug if slug.endswith('/') else slug + '.html')


def git_dates():
    log = subprocess.run(['git', 'log', '--format=@%ad', '--date=short', '--name-only', '--', '*.html'],
                         capture_output=True, text=True, encoding='utf-8').stdout
    last, d = {}, None
    for line in log.splitlines():
        if line.startswith('@'):
            d = line[1:]
        elif line.strip() and line not in last:
            last[line] = d
    return last


def clean(s):
    s = html.unescape(re.sub(r'<[^>]+>', '', s))
    return re.sub(r'\s+', ' ', s).strip()


def page_info(slug):
    src = open(file_of(slug), encoding='utf-8', errors='ignore').read()
    title = TITLE_OVERRIDE.get(slug)
    if not title:
        m = re.search(r'<title>(.*?)</title>', src, re.S | re.I)
        title = clean(m.group(1)) if m else slug
        title = re.sub(r'\s*[—·|–-]\s*Neural Nations.*$', '', title).strip() or slug
    m = (re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', src, re.S | re.I)
         or re.search(r'<meta\s+content=["\'](.*?)["\']\s+name=["\']description["\']', src, re.S | re.I))
    desc = clean(m.group(1)) if m else ''
    if not desc:
        body = src.split('</nav>', 1)[-1]
        for p in re.findall(r'<p[^>]*>(.*?)</p>', body, re.S | re.I):
            t = clean(p)
            if len(t) > 60:
                desc = t
                break
    if len(desc) > 220:
        desc = desc[:217].rsplit(' ', 1)[0] + '…'
    return title, desc


def main():
    tracked = subprocess.run(['git', 'ls-files', '*.html'], capture_output=True, text=True,
                             encoding='utf-8').stdout.splitlines()
    dates = {slug_of(f): d for f, d in git_dates().items() if f.endswith('.html')}
    public = {slug_of(f) for f in tracked if not f.startswith(EXCLUDE_PREFIXES)} - EXCLUDE
    public.add('site-index')

    rows, seen = [], set()
    for group, gtype, freq, prio, entries in GROUPS:
        for e in entries:
            slug, typ = (e if isinstance(e, tuple) else (e, gtype))
            if not os.path.exists(file_of(slug)):
                sys.exit(f'ERROR: {slug!r} listed but {file_of(slug)} does not exist')
            if slug in seen:
                sys.exit(f'ERROR: {slug!r} listed twice')
            seen.add(slug)
            title, desc = page_info(slug)
            rows.append(dict(slug=slug, group=group, type=typ, title=title, desc=desc,
                             date=dates.get(slug, '2026-09-25'),
                             freq=FREQ_OVERRIDE.get(slug, freq), prio=PRIORITY_OVERRIDE.get(slug, prio)))
    missing = sorted(public - seen)
    if missing:
        sys.exit('ERROR: public pages not in GROUPS (add them, or to EXCLUDE):\n  ' + '\n  '.join(missing))

    # ── sitemap.xml ──
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!-- Generated by scripts/build_site_index.py — edit GROUPS there, not this file. -->',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">', '']
    current = None
    for r in rows:
        if r['group'] != current:
            current = r['group']
            out.append(f'  <!-- ── {current} ' + '─' * max(3, 58 - len(current)) + ' -->')
        out += ['  <url>', f'    <loc>{BASE}{r["slug"]}</loc>', f'    <lastmod>{r["date"]}</lastmod>',
                f'    <changefreq>{r["freq"]}</changefreq>', f'    <priority>{r["prio"]}</priority>', '  </url>', '']
    out.append('</urlset>')
    open('sitemap.xml', 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')

    # ── site-index.html page list (between markers) ──
    esc = lambda s: html.escape(s, quote=True)
    parts = []
    recent = sorted((r for r in rows if r['slug'] != 'site-index'), key=lambda r: r['date'], reverse=True)[:10]
    parts.append('<section class="si-recent" aria-labelledby="si-recent-h">'
                 '<h2 id="si-recent-h">Recently updated</h2><ol>')
    for r in recent:
        parts.append(f'<li><a href="{esc(href_of(r["slug"]))}">{esc(r["title"])}</a>'
                     f' <time datetime="{r["date"]}">{r["date"]}</time></li>')
    parts.append('</ol></section>')
    for group, *_ in GROUPS:
        items = [r for r in rows if r['group'] == group]
        gid = re.sub(r'[^a-z]+', '-', group.lower()).strip('-')
        parts.append(f'<details class="si-group" id="{gid}" open><summary><h2>{esc(group)}</h2>'
                     f'<span class="si-count">{len(items)}</span></summary><ul>')
        for r in items:
            parts.append(
                f'<li class="si-item" data-type="{esc(r["type"])}">'
                f'<a href="{esc(href_of(r["slug"]))}">{esc(r["title"])}</a>'
                f'<span class="si-badge si-{r["type"].lower()}">{esc(r["type"])}</span>'
                f'<time datetime="{r["date"]}">updated {r["date"]}</time>'
                + (f'<p>{esc(r["desc"])}</p>' if r['desc'] else '') + '</li>')
        parts.append('</ul></details>')
    block = '\n'.join(parts)

    page = open('site-index.html', encoding='utf-8').read()
    page, n = re.subn(r'(<!-- INDEX:START -->).*?(<!-- INDEX:END -->)',
                      lambda m: m.group(1) + '\n' + block + '\n' + m.group(2), page, flags=re.S)
    if n != 1:
        sys.exit('ERROR: INDEX markers not found in site-index.html')
    types = sorted({r['type'] for r in rows})
    chips = ''.join(f'<button type="button" class="si-chip" data-type="{t}" aria-pressed="false">{t}</button>'
                    for t in types)
    page = re.sub(r'(<!-- CHIPS:START -->).*?(<!-- CHIPS:END -->)',
                  lambda m: m.group(1) + chips + m.group(2), page, flags=re.S)
    page = re.sub(r'<span id="si-total">\d*</span>', f'<span id="si-total">{len(rows)}</span>', page)
    open('site-index.html', 'w', encoding='utf-8', newline='\n').write(page)

    print(f'{len(rows)} pages in {len(GROUPS)} groups -> sitemap.xml + site-index.html')


if __name__ == '__main__':
    main()
