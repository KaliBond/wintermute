"""Build nations.html — every report, artefact and dataset, assembled by nation.

Usage (from repo root):  python scripts/build_nations.py
Run after scripts/build_site_index.py (it reuses that script's page list).
Maintenance checklist: SITE_INDEX_MAINTENANCE.md

Sources:
  1. Site pages listed in build_site_index.GROUPS (title, slug, description)
  2. External artefacts listed in explore-index.html
  3. Datasets in datasets.json and the per-nation cards on datasets.html
Matching is by the NATIONS alias patterns below; PAGE_EXTRA / PAGE_SKIP fix
pages the patterns get wrong.
"""
import html, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_site_index as si  # noqa: E402  (also chdirs to repo root)

# (name, region, alias regex). Case-sensitive, so short codes (US, UK, NZ) are safe.
NATIONS = [
    ('United States', 'Americas', r'\bUSA\b|\bUS\b|\bU\.S\.|United States|(?<!South )(?<!Latin )\bAmerica(?:n|ns)?\b|Empty Pulpit'),
    ('Canada', 'Americas', r'Canad(?:a|ian)'),
    ('Argentina', 'Americas', r'Argentin(?:a|e|ian)'),
    ('Brazil', 'Americas', r'Brazil'),
    ('Chile', 'Americas', r'\bChile(?:an)?\b'),
    ('Colombia', 'Americas', r'Colombia'),
    ('Venezuela', 'Americas', r'Venezuela'),
    ('United Kingdom', 'Europe', r'\bUK\b|United Kingdom|Brit(?:ain|ish)|\bEngland\b|Soft Landing'),
    ('France', 'Europe', r'France|French'),
    ('Germany', 'Europe', r'German(?:y)?|\bBRD\b|Weimar'),
    ('Netherlands', 'Europe', r'Netherlands|Dutch'),
    ('Norway', 'Europe', r'Norw(?:ay|egian)'),
    ('Sweden', 'Europe', r'Swed(?:en|ish)'),
    ('Denmark', 'Europe', r'Denmark|Danish'),
    ('Finland', 'Europe', r'Finland|Finnish'),
    ('Italy', 'Europe', r'Ital(?:y|ian)'),
    ('Spain', 'Europe', r'Spain|Spanish'),
    ('Greece', 'Europe', r'Gree(?:ce|k)|Athens|Sparta|Peloponnes'),
    ('Poland', 'Europe', r'Poland|Polish'),
    ('Latvia', 'Europe', r'Latvia'),
    ('Russia', 'Europe', r'Russia'),
    ('Ukraine', 'Europe', r'Ukrain'),
    ('Rome (historical)', 'Europe', r'\bRom(?:e|an Empire)\b|Latium'),
    ('Iran', 'Middle East & Africa', r'\bIran'),
    ('Israel', 'Middle East & Africa', r'Israel'),
    ('Iraq', 'Middle East & Africa', r'\bIraq'),
    ('Syria', 'Middle East & Africa', r'Syria'),
    ('Lebanon', 'Middle East & Africa', r'Leban'),
    ('Saudi Arabia', 'Middle East & Africa', r'Saudi'),
    ('UAE', 'Middle East & Africa', r'\bUAE\b|Emirates'),
    ('Türkiye', 'Middle East & Africa', r'Türkiye|Turkey'),
    ('Egypt', 'Middle East & Africa', r'Egypt'),
    ('Ethiopia', 'Middle East & Africa', r'Ethiopia'),
    ('Nigeria', 'Middle East & Africa', r'Nigeria'),
    ('Mauritania', 'Middle East & Africa', r'Mauritania'),
    ('DR Congo', 'Middle East & Africa', r'Congo|\bDRC\b'),
    ('South Africa', 'Middle East & Africa', r'South Africa'),
    ('China', 'Asia-Pacific', r'China|Chinese'),
    ('Hong Kong', 'Asia-Pacific', r'Hong ?Kong'),
    ('Taiwan', 'Asia-Pacific', r'Taiwan'),
    ('Japan', 'Asia-Pacific', r'Japan'),
    ('India', 'Asia-Pacific', r'\bIndia(?:n)?\b'),
    ('Pakistan', 'Asia-Pacific', r'Pakistan'),
    ('Afghanistan', 'Asia-Pacific', r'Afghan'),
    ('Mongolia', 'Asia-Pacific', r'Mongolia'),
    ('Thailand', 'Asia-Pacific', r'Thai(?:land)?\b'),
    ('Cambodia', 'Asia-Pacific', r'Cambodia'),
    ('Laos', 'Asia-Pacific', r'\bLaos\b'),
    ('Indonesia', 'Asia-Pacific', r'Indonesia'),
    ('Philippines', 'Asia-Pacific', r'Philippin'),
    ('Singapore', 'Asia-Pacific', r'Singapore'),
    ('Australia', 'Asia-Pacific', r'Australia|Burning House|Dutton|Trove'),
    ('New Zealand', 'Asia-Pacific', r'New Zealand|\bNZ\b|Aotearoa'),
]
REGIONS = ['Americas', 'Europe', 'Middle East & Africa', 'Asia-Pacific']

# Dataset society names that differ from NATIONS names.
DATASET_ALIAS = {
    'USA': 'United States', 'England': 'United Kingdom', 'Indigenous Australia': 'Australia',
    'Democratic Republic of the Congo': 'DR Congo', 'Roman Empire': 'Rome (historical)',
    'Rome': 'Rome (historical)', 'Latium Vetus': 'Rome (historical)',
}
DATASET_IGNORE = {'SpaceX', 'WorldCom'}

# Site-page groups whose pages are generic (tools, site furniture) — never nation reports.
SKIP_GROUPS = {'Start Here', 'Datasets', 'Tools & Explorers', 'Diary & About'}
ALL = {n for n, _, _ in NATIONS}
# Manual fixes: slug -> nations to add / nations to drop.
PAGE_EXTRA = {}
PAGE_SKIP = {
    'mindscapes': ALL,                          # hub page; the per-nation reports are listed instead
    'research/pr-gap2-findings': ALL,           # 44-society paper; names South Africa only as an example
    **{s: {'China'} for s in ['the-borrowed-nation', 'the-borrowed-nation-4', 'the-borrowed-nation-5']},
}


def slugify(name):
    return re.sub(r'[^a-z]+', '-', name.lower()).strip('-')


def matches(text, slug=''):
    # Titles/descriptions match case-sensitively; slugs are lowercase, so match them case-insensitively.
    return [n for n, _, pat in NATIONS
            if re.search(pat, text) or (slug and re.search(pat, slug.replace('-', ' ').replace('_', ' '), re.I))]


def main():
    by = {n: {'site': [], 'ext': [], 'data': []} for n, _, _ in NATIONS}
    region = {n: r for n, r, _ in NATIONS}

    # 1. Site pages
    dates = {si.slug_of(f): d for f, d in si.git_dates().items() if f.endswith('.html')}
    for group, gtype, _, _, entries in si.GROUPS:
        if group in SKIP_GROUPS:
            continue
        for e in entries:
            slug, typ = (e if isinstance(e, tuple) else (e, gtype))
            title, desc = si.page_info(slug)
            found = set(matches(f'{title} {desc}', slug)) | set(PAGE_EXTRA.get(slug, ()))
            found -= set(PAGE_SKIP.get(slug, ()))
            for n in found:
                by[n]['site'].append(dict(href=si.href_of(slug), title=title, desc=desc, type=typ,
                                          date=dates.get(slug, ''), multi=len(found) > 1))

    # 2. External artefacts (explore-index.html)
    src = open('explore-index.html', encoding='utf-8').read()
    for d, href, title, summ in re.findall(
            r'<tr>\s*<td class="date">(.*?)</td>\s*<td><a href="(.*?)"[^>]*>(.*?)</a></td>\s*<td class="summary">(.*?)</td>',
            src, re.S):
        title, summ = si.clean(title), si.clean(summ)
        found = matches(f'{title} {summ}')
        for n in found:
            by[n]['ext'].append(dict(href=html.unescape(href), title=title, desc=summ, date=si.clean(d),
                                     multi=len(found) > 1))

    # 3. Datasets
    seen_urls = set()
    for ds in json.load(open('datasets.json', encoding='utf-8'))['datasets']:
        soc = ds['society']
        if soc in DATASET_IGNORE:
            continue
        n = DATASET_ALIAS.get(soc, soc)
        if n not in by:
            sys.exit(f'ERROR: dataset society {soc!r} has no nation — add to NATIONS or DATASET_ALIAS')
        yrs = f"{ds['year_start']}–{ds['year_end']}" if ds.get('year_start') else ''
        label = ds['filename'].replace('_cleaned.csv', '').replace('_', ' ')
        by[n]['data'].append(dict(href=ds['github_url'], title=label, meta=f"{yrs} · {ds['records']:,} records"))
        seen_urls.add(ds['github_url'])
    dsrc = open('datasets.html', encoding='utf-8').read()
    for soc, meta, href in re.findall(
            r'<div class="dataset-card">\s*<span class="society">(.*?)</span>\s*<span class="meta">(.*?)</span>.*?href="(.*?)"',
            dsrc, re.S):
        soc, meta = si.clean(soc), si.clean(meta)
        base = soc.split(' — ')[0]
        n = DATASET_ALIAS.get(base, base)
        if n not in by or href in seen_urls:
            continue
        by[n]['data'].append(dict(href=href, title=soc, meta=meta.split(' · ')[0] + ' · JUNO nation series'))

    nations = [n for n, _, _ in NATIONS if any(by[n].values())]
    for n in nations:
        by[n]['site'].sort(key=lambda x: (x['multi'], x['title']))
        by[n]['ext'].sort(key=lambda x: (x['multi'], x['title']))
        by[n]['data'].sort(key=lambda x: x['title'])

    # ── render ──
    esc = lambda s: html.escape(s, quote=True)
    out = []
    # Overview: one compact table per region
    out.append('<div class="nx-overview">')
    for r in REGIONS:
        members = sorted(n for n in nations if region[n] == r)
        out.append(f'<table><caption>{esc(r)}</caption><thead><tr><th scope="col">Nation</th>'
                   '<th scope="col" title="Reports on this site">Site</th><th scope="col" title="External artefacts">Ext.</th>'
                   '<th scope="col" title="Datasets">Data</th></tr></thead><tbody>')
        for n in members:
            b = by[n]
            out.append(f'<tr data-nation="{esc(n.lower())}"><td><a href="#{slugify(n)}">{esc(n)}</a></td>'
                       f'<td>{len(b["site"]) or ""}</td><td>{len(b["ext"]) or ""}</td><td>{len(b["data"]) or ""}</td></tr>')
        out.append('</tbody></table>')
    out.append('</div>')

    for r in REGIONS:
        members = sorted(n for n in nations if region[n] == r)
        out.append(f'<h2 class="nx-region" id="{slugify(r)}">{esc(r)}</h2>')
        for n in members:
            b = by[n]
            counts = ' · '.join(x for x in [
                f'{len(b["site"])} on this site' if b['site'] else '',
                f'{len(b["ext"])} external' if b['ext'] else '',
                f'{len(b["data"])} dataset{"s" if len(b["data"]) != 1 else ""}' if b['data'] else ''] if x)
            out.append(f'<details class="nx-nation" id="{slugify(n)}" data-nation="{esc(n.lower())}">'
                       f'<summary><h3>{esc(n)}</h3><span class="nx-counts">{counts}</span></summary><div class="nx-body">')
            if b['site']:
                out.append('<h4>Reports on this site</h4><ul class="nx-list">')
                for x in b['site']:
                    tag = ' <span class="nx-multi">comparative</span>' if x['multi'] else ''
                    out.append(f'<li><a href="{esc(x["href"])}">{esc(x["title"])}</a>'
                               f'<span class="nx-type">{esc(x["type"])}</span>{tag}'
                               + (f'<p>{esc(x["desc"])}</p>' if x['desc'] else '') + '</li>')
                out.append('</ul>')
            if b['ext']:
                out.append('<h4>External artefacts <span class="nx-note">(Claude &amp; other AI artefacts, open in a new tab)</span></h4><ul class="nx-list">')
                for x in b['ext']:
                    tag = ' <span class="nx-multi">comparative</span>' if x['multi'] else ''
                    out.append(f'<li><a href="{esc(x["href"])}" target="_blank" rel="noopener">{esc(x["title"])} ↗</a>{tag}'
                               f'<p>{esc(x["desc"])}</p></li>')
                out.append('</ul>')
            if b['data']:
                out.append('<h4>Datasets</h4><ul class="nx-data">')
                for x in b['data']:
                    out.append(f'<li><a href="{esc(x["href"])}" target="_blank" rel="noopener">{esc(x["title"])}</a>'
                               f' <span>{esc(x["meta"])}</span></li>')
                out.append('</ul>')
            out.append('</div></details>')
    block = '\n'.join(out)

    page = open('nations.html', encoding='utf-8').read()
    page, k = re.subn(r'(<!-- NATIONS:START -->).*?(<!-- NATIONS:END -->)',
                      lambda m: m.group(1) + '\n' + block + '\n' + m.group(2), page, flags=re.S)
    if k != 1:
        sys.exit('ERROR: NATIONS markers not found in nations.html')
    tot = lambda key: sum(len(by[n][key]) for n in nations)
    page = re.sub(r'<span id="nx-stats">.*?</span>',
                  f'<span id="nx-stats">{len(nations)} nations · {tot("site")} site reports · '
                  f'{tot("ext")} external artefacts · {tot("data")} datasets</span>', page)
    open('nations.html', 'w', encoding='utf-8', newline='\n').write(page)
    print(f'{len(nations)} nations; site={tot("site")} ext={tot("ext")} data={tot("data")}')
    if '--report' in sys.argv:
        for n in nations:
            print(f'\n## {n}')
            for x in by[n]['site']:
                print('  S', x['href'], '|', x['title'])
            for x in by[n]['ext']:
                print('  E', x['title'])


if __name__ == '__main__':
    main()
