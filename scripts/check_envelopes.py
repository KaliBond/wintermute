"""Check that every ensemble (ENS) file has its envelope (ENV) and no ENV is orphaned.

Usage (from repo root):  python scripts/check_envelopes.py
Writes data/ENVELOPE_STATUS.md and exits 1 if any rule is broken.

Scope: the published folders cleaned_datasets/ and data/nations/ (incl. data/nations/5yr/).
data/v1.0_final_recompute/ and data/v2.3*/ are frozen script outputs / input snapshots.

Rules
  1. Every ENV has an ENS partner (same name with ENS in place of ENV) covering exactly
     the same Society/Year/Node rows.
  2. Every ENV actually carries spread columns (C_sd... / SD_... / V_range) with values.
  3. Every ENS has an ENV, unless it is listed in NO_ENVELOPE with a reason.
"""
import csv, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DIRS = ['cleaned_datasets', 'data/nations', 'data/nations/5yr']
TOKEN = re.compile(r'(?<![A-Za-z])(ENS|ENV)(?![A-Za-z])')

# Ensemble files whose name doesn't carry the ENS token -> the ENV that belongs to them.
ENSEMBLE_ALIASES = {
    # Same values as Argentina_ENS_1950_2026 (verified 2026-09-25); shares its envelope.
    'cleaned_datasets/Argentina_CAMS5_ensemble_1950_2026_cleaned.csv':
        'cleaned_datasets/Argentina_ENV_1950_2026_cleaned.csv',
    'cleaned_datasets/MARKER_USA_1900_2026_ENSEMBLE_MEAN_cleaned.csv': None,
}

# ENS files with no envelope, and why. Remove an entry once its envelope is added.
SINGLE = 'not an ensemble: identical to single-scorer run {}; no envelope can exist'
MISSING = 'envelope missing: no matching envelope anywhere in the repo; five-scorer re-run needed'
NO_ENVELOPE = {
    'data/nations/Brazil_ENS.csv': SINGLE.format('data/cleaned/brazil_grok_jan.csv (Grok)'),
    'data/nations/Saudi_Arabia_ENS.csv': SINGLE.format('data/new_datasets/SaudiA_GPT_feb.csv (GPT)'),
    'data/nations/Singapore_ENS.csv': SINGLE.format('data/cleaned/Singapore_gem_d3c.csv (Gemini)'),
    'data/nations/South_Africa_ENS.csv': SINGLE.format('data/new_datasets/BLINDS1.csv'),
    'data/nations/UAE_ENS.csv': SINGLE.format('data/cleaned/uae_gem_marc.csv (Gemini)'),
    'data/nations/Ukraine_ENS.csv': SINGLE.format('data/cleaned/Ukraine_gem_1930_Jan26.csv (Gemini)'),
    'data/nations/Venezuela_ENS.csv': SINGLE.format('data/cleaned/Venezuela_gem_dec1970_2025.csv (Gemini)'),
    **{f'data/nations/{n}_ENS.csv': MISSING for n in [
        'Denmark', 'Hong_Kong', 'Indonesia', 'Iraq', 'Israel', 'Latvia', 'Lebanon',
        'Pakistan', 'Philippines', 'Syria']},
    **{f'data/nations/{n}_ENS.csv': MISSING + f' (kept as loaded by the tools; a second full set with envelope, 1850-2026, is cleaned_datasets/{n}_ENS_1850_2026)'
       for n in ['Italy', 'Netherlands']},
    'data/nations/Japan_ENS.csv': MISSING + ' (annual 1875-2026 series, kept as loaded by the tools; the 5-year run is in data/nations/5yr/, and a second full set with envelope, 1850-2026, is cleaned_datasets/Japan_ENS_1850_2026)',
    'data/nations/Sweden_ENS.csv': MISSING + ' (annual series; the 5-year run and its envelope are in data/nations/5yr/)',
    'cleaned_datasets/Latvia_ENS_1901_2025_cleaned.csv': MISSING,
    'cleaned_datasets/Philippines_ENS_1880_2016_cleaned.csv': MISSING,
    'cleaned_datasets/LatimVetus_ENS_460_2010_cleaned.csv': MISSING + ' (the old ENV file was empty and has been removed)',
    'cleaned_datasets/MARKER_USA_1900_2026_ENSEMBLE_MEAN_cleaned.csv': MISSING,
}
SPREAD = re.compile(r'_sd$|^SD_|^V_range$')


def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    keys = {(r['Society'].strip(), str(int(float(r['Year']))), r['Node'].strip()) for r in rows}
    return rows, keys


def main():
    files = sorted(f'{d}/{n}' for d in DIRS if os.path.isdir(d) for n in os.listdir(d)
                   if n.endswith('.csv') and TOKEN.search(n))
    ens = {f for f in files if TOKEN.search(os.path.basename(f)).group(1) == 'ENS'} | set(ENSEMBLE_ALIASES)
    env = [f for f in files if TOKEN.search(os.path.basename(f)).group(1) == 'ENV']
    partner = lambda f, want: f'{os.path.dirname(f)}/{TOKEN.sub(want, os.path.basename(f), count=1)}'

    errors, paired, flagged = [], [], []
    for v in env:
        e = partner(v, 'ENS')
        rows, vkeys = load(v)
        spread = [c for c in rows[0] if SPREAD.search(c)]
        if not spread or not any(r[c].strip() for r in rows for c in spread):
            errors.append(f'{v}: no spread values; not a real envelope')
        if not os.path.exists(e):
            errors.append(f'{v}: ENV without ENS (expected {e})')
            continue
        if load(e)[1] != vkeys:
            errors.append(f'{v}: rows do not match {e}')
            continue
        paired.append((e, v, len(vkeys)))
    for e, v in ENSEMBLE_ALIASES.items():
        if v:
            if load(e)[1] != load(v)[1]:
                errors.append(f'{e}: rows do not match its envelope {v}')
            else:
                paired.append((e, v, len(load(e)[1])))
    have_env = {p[0] for p in paired}
    for e in sorted(ens - have_env):
        if e in NO_ENVELOPE:
            flagged.append((e, NO_ENVELOPE[e]))
        else:
            errors.append(f'{e}: ENS without ENV and not listed in NO_ENVELOPE')
    for e in NO_ENVELOPE:
        if e in have_env:
            errors.append(f'{e}: now has an envelope; remove it from NO_ENVELOPE')

    out = ['# Envelope status', '',
           'Generated by `scripts/check_envelopes.py`. Do not edit by hand.', '',
           'Ensemble scoring was introduced to strengthen the CAMS methodology. Early series were scored by '
           'a single AI model. An ensemble run has several scorers rate each node-year independently, then '
           'publishes the **ensemble mean** (ENS), which averages out the quirks of any one scorer, and the '
           '**envelope** (ENV), which records how far the scorers disagreed (SD per dimension, Node Value '
           'range). The envelope shows where a score is firm and where it is contested. The scorers are all '
           'large language models trained on overlapping text, so their agreement shows reproducibility, '
           'not full independence.', '',
           'Some older single-scorer series keep the ENS filename because the JUNO tools and CAMS Compare '
           'load them by that name. They are listed below as "not an ensemble".', '',
           'Rule: every ensemble (ENS) file has its envelope (ENV), covering exactly the same '
           'society/year/node rows, and no ENV exists without its ENS. Scope: `cleaned_datasets/` '
           'and `data/nations/` (the published data).', '',
           f'## Paired ({len(paired)})', '', '| Ensemble | Envelope | Rows |', '|---|---|---|']
    out += [f'| `{e}` | `{v}` | {n:,} |' for e, v, n in sorted(paired)]
    out += ['', f'## No envelope ({len(flagged)})', '',
            'Kept and flagged rather than deleted: the JUNO tools and the published dataset use them.', '',
            '| Ensemble | Status |', '|---|---|']
    out += [f'| `{e}` | {why} |' for e, why in flagged]
    out += ['', f'## Problems ({len(errors)})', ''] + ([f'- {x}' for x in errors] or ['None.'])
    open('data/ENVELOPE_STATUS.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')

    print(f'{len(paired)} paired, {len(flagged)} flagged without envelope, {len(errors)} problems')
    for x in errors:
        print('  PROBLEM', x)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
