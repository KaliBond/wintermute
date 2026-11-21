"""SA-PEP-001 R3.5 task 1: updated census (scores only, no outcome data).
Union = R3.4-collation union (JUNO v3 + nation_blind additions) + fetch2 new societies.
Rules unchanged from census_2026_11_21.py (complete canonical 8x4, consecutive years
t-2..t, outcome window t+1..t+3 closing by 2025; stride-3 and stride-6 greedy counts).
Primary file per new society (latest copy; alternatives kept for the halo analysis only):
  DRC, Egypt, Eritrea, Ethiopia, Mauritania, Nigeria : wt-best/analysis/camnations5_*  (identical to wt-war-thesis, wt-step1 copies)
  Vietnam   : wt-best/juno/Vietnam_CAMS_mfin_1965-2026_block1 (DRV 1965-75 and SRV 1976- merged as one society 'Vietnam')
  Latvia    : cleaned_datasets/Latvia_ENS_1901_2025_cleaned (Latvia_R2 rescore = alternative scoring)
  Philippines: cleaned_datasets/Philippines_ENS_1880_2016_cleaned (phil_gem_dec = alternative scoring)
Excluded: Turkiye_CAMS5 (5-yearly, 0 annual anchors); kimi blinded_001..004 (no key, rescaled
values, correlate with known JUNO series, e.g. blinded_003 vs Italy r=0.95: probable derivatives);
JUNO_Blind_Test/keyed_* (100% row match to JUNO: duplicate).
"""
import glob, json, sys
import pandas as pd, numpy as np
sys.path.insert(0, '/workspace/SA-PEP')
from census_2026_11_21 import norm, report, NODES
D = '/workspace/SA-PEP-collate'
F = D + '/fetch2/'

def load_union_R3_4():
    j = pd.read_csv(f'{D}/JUNO_Unified_Dataset_v3_2026-10-19.csv')
    J = norm(j, 'Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'); J['src'] = 'JUNO_v3'
    nb = pd.read_csv(f'{D}/nation_blind_ensemble_primitives.csv')
    key = {s['set_id']: s['nation'] for s in json.load(open(f'{D}/nation_blind_ensemble_KEY.json'))['sets']}
    nb['soc'] = nb.Set_ID.map(key); B = norm(nb, 'soc', 'Year', 'Node', 'C', 'K', 'S', 'A'); B['src'] = 'nation_blind'
    jk = set(zip(J.soc, J.year))
    return pd.concat([J, B[[(s, y) not in jk for s, y in zip(B.soc, B.year)]]])

NEW = {'Democratic Republic of the Congo': 'Users__julie__wt-best__analysis__camnations5_drc_1960_2026.csv',
       'Egypt': 'Users__julie__wt-best__analysis__camnations5_egypt_1952_2026.csv',
       'Eritrea': 'Users__julie__wt-best__analysis__camnations5_eritrea_1993_2026.csv',
       'Ethiopia': 'Users__julie__wt-best__analysis__camnations5_ethiopia_1960_2026.csv',
       'Mauritania': 'Users__julie__wt-best__analysis__camnations5_mauritania_1960_2026.csv',
       'Nigeria': 'Users__julie__wt-best__analysis__camnations5_nigeria_1960_2026.csv',
       'Vietnam': 'Users__julie__wt-best__juno__Vietnam_CAMS_mfin_1965-2026_block1.csv',
       'Latvia': 'Users__julie__wt-best__cleaned_datasets__Latvia_ENS_1901_2025_cleaned.csv',
       'Philippines': 'Users__julie__wt-best__cleaned_datasets__Philippines_ENS_1880_2016_cleaned.csv'}

def load_new():
    out = []
    for soc, f in NEW.items():
        d = pd.read_csv(F + f); d.columns = [c.strip() for c in d.columns]
        d = d[d.Node.isin(NODES)].copy(); d['Society'] = soc
        d['Year'] = pd.to_numeric(d.Year, errors='coerce')
        for c in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=['Year'])
        X = norm(d, 'Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'); X['src'] = f
        out.append(X)
    return pd.concat(out)

def union():
    U0 = load_union_R3_4(); N = load_new()
    assert not set(N.soc) & set(U0.soc)
    U = pd.concat([U0, N]); U['year'] = U.year.astype(int)
    return U

if __name__ == '__main__':
    U = union()
    U.to_csv('/workspace/SA-PEP/union_scores_R3_5.csv', index=False)
    r = report('union_R3_5', U)
    r.to_csv('/workspace/SA-PEP/census_R3_5_cases.csv', index=False)
    tot = r[['complete_years', 'anchors_all', 'anchors_stride3', 'anchors_stride6']].sum()
    print(r[r.soc.isin(list(NEW))].to_string()); print(tot.to_dict())
    print('societies', int((r.complete_years > 0).sum()), 'with stride6>=1', int((r.anchors_stride6 > 0).sum()))
    print('stride6 excl Denmark', int(tot.anchors_stride6 - r[r.soc == 'Denmark'].anchors_stride6.sum()))
    print('2026 prospective anchors', sum(2026 in x for x in r.prospective_anchors))
