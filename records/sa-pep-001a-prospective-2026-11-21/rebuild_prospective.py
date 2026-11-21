"""Rebuild prospective_scores_2024_2026.csv and prospective_predictor_D_Helm_2026.csv from sources/.
Scores only.  Run from this folder: python3 rebuild_prospective.py ; outputs *_rebuilt.csv for byte comparison.
Normalisation and union rules: census_2026_11_21.norm and census_R3_5 (JUNO v3 first; nation_blind
society-years not in JUNO; new societies from the camnations5 and Vietnam files).
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from census_2026_11_21 import norm, NODES
from decoupling_R3_4 import decoupling
SRC = 'sources/'
COLLATION_25 = ['Argentina', 'Australia', 'Belgium', 'Cambodia', 'Chile', 'Colombia', 'France', 'Germany', 'Greece',
                'Hong Kong', 'Iran', 'Israel', 'Japan', 'Laos', 'Mongolia', 'New Zealand', 'Norway', 'Poland', 'Russia',
                'Saudi Arabia', 'Singapore', 'Sri Lanka', 'UK', 'USA', 'Ukraine']
NEW = {'Democratic Republic of the Congo': 'Users__julie__wt-best__analysis__camnations5_drc_1960_2026.csv',
       'Egypt': 'Users__julie__wt-best__analysis__camnations5_egypt_1952_2026.csv',
       'Eritrea': 'Users__julie__wt-best__analysis__camnations5_eritrea_1993_2026.csv',
       'Ethiopia': 'Users__julie__wt-best__analysis__camnations5_ethiopia_1960_2026.csv',
       'Mauritania': 'Users__julie__wt-best__analysis__camnations5_mauritania_1960_2026.csv',
       'Nigeria': 'Users__julie__wt-best__analysis__camnations5_nigeria_1960_2026.csv',
       'Vietnam': 'Users__julie__wt-best__juno__Vietnam_CAMS_mfin_1965-2026_block1.csv'}
J = norm(pd.read_csv(SRC + 'JUNO_Unified_Dataset_v3_2026-10-19.csv'), 'Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'); J['src'] = 'JUNO_v3'
nb = pd.read_csv(SRC + 'nation_blind_ensemble_primitives.csv')
key = {s['set_id']: s['nation'] for s in json.load(open(SRC + 'nation_blind_ensemble_KEY.json'))['sets']}
nb['soc'] = nb.Set_ID.map(key); B = norm(nb, 'soc', 'Year', 'Node', 'C', 'K', 'S', 'A'); B['src'] = 'nation_blind'
jk = set(zip(J.soc, J.year)); parts = [J, B[[(s, y) not in jk for s, y in zip(B.soc, B.year)]]]
for soc, f in NEW.items():
    d = pd.read_csv(SRC + f); d.columns = [c.strip() for c in d.columns]
    d = d[d.Node.isin(NODES)].copy(); d['Society'] = soc; d['Year'] = pd.to_numeric(d.Year, errors='coerce')
    for c in ['Coherence', 'Capacity', 'Stress', 'Abstraction']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['Year'])
    X = norm(d, 'Society', 'Year', 'Node', 'Coherence', 'Capacity', 'Stress', 'Abstraction'); X['src'] = f; parts.append(X)
U = pd.concat(parts); U['year'] = U.year.astype(int)
P = COLLATION_25 + list(NEW)
S = U[U.soc.isin(P) & U.year.between(2024, 2026)].copy()
S['set'] = np.where(S.soc.isin(COLLATION_25), 'collation_25', 'R3_5_added_7')
S['node'] = pd.Categorical(S.node, NODES, ordered=True)
S = S.sort_values(['soc', 'year', 'node'])[['set', 'soc', 'year', 'node', 'C', 'K', 'S', 'A', 'src']]
S.to_csv('prospective_scores_2024_2026_rebuilt.csv', index=False)
rows = []
for s, g in S.groupby('soc', observed=True):
    M = {y: g[g.year == y].set_index('node').loc[NODES][['C', 'K', 'A', 'S']].values.astype(float) for y in (2024, 2025, 2026)}
    rows.append({'set': g.set.iloc[0], 'soc': s, 'D_Helm_2024_2026': round(float(decoupling(M[2024][None], M[2026][None])[0]), 6),
                 'copy_2025_eq_2024': bool(np.allclose(M[2025], M[2024])), 'copy_2026_eq_2025': bool(np.allclose(M[2026], M[2025])), 'src': g.src.iloc[0]})
R = pd.DataFrame(rows)
R['tercile'] = pd.qcut(R.D_Helm_2024_2026.rank(method='first'), 3, labels=['low', 'mid', 'top'])
R.to_csv('prospective_predictor_D_Helm_2026_rebuilt.csv', index=False)
print(len(S), 'score rows;', len(R), 'societies')
