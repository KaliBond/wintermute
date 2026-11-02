"""Recompute every dataset figure quoted in the Lab Chief commentary (records/PUTATIVE_ISSUES.md, LC-008).

Reads ../juno/JUNO_Unified_Dataset.csv (SHA-256 49ee7b173abc5856...) and prints the numbers.
Node Value is read as stored (Node_Value_Calc = C + K - S + 0.5A). Run from the repo root:
    python3 records/lab-chief-checks/recompute.py
"""
import numpy as np, pandas as pd

j = pd.read_csv('juno/JUNO_Unified_Dataset.csv', low_memory=False)
print('rows', len(j), '| societies', j.Society.nunique(), '| years', j.Year.min(), '-', j.Year.max())
gy = j.groupby(['Society', 'Year'])
a = gy.agg(V_mean=('Node_Value_Calc', 'mean'), B_eff=('Bond_Strength_Calc', 'mean'),
           lam2=('Lambda2_Calc', 'first'), S_mean=('Stress', 'mean')).reset_index()
print('society-years', len(a))

# 1. bond strength against lambda2
pooled = a[['B_eff', 'lam2']].corr().iloc[0, 1]
within = a.groupby('Society').apply(lambda g: g[['B_eff', 'lam2']].corr().iloc[0, 1] if len(g) > 3 else np.nan)
print(f'bond vs lambda2: pooled r = {pooled:.3f}; median within-society r = {within.median():.3f}')

# 2. the slide, 2000 -> 2025
p = a.pivot(index='Year', columns='Society', values='V_mean')
s = a.pivot(index='Year', columns='Society', values='S_mean')
ok = [c for c in p.columns if pd.notna(p.loc[2000, c]) and pd.notna(p.loc[2025, c])]
print(f'2000->2025: {len(ok)} comparable societies; V declined in {sum(p.loc[2025,c] < p.loc[2000,c] for c in ok)};'
      f' mean Stress rose in {sum(s.loc[2025,c] > s.loc[2000,c] for c in ok)}')

# 3. responsiveness: slope of each node's yearly change on the mean change of the other seven
nv = j.pivot_table(index=['Society', 'Year'], columns='Node', values='Node_Value_Calc')
d = nv.groupby(level=0).diff().dropna()
tot = d.sum(axis=1)
resp = {n: np.cov((tot - d[n]) / 7, d[n])[0, 1] / np.var((tot - d[n]) / 7, ddof=1) for n in d.columns}
print('responsiveness (leave-self-out):', {k: round(v, 2) for k, v in sorted(resp.items(), key=lambda kv: kv[1])})
X = d.values - d.values.mean(0)
sv = np.linalg.svd(X, compute_uv=False)
print(f'one shared movement: PC1 share of node-change variance = {sv[0]**2 / (sv**2).sum():.3f} ({len(d)} node-transition rows)')

# 4. common pattern in stress by era (balanced societies only, z-scored, no interpolation)
st = s.copy()
for lo, hi in [(1900, 1949), (1950, 1999), (2000, 2025)]:
    sub = st.loc[lo:hi]
    cols = [c for c in sub.columns if sub[c].notna().sum() == len(sub)]
    z = (sub[cols] - sub[cols].mean()) / sub[cols].std().replace(0, 1)
    sv = np.linalg.svd(z.dropna(axis=1).values, compute_uv=False)
    print(f'stress PC1 share {lo}-{hi}: {sv[0]**2 / (sv**2).sum():.3f} ({len(cols)} balanced societies)')

# 5. United States against its own 1900-1999 readings
g = a[a.Society == 'USA'].set_index('Year').V_mean
ref = g.loc[1900:1999]
print(f'USA: {len(g)} society-years; 1900-1999 n={len(ref)} mean {ref.mean():.2f} SD {ref.std():.2f};'
      f' 2026 V = {g.iloc[-1]:.2f}; z = {(g.iloc[-1] - ref.mean()) / ref.std():.2f}')

# 6. Cambodia 1973-1981
c = a[(a.Society == 'Cambodia') & a.Year.between(1973, 1981)].set_index('Year')
print(c[['V_mean', 'B_eff', 'lam2']].round(3).to_string())
