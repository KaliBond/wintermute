#!/usr/bin/env python3
"""Frozen PREREG.md rules applied UNCHANGED to the annual USA ENS file supplied by Kari (29 Oct 2026, after hi-fi and 5-yearly results were seen).
Machinery imported from ../run.py. 2026 row is dropped for the frozen computations (frozen window ends 2025; X, N1 grid 1906-2025)."""
import sys, hashlib
sys.path.insert(0, '/workspace/analysis/usa_ensemble_vs_periods')
import numpy as np, pandas as pd
import run as RUN
from run import *

F = '/workspace/uploads/usa_ens_annual/USA_ENS_annual.csv'
HERE = '/workspace/analysis/usa_ensemble_vs_periods/canonical_usa_ens_annual/'
OUTL = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUTL.append(s)

# frozen checks
for p, h in RUN.MD5.items():
    if p != RUN.ENS: assert RUN.md5(p) == h
fro = dict(l.split(None, 1) for l in open(RUN.OUT + 'FROZEN.txt').read().splitlines() if l.strip() and not l.startswith('#'))
assert fro['PREREG_sha256'].strip() == RUN.sha256(RUN.OUT + 'PREREG.md'), 'PREREG changed'
d = pd.read_csv(F)
P('# USA ENS ANNUAL canonical run (frozen PREREG, sha256 matched FROZEN.txt)')
P(f'file md5 {RUN.md5(F)} (box reconstruction of the uploaded file; see DEVIATIONS.md item 2); rows {len(d)}; years {d.Year.min()}-{d.Year.max()} ({d.Year.nunique()}); rows/year {sorted(d.groupby("Year").size().unique())}; NaN {int(d.isna().sum().sum())}; duplicates {int(d.duplicated(["Year","Node"]).sum())}; Society {list(d.Society.unique())}')
d['Q'] = d.Abstraction * d.Coherence + d.Capacity - d.Stress
Qall = d.pivot(index='Year', columns='Node', values='Q')[NODES]
Q = Qall.loc[1900:2025]
assert Q.shape == (126, 8)
ys = [y for v in period_years().values() for y in v]
assert set(range(1900, 2026)) - set(ys) == GAP_YEARS
P('frozen computations use 1900-2025 (126 x 8); 2026 row excluded (no frozen period, outside X/N1 grid)')

rng = np.random.default_rng(SEED)
prof = period_profiles(Q)
raters = {k: rater_Q(p) for k, p in RATERS.items()}
cons = consensus(raters)
R1 = h1(prof, cons, rng)
X = drift_X(Q); thr = float(np.percentile(X.dropna(), H2_PCT))
U = upcrossings(X, thr)
Sl, Sg, tab = hits(U, TRANSITIONS)
n1 = null_N1(U, TRANSITIONS); n2 = null_N2(U, TRANSITIONS, rng)
p_n1, p_n2 = pval_ge(Sl, n1), pval_ge(Sl, n2)
T3 = h3(Q, prof, raters, rng); p3 = float(T3.p.max())
hm = holm({'H1': R1['p_H1'], 'H2': p_n1, 'H3': p3})
v1 = R1['thresholds_met'] and hm['H1']
v2 = Sl >= H2_MIN_HITS and Sl > Sg and hm['H2'] and p_n2 <= 0.05
v3 = bool(T3.passes.all()) and hm['H3']
P(f"\nH1: rho={R1['rho']:.3f} p_a={R1['p_a']:.4f}; median r={R1['median_r']:.3f} p_b={R1['p_b']:.4f}; thresholds met={R1['thresholds_met']}; dropped={R1['dropped']}; supported={v1}")
P('H1 per-period r:', {k: round(float(v), 3) for k, v in R1['per_period_r'].items()})
P(f"H2: X defined {X.index.min()}-{X.index.max()} (n={len(X)}); X==0 in {int((X<1e-12).sum())} years; distinct X values {X.round(10).nunique()}; thr(P90)={thr:.4f}; upcrossings={int(U.sum())} in {[int(t) for t in U.index[U]]}")
P(f"H2: S_lead={Sl}/14; S_lag={Sg}/14; p_N1={p_n1:.4f} (n shifts {len(n1)}, null mean {n1.mean():.2f}, max {n1.max()}); p_N2={p_n2:.4f} (null mean {n2.mean():.2f}); supported={v2}")
P('H2 per transition:'); P(tab.to_string(index=False))
P('H3:'); P(T3.drop(columns=['per_rater']).to_string(index=False))
P('H3 per-rater deltas:', {r.transition: {k: round(float(v), 2) for k, v in r.per_rater.items()} for r in T3.itertuples()})
P(f"H3: all pass={bool(T3.passes.all())}; max p={p3:.4f}; supported={v3}")
P('Holm p used:', {'H1': R1['p_H1'], 'H2': p_n2 if False else p_n1, 'H3': p3}); P('Holm:', hm)
P(f'\nSUMMARY H1 supported={v1}  H2 supported={v2}  H3 supported={v3}')
prof.to_csv(HERE + 'period_profiles.csv'); X.to_csv(HERE + 'drift_by_year.csv'); U.to_csv(HERE + 'upcrossings.csv'); tab.to_csv(HERE + 'h2.csv', index=False); T3.to_csv(HERE + 'h3.csv', index=False)

# ---- secondary / exploratory (same set as secondary.py; descriptive) ----
P('\n# Secondary / exploratory (descriptive only)')
L = []
def h2stat(Xs, thr_, lead=H2_LEAD, lag=H2_LAG, name=''):
    Uu = upcrossings(Xs, thr_); sl, sg, _ = hits(Uu, TRANSITIONS, lead=lead, lag=lag)
    yy = list(Uu.index); N = len(yy); y0 = yy[0]; nul = []
    for s in range(1, N):
        if min(s, N - s) <= N1_EXCL: continue
        sh = [y0 + ((T - y0 + s) % N) for T in TRANSITIONS]
        nul.append(hits(Uu, sh, lead=lead, lag=lag, circular=True)[0])
    L.append(dict(item=name, thr=round(float(thr_), 4), upcrossings=int(Uu.sum()), S_lead=sl, S_lag=sg, p_N1=round(pval_ge(sl, np.array(nul)), 4)))
h2stat(X, np.percentile(X, 85), name='X P85'); h2stat(X, np.percentile(X, 95), name='X P95')
h2stat(X, np.percentile(X, 90), lead=3, name='X P90 lead3'); h2stat(X, np.percentile(X, 90), lead=8, name='X P90 lead8')
thr_t = {t: (np.percentile(X.loc[t-20:t-1], 90) if len(X.loc[t-20:t-1]) >= 10 else np.nan) for t in X.index}
ev = {t: bool(X[t-1] < thr_t[t-1] and X[t] >= thr_t[t]) for t in X.index if (t-1) in X.index and not np.isnan(thr_t.get(t, np.nan)) and not np.isnan(thr_t.get(t-1, np.nan))}
Uc = pd.Series(ev); sl, sg, _ = hits(Uc, TRANSITIONS)
L.append(dict(item=f'X causal trailing-20 P90 (evaluable {Uc.index[0]}-{Uc.index[-1]})', thr=np.nan, upcrossings=int(Uc.sum()), S_lead=sl, S_lag=sg, p_N1=np.nan))
z = acute_z(Q); anyac = (z.abs() >= Z_ACUTE).any(axis=1) & z.notna().all(axis=1)
ac = [int(t) for t in anyac.index[anyac]]
L.append(dict(item=f'acute node-years |z|>=1.96: {len(ac)} years with >=1 acute node', thr=np.nan, upcrossings=len(ac), S_lead=sum(any(T-H2_LEAD<=y<=T-1 for y in ac) for T in TRANSITIONS), S_lag=sum(any(T<=y<=T+H2_LAG for y in ac) for T in TRANSITIONS), p_N1=np.nan))
Dg = regime_D(Q); Kg = coupling_shift(Q)
for nm, S in [('Theta drift D (P90)', Dg), ('coupling shift K (P90)', Kg)]:
    h2stat(S, np.percentile(S.dropna(), 90), name=nm + f' [{S.index.min()}-{S.index.max()}]')
sec = pd.DataFrame(L); sec.to_csv(HERE + 'secondary_h2.csv', index=False); P(sec.to_string(index=False))
P('acute years:', ac)
P(f'Theta D range {Dg.min():.3f} to {Dg.max():.3f}; years D>=0.40: {[int(t) for t in Dg.index[Dg>=0.40]]}; years D<0.15: {int((Dg<0.15).sum())} of {len(Dg)}')
cases = [x for x, *_ in PERIODS]
for nm, ex in [('without CLAUDE', ('CLAUDE',)), ('FROZEN_GPT only', ('CLAUDE', 'GROK')), ('CLAUDE only', ('FROZEN_GPT', 'GROK')), ('GROK only', ('FROZEN_GPT', 'CLAUDE'))]:
    use = {k: v for k, v in raters.items() if k not in ex}
    c = consensus(use) if len(use) >= 2 else list(use.values())[0]
    em = prof.loc[cases].where(c.loc[cases].notna()).mean(axis=1); rm = c.loc[cases].mean(axis=1); rs = []
    for cs in cases:
        m = c.loc[cs].notna()
        if m.sum() >= 6 and prof.loc[cs][m].std() > 0 and c.loc[cs][m].std() > 0: rs.append(np.corrcoef(prof.loc[cs][m], c.loc[cs][m])[0, 1])
    P(f'H1 sens {nm}: rho={stats.spearmanr(em, rm)[0]:.3f}; median r={np.median(rs):.3f} (n periods {len(rs)})')
e = prof.loc[cases].where(cons.loc[cases].notna()); r = cons.loc[cases]
P(f'H1 descriptive: mean ensemble Q {np.nanmean(e.values):.2f}, mean rater consensus Q {np.nanmean(r.values):.2f}')
ev_, rv_ = e.values.ravel(), r.values.ravel(); mk = ~np.isnan(ev_) & ~np.isnan(rv_)
M = np.column_stack([ev_[mk], rv_[mk]]); n, k = M.shape; gm = M.mean(); rm_ = M.mean(1); cm = M.mean(0)
SSR = k*((rm_-gm)**2).sum(); SSC = n*((cm-gm)**2).sum(); SST = ((M-gm)**2).sum(); SSE = SST-SSR-SSC
MSR = SSR/(n-1); MSC = SSC/(k-1); MSE = SSE/((n-1)*(k-1))
P(f'ICC(3,1)={(MSR-MSE)/(MSR+(k-1)*MSE):.3f}; ICC(2,1)={(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n):.3f} on {n} cells')

# ---- structure descriptives ----
P('\n# Series structure (descriptive)')
same = int(sum((Q.iloc[i] == Q.iloc[i-1]).all() for i in range(1, len(Q))))
P(f'years whose eight Q equal the previous year exactly: {same} of {len(Q)-1}; distinct annual states: {len(Q.drop_duplicates())} of {len(Q)}')
dh, Qh = load_ensemble()
a, b = Q.values.ravel(), Qh.values.ravel()
P(f'vs hi-fi annual (126x8 cells): Pearson r={np.corrcoef(a,b)[0,1]:.3f}; mean Q annual-ENS {a.mean():.2f}, hi-fi {b.mean():.2f}; per-node r: ' + ', '.join(f'{n_}={np.corrcoef(Q[n_],Qh[n_])[0,1]:.2f}' for n_ in NODES))
Xh = pd.read_csv(RUN.OUT + 'drift_by_year.csv', index_col=0).iloc[:, 0]
P(f'drift X correlation with hi-fi X (1905-2025): Spearman {stats.spearmanr(X, Xh.reindex(X.index))[0]:.3f}')
open(HERE + 'results.md', 'w').write('\n'.join(OUTL) + '\n')
