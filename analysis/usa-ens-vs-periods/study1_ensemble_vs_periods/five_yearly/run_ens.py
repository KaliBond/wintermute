#!/usr/bin/env python3
"""Frozen PREREG.md rules applied to USA_ENS.csv (canonical per Kari, 29 Oct 2026, after hi-fi test seen).
Imports the frozen machinery from ../run.py unchanged. USA_ENS has 26 sampled years (5-year spacing, plus 2026), not annual,
so rules that need annual data are reported as NOT EVALUABLE, never imputed."""
import sys, hashlib
sys.path.insert(0, '/workspace/analysis/usa_ensemble_vs_periods')
import numpy as np, pandas as pd
import run as RUN
from run import *

ENSF = '/workspace/wintermute/data/nations/USA_ENS.csv'
HERE = '/workspace/analysis/usa_ensemble_vs_periods/canonical_usa_ens/'
EXP_MD5 = '2f51ff07a507324e6cfcc272e00e1fa4'
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# checks
assert hashlib.md5(open(ENSF, 'rb').read()).hexdigest() == EXP_MD5
for p, h in RUN.MD5.items():
    if p != RUN.ENS: assert RUN.md5(p) == h
fro = dict(l.split(None, 1) for l in open('/workspace/analysis/usa_ensemble_vs_periods/FROZEN.txt').read().splitlines() if l.strip() and not l.startswith('#'))
assert fro['PREREG_sha256'].strip() == RUN.sha256('/workspace/analysis/usa_ensemble_vs_periods/PREREG.md'), 'PREREG changed'
d = pd.read_csv(ENSF); ens_cols = list(d.columns); d['Q'] = d.Abstraction * d.Coherence + d.Capacity - d.Stress
Q = d.pivot(index='Year', columns='Node', values='Q')[NODES]
yrs = list(Q.index)
P('# USA_ENS canonical run (frozen PREREG, sha256 matched FROZEN.txt)')
P(f'file md5 {EXP_MD5}; rows {len(d)}; years {len(yrs)}: {yrs}; NaN cells {int(d.isna().sum().sum())}; Society values {list(d.Society.unique())}; no # header lines')
P('columns identical to hi-fi file: ', ens_cols == list(pd.read_csv(RUN.ENS, comment="#").columns))
ys_all = [y for v in period_years().values() for y in v]
gap_in_ens = [y for y in yrs if y not in ys_all]
P('sampled years falling in no frozen period (gap years):', gap_in_ens)

def prof_sparse(Q):
    rows, n = {}, {}
    for c, ys in period_years().items():
        have = [y for y in ys if y in Q.index]
        n[c] = len(have)
        rows[c] = Q.loc[have].mean() if have else pd.Series(np.nan, index=NODES)
    return pd.DataFrame(rows).T[NODES], pd.Series(n)

prof, nyr = prof_sparse(Q)
lab = {c: l for c, l, a, b in PERIODS}
P('\n## Sampled years per frozen period')
P(pd.DataFrame({'label': pd.Series(lab), 'n_sampled_years': nyr, 'years': {c: [y for y in ys if y in Q.index] for c, ys in period_years().items()}}).to_string())
raters = {k: rater_Q(p) for k, p in RATERS.items()}
cons = consensus(raters)

def h1_on(cases, prof, tag):
    sv = RUN.PERIODS
    RUN.PERIODS = [p for p in sv if p[0] in cases]
    try:
        r = RUN.h1(prof, cons, np.random.default_rng(SEED))
    finally:
        RUN.PERIODS = sv
    r['n_periods'] = len(cases)
    P(f"{tag}: n periods={len(cases)} rho={r['rho']:.3f} p_a={r['p_a']:.4f}; median r={r['median_r']:.3f} p_b={r['p_b']:.4f}; thresholds met={r['thresholds_met']}; dropped from H1b={r['dropped']}")
    return r

strict = [c for c in nyr.index if nyr[c] >= 2]
relax = [c for c in nyr.index if nyr[c] >= 1]
P('\n## H1 STRICT (frozen: period needs >=2 sampled years; section 6)')
R1s = h1_on(strict, prof, 'H1 strict')
P('\n## H1 RELAXED (POST HOC, labelled; periods with >=1 sampled year; not confirmatory)')
R1r = h1_on(relax, prof, 'H1 relaxed')

P('\n## H2 (annual drift X needs 5 consecutive annual years before t)')
X = drift_X(Q)
P('drift_X values computable on USA_ENS:', len(X), '(every year has no 5 preceding annual years; spacing is 5 years)')
P('=> H2 NOT EVALUABLE. No threshold, upcrossing, lead or lag count, or null p can be computed without imputing years. Counted as not supported.')
nb = [t for t in Q.index if sum(Q[NODES[0]].reindex(range(t - BASE_W, t)).notna()) >= BASE_MIN]
P('years with >=10 observed baseline years in the trailing 20 (needed by D, acute z):', len(nb), '=> secondaries D, acute z, coupling (10 annual obs in window) NOT EVALUABLE')

P('\n## H3 (frozen needs both periods retained, i.e. >=2 sampled years; plus bootstrap CI)')
rows = []
for name, a, b, exp in NAMED:
    nodes = [n for n in NODES if not (np.isnan(cons.loc[a, n]) or np.isnan(cons.loc[b, n]))]
    dE = prof.loc[b, nodes].mean() - prof.loc[a, nodes].mean()
    dR = cons.loc[b, nodes].mean() - cons.loc[a, nodes].mean()
    per = {k: round(float(v.loc[b, nodes].mean() - v.loc[a, nodes].mean()), 3) for k, v in raters.items()}
    nm = sum(np.sign(prof.loc[b, n] - prof.loc[a, n]) == np.sign(cons.loc[b, n] - cons.loc[a, n]) for n in nodes)
    rows.append(dict(transition=name, years_a=nyr[a], years_b=nyr[b], strict_testable=(nyr[a] >= 2 and nyr[b] >= 2),
                     delta_ens=round(float(dE), 3), delta_rater=round(float(dR), 3), per_rater=per,
                     same_sign=bool(np.sign(dE) == np.sign(dR)), nodes_matching=f'{nm}/{len(nodes)}',
                     CI='degenerate (a period has 1 sampled year, no within-period resampling variation)'))
H3 = pd.DataFrame(rows); P(H3.to_string())
P('=> H3 STRICT: NOT EVALUABLE for any of the three transitions (each involves a 1-year period). Not supported. Signs above are descriptive only (relaxed, post hoc).')

holm_p = {'H1': R1s['p_H1'], 'H2': 1.0, 'H3': 1.0}
P('\n## Holm (convention for non-evaluable tests: p set to 1.0, i.e. cannot pass)')
hm = holm(holm_p); P('p used:', holm_p); P('Holm:', hm)
v1 = R1s['thresholds_met'] and hm['H1']
P(f'H1 strict supported={v1}  H2 supported=False (not evaluable)  H3 supported=False (not evaluable)')

# like-for-like control: hi-fi annual series subsampled to the ENS years (descriptive, post hoc)
P('\n## Control (POST HOC, descriptive): hi-fi annual series subsampled to the USA_ENS sampled years')
dh, Qh = load_ensemble()
common = [y for y in yrs if y in Qh.index]
Qs = Qh.loc[common]
profh, nyh = prof_sparse(Qs)
cs = [c for c in nyh.index if nyh[c] >= 2]
R1h = h1_on(cs, profh, 'hi-fi subsampled H1 strict')
a, b = Qs.values.ravel(), Q.loc[common].values.ravel()
P(f'shared year-node cells {len(a)} (years {common[0]}-{common[-1]}, {len(common)} years): Pearson r hi-fi vs ENS Q = {np.corrcoef(a, b)[0, 1]:.3f}; mean Q hi-fi {a.mean():.2f}, ENS {b.mean():.2f}')
# like-for-like named signs on subsample
for name, aa, bb, exp in NAMED:
    nodes = [n for n in NODES if not (np.isnan(cons.loc[aa, n]) or np.isnan(cons.loc[bb, n]))]
    P(f'  {name}: delta hi-fi subsampled {profh.loc[bb, nodes].mean() - profh.loc[aa, nodes].mean():.3f}')
# full hi-fi annual H1 on periods retained under ENS strict set (for side-by-side)
Hfull = pd.read_csv('/workspace/analysis/usa_ensemble_vs_periods/period_profiles.csv', index_col=0)
R1hf = h1_on(strict, Hfull, 'hi-fi annual, same 7 periods as ENS strict')
open(HERE + 'results.md', 'w').write('\n'.join(out) + '\n')
H3.to_csv(HERE + 'h3.csv', index=False); prof.assign(n_years=nyr).to_csv(HERE + 'period_profiles.csv')
