#!/usr/bin/env python3
"""Sensitivity, secondary and exploratory items exactly as specified in PREREG.md sections 4 and 5. Descriptive only."""
import numpy as np, pandas as pd
import run
from run import *

def main():
    rng = np.random.default_rng(SEED)
    d, Q = load_ensemble(); prof = period_profiles(Q)
    raters = {k: rater_Q(p) for k, p in RATERS.items()}
    X = drift_X(Q); L = []
    def h2stat(Xs, thr, lead=H2_LEAD, lag=H2_LAG, name=''):
        U = upcrossings(Xs, thr)
        Sl, Sg, _ = hits(U, TRANSITIONS, lead=lead, lag=lag)
        # N1 with this window
        yrs = list(U.index); N = len(yrs); y0 = yrs[0]
        nul = []
        for s in range(1, N):
            if min(s, N - s) <= N1_EXCL: continue
            sh = [y0 + ((T - y0 + s) % N) for T in TRANSITIONS]
            nul.append(hits(U, sh, lead=lead, lag=lag, circular=True)[0])
        L.append(dict(item=name, upcrossings=int(U.sum()), S_lead=Sl, S_lag=Sg, p_N1=pval_ge(Sl, np.array(nul))))
    h2stat(X, np.percentile(X, 85), name='X P85')
    h2stat(X, np.percentile(X, 95), name='X P95')
    h2stat(X, np.percentile(X, 90), lead=3, name='X P90 lead3')
    h2stat(X, np.percentile(X, 90), lead=8, name='X P90 lead8')
    # causal threshold
    thr_t = {t: (np.percentile(X.loc[t-20:t-1], 90) if len(X.loc[t-20:t-1]) >= 10 else np.nan) for t in X.index}
    ev = {t: bool(X[t-1] < thr_t[t-1] and X[t] >= thr_t[t]) for t in X.index if (t-1) in X.index and not np.isnan(thr_t.get(t, np.nan)) and not np.isnan(thr_t.get(t-1, np.nan))}
    Uc = pd.Series(ev); Sl, Sg, _ = hits(Uc, TRANSITIONS)
    yrs = list(Uc.index); N = len(yrs); y0 = yrs[0]
    # transitions before the first causal year cannot be evaluated; count only those with a full lead window
    L.append(dict(item=f'X causal trailing-20 P90 (evaluable {yrs[0]}-{yrs[-1]})', upcrossings=int(Uc.sum()), S_lead=Sl, S_lag=Sg, p_N1=np.nan))
    # node acute firings (|z|>=1.96) counted as events: year with at least one acute node, upcrossing of the any-acute indicator
    z = acute_z(Q); anyac = (z.abs() >= Z_ACUTE).any(axis=1) & z.notna().all(axis=1)
    ac_years = [int(t) for t in anyac.index[anyac]]
    L.append(dict(item=f'acute node-years (|z|>=1.96), years 1910-2025: {len(ac_years)} year(s) with >=1 acute node', upcrossings=len(ac_years), S_lead=sum(any(T-H2_LEAD<=y<=T-1 for y in ac_years) for T in TRANSITIONS), S_lag=sum(any(T<=y<=T+H2_LAG for y in ac_years) for T in TRANSITIONS), p_N1=np.nan))
    for nm, S, pct in [('Theta drift D (P90)', regime_D(Q), 90), ('coupling shift K (P90)', coupling_shift(Q), 90)]:
        h2stat(S, np.percentile(S.dropna(), pct), name=nm + f' [{S.index.min()}-{S.index.max()}]')
    out = pd.DataFrame(L); out.to_csv(OUT + 'secondary_h2.csv', index=False)
    Dg = regime_D(Q)
    lines = ['# Secondary / exploratory (descriptive only)', out.to_string(index=False),
             f'Theta D range {Dg.min():.3f} to {Dg.max():.3f}; years with D>=0.40: {[int(t) for t in Dg.index[Dg>=0.40]]}; years D<0.15: {int((Dg<0.15).sum())} of {len(Dg)}']
    # H1 sensitivity
    for nm, ex in [('without CLAUDE', ('CLAUDE',)), ('FROZEN_GPT only', ('CLAUDE', 'GROK')), ('CLAUDE only', ('FROZEN_GPT', 'GROK')), ('GROK only', ('FROZEN_GPT', 'CLAUDE'))]:
        use = {k: v for k, v in raters.items() if k not in ex}
        c = consensus(use) if len(use) >= 2 else list(use.values())[0]
        cases = [x for x, *_ in PERIODS]
        em = prof.loc[cases].where(c.loc[cases].notna()).mean(axis=1); rm = c.loc[cases].mean(axis=1)
        rs = []
        for cs in cases:
            m = c.loc[cs].notna()
            if m.sum() >= 6 and prof.loc[cs][m].std() > 0 and c.loc[cs][m].std() > 0: rs.append(np.corrcoef(prof.loc[cs][m], c.loc[cs][m])[0, 1])
        lines.append(f'H1 sens {nm}: rho={stats.spearmanr(em, rm)[0]:.3f}; median r={np.median(rs):.3f} (n periods {len(rs)})')
    cons = consensus(raters); cases = [x for x, *_ in PERIODS]
    e = prof.loc[cases].where(cons.loc[cases].notna()); r = cons.loc[cases]
    lines.append(f'H1 descriptive: mean ensemble Q {np.nanmean(e.values):.2f}, mean rater consensus Q {np.nanmean(r.values):.2f}')
    ev_, rv_ = e.values.ravel(), r.values.ravel(); mk = ~np.isnan(ev_) & ~np.isnan(rv_)
    M = np.column_stack([ev_[mk], rv_[mk]]); n, k = M.shape
    gm = M.mean(); rm_ = M.mean(1); cm = M.mean(0)
    SSR = k*((rm_-gm)**2).sum(); SSC = n*((cm-gm)**2).sum(); SST = ((M-gm)**2).sum(); SSE = SST-SSR-SSC
    MSR = SSR/(n-1); MSC = SSC/(k-1); MSE = SSE/((n-1)*(k-1))
    lines.append(f'ICC(3,1)={(MSR-MSE)/(MSR+(k-1)*MSE):.3f}; ICC(2,1)={(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n):.3f} on {n} cells')
    open(OUT + 'secondary.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
main()
