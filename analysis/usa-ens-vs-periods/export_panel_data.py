#!/usr/bin/env python3
"""Export the small data set behind usa-ens-vs-periods.html (29 Oct 2026).
Reads ONLY computed outputs of the two frozen runs, plus the three rater files to rebuild the rater-consensus
node profiles with the frozen consensus() function (the runs did not save them). Writes panel_data.json and
period_profiles_compare.csv beside this script. No statistic is computed here that the runs did not compute,
except the per-period Pearson r, which is asserted equal to the value in run_log.txt.
Paths are the box paths used on 29 Oct 2026."""
import sys, json, re, ast
sys.path.insert(0, '/workspace/analysis/usa_ensemble_vs_periods')
import numpy as np, pandas as pd
import run as RUN

S1 = '/workspace/analysis/usa_ensemble_vs_periods/'
A = S1 + 'canonical_usa_ens_annual/'
S2 = '/workspace/analysis/usa_external_marker_lead/'
HERE = '/workspace/wm_site/analysis/usa-ens-vs-periods/'
NODES = RUN.NODES

log = open(A + 'run_log.txt').read()
log2 = open(S2 + 'run_log.txt').read()

# ---- Study 1, H1: period profiles (canonical annual) vs rater consensus
ens = pd.read_csv(A + 'period_profiles.csv', index_col=0)
raters = {k: RUN.rater_Q(p) for k, p in RUN.RATERS.items()}
cons = RUN.consensus(raters)
logr = ast.literal_eval(re.search(r"H1 per-period r: (\{.*\})", log).group(1))
periods, rows = [], []
for c, label, y0, y1 in RUN.PERIODS:
    e, r = ens.loc[c, NODES].astype(float), cons.loc[c, NODES].astype(float)
    m = r.notna() & e.notna()
    rr = float(np.corrcoef(e[m], r[m])[0, 1])
    assert abs(rr - logr[c]) < 5e-4, (c, rr, logr[c])
    periods.append(dict(id=c, label=label, y0=y0, y1=y1,
                        ens=[round(float(x), 2) for x in e],
                        cons=[None if np.isnan(x) else round(float(x), 2) for x in r],
                        raters={k: [None if np.isnan(x) else round(float(x), 2) for x in v.loc[c, NODES]] for k, v in raters.items()},
                        r=round(float(logr[c]), 3), n_cons=int(m.sum()),
                        mean_ens=round(float(e.mean()), 2), mean_cons=round(float(r.mean()), 2)))
    for n in NODES:
        rows.append(dict(period=c, label=label, y0=y0, y1=y1, node=n, ensemble_Q=round(float(e[n]), 3),
                         consensus_Q=None if np.isnan(r[n]) else round(float(r[n]), 3)))
pd.DataFrame(rows).to_csv(HERE + 'period_profiles_compare.csv', index=False)
m1 = re.search(r"H1: rho=([\d.]+) p_a=([\d.]+); median r=([\d.]+) p_b=([\d.]+)", log)
h1 = dict(rho=float(m1[1]), p_a=float(m1[2]), median_r=float(m1[3]), p_b=float(m1[4]))

# ---- Study 1, H3
h3 = pd.read_csv(A + 'h3.csv')
H3 = []
for r in h3.itertuples():
    per = ast.literal_eval(r.per_rater.replace('np.float64', ''))
    H3.append(dict(name=r.transition, expected_sign=int(r.expected_sign), delta_ens=round(r.delta_ensemble, 2),
                   delta_cons=round(r.delta_rater, 2), per_rater={k: round(float(v), 2) for k, v in per.items()},
                   ci_lo=round(r.ci_lo, 2), ci_hi=round(r.ci_hi, 2), p=round(r.p, 4), nodes=r.nodes_matching, passes=bool(r.passes)))

# ---- Study 1, H2 (annual drift X^A)
XA = pd.read_csv(A + 'drift_by_year.csv', index_col=0).iloc[:, 0]
UA = pd.read_csv(A + 'upcrossings.csv', index_col=0).iloc[:, 0].astype(bool)
thrA = float(re.search(r"thr\(P90\)=([\d.]+)", log)[1])
h2 = pd.read_csv(A + 'h2.csv')
m2 = re.search(r"H2: S_lead=(\d+)/14; S_lag=(\d+)/14; p_N1=([\d.]+).*?p_N2=([\d.]+)", log)
H2 = dict(thr=thrA, S_lead=int(m2[1]), S_lag=int(m2[2]), p_N1=float(m2[3]), p_N2=float(m2[4]), n=14)
trans = [dict(T=int(r.T), lead=bool(r.lead_hit), lag=bool(r.lag_hit)) for r in h2.itertuples()]

# ---- Study 2 (state-based drift X^S)
XS = pd.read_csv(S2 + 'drift_XS_by_year.csv', index_col=0).iloc[:, 0]
UXS = pd.read_csv(S2 + 'upcrossings_XS.csv', index_col=0).iloc[:, 0].astype(bool)
thrS = float(re.search(r"X\^S[^\n]*?thr\(P90\)=([\d.]+)", log2)[1]) if re.search(r"X\^S[^\n]*?thr\(P90\)=([\d.]+)", log2) else None
rt = pd.read_csv(S2 + 'results_table.csv')
markers = {}
for key, f, name in [('E1', 'hits_E1.csv', 'NBER peaks (primary)'), ('E2', 'hits_E2.csv', 'Party-control switches'),
                     ('E3', 'hits_E3.csv', 'Amendment ratifications'), ('C1', 'hits_C1.csv', 'Census years (negative control)')]:
    h = pd.read_csv(S2 + f)
    row = rt[rt.label.str.startswith(key + ' ')].iloc[0]
    markers[key] = dict(name=name, n=int(row.n), S_lead=int(row.S_lead), S_lag=int(row.S_lag), p_N1=float(row.p_N1), p_N2=float(row.p_N2),
                        events=[dict(T=int(r.T), lead=bool(r.lead_hit), lag=bool(r.lag_hit)) for r in h.itertuples()])

def series(s):
    return dict(years=[int(y) for y in s.index], x=[None if np.isnan(v) else round(float(v), 4) for v in s.values])

out = dict(
    generated='2026-10-29',
    nodes=NODES,
    study1=dict(periods=periods, h1=h1, h3=H3, h2=H2, transitions=trans,
                drift=dict(**series(XA), thr=thrA, upcross=[int(y) for y in UA.index[UA]])),
    study2=dict(drift=dict(**series(XS), thr=thrS, upcross=[int(y) for y in UXS.index[UXS]]), markers=markers,
                table=[dict(label=r.label, n=int(r.n), S_lead=int(r.S_lead), S_lag=int(r.S_lag), p_N1=float(r.p_N1), p_N2=float(r.p_N2)) for r in rt.itertuples()]),
)
json.dump(out, open(HERE + 'panel_data.json', 'w'), separators=(',', ':'))
print('thrS', thrS, 'upS', out['study2']['drift']['upcross'], 'upA', out['study1']['drift']['upcross'])
print('H1', h1, 'H2', H2)
