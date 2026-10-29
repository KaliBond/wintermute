#!/usr/bin/env python3
"""USA ensemble vs casebook periods. Implements PREREG.md. Outcomes are NOT computed on import.

  python3 run.py --check   structure and checksum checks only (no outcome statistics)
  python3 run.py --run     full analysis. Only after PREREG.md is signed off and FROZEN.txt exists.

Every rule below mirrors PREREG.md (section numbers in comments). Do not edit thresholds here:
record any change in the PREREG.md Deviation log first.
"""
import argparse, hashlib, itertools, sys
import numpy as np, pandas as pd
from scipy import stats

SEED = 20261029
B = 10000
NODES = ['Helm', 'Shield', 'Lore', 'Stewards', 'Craft', 'Hands', 'Archive', 'Flow']
ENS = '/workspace/wintermute/data/nations/USA_v12opt_hifi_block1.csv'
RD = '/workspace/wt-blindrater/analysis/blind_rater_accidental_test/data/'
MD5 = {
    ENS: 'cd8619ebddeef95925dd6ad5ec90a064',
    RD + 'BLIND_CASEBOOK_200.csv': '6a3fdb6335d4d3fe475c63e614b9757a',
    RD + 'BLIND_RESPONSE_FROZEN_GPT_200.csv': '3e30d87f7a993594f9ed0135296c0367',
    RD + 'BLIND_RESPONSE_CLAUDE_200.csv': 'b263cb6b3d3318be080e9a938d85263d',
    RD + 'BLIND_RESPONSE_GROK_200.csv': '0a3f30b58e3e31490e13196fbf4ffa00',
}
RATERS = {'FROZEN_GPT': RD + 'BLIND_RESPONSE_FROZEN_GPT_200.csv',
          'CLAUDE': RD + 'BLIND_RESPONSE_CLAUDE_200.csv',
          'GROK': RD + 'BLIND_RESPONSE_GROK_200.csv'}
OUT = '/workspace/analysis/usa_ensemble_vs_periods/'

# Section 3: frozen period table (CaseID, label, first year, last year)
PERIODS = [
    ('CASE-18', 'Progressive era', 1901, 1916), ('CASE-01', 'WWI', 1917, 1918),
    ('CASE-16', '1920s boom', 1921, 1929), ('CASE-14', 'Great Depression', 1930, 1932),
    ('CASE-05', 'New Deal', 1933, 1940), ('CASE-09', 'WWII', 1941, 1945),
    ('CASE-04', 'Postwar', 1946, 1960), ('CASE-21', '1960s', 1961, 1972),
    ('CASE-17', '1970s', 1973, 1980), ('CASE-06', 'Reagan 1980s', 1981, 1989),
    ('CASE-23', '1991-2001', 1991, 2000), ('CASE-10', 'Post-9/11', 2001, 2008),
    ('CASE-08', 'GFC aftermath', 2009, 2015), ('CASE-03', 'Realignment 2016-20', 2016, 2019),
    ('CASE-24', '2020-25', 2020, 2025)]
GAP_YEARS = {1900, 1919, 1920, 1990}
TRANSITIONS = [1917, 1921, 1930, 1933, 1941, 1946, 1961, 1973, 1981, 1991, 2001, 2009, 2016, 2020]
NAMED = [('N-a 1920s boom -> Great Depression', 'CASE-16', 'CASE-14', -1),
         ('N-b New Deal -> WWII', 'CASE-05', 'CASE-09', +1),
         ('N-c 1991-2001 -> 2020-25', 'CASE-23', 'CASE-24', -1)]

# Thresholds (section 5)
H1A_RHO, H1B_R = 0.60, 0.30
H2_PCT, H2_LEAD, H2_LAG, H2_MIN_HITS = 90, 5, 4, 7
N1_EXCL = 2
N2_MIN_SPACING = 4
Z_ACUTE, SD_FLOOR, BASE_W, BASE_MIN = 1.96, 0.25, 20, 10
HOLM_ALPHA = 0.05


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------- loading ----------
def load_ensemble(path=ENS):
    d = pd.read_csv(path, comment='#')
    d = d.rename(columns={'Node Value': 'NodeValue', 'Bond Strength': 'Bond'})
    d['Q'] = d.Abstraction * d.Coherence + d.Capacity - d.Stress  # Node Value is NOT used
    Q = d.pivot(index='Year', columns='Node', values='Q')[NODES]
    return d, Q


def rater_Q(path):
    r = pd.read_csv(path)
    r['Q'] = r.Abstraction * r.Coherence + r.Capacity - r.Stress  # NaN if any score empty
    return r.pivot(index='CaseID', columns='Node', values='Q')[NODES]


def consensus(raters, exclude=()):
    """Mean of available rater Qs, at least 2 raters present (section 4)."""
    use = [k for k in raters if k not in exclude]
    arr = np.stack([raters[k].values for k in use])
    n = np.sum(~np.isnan(arr), axis=0)
    m = np.where(n >= 2, np.nanmean(np.where(np.isnan(arr), np.nan, arr), axis=0), np.nan)
    return pd.DataFrame(m, index=raters[use[0]].index, columns=NODES)


def period_years():
    return {c: list(range(a, b + 1)) for c, _, a, b in PERIODS}


def period_profiles(Q):
    return pd.DataFrame({c: Q.loc[ys].mean() for c, ys in period_years().items()}).T[NODES]


# ---------- relational measures (section 4) ----------
def cosdist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or np.isnan(na) or np.isnan(nb):
        return np.nan
    return 1 - float(np.dot(a, b) / (na * nb))


def tilt(Q):
    return Q.sub(Q.mean(axis=1), axis=0)


def drift_X(Q, k=5):
    th = tilt(Q)
    yrs = list(Q.index)
    out = {}
    for t in yrs:
        w = [y for y in range(t - k, t) if y in th.index]
        if len(w) == k:
            out[t] = cosdist(th.loc[t].values, th.loc[w].mean().values)
    return pd.Series(out, name='X')


def baseline_stats(s, t, W=BASE_W, mn=BASE_MIN):
    w = s.reindex(range(t - W, t)).dropna()
    if len(w) < mn:
        return np.nan, np.nan
    return w.mean(), max(w.std(ddof=1), SD_FLOOR)


def regime_D(Q, lag=10):
    base = {}
    for t in Q.index:
        row = {}
        for n in NODES:
            m, _ = baseline_stats(Q[n], t)
            row[n] = m
        v = pd.Series(row)
        base[t] = v - v.mean() if not v.isna().any() else v * np.nan
    out = {}
    for t in Q.index:
        if (t - lag) in base and not base[t].isna().any() and not base[t - lag].isna().any():
            out[t] = cosdist(base[t].values, base[t - lag].values)
    return pd.Series(out, name='D')


def acute_z(Q):
    z = pd.DataFrame(index=Q.index, columns=NODES, dtype=float)
    for n in NODES:
        for t in Q.index:
            m, sd = baseline_stats(Q[n], t)
            if not np.isnan(m):
                z.loc[t, n] = (Q.loc[t, n] - m) / sd
    return z


def coupling_shift(Q, win=10):
    mats = {t: Q.loc[t - win + 1:t].corr().values for t in Q.index if t - win + 1 >= Q.index.min()}
    out = {}
    for t in mats:
        if (t - win) in mats:
            d = mats[t] - mats[t - win]
            out[t] = float(np.sqrt(np.nansum(d ** 2)))
    return pd.Series(out, name='K')


# ---------- H2 machinery (section 5) ----------
def upcrossings(X, thr):
    yrs = [t for t in X.index if (t - 1) in X.index]
    return pd.Series({t: bool(X[t - 1] < thr <= X[t]) for t in yrs})


def hits(U, Ts, lead=H2_LEAD, lag=H2_LAG, circular=False):
    """U: boolean Series indexed by year. Returns (S_lead, S_lag, per-transition table)."""
    yrs = list(U.index)
    y0, N = yrs[0], len(yrs)
    arr = U.values
    rows = []
    for T in Ts:
        i = T - y0
        lw = [(i - j) for j in range(lead, 0, -1)]
        gw = [(i + j) for j in range(0, lag + 1)]
        if circular:
            lw = [j % N for j in lw]; gw = [j % N for j in gw]
        else:
            lw = [j for j in lw if 0 <= j < N]; gw = [j for j in gw if 0 <= j < N]
        lh = [yrs[j] for j in lw if arr[j]]
        gh = [yrs[j] for j in gw if arr[j]]
        rows.append({'T': T, 'lead_hit': bool(lh), 'lag_hit': bool(gh),
                     'lead_years': (T - lh[0]) if lh else np.nan})
    tab = pd.DataFrame(rows)
    return int(tab.lead_hit.sum()), int(tab.lag_hit.sum()), tab


def null_N1(U, Ts):
    yrs = list(U.index); N = len(yrs); y0 = yrs[0]
    res = []
    for s in range(1, N):
        if min(s, N - s) <= N1_EXCL:
            continue
        shifted = [y0 + ((T - y0 + s) % N) for T in Ts]
        res.append(hits(U, shifted, circular=True)[0])
    return np.array(res)


def null_N2(U, Ts, rng, nrep=B, lo=1912, hi=2021, n=None):
    n = n or len(Ts)
    pool = np.arange(lo, hi + 1)
    res = []
    while len(res) < nrep:
        pick = np.sort(rng.choice(pool, n, replace=False))
        if np.all(np.diff(pick) >= N2_MIN_SPACING):
            res.append(hits(U, list(pick))[0])
    return np.array(res)


def pval_ge(obs, null):
    return (1 + int(np.sum(null >= obs))) / (len(null) + 1)


# ---------- H1 ----------
def h1(ens_prof, rat_cons, rng):
    cases = [c for c, *_ in PERIODS]
    E = ens_prof.loc[cases]; R = rat_cons.loc[cases]
    Em = E.where(R.notna()); Rm = R
    em, rm = Em.mean(axis=1), Rm.mean(axis=1)
    rho = stats.spearmanr(em, rm)[0]
    perm = np.array([stats.spearmanr(em, rng.permutation(rm.values))[0] for _ in range(B)])
    p_a = (1 + int(np.sum(perm >= rho))) / (B + 1)
    rs, used = {}, []
    for c in cases:
        m = R.loc[c].notna() & E.loc[c].notna()
        if m.sum() >= 6 and E.loc[c][m].std() > 0 and R.loc[c][m].std() > 0:
            rs[c] = np.corrcoef(E.loc[c][m], R.loc[c][m])[0, 1]; used.append(c)
    med = float(np.median(list(rs.values())))
    pm = []
    for _ in range(B):
        v = []
        for c in used:
            m = R.loc[c].notna() & E.loc[c].notna()
            e = E.loc[c][m].values; r = rng.permutation(R.loc[c][m].values)
            v.append(np.corrcoef(e, r)[0, 1])
        pm.append(np.median(v))
    p_b = (1 + int(np.sum(np.array(pm) >= med))) / (B + 1)
    ok = (rho >= H1A_RHO) and (med >= H1B_R)
    return dict(rho=rho, p_a=p_a, median_r=med, p_b=p_b, p_H1=max(p_a, p_b),
                thresholds_met=ok, per_period_r=rs, dropped=[c for c in cases if c not in used])


# ---------- H3 ----------
def block_boot_mean(vals, rng, blk=3):
    n = len(vals); b = min(blk, n)
    starts = np.arange(n - b + 1)
    out = []
    while len(out) < n:
        s = rng.choice(starts); out.extend(vals[s:s + b])
    return float(np.mean(out[:n]))


def h3(Q, ens_prof, raters, rng):
    rows = []
    cons = consensus(raters)
    for name, a, b, exp in NAMED:
        nodes = [n for n in NODES if not (np.isnan(cons.loc[a, n]) or np.isnan(cons.loc[b, n]))]
        dE = ens_prof.loc[b, nodes].mean() - ens_prof.loc[a, nodes].mean()
        dR = cons.loc[b, nodes].mean() - cons.loc[a, nodes].mean()
        per = {k: (v.loc[b, nodes].mean() - v.loc[a, nodes].mean()) for k, v in raters.items()}
        disputed = any(np.sign(x) != np.sign(dR) for x in per.values() if not np.isnan(x))
        ya = Q.loc[period_years()[a], nodes].mean(axis=1).values
        yb = Q.loc[period_years()[b], nodes].mean(axis=1).values
        bs = np.array([block_boot_mean(yb, rng) - block_boot_mean(ya, rng) for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = min(1.0, 2 * min((1 + np.sum(bs <= 0)) / (B + 1), (1 + np.sum(bs >= 0)) / (B + 1)))
        nm = int(sum(np.sign(ens_prof.loc[b, n] - ens_prof.loc[a, n]) == np.sign(cons.loc[b, n] - cons.loc[a, n]) for n in nodes))
        rows.append(dict(transition=name, expected_sign=exp, delta_ensemble=dE, delta_rater=dR,
                         per_rater=per, rater_disputed=disputed, ci_lo=lo, ci_hi=hi, p=p,
                         nodes_matching=f'{nm}/{len(nodes)}',
                         passes=(not disputed) and np.sign(dE) == np.sign(dR) and (lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def holm(ps, alpha=HOLM_ALPHA):
    order = sorted(ps, key=ps.get); m = len(order); res = {}; alive = True
    for i, k in enumerate(order):
        ok = alive and ps[k] <= alpha / (m - i)
        res[k] = ok; alive = ok
    return res


# ---------- modes ----------
def check():
    bad = [p for p, h in MD5.items() if md5(p) != h]
    if bad:
        print('CHECKSUM MISMATCH:', bad); sys.exit(1)
    print('checksums OK')
    d, Q = load_ensemble()
    assert Q.shape == (126, 8) and Q.index.min() == 1900 and Q.index.max() == 2025
    assert not d.isna().any().any() and not d.duplicated(['Year', 'Node']).any()
    ys = [y for v in period_years().values() for y in v]
    assert len(PERIODS) == 15 and len(set(ys)) == len(ys)
    assert set(range(1900, 2026)) - set(ys) == GAP_YEARS
    assert len(TRANSITIONS) == 14
    print('structure OK: 126 years x 8 nodes, 15 periods, 14 transitions, gap years', sorted(GAP_YEARS))
    for k, p in RATERS.items():
        r = pd.read_csv(p); print(k, r.shape, 'empty score cells', int(r[['Coherence', 'Capacity', 'Stress', 'Abstraction']].isna().sum().sum()))


def run():
    try:
        frozen = dict(l.split(None, 1) for l in open(OUT + 'FROZEN.txt').read().splitlines() if l.strip() and not l.startswith('#'))
    except FileNotFoundError:
        print('FROZEN.txt missing: PREREG.md not signed off. Refusing to run.'); sys.exit(1)
    if frozen.get('PREREG_sha256', '').strip() != sha256(OUT + 'PREREG.md'):
        print('PREREG.md changed since freeze. Refusing to run.'); sys.exit(1)
    check()
    rng = np.random.default_rng(SEED)
    d, Q = load_ensemble()
    prof = period_profiles(Q)
    raters = {k: rater_Q(p) for k, p in RATERS.items()}
    cons = consensus(raters)
    R1 = h1(prof, cons, rng)
    X = drift_X(Q); thr = float(np.percentile(X.dropna(), H2_PCT))
    U = upcrossings(X, thr)
    Sl, Sg, tab = hits(U, TRANSITIONS)
    n1 = null_N1(U, TRANSITIONS); n2 = null_N2(U, TRANSITIONS, rng)
    p_n1, p_n2 = pval_ge(Sl, n1), pval_ge(Sl, n2)
    T3 = h3(Q, prof, raters, rng)
    p3 = float(T3.p.max())
    hm = holm({'H1': R1['p_H1'], 'H2': p_n1, 'H3': p3})
    v1 = R1['thresholds_met'] and hm['H1']
    v2 = Sl >= H2_MIN_HITS and Sl > Sg and hm['H2'] and p_n2 <= 0.05
    v3 = bool(T3.passes.all()) and hm['H3']
    prof.to_csv(OUT + 'period_profiles.csv'); X.to_csv(OUT + 'drift_by_year.csv')
    U.to_csv(OUT + 'upcrossings.csv'); tab.to_csv(OUT + 'h2.csv', index=False); T3.to_csv(OUT + 'h3.csv', index=False)
    with open(OUT + 'results.md', 'w') as f:
        f.write('# Results (generated by run.py)\n\n')
        f.write(f"H1: rho={R1['rho']:.3f} p_a={R1['p_a']:.4f}; median r={R1['median_r']:.3f} p_b={R1['p_b']:.4f}; supported={v1}\n")
        f.write(f"H2: thr={thr:.4f}; S_lead={Sl}/14; S_lag={Sg}/14; p_N1={p_n1:.4f}; p_N2={p_n2:.4f}; supported={v2}\n")
        f.write(f"H3: all pass={bool(T3.passes.all())}; max p={p3:.4f}; supported={v3}\n")
        f.write(f'Holm: {hm}\n')
    print(open(OUT + 'results.md').read())
    # Sensitivity, secondary and exploratory runs (section 5) are added here ONLY as frozen in PREREG.md.


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.run: run()
    else: check()
