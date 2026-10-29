#!/usr/bin/env python3
"""External-marker lead test. Implements PREREG.md (FROZEN, see FROZEN.txt).

  python3 run.py --check   structure, checksum and marker-eligibility checks only (no outcome statistics).
  python3 run.py --run     full analysis per PREREG.md. Requires FROZEN.txt hashes to match.

Frozen machinery imported unchanged from /workspace/analysis/usa_ensemble_vs_periods/run.py:
drift_X, upcrossings, hits, null_N1, pval_ge, holm, cosdist, tilt, load_ensemble. null_N2 is re-implemented with parameters (PREREG 6.5a).
"""
import argparse, csv, hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/workspace/analysis/usa_ensemble_vs_periods')
import run as RUN

NODES = ['Helm', 'Shield', 'Lore', 'Stewards', 'Craft', 'Hands', 'Archive', 'Flow']
DATA = '/workspace/uploads/usa_ens_annual/USA_ENS_annual.csv'
DATA_MD5 = '1c9626e9481662df6c00974f720e0ec0'   # box reconstruction of the upload; no original md5 (see PREREG section 3)
HERE = '/workspace/analysis/usa_external_marker_lead/'
MK = HERE + 'markers/'
SEED, B = 20261029, 10000
LEAD, LAG, PCT, MIN_PREV_STATES, WIN_STATES = 5, 4, 90, 3, 5
HIT_FRAC = 0.50
SETS = {'E1_nber_peaks': 'primary', 'E2_party_control': 'secondary', 'E3_amendments': 'secondary', 'C1_census_years': 'negative_control'}


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load_Q():
    d = pd.read_csv(DATA)
    d['Q'] = d.Abstraction * d.Coherence + d.Capacity - d.Stress   # Node Value and Bond Strength not used
    Q = d.pivot(index='Year', columns='Node', values='Q')[NODES]
    return d, Q.loc[1900:2025]                                      # 2026 excluded (frozen window ends 2025)


def states(Q):
    """State id per year: a new state starts when any of the eight Q differs from the previous year (exact equality)."""
    new = np.r_[True, (Q.diff().abs().iloc[1:] > 0).any(axis=1).values]
    return pd.Series(np.cumsum(new) - 1, index=Q.index, name='state')


def marker_years():
    out = {}
    out['E1_nber_peaks'] = sorted(pd.read_csv(MK + 'nber_peaks.csv').peak_year)
    out['E2_party_control'] = sorted(pd.read_csv(MK + 'party_control_switches.csv').year)
    out['E3_amendments'] = sorted(set(pd.read_csv(MK + 'amendment_ratifications.csv').year))
    out['C1_census_years'] = sorted(pd.read_csv(MK + 'census_years_negative_control.csv').year)
    return out


def check():
    assert md5(DATA) == DATA_MD5, 'data md5 mismatch'
    d, Q = load_Q()
    assert Q.shape == (126, 8) and not d.isna().any().any() and not d.duplicated(['Year', 'Node']).any()
    print('data OK: 126 years (1900-2025) x 8 nodes; 2026 row present in file and excluded')
    st = states(Q)
    n_states = int(st.nunique()); zero = int((st.diff().iloc[1:] == 0).sum())
    first_start = {s: int(Q.index[st == s][0]) for s in range(n_states)}
    # X^S needs >= MIN_PREV_STATES previous distinct states, so it is first defined in state index MIN_PREV_STATES
    support_start = first_start[MIN_PREV_STATES]
    L = support_start + 1            # first year an upcrossing can occur (needs X^S[t-1] and X^S[t])
    lo, hi = L + 6, 2021            # eligible marker years (same rule as frozen 1912-2021 with L=1906)
    print(f'states: {n_states}; unchanged-year transitions: {zero} of 125; first {MIN_PREV_STATES+1} state start years: {[first_start[s] for s in range(MIN_PREV_STATES+1)]}')
    print(f'X^S support starts {support_start}; first upcrossing-capable year L={L}; eligible marker years T in [{lo}, {hi}]')
    mk = marker_years()
    for k, ys in mk.items():
        el = [y for y in ys if lo <= y <= hi]
        gaps = np.diff(sorted(set(el))) if len(el) > 1 else []
        print(f'{k} [{SETS[k]}]: listed {len(ys)}, distinct {len(set(ys))}, eligible {len(el)}; min gap {min(gaps) if len(gaps) else None}; N2 spacing = min(4, max(1, min gap))')
    ps = pd.read_csv(MK + 'party_control_states.csv'); assert list(ps.year) == list(range(1899, 2026))
    print('marker files OK. No event counts or drift statistics are computed in --check.')



def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


FROZEN_FILES = ['PREREG.md'] + ['markers/' + f for f in ['nber_peaks.csv', 'nber_troughs.csv', 'party_control_states.csv', 'party_control_switches.csv',
                                                          'amendment_ratifications.csv', 'census_years_negative_control.csv']]


def verify_frozen():
    fro = dict(l.split(None, 1) for l in open(HERE + 'FROZEN.txt').read().splitlines() if l.strip() and not l.startswith('#'))
    for f in FROZEN_FILES:
        assert fro[f].strip() == sha256(HERE + f), f'{f} changed since freeze'
    assert fro['frozen_machinery_PREREG'].strip() == sha256('/workspace/analysis/usa_ensemble_vs_periods/PREREG.md')
    return fro


# ---------- X^S ----------
def drift_XS(Q, minprev=MIN_PREV_STATES, win=WIN_STATES):
    th = RUN.tilt(Q); st = states(Q); n = int(st.nunique())
    tv = [th.loc[Q.index[st == k][0]].values for k in range(n)]
    xs = {k: RUN.cosdist(tv[k], np.mean(tv[max(0, k - win):k], axis=0)) for k in range(minprev, n)}
    start = {k: int(Q.index[st == k][0]) for k in range(n)}
    X = pd.Series({y: xs[st[y]] for y in Q.index if st[y] >= minprev}, name='XS')
    return X, st, xs, start


def null_N2(U, Ts, rng, lo, hi, spacing, nrep=B):
    pool = np.arange(lo, hi + 1); res = []
    while len(res) < nrep:
        pick = np.sort(rng.choice(pool, len(Ts), replace=False))
        if np.all(np.diff(pick) >= spacing):
            res.append(RUN.hits(U, list(pick))[0])
    return np.array(res)


def evaluate(U, years, lo, hi, rng, label):
    Ts = sorted({int(y) for y in years if lo <= y <= hi}); n = len(Ts)
    gaps = np.diff(Ts); m = int(min(4, max(1, gaps.min()))) if len(gaps) else 1
    Sl, Sg, tab = RUN.hits(U, Ts)
    n1 = RUN.null_N1(U, Ts); n2 = null_N2(U, Ts, rng, lo, hi, m)
    return dict(label=label, n=n, spacing=m, S_lead=Sl, S_lag=Sg, frac_lead=Sl / n, n1_mean=float(n1.mean()), n1_rate=float(n1.mean() / n),
                n2_mean=float(n2.mean()), n_shifts=len(n1), p_N1=RUN.pval_ge(Sl, n1), p_N2=RUN.pval_ge(Sl, n2), tab=tab, Ts=Ts)


def row(r):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('tab', 'Ts')}


def run():
    fro = verify_frozen()
    out = []
    def P(*a):
        t = ' '.join(str(x) for x in a); print(t); out.append(t)
    d, Q = load_Q()
    assert md5(DATA) == DATA_MD5
    mk = marker_years(); rng = np.random.default_rng(SEED)
    P('# External-marker lead test: run of FROZEN PREREG (hashes verified against FROZEN.txt)')
    P(f'data md5 {md5(DATA)} (box reconstruction; original checksum unavailable); 1900-2025 x 8')
    X, st, xs, start = drift_XS(Q)
    thr = float(np.percentile(X.values, PCT)); U = RUN.upcrossings(X, thr)
    L = int(U.index.min()); lo, hi = L + 6, 2021
    ups = [int(t) for t in U.index[U]]
    P(f'X^S: states {int(st.nunique())}; support {X.index.min()}-{X.index.max()} ({len(X)} years); thr(P90)={thr:.4f}; upcrossings={len(ups)} in {ups}; L={L}; eligible T [{lo},{hi}]')
    R = {}
    R['E1'] = evaluate(U, mk['E1_nber_peaks'], lo, hi, rng, 'E1 NBER peaks (primary, X^S)')
    R['E2'] = evaluate(U, mk['E2_party_control'], lo, hi, rng, 'E2 party control (secondary)')
    R['E3'] = evaluate(U, mk['E3_amendments'], lo, hi, rng, 'E3 amendments (secondary)')
    R['C1'] = evaluate(U, mk['C1_census_years'], lo, hi, rng, 'C1 census (negative control)')
    # X^A for F5
    XA = RUN.drift_X(Q); thrA = float(np.percentile(XA.dropna().values, PCT)); UA = RUN.upcrossings(XA, thrA)
    LA = int(UA.index.min()); loA, hiA = LA + 6, 2021
    RA = evaluate(UA, mk['E1_nber_peaks'], loA, hiA, rng, 'E1 under X^A (for F5)')
    # F criteria
    e = R['E1']; Ts = e['Ts']
    c = {j: int(sum(bool(U.get(T - j, False)) for T in Ts)) for j in range(0, 6)}
    passA = RA['S_lead'] > RA['S_lag'] and RA['p_N1'] <= 0.05
    passXS_123 = e['frac_lead'] >= 0.60 and e['S_lead'] > e['S_lag'] and e['p_N1'] <= 0.05 and e['p_N2'] <= 0.05
    F = {'F1 S_lead/n < 0.60': e['frac_lead'] < 0.60,
         'F2 S_lead <= S_lag': e['S_lead'] <= e['S_lag'],
         'F3 p_N1 > 0.05 or p_N2 > 0.05': e['p_N1'] > 0.05 or e['p_N2'] > 0.05,
         'F4 census control S_lead/n >= 0.50 and p_N1 <= 0.05': R['C1']['frac_lead'] >= 0.50 and R['C1']['p_N1'] <= 0.05,
         'F5 passes F1-F3 under X^S but fails under X^A': passXS_123 and not passA,
         'F6 N1 null-mean hit rate >= 0.45': e['n1_rate'] >= 0.45,
         'F7 anchoring: c0>=1 and c0>=max(c1..c5)': c[0] >= 1 and c[0] >= max(c[j] for j in range(1, 6)),
         'F8 fewer than 5 X^S upcrossings': len(ups) < 5}
    supported = not any(F.values())
    P('\n## PRIMARY: E1 NBER peaks, X^S')
    P(f"n scored={e['n']} {e['Ts']}; S_lead={e['S_lead']} ({e['frac_lead']:.3f}); S_lag={e['S_lag']}; N1: {e['n_shifts']} shifts, null mean {e['n1_mean']:.3f} (rate {e['n1_rate']:.3f}), p_N1={e['p_N1']:.4f}; N2 spacing {e['spacing']}, null mean {e['n2_mean']:.3f}, p_N2={e['p_N2']:.4f}")
    P('per-marker table:'); P(e['tab'].to_string(index=False))
    P(f'c_j (markers with upcrossing j years before T): {c}')
    P(f"X^A check (F5): thr(P90)={thrA:.4f}; upcrossings={int(UA.sum())} in {[int(t) for t in UA.index[UA]]}; n={RA['n']}; S_lead={RA['S_lead']}; S_lag={RA['S_lag']}; p_N1={RA['p_N1']:.4f}; p_N2={RA['p_N2']:.4f}; passes direction+p_N1={passA}")
    P('\nF criteria (True = triggered):')
    for k, v in F.items(): P(f'  {k}: {"TRIGGERED" if v else "pass"}')
    P(f'PRIMARY VERDICT: {"SUPPORTED" if supported else "NOT SUPPORTED"}')
    P('\n## SECONDARY (Holm over E2, E3 using the larger of p_N1, p_N2; alpha 0.05)')
    ps = {k: max(R[k]['p_N1'], R[k]['p_N2']) for k in ('E2', 'E3')}; hm = RUN.holm(ps)
    for k in ('E2', 'E3'):
        r = R[k]
        sup = r['frac_lead'] >= 0.60 and r['S_lead'] > r['S_lag'] and hm[k]
        P(f"{r['label']}: n={r['n']}; S_lead={r['S_lead']} ({r['frac_lead']:.3f}); S_lag={r['S_lag']}; N1 mean {r['n1_mean']:.3f}; N2 mean {r['n2_mean']:.3f} (spacing {r['spacing']}); p_N1={r['p_N1']:.4f}; p_N2={r['p_N2']:.4f}; Holm p used={ps[k]:.4f}; Holm pass={hm[k]}; supportive={sup}")
        P(r['tab'].to_string(index=False))
    r = R['C1']
    P(f"\n## NEGATIVE CONTROL C1 census: n={r['n']}; S_lead={r['S_lead']} ({r['frac_lead']:.3f}); S_lag={r['S_lag']}; N1 mean {r['n1_mean']:.3f}; N2 mean {r['n2_mean']:.3f} (spacing {r['spacing']}); p_N1={r['p_N1']:.4f}; p_N2={r['p_N2']:.4f}")
    P(r['tab'].to_string(index=False))
    # exploratory
    P('\n## EXPLORATORY (no verdicts)'); ex = []
    # (a) X^C
    xc = pd.Series({start[k]: xs[k] for k in sorted(xs)}); thrC = float(np.percentile(xc.values, PCT))
    ev = {int(y): bool(xc.iloc[i] >= thrC and xc.iloc[i - 1] < thrC) for i, y in enumerate(xc.index) if i > 0}
    UC = pd.Series({y: ev.get(y, False) for y in U.index})
    rC = evaluate(UC, mk['E1_nber_peaks'], lo, hi, rng, 'a: E1 under X^C'); ex.append(rC)
    P(f"(a) X^C: states with value {len(xc)}; thr(P90 over states)={thrC:.4f}; events={int(UC.sum())} in {[int(t) for t in UC.index[UC]]}")
    for pc in (85, 95):
        tp = float(np.percentile(X.values, pc)); Up = RUN.upcrossings(X, tp)
        r2 = evaluate(Up, mk['E1_nber_peaks'], lo, hi, rng, f'b: E1 X^S P{pc} (thr {tp:.4f}, {int(Up.sum())} upcrossings)'); ex.append(r2)
    tr = sorted(pd.read_csv(MK + 'nber_troughs.csv').trough_year)
    ex.append(evaluate(U, tr, lo, hi, rng, 'c: NBER troughs X^S'))
    sw = pd.read_csv(MK + 'party_control_switches.csv')
    for body in ('house', 'senate', 'president'):
        ys = sw[sw.bodies_switched.str.contains(body)].year.tolist()
        ex.append(evaluate(U, ys, lo, hi, rng, f'd: party {body} alone X^S'))
    # (f) hi-fi
    dh, Qh = RUN.load_ensemble(); Qh = Qh.loc[1900:2025]
    XH = RUN.drift_X(Qh); thrH = float(np.percentile(XH.dropna().values, PCT)); UH = RUN.upcrossings(XH, thrH)
    ex.append(evaluate(UH, mk['E1_nber_peaks'], 1912, 2021, rng, f'f: hi-fi annual E1 under X^A (thr {thrH:.4f}, {int(UH.sum())} upcrossings)'))
    for r2 in ex:
        P(f"{r2['label']}: n={r2['n']}; S_lead={r2['S_lead']} ({r2['frac_lead']:.3f}); S_lag={r2['S_lag']}; N1 mean {r2['n1_mean']:.3f}; N2 mean {r2['n2_mean']:.3f}; p_N1={r2['p_N1']:.4f}; p_N2={r2['p_N2']:.4f}")
    P('(e) E1 lead-year table (years before T of the earliest lead upcrossing): ' + str({int(a): b for a, b in zip(e['tab']['T'], e['tab'].lead_years)}))
    # outputs
    X.to_csv(HERE + 'drift_XS_by_year.csv'); U.to_csv(HERE + 'upcrossings_XS.csv')
    pd.DataFrame([row(r) for r in list(R.values()) + [RA] + ex]).to_csv(HERE + 'results_table.csv', index=False)
    for k in R: R[k]['tab'].to_csv(HERE + f'hits_{k}.csv', index=False)
    open(HERE + 'run_log.txt', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--run', action='store_true'); a = ap.parse_args()
    if a.check:
        check()
    elif a.run:
        run()
    else:
        sys.exit('use --check or --run')
