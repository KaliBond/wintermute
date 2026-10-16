#!/usr/bin/env python3
"""
T-M series consolidated analysis script v2 (CAMS/JUNO morphology programme, 2026-10-16).
v2 changelog (post clean-machine audit by K. McKern):
  - pivots() signature fixed (unused argument removed)
  - T-M3d computed on the 149-episode analysis set (primary) AND the 158 censored set (sensitivity)
  - displacement signature alignment made explicit: society-peak-year (canonical, as T-M1/T-M3b original)
    vs trough-aligned (variant); both reported where they differ
  - failure locus: plain argmin of episode minimum (deterministic; the ad-hoc tie-break used in the
    original session is dropped)
  - added: T-M2b/T-M3b original-alignment kappas, full-vs-rivals AUCs, T-M4 concordance,
    T-M3d-2 original node-FE panel (all three panel specs labelled)
  - intermediate outputs saved: episodes_censored_158.csv, episodes_analysis_149.csv,
    T-M4_blind_set_{n}_episodes.csv, T-M4_per_set_kappa.csv; episodes.csv NOT overwritten
Seeds: CV=0, T-M1c=42, T-M1 AUC null=7, T-M2a=11, T-M3a=23, T-M3d null=5, floor=9, T-M4=31.
"""
import pandas as pd, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

NODES = ['Archive','Craft','Flow','Hands','Helm','Lore','Shield','Stewards']

def pivots(df, soc_col, yr_col, c, k, s, a):
    pv = {m: df.pivot_table(index=[soc_col,yr_col], columns='Node', values=v)
          for m,v in dict(C=c,K=k,S=s,A=a).items()}
    pv['V'] = pv['C'] + pv['K'] - pv['S'] + 0.5*pv['A']
    return pv

def extract_episodes(Sbar, min_len=15):
    eps = []
    for soc in Sbar.index.get_level_values(0).unique():
        s = Sbar.loc[soc].sort_index(); mu, sd = s.mean(), s.std()
        if sd == 0 or len(s) < min_len: continue
        z = ((s-mu)/sd).values; yrs = s.index.values
        in_ep = False; start = 0
        for i in range(len(yrs)):
            if z[i] >= 1.0 and not in_ep and i >= 2 and (z[i-2:i] < 1.0).all():
                start = i; in_ep = True
            elif in_ep and z[i] < 0.5:
                eps.append((soc, yrs[start], yrs[i-1], yrs[i])); in_ep = False
        if in_ep: eps.append((soc, yrs[start], yrs[-1], None))
    return eps

def episode_features(pv, eps, censor_years=5):
    rows = []
    for soc, onset, last, end in eps:
        if end is None: continue
        yrs = pv['S'].loc[soc].index.values
        if last + censor_years > yrs[-1]: continue
        ep = (yrs >= onset) & (yrs <= last); pre = (yrs >= onset-2) & (yrs < onset)
        if ep.sum() < 1 or pre.sum() < 1: continue
        rec = dict(society=soc, onset=int(onset), end=int(last))
        preV = pv['V'].loc[soc].loc[pre].mean().mean()
        spk = pv['S'].loc[soc].loc[ep].mean(axis=1).idxmax()   # society stress-peak year
        dS, dV, R = {}, {}, {}
        for n in NODES:
            vn = pv['V'].loc[soc].loc[ep, n]; tstar = vn.idxmin()
            rec[f'trough_{n}'] = int(tstar)
            dS[n] = pv['S'].loc[soc].loc[tstar,n] - pv['S'].loc[soc].loc[pre,n].mean()
            dV[n] = vn.loc[tstar] - pv['V'].loc[soc].loc[pre,n].mean()
            R[n]  = (pv['C'].loc[soc].loc[tstar,n] - pv['C'].loc[soc].loc[pre,n].mean()) \
                  + (pv['K'].loc[soc].loc[tstar,n] - pv['K'].loc[soc].loc[pre,n].mean()) \
                  + 0.5*(pv['A'].loc[soc].loc[tstar,n] - pv['A'].loc[soc].loc[pre,n].mean())
            rec[f'sig_{n}'] = pv['S'].loc[soc].loc[spk,n] - pv['S'].loc[soc].loc[pre,n].mean()
        L1 = sum(abs(rec[f'sig_{n}']) for n in NODES)
        for n in NODES: rec[f'd_{n}'] = rec[f'sig_{n}']/L1 if L1 else 0.0   # canonical: society-peak signature
        rec['pre_Vbar'] = preV
        rec['uptake_locus_trough'] = NODES[int(np.argmax([dS[n] for n in NODES]))]
        rec['uptake_locus_peak']   = NODES[int(np.argmax([rec[f'sig_{n}'] for n in NODES]))]
        rec['locus_level']    = pv['V'].loc[soc].loc[ep].min().idxmin()   # plain argmin (canonical)
        rec['locus_dV']       = NODES[int(np.argmin([dV[n] for n in NODES]))]
        rec['damage_locus_R'] = NODES[int(np.argmin([R[n] for n in NODES]))]
        recV = pv['V'].loc[soc].loc[(yrs > last) & (yrs <= last+censor_years)].mean(axis=1)
        rec['recovered'] = bool(recV.max() >= preV)   # AMENDED outcome: full regain of trough loss
        s = pv['S'].loc[soc].mean(axis=1)
        rec['zpeak'] = float(((s.loc[ep]) - s.mean()).max() / s.std())
        rows.append(rec)
    return pd.DataFrame(rows)

def trough_deltas(pv, E):
    dS_, dV_, R_ = [], [], []
    for i, e in E.iterrows():
        soc = e.society; yrs = pv['S'].loc[soc].index.values
        pre = (yrs >= e.onset-2) & (yrs < e.onset)
        ds, dv, rr = [], [], []
        for n in NODES:
            ts = int(e[f'trough_{n}'])
            ds.append(pv['S'].loc[soc].loc[ts,n] - pv['S'].loc[soc].loc[pre,n].mean())
            dv.append(pv['V'].loc[soc].loc[ts,n] - pv['V'].loc[soc].loc[pre,n].mean())
            rr.append((pv['C'].loc[soc].loc[ts,n] - pv['C'].loc[soc].loc[pre,n].mean())
                     +(pv['K'].loc[soc].loc[ts,n] - pv['K'].loc[soc].loc[pre,n].mean())
                     +0.5*(pv['A'].loc[soc].loc[ts,n] - pv['A'].loc[soc].loc[pre,n].mean()))
        dS_.append(ds); dV_.append(dv); R_.append(rr)
    return map(np.array, (dS_, dV_, R_))

def cos(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return a@b/(na*nb) if na*nb > 0 else 0.0

def cv_auc(X, y, seed=0):
    if X.ndim == 1: X = X.reshape(-1,1)
    return np.mean([roc_auc_score(y[te], LogisticRegression(max_iter=1000).fit(X[tr],y[tr]).predict_proba(X[te])[:,1])
                    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=seed).split(X,y)])

def loeo_modal_test(df_, col, seed):
    df_ = df_.copy()
    df_['pred'] = [None if (o:=df_[(df_.society==e.society)&(df_.index!=i)]).empty else o[col].mode().iloc[0]
                   for i,e in df_.iterrows()]
    M = df_.dropna(subset=['pred'])
    acc = (M.pred==M[col]).mean(); marg = df_[col].value_counts(normalize=True).max()
    loc = M[col].values; labs = M.society.values
    rng = np.random.default_rng(seed); nulls = []
    for _ in range(500):
        sh = rng.permutation(labs); c = 0
        for i in range(len(M)):
            o = loc[(sh==sh[i])]; o = o[np.arange(len(M))[sh==sh[i]]!=i]
            if len(o)==0: continue
            v_,c_ = np.unique(o, return_counts=True); c += v_[c_.argmax()]==loc[i]
        nulls.append(c/len(M))
    return acc, marg, np.mean(nulls), (np.sum(np.array(nulls)>=acc)+1)/501

# ================= unified dataset =================
df = pd.read_csv('JUNO_Unified_Dataset.csv', low_memory=False)
drift = (df.Coherence+df.Capacity-df.Stress+0.5*df.Abstraction - pd.to_numeric(df.Node_Value_Calc, errors='coerce')).abs().max()
assert drift < 1e-9, f"canonical drift {drift}"
pv = pivots(df, 'Society','Year','Coherence','Capacity','Stress','Abstraction')
eps = extract_episodes(pv['S'].mean(axis=1))
print(f"extracted: {len(eps)} episodes, {len(set(e[0] for e in eps))} societies")     # 201 / 48
E = episode_features(pv, eps)
print(f"eligible: {len(E)} / {E.society.nunique()} societies")                        # 158 / 42
E.to_csv('episodes_censored_158.csv', index=False)

dcols = [f'd_{n}' for n in NODES]
E['typicality'] = [np.nan if (o:=E[(E.society==e.society)&(E.index!=i)]).empty
                   else cos(e[dcols].values.astype(float), o[dcols].values.mean(axis=0))
                   for i,e in E.iterrows()]
Ev2 = E.dropna(subset=['typicality']).copy()
print(f"T-M1 set: {len(Ev2)} / {Ev2.society.nunique()}")                              # 149 / 33
Ev2.to_csv('episodes_analysis_149.csv', index=False)

# T-M1 horse race
vd = pd.read_csv('vdem_core_v16.csv')
vd['country_name'] = vd.country_name.replace({'United Kingdom':'UK','United States of America':'USA','United Arab Emirates':'UAE'})
lib = {soc: dict(zip(g.year, g.v2x_libdem)) for soc,g in vd.groupby('country_name')}
Ev2['libdem'] = [lib[e.society][max(y for y in lib[e.society] if y <= e.onset)] if e.society in lib and any(y <= e.onset for y in lib[e.society]) else np.nan for e in Ev2.itertuples()]
D = Ev2.dropna(subset=['libdem']); y = D.recovered.astype(int).values
print("T-M1a morphology:", round(cv_auc(D.typicality.values.astype(float), y),3),     # 0.492
      " libdem:", round(cv_auc(D.libdem.values.astype(float), y),3),                  # 0.547
      " baseline:", round(cv_auc(D[['pre_Vbar','zpeak']].values.astype(float), y),3)) # 0.553
Xfull = np.column_stack([D.typicality,D.libdem,D.pre_Vbar,D.zpeak]).astype(float)
Xriv  = np.column_stack([D.libdem,D.pre_Vbar,D.zpeak]).astype(float)
print("T-M1b full:", round(cv_auc(Xfull,y),3), " rivals_only:", round(cv_auc(Xriv,y),3))  # 0.548 / 0.555

# T-M1c (seed 42)
sigs = D[dcols].values.astype(float); labs = D.society.values
def wb(labels):
    w,b = [],[]
    for a_ in range(len(D)):
        for b_ in range(a_+1,len(D)):
            c = cos(sigs[a_],sigs[b_]); (w if labels[a_]==labels[b_] else b).append(c)
    return np.mean(w)-np.mean(b), np.mean(w), np.mean(b)
obs,wm,bm = wb(labs); rng = np.random.default_rng(42)
nulls = [wb(rng.permutation(labs))[0] for _ in range(500)]
print(f"T-M1c within {wm:.3f} between {bm:.3f} gap {obs:.4f} p {(np.sum(np.array(nulls)>=obs)+1)/501:.4f}")  # 0.849/0.823/0.0267/0.004

# T-M2 (plain-argmin locus, canonical)
acc,marg,nm,p = loeo_modal_test(Ev2,'locus_level',11)
print(f"T-M2a acc {acc:.3f} marginal {marg:.3f} ({acc/marg:.2f}x) null {nm:.3f} p {p:.4f}")   # 0.510/0.409/1.25x/0.287/0.002
print("T-M2 locus counts:", Ev2.locus_level.value_counts().to_dict())                          # Hands 61
print("T-M2b kappa (peak-uptake vs level):", round(cohen_kappa_score(Ev2.uptake_locus_peak, Ev2.locus_level),3))  # 0.154

# T-M3 (ΔV locus)
acc3,marg3,nm3,p3 = loeo_modal_test(Ev2,'locus_dV',23)
print(f"T-M3a acc {acc3:.3f} marginal {marg3:.3f} ({acc3/marg3:.2f}x) null {nm3:.3f} p {p3:.4f}")  # 0.309/0.329/0.94x/0.223/0.034
print("T-M3 relocated share:", round((Ev2.locus_level!=Ev2.locus_dV).mean(),3))                    # 0.624
print("T-M3b kappa (peak-uptake vs dV):", round(cohen_kappa_score(Ev2.uptake_locus_peak, Ev2.locus_dV),3))  # 0.476
_,p3c,_,_ = stats.chi2_contingency(pd.crosstab(Ev2.locus_dV, Ev2.recovered))
print("T-M3c chi2 p:", round(p3c,4))                                                               # 0.193

# T-M3d: trough-aligned, PRIMARY = 149 set, sensitivity = 158 set (seeds 5, 9)
for name, E_ in [("149 (primary)", Ev2), ("158 (sensitivity)", E)]:
    dS,dV,R = trough_deltas(pv, E_)
    up_i, dmg_i, res_i = dS.argmax(1), dV.argmin(1), R.argmin(1)
    k_full = cohen_kappa_score(up_i, dmg_i); k_R = cohen_kappa_score(up_i, res_i)
    rng = np.random.default_rng(5)
    nullk = [cohen_kappa_score(up_i, np.array([rng.permutation(r) for r in R]).argmin(1)) for _ in range(1000)]
    rng = np.random.default_rng(9)
    floor = [cohen_kappa_score((sp:=np.array([rng.permutation(r) for r in dS])).argmax(1), (R-sp).argmin(1)) for _ in range(1000)]
    print(f"T-M3d {name}: kappa {k_full:.3f} | kappa_R {k_R:.3f} null95 {np.percentile(nullk,95):.3f} "
          f"p {(np.sum(np.array(nullk)>=k_R)+1)/1001:.4f} | floor {np.mean(floor):.3f}")

# Lag panels — three specs, explicitly labelled
def build_panel(back_gap_filter, need_prev):
    pan = []
    for soc in pv['S'].index.get_level_values(0).unique():
        s,c,k = pv['S'].loc[soc], pv['C'].loc[soc], pv['K'].loc[soc]
        yrs = s.index.values; ds,dc,dk = s.diff(), c.diff(), k.diff()
        for t in range(1, len(yrs)-1):
            if yrs[t+1]-yrs[t] > 2: continue
            if back_gap_filter and yrs[t]-yrs[t-1] > 2: continue
            for n in NODES:
                pan.append((soc, yrs[t], n, ds.iloc[t][n], dc.iloc[t+1][n], dk.iloc[t+1][n],
                            dc.iloc[t][n], dk.iloc[t][n], dc.iloc[t-1][n], dk.iloc[t-1][n]))
    cols = ['soc','year','node','dS','dC_next','dK_next','dC_now','dK_now','dC_prev','dK_prev']
    P = pd.DataFrame(pan, columns=cols)
    return P.dropna(subset=cols[3:] if need_prev else cols[3:8])
P_orig = build_panel(False, False)   # T-M3d-2 original: node FE, node clustering, forward-gap only
print("T-M3d-2 panel n:", len(P_orig))
mC = smf.ols('dC_next ~ dS + dC_now + C(node)', data=P_orig).fit(cov_type='cluster', cov_kwds={'groups':P_orig.node})
mK = smf.ols('dK_next ~ dS + dK_now + C(node)', data=P_orig).fit(cov_type='cluster', cov_kwds={'groups':P_orig.node})
print(f"  fwd C {mC.params['dS']:.4f} p {mC.pvalues['dS']:.1e} | fwd K {mK.params['dS']:.4f} p {mK.pvalues['dS']:.1e}")
P2 = build_panel(True, True)         # T-M3e: both gaps filtered, placebo columns present
print("T-M3e panel n:", len(P2))     # 40472
Q = P2.copy()
for cn in ['dS','dC_next','dK_next','dC_now','dK_now','dC_prev','dK_prev']:
    Q[cn] = P2[cn] - P2.groupby(['soc','year'])[cn].transform('mean')
    Q[cn] = Q[cn] - Q.groupby('node')[cn].transform('mean')
def fwl(y_, xs):
    X = sm.add_constant(Q[xs].values)
    m = sm.OLS(Q[y_].values, X).fit(cov_type='cluster', cov_kwds={'groups': P2.soc})
    return round(float(m.params[1]),4), f"{m.pvalues[1]:.1e}"
print("  fwd C:", fwl('dC_next',['dS','dC_now']), " rev C:", fwl('dC_prev',['dS','dC_now']))
print("  fwd K:", fwl('dK_next',['dS','dK_now']), " rev K:", fwl('dK_prev',['dS','dK_now']))

# ================= T-M4 blind replication (seed 31) =================
rows = []; allup, allres = [], []
for n in [1,2,3,5,6,7]:
    b = pd.read_csv(f'JUNO_Blind_Set_{n}.csv', low_memory=False)
    b['yr'] = b.T_Offset.str.replace('T+','').astype(int)
    pvB = pivots(b, 'Society_ID','yr','C','K','S','A')
    EB = episode_features(pvB, extract_episodes(pvB['S'].mean(axis=1)))
    dSb, dVb, Rb = trough_deltas(pvB, EB)
    u, r = dSb.argmax(1), Rb.argmin(1)
    allup += list(u); allres += list(r)
    EB['uptake_idx'] = u; EB['R_locus_idx'] = r
    EB.to_csv(f'T-M4_blind_set_{n}_episodes.csv', index=False)
    rows.append(dict(set=n, episodes=len(EB), kappa=round(cohen_kappa_score(u,r),3), concordance=round(float((u==r).mean()),3)))
obsB = cohen_kappa_score(allup, allres); concB = float((np.array(allup)==np.array(allres)).mean())
rng = np.random.default_rng(31)
nullB = [cohen_kappa_score(allup, rng.permutation(allres)) for _ in range(1000)]
print(f"T-M4 pooled n={len(allup)} kappa={obsB:.3f} concordance={concB:.3f} p={(np.sum(np.array(nullB)>=obsB)+1)/1001:.4f}")
pd.DataFrame(rows).to_csv('T-M4_per_set_kappa.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
