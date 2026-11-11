"""Supplementary pooled evaluation of p3_compare.py out-of-fold predictions (per prereg)."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
pred=pd.read_csv(sys.argv[1]); a,b=sys.argv[2],sys.argv[3]
W=pred.pivot_table(index=['society','year','actual'],columns='model',values='probability').reset_index()
def m(df):
    return {k:(roc_auc_score(df.actual,df[k]),brier_score_loss(df.actual,df[k])) for k in [a,b,'base_rate']}
pt=m(W); socs=W.society.unique(); g={s:W[W.society==s] for s in socs}
rng=np.random.default_rng(20261017); dA,dB=[],[]
for _ in range(2000):
    S=pd.concat([g[s] for s in rng.choice(socs,len(socs))])
    if S.actual.nunique()<2: continue
    r=m(S); dA.append(r[b][0]-r[a][0]); dB.append(r[b][1]-r[a][1])
out={'n_society_years':len(W),'n_societies':len(socs),'n_positive_targets':int(W.actual.sum()),
     'base_rate':float(W.actual.mean()),
     'pooled_auc':{k:round(v[0],4) for k,v in pt.items()},'pooled_brier':{k:round(v[1],5) for k,v in pt.items()},
     f'delta_auc_{b}_minus_{a}':round(pt[b][0]-pt[a][0],4),'delta_auc_ci95':[round(x,4) for x in np.percentile(dA,[2.5,97.5])],
     f'delta_brier_{b}_minus_{a}':round(pt[b][1]-pt[a][1],5),'delta_brier_ci95':[round(x,5) for x in np.percentile(dB,[2.5,97.5])],
     'bootstrap_reps_used':len(dA)}
print(json.dumps(out,indent=1)); json.dump(out,open(sys.argv[4],'w'),indent=1)
