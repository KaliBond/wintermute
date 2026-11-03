"""Exploratory P7: aggregate + Lambda2 vs aggregate, same LOSO/fit as p3_compare.py (imports its functions)."""
import sys; sys.path.insert(0,'cams_p3_pr')
import pandas as pd, numpy as np
from analysis.p3_compare import load_panel, make_target, fit_predict, DIMS
w=make_target(load_panel('FROZEN_CAMS.csv'),'INDEPENDENT_EVENTS.csv',3)
w=w.merge(pd.read_csv('LAMBDA2.csv'),on=['Society','Year'],how='left'); assert w.Lambda2.notna().all()
A=[f'mean_{d}' for d in DIMS]; rows=[]
for s in sorted(w.Society.unique()):
    te,tr=w[w.Society==s],w[w.Society!=s]
    if len(tr)<15 or len(te)<2: continue
    for name,cols in [('baseline',A),('baseline_plus_lambda2',A+['Lambda2'])]:
        for yr,y,p in zip(te.Year,te.target.astype(int),fit_predict(tr,te,cols)):
            rows.append(dict(society=s,year=int(yr),actual=int(y),model=name,probability=float(p)))
    for yr,y in zip(te.Year,te.target.astype(int)):
        rows.append(dict(society=s,year=int(yr),actual=int(y),model='base_rate',probability=float(tr.target.mean())))
pd.DataFrame(rows).to_csv('results/p7_lambda2_E1_h3_predictions.csv',index=False)
print('lambda2 summary',w.Lambda2.describe().round(3).to_dict())
