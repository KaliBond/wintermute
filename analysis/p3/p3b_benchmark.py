"""P3b per PREREG_P3b_17Oct2026.md."""
import sys, json; sys.path.insert(0,'cams_p3_pr')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from analysis.p3_compare import load_panel, make_target, fit_predict, DIMS
OUT='results/benchmark/'
ev=pd.read_csv('INDEPENDENT_EVENTS.csv')
v=pd.read_csv('/workspace/analysis/tm_kimi3/morphology_test/vdem_core_v16.csv')
VAL={'UK':'United Kingdom','USA':'United States of America','UAE':'United Arab Emirates','Turkey':'Türkiye'}
def features(h):
    w=make_target(load_panel('FROZEN_CAMS.csv'),'INDEPENDENT_EVENTS.csv',h)
    coup={s:set(g.Year[g.Event==1]) for s,g in ev.groupby('Society')}
    start=ev.groupby('Society').Year.min().to_dict()
    c10,ysl=[],[]
    for r in w.itertuples():
        cs=[y for y in coup[r.Society] if y<=r.Year]
        c10.append(int(any(y>=r.Year-9 for y in cs)))
        ysl.append(min(30, r.Year-max(cs)) if cs else min(30, max(0,r.Year-start[r.Society])) if False else (min(30,r.Year-max(cs)) if cs else 30))
    w['coup10'],w['ysl']=c10,ysl
    lib={}
    for s in w.Society.unique():
        sub=v[v.country_name==VAL.get(s,s)].dropna(subset=['v2x_libdem']).drop_duplicates('year').set_index('year').v2x_libdem
        lib[s]=sub
    w['libdem']=[lib[r.Society].get(r.Year,np.nan) for r in w.itertuples()]
    w['dlibdem']=w.libdem-[lib[r.Society].get(r.Year-1,np.nan) for r in w.itertuples()]
    n0=len(w); w=w.dropna(subset=['libdem','dlibdem']).reset_index(drop=True)
    dec=pd.get_dummies((w.Year//10*10).astype(str),prefix='dec').astype(float)
    return pd.concat([w,dec],axis=1), n0, list(dec.columns)
C=[f'mean_{d}' for d in DIMS]; B1=['coup10','ysl']; B2=['libdem','dlibdem']
MODELS={'B1':B1,'B2':B2,'B3':B1+B2,'C':C,'C+B3':C+B1+B2}
def run(h,decade):
    w,n0,D=features(h); P={m:np.full(len(w),np.nan) for m in MODELS}
    for s in w.Society.unique():
        te=w.Society==s; tr=~te
        for m,cols in MODELS.items(): P[m][te.values]=fit_predict(w[tr],w[te],cols+(D if decade else []))
    P=pd.DataFrame(P); P['actual']=w.target.astype(int); P['society']=w.Society; P['year']=w.Year
    return P,n0
def met(df,m): return roc_auc_score(df.actual,df[m]), brier_score_loss(df.actual,df[m])
def report(P,n0,tag):
    socs=P.society.unique(); g={s:P[P.society==s] for s in socs}
    rng=np.random.default_rng(20261017); boots=[]
    for _ in range(2000):
        S=pd.concat([g[s] for s in rng.choice(socs,len(socs))])
        if S.actual.nunique()==2: boots.append({m:met(S,m) for m in MODELS})
    res={'tag':tag,'n_society_years_P3_set':n0,'n_common_sample':len(P),'n_societies':len(socs),
         'n_positive':int(P.actual.sum()),'base_rate':round(float(P.actual.mean()),4),
         'auc':{m:round(met(P,m)[0],4) for m in MODELS},'brier':{m:round(met(P,m)[1],5) for m in MODELS},'contrasts':{}}
    for a,b in [('C','B3'),('C+B3','B3'),('C','B1')]:
        pa,pb=met(P,a),met(P,b)
        dA=[x[a][0]-x[b][0] for x in boots]; dB=[x[a][1]-x[b][1] for x in boots]
        res['contrasts'][f'{a} minus {b}']={'dAUC':round(pa[0]-pb[0],4),'dAUC_ci95':[round(q,4) for q in np.percentile(dA,[2.5,97.5])],
            'dBrier':round(pa[1]-pb[1],5),'dBrier_ci95':[round(q,5) for q in np.percentile(dB,[2.5,97.5])]}
    res['bootstrap_reps']=len(boots)
    json.dump(res,open(OUT+f'{tag}.json','w'),indent=1); P.to_csv(OUT+f'{tag}_predictions.csv',index=False)
    return res
for h,dec,tag in [(3,False,'primary_h3'),(5,False,'S1_h5'),(3,True,'R1_h3_decadeFE'),(5,True,'R1_h5_decadeFE')]:
    r=report(*run(h,dec),tag); print(json.dumps(r))
