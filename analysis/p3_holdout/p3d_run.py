import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score, brier_score_loss
P3='/workspace/analysis/p3'
H=open('holdout_societies.txt').read().split('\n'); H=[h for h in H if h]
F=pd.read_csv(P3+'/FROZEN_CAMS.csv'); F[F.Society.isin(H)].to_csv('HOLDOUT_CAMS.csv',index=False)
E=pd.read_csv(P3+'/INDEPENDENT_EVENTS.csv'); E[E.Society.isin(H)].to_csv('HOLDOUT_EVENTS.csv',index=False)
import sys; sys.path.insert(0,P3+'/cams_p3_pr'); 
src=open(P3+'/p3b_benchmark.py').read().split('for h,dec,tag in')[0]
src=src.replace("'FROZEN_CAMS.csv'","'HOLDOUT_CAMS.csv'").replace("'INDEPENDENT_EVENTS.csv'","'HOLDOUT_EVENTS.csv'").replace("OUT='results/benchmark/'","OUT='results/'")
import os; cwd=os.getcwd(); os.chdir(P3); g={}; exec(compile(src.replace("'HOLDOUT_CAMS.csv'",repr(cwd+'/HOLDOUT_CAMS.csv')).replace("'HOLDOUT_EVENTS.csv'",repr(cwd+'/HOLDOUT_EVENTS.csv')).replace("OUT='results/'",f"OUT={cwd+'/results/'!r}"),'p3b','exec'),g); os.chdir(cwd)
R={}
for h in (3,5):
    P,n0=g['run'](h,False); R[h]=g['report'](P,n0,f'P3d_h{h}')
    print(json.dumps(R[h]))
# secondary split, h=3
P=pd.read_csv('results/P3d_h3_predictions.csv')
per={}
for s,d in P.groupby('society'):
    if d.actual.nunique()==2: per[s]=roc_auc_score(d.actual,d.B3)
excl=sorted(set(P.society)-set(per)); med=np.median(list(per.values()))
weak=[s for s,v in per.items() if v<=med]; strong=[s for s,v in per.items() if v>med]
def dauc(df): return roc_auc_score(df.actual,df['C+B3'])-roc_auc_score(df.actual,df.B3)
def dbr(df): return brier_score_loss(df.actual,df['C+B3'])-brier_score_loss(df.actual,df.B3)
g2={s:P[P.society==s] for s in per}; rng=np.random.default_rng(20261017); bw,bs,bd=[],[],[]
for _ in range(2000):
    W=pd.concat([g2[s] for s in rng.choice(weak,len(weak))]); S=pd.concat([g2[s] for s in rng.choice(strong,len(strong))])
    if W.actual.nunique()<2 or S.actual.nunique()<2: continue
    a,b=dauc(W),dauc(S); bw.append(a); bs.append(b); bd.append(a-b)
Wd=pd.concat([g2[s] for s in weak]); Sd=pd.concat([g2[s] for s in strong])
ci=lambda x:[round(q,4) for q in np.percentile(x,[2.5,97.5])]
sec={'n_with_both_classes':len(per),'excluded_one_class':excl,'median_B3_auc':round(med,4),
 'per_society_B3_auc':{k:round(v,3) for k,v in sorted(per.items(),key=lambda x:x[1])},
 'weak':{'societies':weak,'n_sy':len(Wd),'pos':int(Wd.actual.sum()),'auc_B3':round(roc_auc_score(Wd.actual,Wd.B3),4),'auc_CB3':round(roc_auc_score(Wd.actual,Wd['C+B3']),4),'dAUC':round(dauc(Wd),4),'ci':ci(bw),'dBrier':round(dbr(Wd),5)},
 'strong':{'societies':strong,'n_sy':len(Sd),'pos':int(Sd.actual.sum()),'auc_B3':round(roc_auc_score(Sd.actual,Sd.B3),4),'auc_CB3':round(roc_auc_score(Sd.actual,Sd['C+B3']),4),'dAUC':round(dauc(Sd),4),'ci':ci(bs),'dBrier':round(dbr(Sd),5)},
 'weak_minus_strong':{'dAUC_diff':round(dauc(Wd)-dauc(Sd),4),'ci':ci(bd)},'reps':len(bd)}
json.dump(sec,open('results/P3d_secondary_split_h3.json','w'),indent=1); print(json.dumps(sec,indent=1))
