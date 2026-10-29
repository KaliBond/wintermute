"""P3c driver: P3, P3b, P7 on CURATED_CAMS and on JUNO restricted to the same society-years. Reuses p3_compare.py and p3b_benchmark.py logic."""
import sys, json, os, subprocess
import pandas as pd, numpy as np
P3='/workspace/analysis/p3'; sys.path.insert(0,P3+'/cams_p3_pr')
from analysis.p3_compare import evaluate, load_panel, make_target, fit_predict, DIMS
from sklearn.metrics import roc_auc_score, brier_score_loss
C=pd.read_csv('CURATED_CAMS.csv'); F=pd.read_csv(P3+'/FROZEN_CAMS.csv')
sy=set(zip(C.Society,C.Year)); Jr=F[[ (s,y) in sy for s,y in zip(F.Society,F.Year)]]
Jr.to_csv('JUNO_RESTRICTED.csv',index=False)
print('JUNO restricted: societies',Jr.Society.nunique(),'node-rows',len(Jr))
# p3b logic, re-pointed
src=open(P3+'/p3b_benchmark.py').read().split('for h,dec,tag in')[0]
def p3b(panel,events,tag_prefix,out):
    s=src.replace("'FROZEN_CAMS.csv'",repr(panel)).replace("'INDEPENDENT_EVENTS.csv'",repr(events)).replace("OUT='results/benchmark/'",f"OUT={out!r}")
    g={}; exec(s,g); res={}
    for h in (3,5):
        r=g['report'](*g['run'](h,False),f'{tag_prefix}_P3b_h{h}'); res[h]=r
    return res
def pooled(pred_csv,a,b,out):
    r=subprocess.run(['/workspace/.venv_p3/bin/python',P3+'/p3_pooled.py',pred_csv,a,b,out],capture_output=True,text=True); return json.load(open(out))
def p7(panel,events,lam,h,tag):
    w=make_target(load_panel(panel),events,h).merge(pd.read_csv(lam),on=['Society','Year'],how='left'); assert w.Lambda2.notna().all()
    A=[f'mean_{d}' for d in DIMS]; rows=[]
    for s in sorted(w.Society.unique()):
        te,tr=w[w.Society==s],w[w.Society!=s]
        if len(tr)<15 or len(te)<2: continue
        for name,cols in [('baseline',A),('baseline_plus_lambda2',A+['Lambda2'])]:
            for yr,y,p in zip(te.Year,te.target.astype(int),fit_predict(tr,te,cols)): rows.append(dict(society=s,year=int(yr),actual=int(y),model=name,probability=float(p)))
        for yr,y in zip(te.Year,te.target.astype(int)): rows.append(dict(society=s,year=int(yr),actual=int(y),model='base_rate',probability=float(tr.target.mean())))
    f=f'results/{tag}_P7_h{h}_predictions.csv'; pd.DataFrame(rows).to_csv(f,index=False)
    return pooled(f,'baseline','baseline_plus_lambda2',f'results/{tag}_P7_h{h}.json')
# lambda2 for JUNO-restricted from canonical stored column
pd.read_csv(P3+'/LAMBDA2.csv').merge(Jr[['Society','Year']].drop_duplicates()).to_csv('LAMBDA2_JUNO_RESTRICTED.csv',index=False)
ALL={}
for tag,panel,lam in [('curated','CURATED_CAMS.csv','LAMBDA2_CURATED.csv'),('junoR','JUNO_RESTRICTED.csv','LAMBDA2_JUNO_RESTRICTED.csv')]:
    ALL[tag]={}
    for h in (3,5):
        out,preds=evaluate(panel,'EVENTS_CURATED.csv',h)
        json.dump(out,open(f'results/{tag}_P3_h{h}.json','w'),indent=1)
        f=f'results/{tag}_P3_h{h}_predictions.csv'; pd.DataFrame(preds).to_csv(f,index=False)
        ALL[tag][f'P3_h{h}']=pooled(f,'baseline','configuration',f'results/{tag}_P3_h{h}_pooled.json')
        ALL[tag][f'P7_h{h}']=p7(panel,'EVENTS_CURATED.csv',lam,h,tag)
    ALL[tag]['P3b']=p3b(panel,'EVENTS_CURATED.csv',tag,'results/')
json.dump(ALL,open('results/ALL.json','w'),indent=1)
# score agreement curated vs JUNO on shared society-years
m=C.merge(F,on=['Society','Year','Node'],suffixes=('_cur','_juno'))
agg=m.groupby(['Society','Year'])[[f'{d}_{k}' for d in DIMS for k in ('cur','juno')]].mean()
print('shared node-rows',len(m),'society-years',len(agg))
print({d:round(float(np.corrcoef(agg[f'{d}_cur'],agg[f'{d}_juno'])[0,1]),3) for d in DIMS})
print('identical node-rows share',round(float(np.mean(np.all(np.isclose(m[[f'{d}_cur' for d in DIMS]].values,m[[f'{d}_juno' for d in DIMS]].values),axis=1))),3))
print('per-society identical share:',m.assign(same=np.all(np.isclose(m[[f'{d}_cur' for d in DIMS]].values,m[[f'{d}_juno' for d in DIMS]].values),axis=1)).groupby('Society').same.mean().round(2).to_dict())
