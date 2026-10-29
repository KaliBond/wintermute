import pandas as pd, numpy as np, itertools, zipfile, hashlib
from scipy import stats
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
U='data/'
X='data/'
files={'FROZEN':U+'BLIND_RESPONSE_FROZEN_GPT_200.csv','UPLOAD_FILLED':U+'BLIND_RESPONSE_GROK_200.csv','CLAUDE_files27':X+'BLIND_RESPONSE_CLAUDE_200.csv'}
tmpl=pd.read_csv(X+'BLIND_RESPONSE_TEMPLATE_200.csv')
keys=list(zip(tmpl.CaseID,tmpl.Node))
R={}
for k,f in files.items():
    d=pd.read_csv(f).set_index(['CaseID','Node']).loc[keys]
    d['Q']=d.Abstraction*d.Coherence+d.Capacity-d.Stress
    R[k]=d
    print(k,'rows',len(d),'Q nonnull',d.Q.notna().sum())
Q=pd.DataFrame({k:v.Q for k,v in R.items()})
print('\nFile order identical to template?',{k:list(pd.read_csv(f)[['CaseID','Node']].itertuples(index=False,name=None))==keys for k,f in files.items()})
# Pearson
print('\nPearson on Q')
c=Q.dropna()
print('complete triples:',len(c))
for a,b in itertools.combinations(Q.columns,2):
    p=Q[[a,b]].dropna()
    print(f'{a}-{b}: pairwise-complete n={len(p)} r={p[a].corr(p[b]):.4f} | 175-triples r={c[a].corr(c[b]):.4f} | Spearman(175)={stats.spearmanr(c[a],c[b])[0]:.4f}')
# ICC
def icc(M):
    M=np.asarray(M,float); n,k=M.shape
    gm=M.mean(); rm=M.mean(1); cm=M.mean(0)
    SSR=k*((rm-gm)**2).sum(); SSC=n*((cm-gm)**2).sum(); SST=((M-gm)**2).sum(); SSE=SST-SSR-SSC
    MSR=SSR/(n-1); MSC=SSC/(k-1); MSE=SSE/((n-1)*(k-1)); MSW=(SSC+SSE)/(n*(k-1))
    return {'ICC(1,1)':(MSR-MSW)/(MSR+(k-1)*MSW),
     'ICC(2,1)/ICC(A,1) abs-agree single':(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n),
     'ICC(3,1)/ICC(C,1) consistency single':(MSR-MSE)/(MSR+(k-1)*MSE),
     'ICC(1,k)':(MSR-MSW)/MSR,
     'ICC(2,k)/ICC(A,k) abs-agree average':(MSR-MSE)/(MSR+(MSC-MSE)/n),
     'ICC(3,k)/ICC(C,k) consistency average':(MSR-MSE)/MSR}
print('\nICC on Q, 175 complete triples')
for k,v in icc(c).items(): print(f'  {k}: {v:.4f}')
print('rater means (175):',c.mean().round(3).to_dict(),'SD:',c.std().round(3).to_dict())
# all 3 raters ICC on 200 would need filling; only for the two complete raters ICC on 200
c2=Q[['FROZEN','UPLOAD_FILLED']]
print('ICC FROZEN vs UPLOAD_FILLED on 200:',{k:round(v,4) for k,v in icc(c2).items()})
# component-level agreement
print('\nPearson per construct (rows where all three present)')
for m in ['Coherence','Capacity','Stress','Abstraction']:
    T=pd.DataFrame({k:v[m] for k,v in R.items()}).dropna()
    print(m,len(T),{f'{a[:6]}-{b[:6]}':round(T[a].corr(T[b]),3) for a,b in itertools.combinations(T.columns,2)}, 'means',T.mean().round(2).to_dict())
# Bootstrap CI for 175-triple ICC(A,1) & r
rng=np.random.default_rng(1); arr=c.values
bs=[icc(arr[rng.integers(0,len(arr),len(arr))])['ICC(2,1)/ICC(A,1) abs-agree single'] for _ in range(2000)]
print('bootstrap 95% CI ICC(A,1):',np.percentile(bs,[2.5,97.5]).round(3))
# case mapping (my inference from casebook cue text; no sealed map supplied)
M={'CASE-15':'Colonial 1700-63','CASE-20':'Mature colonial to imperial war','CASE-25':'Early settlement','CASE-13':'Revolution/Confederation','CASE-07':'Early Republic','CASE-22':'Jacksonian expansion','CASE-12':'Antebellum 1845-61','CASE-02':'Civil War','CASE-19':'Reconstruction','CASE-11':'Gilded Age','CASE-18':'Progressive era','CASE-01':'WWI','CASE-16':'1920s boom','CASE-14':'Great Depression','CASE-05':'New Deal','CASE-09':'WWII','CASE-04':'Postwar 1945-65','CASE-21':'1960s','CASE-17':'1970s','CASE-06':'Reagan 1980s','CASE-23':'1991-2001','CASE-10':'Post-9/11 2001-08','CASE-08':'GFC aftermath','CASE-03':'Realignment 2016-20','CASE-24':'2020-25'}
Qs=Q.copy()
case=Qs.groupby(level=0).mean()  # mean node Q per case (NaN skipped)
case['label']=case.index.map(M)
pd.set_option('display.width',200)
print('\nMean Q per case (nodes with data; per-rater skipna)\n',case.round(2).to_string())
# Fair comparison: only nodes complete for all 3
cc=c.groupby(level=0).mean(); cc['n_nodes']=c.groupby(level=0).size(); cc['label']=cc.index.map(M)
cc.round(2).to_csv('case_mean_Q_175.csv'); case.round(3).to_csv('case_mean_Q_skipna.csv')
Q.assign(**{}).to_csv('node_Q_all_raters.csv')
print('\nMean Q per case, 175 complete triples\n',cc.round(2).to_string())
def chg(a,b,tbl):
    return {r:round(tbl.loc[b,r]-tbl.loc[a,r],2) for r in ['FROZEN','UPLOAD_FILLED','CLAUDE_files27']}
for a,b,name in [('CASE-16','CASE-14','1920s->Depression'),('CASE-05','CASE-09','New Deal->WWII'),('CASE-23','CASE-24','1991-2001->2020-25'),('CASE-02','CASE-19','Civil War->Reconstruction'),('CASE-14','CASE-05','Depression->New Deal'),('CASE-12','CASE-02','Antebellum->Civil War')]:
    print(name,'(175 triples)',chg(a,b,cc),'(all avail)',chg(a,b,case))
# rank correlations of case means
print('\nCase-mean correlation (25 cases, 175-trip means)')
print(cc[['FROZEN','UPLOAD_FILLED','CLAUDE_files27']].corr().round(3))
# contemporary node profile
print('\nCASE-24 node Q and raw by rater')
for k,v in R.items():
    d=v.loc['CASE-24'][['Coherence','Capacity','Stress','Abstraction','Q']]; print(k); print(d.to_string())
print('\nCASE-24 node Q rank (within case); mean over raters')
n24=Q.loc['CASE-24']; print(n24.assign(mean=n24.mean(1)).round(1).to_string())
# Is the 24 Helm/Lore low relative to other cases? percentile of node within same-node across 25 cases
print('\nCASE-24 node Q vs same node across other 24 cases (percentile rank, per rater) and vs own case mean')
for k in Q.columns:
    row=[]
    for nd in ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']:
        s=Q[k].xs(nd,level='Node'); v=s['CASE-24']
        pr=(s.dropna()<v).mean()*100 if pd.notna(v) else np.nan
        row.append(f'{nd}:{v:.0f}(pct {pr:.0f})' if pd.notna(v) else f'{nd}:NA')
    print(k,' '.join(row))
