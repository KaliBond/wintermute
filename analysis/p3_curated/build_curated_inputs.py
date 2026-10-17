import pandas as pd, numpy as np
C=pd.read_csv('CURATED_CAMS.csv')
ev=pd.read_csv('../p3/INDEPENDENT_EVENTS.csv'); coups=pd.read_csv('../p3/sources/coups-by-country.csv')
NEW={'Eritrea':1993}
extra=[]
for s,y0 in NEW.items():
    yrs=set(coups.year[coups.country==s])
    extra+=[(s,y,int(y in yrs)) for y in range(max(1950,y0),2026)]
E=pd.concat([ev[ev.Society.isin(C.Society.unique())],pd.DataFrame(extra,columns=ev.columns)])
assert set(C.Society.unique())<=set(E.Society.unique()), set(C.Society.unique())-set(E.Society.unique())
E.to_csv('EVENTS_CURATED.csv',index=False)
def lam2(df):
    out=[]
    for (s,y),g in df.groupby(['Society','Year']):
        q=(0.6*g.Coherence.values+0.4*g.Abstraction.values)/10; S=g.Stress.values
        w=np.sqrt(np.clip(q,0,None))*2**(-S/10); B=np.outer(w,w); np.fill_diagonal(B,0)
        L=np.diag(B.sum(1))-B; out.append((s,y,np.sort(np.linalg.eigvalsh(L))[1]))
    return pd.DataFrame(out,columns=['Society','Year','Lambda2'])
lam2(C).to_csv('LAMBDA2_CURATED.csv',index=False)
J=pd.read_csv('../p3/LAMBDA2.csv'); F=pd.read_csv('../p3/FROZEN_CAMS.csv')
chk=lam2(F[F.Society.isin(['Argentina','USA','Iran'])]).merge(J,on=['Society','Year'],suffixes=('_re','_juno'))
print('lambda2 recompute vs JUNO: n',len(chk),'max abs diff',float((chk.Lambda2_re-chk.Lambda2_juno).abs().max()))
print('events rows',len(E),'coup-years',int(E.Event.sum()))
