"""Build FROZEN_CAMS.csv and INDEPENDENT_EVENTS*.csv per PREREG_P3_17Oct2026.md."""
import pandas as pd, numpy as np, json, hashlib
J='/workspace/wm_pr/juno/JUNO_Unified_Dataset.csv'
V='/workspace/analysis/tm_kimi3/morphology_test/vdem_core_v16.csv'
C='sources/coups-by-country.csv'
d=pd.read_csv(J,low_memory=False)
F=d[['Society','Year','Node','Coherence','Capacity','Stress','Abstraction']].copy()
F=F[F.Society!='Hong Kong']
F.to_csv('FROZEN_CAMS.csv',index=False)
# P7 feature, one value per society-year (check constancy)
l2=d.groupby(['Society','Year']).Lambda2_Calc.agg(['min','max'])
assert (l2['max']-l2['min']).abs().max()<1e-9
l2['min'].rename('Lambda2').reset_index().to_csv('LAMBDA2.csv',index=False)

COVERAGE={'Cambodia':1953,'Laos':1953,'Democratic Republic of the Congo':1960,'Mauritania':1960,
          'Nigeria':1960,'Singapore':1965,'UAE':1971,'Latvia':1991,'Ukraine':1991}
ALIAS={'UAE':'United Arab Emirates'}
c=pd.read_csv(C)
socs=sorted(F.Society.unique())
rows=[]
for s in socs:
    name=ALIAS.get(s,s); yrs=set(c.loc[c.country==name,'year'])
    for y in range(max(1950,COVERAGE.get(s,1950)),2026):
        rows.append((s,y,int(y in yrs)))
E1=pd.DataFrame(rows,columns=['Society','Year','Event']); E1.to_csv('INDEPENDENT_EVENTS.csv',index=False)
# check every coup row for our societies is inside coverage
for s in socs:
    name=ALIAS.get(s,s); out=c[(c.country==name)&((c.year<max(1950,COVERAGE.get(s,1950)))|(c.year>2025))]
    if len(out): print('outside coverage:',s,out.year.tolist())
v=pd.read_csv(V)
VAL={'UK':'United Kingdom','USA':'United States of America','UAE':'United Arab Emirates','Turkey':'Türkiye'}
rows=[]
for s in socs:
    sub=v[v.country_name==VAL.get(s,s)].dropna(subset=['v2x_libdem']).set_index('year').v2x_libdem.sort_index()
    sub=sub[~sub.index.duplicated()]
    for y in sub.index:
        if y-1 in sub.index: rows.append((s,int(y),int(sub[y]-sub[y-1]<=-0.05)))
E2=pd.DataFrame(rows,columns=['Society','Year','Event']); E2.to_csv('INDEPENDENT_EVENTS_S2_vdem.csv',index=False)
print('E1',len(E1),E1.Society.nunique(),E1.Event.sum(),' E2',len(E2),E2.Society.nunique(),E2.Event.sum())
print('E2 missing socs',[s for s in socs if s not in set(E2.Society)])
for f in ['FROZEN_CAMS.csv','INDEPENDENT_EVENTS.csv','INDEPENDENT_EVENTS_S2_vdem.csv','LAMBDA2.csv']:
    print(f,hashlib.sha256(open(f,'rb').read()).hexdigest())
