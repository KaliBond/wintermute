import pandas as pd, numpy as np, json
NODES=['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']
def norm(df,soc,yr,node,C,K,S,A):
    d=df[[soc,yr,node,C,K,S,A]].copy(); d.columns=['soc','year','node','C','K','S','A']
    d=d[d.node.isin(NODES)]
    return d
def complete_years(d):
    out={}
    for s,g in d.groupby('soc'):
        ok=[]
        for y,h in g.groupby('year'):
            if set(h.node)==set(NODES) and len(h)==8 and h[['C','K','S','A']].notna().all().all(): ok.append(int(y))
        out[s]=sorted(ok)
    return out
def anchors(years,last_outcome=2025):
    ys=set(years); a=[t for t in years if {t-1,t-2}<=ys and t+3<=last_outcome]
    # greedy stride-6 non-overlapping
    sel=[];nxt=-10**9
    for t in a:
        if t-2>=nxt: sel.append(t); nxt=t+4
    sel3=[];nxt=-10**9
    for t in a:
        if t-2>=nxt: sel3.append(t); nxt=t+1
    pros=[t for t in years if {t-1,t-2}<=ys and t>=2025]
    return len(a),len(sel3),len(sel),pros
def report(name,d):
    cy=complete_years(d); rows=[]
    for s,ys in cy.items():
        a,a3,a6,p=anchors(ys)
        rows.append(dict(set=name,soc=s,ymin=min(ys) if ys else None,ymax=max(ys) if ys else None,complete_years=len(ys),anchors_all=a,anchors_stride3=a3,anchors_stride6=a6,prospective_anchors=p))
    return pd.DataFrame(rows)

if __name__=='__main__':
    # Collation 2026-11-21 data census. Scores only; no outcome data opened.
    import os
    D='/workspace/SA-PEP-collate'
    j=pd.read_csv(f'{D}/JUNO_Unified_Dataset_v3_2026-10-19.csv')
    J=norm(j,'Society','Year','Node','Coherence','Capacity','Stress','Abstraction')
    nb=pd.read_csv(f'{D}/nation_blind_ensemble_primitives.csv')
    key={s['set_id']:s['nation'] for s in json.load(open(f'{D}/nation_blind_ensemble_KEY.json'))['sets']}
    nb['soc']=nb.Set_ID.map(key); B=norm(nb,'soc','Year','Node','C','K','S','A')
    # Deduplicated union: JUNO first; add blind-set society-years not present in JUNO (Belgium, Sri Lanka, Japan 1850-1874 etc.)
    jk=set(zip(J.soc,J.year)); Bx=B[[ (s,y) not in jk for s,y in zip(B.soc,B.year)]]
    U=pd.concat([J,Bx])
    r=report('union_JUNO+blind',U)
    r.to_csv('/workspace/SA-PEP/census_2026-11-21_cases.csv',index=False)
    tot=r[['complete_years','anchors_all','anchors_stride3','anchors_stride6']].sum()
    print(r.to_string()); print(tot)
    print('societies',len(r[r.complete_years>0]),'with >=1 stride6 case',int((r.anchors_stride6>0).sum()))
    print('stride6 excl Denmark',int(tot.anchors_stride6-r[r.soc=='Denmark'].anchors_stride6.sum()))
    print('prospective anchor 2026 count',sum(2026 in x for x in r.prospective_anchors))
