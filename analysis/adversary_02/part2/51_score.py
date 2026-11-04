# ADVERSARY-02 Part 2 central uniform recompute + Section 7 scoring. Derived values from PRIMITIVES ONLY (V=C+K-S; Bond per sealed formula).
# Thresholds are copied verbatim from the SIGNED Section 7 (manifest 766a1cbd...). Nothing imputed. Run: python3 51_score.py
import pandas as pd, numpy as np, math, itertools, json
O='/workspace/evidence/runs/part2/analysis_v2_manus_2026-11-04/'
S7='/workspace/evidence/ADVERSARY-02/analysis/section7/'
P=pd.read_csv(O+'all_passes_long.csv'); D=['C','K','S','A']
NAT=['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']
YRS=[2021,2022,2023,2024]; COMP=['Evergrande','Vanke','Longfor','Huawei','SMIC','BYD','CATL']
PRIMARY=['Claude_agent','Codex','Kimi','Gemini','Claude_chat','Perplexity_unconf','Manus']
# ---------- cells ----------
def run_cells(run,nmin_override=None,passes=None):
    g=P[P.Run==run]
    if passes is not None: g=g[g.Pass.isin(passes)]
    npass=g.Pass.nunique()
    nmin=nmin_override if nmin_override else (3 if npass>=3 else 1)   # sealed n_eff>=3 applies where >=3 passes exist; 1 otherwise (see methods note)
    a=g.dropna(subset=['C']).groupby(['Entity','Year','Node'])
    c=a[D].mean(); c['n_eff']=a.size(); c=c[c.n_eff>=nmin].reset_index(); return c
def pool(cellsets):
    x=pd.concat([c.assign(Run=k) for k,c in cellsets.items()])
    p=x.groupby(['Entity','Year','Node'])[D].mean(); p['n_runs']=x.groupby(['Entity','Year','Node']).Run.nunique()
    p=p.reset_index(); p['V']=p.C+p.K-p.S; return p
def bond_ey(g):
    g=g.reset_index(drop=True); vals=[]
    for i in g.index:
        bs=[(0.6*g.C[i]*g.C[j]+0.4*g.A[i]*g.A[j])*math.exp(-(g.S[i]+g.S[j])/20) for j in g.index if j!=i]; vals.append(np.mean(bs))
    return np.mean(vals)
def ey_table(p):
    rows=[]
    for (e,y),g in p.groupby(['Entity','Year']):
        n=len(g); ok=n>=6
        rows.append(dict(Entity=e,Year=y,n_nodes=n,scorable=ok,M=g.V.mean() if ok else np.nan,meanS=g.S.mean() if ok else np.nan,meanC=g.C.mean() if ok else np.nan,
                         meanK=g.K.mean() if ok else np.nan,meanA=g.A.mean() if ok else np.nan,Bond=bond_ey(g) if ok else np.nan))
    return pd.DataFrame(rows)
class Pool:
    def __init__(s,name,cellsets):
        s.name=name; s.runs=list(cellsets); s.cells=pool(cellsets); s.ey=ey_table(s.cells)
        s.Mt=s.ey.pivot(index='Year',columns='Entity',values='M'); s.Bt=s.ey.pivot(index='Year',columns='Entity',values='Bond'); s.St=s.ey.pivot(index='Year',columns='Entity',values='meanS')
    def M(s,e,y):
        try: return s.Mt.loc[y,e]
        except KeyError: return np.nan
    def B(s,e,y):
        try: return s.Bt.loc[y,e]
        except KeyError: return np.nan
    def Sm(s,e,y):
        try: return s.St.loc[y,e]
        except KeyError: return np.nan
    def V(s,e,y,n):
        g=s.cells[(s.cells.Entity==e)&(s.cells.Year==y)&(s.cells.Node==n)]; return g.V.iloc[0] if len(g) else np.nan
    def nodeV(s,e,y):
        g=s.cells[(s.cells.Entity==e)&(s.cells.Year==y)]
        return g.set_index('Node').V if len(g)>=6 else None
    def cube(s,e):
        X=np.full((4,8,4),np.nan)
        for r in s.cells[s.cells.Entity==e].itertuples(): X[r.Year-2021,NAT.index(r.Node)]=[r.C,r.K,r.S,r.A]
        return X
    def mad(s,a,b): return np.nanmean(np.abs(s.cube(a)-s.cube(b)))
nan=np.isnan
# ---------- verdict helpers ----------
def all3(conds):  # list of True/False/None -> verdict for "all must hold"
    conds=[None if c is None else bool(c) for c in conds]
    if any(c is False for c in conds): return 'FAIL'
    if any(c is None for c in conds): return 'INDETERMINATE'
    return 'PASS'
def cmp(x,op,t):
    if nan(x): return None
    return {'>=':x>=t,'>':x>t,'<=':x<=t,'<':x<t}[op]
# ---------- Part 1 baseline pools (for P12 splice only) ----------
P1={'noGrok':pd.read_csv(S7+'pooled_cells_noGrok.csv'),'withGrok':pd.read_csv(S7+'pooled_cells_withGrok.csv')}
# ---------- items ----------
def thr(pr,a,b): return a if pr=='P' else b
def item_P1(p,t):
    M=[p.M('Evergrande',y) for y in YRS]; lim=thr(t,5.0,5.5); l24=thr(t,4.0,4.5)
    v=all3([cmp(m,'<=',lim) for m in M]+[cmp(M[3],'<=',l24)]); return v,dict(M2021_24=[round(m,3) for m in M],yr_limit=lim,M2024_limit=l24)
def item_P2(p,t):
    dlim=thr(t,2.3,3.0); yv=[];det=[]
    for y in YRS:
        nv=p.nodeV('Evergrande',y); need=['Stewards','Shield','Craft','Flow']
        if nv is None or any(n not in nv.index for n in need): yv.append(None); det.append('missing'); continue
        low=nv.min(); lowset=set(nv[nv==low].index); order=nv.sort_values()
        a=lowset<= {'Stewards','Shield'}; b='Stewards' in list(order.index[:2]); c=min(nv['Craft'],nv['Flow'])>max(nv['Shield'],nv['Stewards']); d=nv['Stewards']<dlim
        yv.append(bool(a and b and c and d)); det.append(f"{y}:a{int(a)}b{int(b)}c{int(c)}d{int(d)}(St={nv['Stewards']:.2f},Sh={nv['Shield']:.2f},low={order.index[0]})")
    ok=sum(x is True for x in yv); bad=sum(x is False for x in yv)
    v='PASS' if ok>=3 else ('FAIL' if bad>=2 else 'INDETERMINATE'); return v,dict(years_all_four=ok,detail=det,d_limit=dlim)
def item_P3(p,t):
    M=[p.M('Huawei',y) for y in YRS]; hi=thr(t,9.15,9.14)
    conds=[cmp(M[3],'>',4.0),cmp(M[3],'<',hi)]+([cmp(m,'<',hi) for m in M] if t=='P' else [])
    sh=[]
    for y in YRS:
        nv=p.nodeV('Huawei',y); sh.append(None if nv is None else nv.idxmin()=='Shield')
    return all3(conds),dict(M2021_24=[round(m,3) for m in M],hi=hi,info_Shield_lowest_years=sum(x is True for x in sh),info_lowest_by_year=[None if p.nodeV('Huawei',y) is None else p.nodeV('Huawei',y).idxmin() for y in YRS])
def item_P4(p,t):
    M=[p.M('BYD',y) for y in YRS]; l24=thr(t,7.2,7.5)
    return all3([cmp(M[3],'>=',l24)]+[cmp(m,'>=',6.5) for m in M]),dict(M2021_24=[round(m,3) for m in M],M2024_min=l24)
def item_P5(p,t):
    M=[p.M('CATL',y) for y in YRS]; B=[p.B('CATL',y) for y in YRS]; l24,ly,lb=thr(t,(8.0,8.3,23.1),(8.1,8.55,24.2))
    return all3([cmp(M[3],'>=',l24)]+[cmp(m,'>=',ly) for m in M]+[cmp(b,'>=',lb) for b in B]),dict(M2021_24=[round(m,3) for m in M],Bond2021_24=[round(b,2) for b in B],lims=(l24,ly,lb))
def item_P6(p,t):
    ok=bad=un=0
    for e in ['Evergrande','Huawei','BYD','CATL']:
        for y in YRS:
            nv=p.nodeV(e,y)
            if nv is None: un+=1
            elif nv.idxmin() in ('Stewards','Shield'): ok+=1
            else: bad+=1
    v='PASS' if ok>=14 else ('FAIL' if ok+un<14 else 'INDETERMINATE'); return v,dict(satisfied=ok,not_satisfied=bad,unscorable=un)
def item_P7(p,t):
    lim=thr(t,10.0,8.0); g=[]
    for y in YRS:
        a,b,c=p.B('Vanke',y),p.B('Longfor',y),p.B('Evergrande',y); g.append(np.nan if nan(a) or nan(b) or nan(c) else min(a,b)-c)
    return all3([cmp(x,'>=',lim) for x in g]),dict(gap2021_24=[round(x,2) for x in g],lim=lim)
def item_P8(p,t):
    lim=thr(t,5.0,3.5); d=p.mad('Vanke','Longfor'); r1=p.mad('Evergrande','Vanke')/d; r2=p.mad('Evergrande','Longfor')/d
    return all3([cmp(r1,'>=',lim),cmp(r2,'>=',lim)]),dict(ratio_vs_Vanke=round(r1,2),ratio_vs_Longfor=round(r2,2),VL_MAD=round(d,3),lim=lim)
def item_P9(p,t):
    g=[abs(p.M('Huawei',y)-p.M('SMIC',y)) for y in YRS]; return all3([cmp(x,'<=',1.5) for x in g]),dict(gap2021_24=[round(x,3) for x in g])
def item_P10(p,t):
    l=p.mad('Evergrande','Longfor'); v=p.mad('Evergrande','Vanke'); return all3([None if nan(l) or nan(v) else l>v]),dict(MAD_Ev_Longfor=round(l,3),MAD_Ev_Vanke=round(v,3))
def item_P11(p,t):
    V=[p.M('Vanke',y) for y in YRS]; L=[p.M('Longfor',y) for y in YRS]
    return all3([cmp(x,'>=',7.49) for x in V]+[cmp(x,'>=',8.66) for x in L]),dict(Vanke=[round(x,3) for x in V],Longfor=[round(x,3) for x in L])
# ---- P12 coupling (33_coupling_codex_v2.py logic unchanged, YMAX=2024) ----
PAIRS=[('Stewards','Stewards',0,1),('Flow','Flow',0,1),('Helm','Helm',1,2),('Lore','Lore',1,3)]
EVENTS=[('Evergrande','Property restraint 2016',2016,-1),('Evergrande','Three Red Lines 2020',2020,-1),('Huawei','US Entity List May 2019',2019,-1),
 ('BYD','NEV plan 2012',2012,+1),('BYD','NEV plan 2020',2020,+1),('CATL','NEV plan 2012',2012,+1),('CATL','NEV plan 2020',2020,+1),('SMIC','US Entity List Dec 2020',2020,-1)]
CFG={'primary':dict(thr_=1,gate='G1',strict=True),'lenient_G2_lag0':dict(thr_=1,gate='G2',strict=False),'thr2':dict(thr_=2,gate='G1',strict=True)}
def p1key(pool_kind): return 'withGrok' if pool_kind=='withGrok' else 'noGrok'
def vseries(p2cells,p1kind,e,n,mask2021):
    p1=P1[p1kind]; a=p1[(p1.Entity==e)&(p1.Node==n)].assign(V=lambda d:d.C+d.K-d.S).set_index('Year').V
    b=p2cells[(p2cells.Entity==e)&(p2cells.Node==n)]; b=b.set_index('Year').V if len(b) else pd.Series(dtype=float)
    s=pd.concat([a[a.index<=2020],b[b.index>=2021]]); s=s[~s.index.duplicated()]
    d=s.reindex(range(2009,2025)).diff()
    if mask2021: d.loc[2021]=np.nan
    return d
def coupling_event(p2cells,p1kind,ent,E,s,mask2021,thr_,gate,strict,YMAX=2024):
    res=[]
    for nn,cn,lo,hi in PAIRS:
        dn=vseries(p2cells,p1kind,'China',nn,mask2021); dc=vseries(p2cells,p1kind,ent,cn,mask2021)
        x=dn.get(E,np.nan); r=None
        if nan(x) or abs(x)<thr_ or (gate=='G1' and np.sign(x)!=s): res.append('FAIL-national-gate'); continue
        for t in range(E-2,E):
            v=dc.get(t,np.nan)
            if not nan(v) and abs(v)>=thr_ and np.sign(v)==s: r='FAIL-company-first'; break
        if r: res.append(r); continue
        lo_eff=max(lo,1) if strict else lo; na=False
        for t in range(E+lo_eff,E+hi+1):
            if t>YMAX: break
            v=dc.get(t,np.nan)
            if nan(v): na=True; continue
            if abs(v)>=thr_: r='PASS' if np.sign(v)==s else 'FAIL-wrong-direction'; break
        if r: res.append(r); continue
        if E+hi>YMAX: res.append('CENSORED')
        else: res.append('UNSCORABLE-NA' if na else 'FAIL-no-response')
    return res
def coupling_scores(p,p1kind,cfg='primary',mask2021=False):
    out=[]
    for ent,ev,E,s in EVENTS:
        res=coupling_event(p.cells,p1kind,ent,E,s,mask2021,**CFG[cfg])
        out.append(dict(Pool=p.name,Config=cfg,Mask2021=mask2021,Entity=ent,Event=ev,Score=res.count('PASS'),Censored=res.count('CENSORED'),Unscorable=res.count('UNSCORABLE-NA'),Detail='; '.join(f'{pp[0]}:{r}' for pp,r in zip(PAIRS,res))))
    return out
def item_P12(p,t):
    sc=coupling_scores(p,'withGrok' if t=='S' else 'noGrok'); 
    mx=max(x['Score'] for x in sc); mxu=max(x['Score']+x['Censored']+x['Unscorable'] for x in sc)
    v='FAIL' if mx>=2 else ('PASS' if mxu<2 else 'INDETERMINATE')
    return v,dict(event_scores={x['Event']+' ('+x['Entity']+')':x['Score'] for x in sc},max_score=mx,max_upper_bound_incl_unscorable=mxu)
# ---- contrasts ----
def item_C1(p,t):
    lim=thr(t,9.30,9.18); m=p.M('China',2024); return all3([cmp(m,'>=',lim)]),dict(M_China_2024=round(m,3),lim=lim)
def item_C2(p,t):
    lim=thr(t,7.39,7.26); m=p.M('Huawei',2024); return all3([cmp(m,'>',lim)]),dict(M_Huawei_2024=round(m,3),lim=lim,info_margin_version_ge=thr(t,8.67,None),margin_pass=None if nan(m) or t=='S' else bool(m>=8.67))
def item_C3(p,t):
    ls,lm=thr(t,(5.54,7.2),(5.15,7.5)); s=p.Sm('BYD',2024); m=p.M('BYD',2024)
    nodeS=np.nan
    g=p.cells[(p.cells.Entity=='BYD')&(p.cells.Year==2024)&p.cells.Node.isin(['Craft','Hands'])]
    if len(g)==2: nodeS=g.S.mean()
    return all3([cmp(s,'>=',ls),cmp(m,'>=',lm)]),dict(S_BYD_2024=round(s,3),M_BYD_2024=round(m,3),lims=(ls,lm),info_CraftHands_S=None if nan(nodeS) else round(nodeS,3),info_nodeS_lim=thr(t,5.48,5.04))
ITEMS=[('P1',item_P1),('P2',item_P2),('P3',item_P3),('P4',item_P4),('P5',item_P5),('P6',item_P6),('P7',item_P7),('P8',item_P8),('P9',item_P9),('P10',item_P10),('P11',item_P11),('P12',item_P12),('C1',item_C1),('C2',item_C2),('C3',item_C3)]
def fc5(verdicts):  # verdicts: dict item->verdict
    f=sum(verdicts[k]=='FAIL' for k in ['P1','P3','P4','P5']); u=sum(verdicts[k]=='INDETERMINATE' for k in ['P1','P3','P4','P5'])
    if f>=3: return 'TRIGGERED'
    if f+u<3: return 'NOT TRIGGERED'
    return 'INDETERMINATE'
def score_pool(p,t):
    V={};X={}
    for k,fn in ITEMS: V[k],X[k]=fn(p,t)
    V['FC5']=fc5(V); X['FC5']=dict(fails_among_P1_P3_P4_P5=[k for k in ['P1','P3','P4','P5'] if V[k]=='FAIL'])
    return V,X
if __name__=='__main__':
    cells={r:run_cells(r) for r in PRIMARY+['Grok']}
    POOLS={'a_primary(7 runs)':(PRIMARY,'P'),'d_primary_noManus':([r for r in PRIMARY if r!='Manus'],'P'),'b_secondary(+Grok)':(PRIMARY+['Grok'],'S'),'c_primary_noPerplexity':([r for r in PRIMARY if r!='Perplexity_unconf'],'P')}
    pools={};allV={};allX={}
    for name,(rs,t) in POOLS.items():
        p=Pool(name,{r:cells[r] for r in rs}); pools[name]=p; allV[name],allX[name]=score_pool(p,t)
        p.cells.to_csv(O+f'derived_pooled_cells_{name.split("(")[0]}.csv',index=False); p.ey.to_csv(O+f'derived_entity_year_{name.split("(")[0]}.csv',index=False)
    # per-scorer (primary thresholds, own cells only)
    per={};perX={};peref=[]
    for r in PRIMARY+['Grok']:
        p=Pool(r,{r:cells[r]}); per[r],perX[r]=score_pool(p,'P'); peref.append(p.ey.assign(Run=r))
    pd.concat(peref).to_csv(O+'derived_entity_year_per_scorer.csv',index=False)
    cellsall=pd.concat([c.assign(Run=k) for k,c in cells.items()]); cellsall['V']=cellsall.C+cellsall.K-cellsall.S; cellsall.to_csv(O+'derived_run_cells.csv',index=False)
    # sensitivities (primary thresholds, primary pool): Codex & Claude_agent n_eff>=1 ; Kimi pass1 only ; splice-masked P12
    sens={}
    def run_sens(label,cs,rs=PRIMARY):
        p=Pool(label,{r:cs[r] for r in rs}); v,x=score_pool(p,'P'); sens[label]=(v,x)
    c2=dict(cells); c2['Codex']=run_cells('Codex',1); c2['Claude_agent']=run_cells('Claude_agent',1); c2['Manus']=run_cells('Manus',1); run_sens('s1_nmin1_for_5pass_runs',c2)
    c3=dict(cells); c3['Kimi']=run_cells('Kimi',passes=[1]); run_sens('s2_Kimi_pass1_only',c3)
    c4=dict(cells); c4['Kimi']=run_cells('Kimi',passes=[2]); run_sens('s3_Kimi_pass2_only',c4)
    p=pools['a_primary(7 runs)']; sens['s4_P12_2021_diff_masked']=({'P12':('PASS' if max(x['Score']+x['Censored']+x['Unscorable'] for x in coupling_scores(p,'noGrok',mask2021=True))<2 else ('FAIL' if max(x['Score'] for x in coupling_scores(p,'noGrok',mask2021=True))>=2 else 'INDETERMINATE'))},{'P12':coupling_scores(p,'noGrok',mask2021=True)})
    # coupling detail tables
    cp=[]
    for name,(rs,t) in POOLS.items():
        for cfg in CFG: cp+=coupling_scores(pools[name],'withGrok' if t=='S' else 'noGrok',cfg)
    for r in PRIMARY+['Grok']:
        cp+=coupling_scores(Pool(r,{r:cells[r]}),'noGrok','primary')
    cp+=coupling_scores(p,'noGrok','primary',mask2021=True)
    pd.DataFrame(cp).to_csv(O+'derived_coupling_event_scores.csv',index=False)
    # scoring table
    ids=[k for k,_ in ITEMS]+['FC5']; rows=[]
    for k in ids:
        row=dict(Item=k,**{ 'a_primary':allV['a_primary(7 runs)'][k],'b_secondary_withGrok':allV['b_secondary(+Grok)'][k],'c_primary_noPerplexity':allV['c_primary_noPerplexity'][k],'d_primary_noManus':allV['d_primary_noManus'][k]})
        for r in PRIMARY+['Grok']: row['scorer_'+r]=per[r][k]
        pv=[per[r][k] for r in PRIMARY]; row['agree_primary7']=f"{pv.count('PASS')}P/{pv.count('FAIL')}F/{pv.count('INDETERMINATE')}I"+('' if k!='FC5' else ' (PASS=NOT TRIGGERED needs read: see verdict words)')
        row['agree_primary7']=f"{sum(v in('PASS','NOT TRIGGERED') for v in pv)} pass / {sum(v in('FAIL','TRIGGERED') for v in pv)} fail / {pv.count('INDETERMINATE')} indet (of 7)"
        row['pools_disagree']=len({row['a_primary'],row['b_secondary_withGrok'],row['c_primary_noPerplexity'],row['d_primary_noManus']})>1
        row['metrics_a']=json.dumps(allX['a_primary(7 runs)'][k],default=str); row['metrics_b']=json.dumps(allX['b_secondary(+Grok)'][k],default=str); row['metrics_c']=json.dumps(allX['c_primary_noPerplexity'][k],default=str); row['metrics_d']=json.dumps(allX['d_primary_noManus'][k],default=str)
        for s,(v,x) in sens.items():
            if k in v: row['sens_'+s]=v[k] if not isinstance(v[k],tuple) else v[k][0]
        rows.append(row)
    T=pd.DataFrame(rows); T.to_csv(O+'scoring_table.csv',index=False)
    # per-scorer metrics file
    pm=[dict(Run=r,Item=k,Verdict=per[r][k],Metrics=json.dumps(perX[r][k],default=str)) for r in PRIMARY+['Grok'] for k in ids]
    pd.DataFrame(pm).to_csv(O+'scoring_per_scorer_metrics.csv',index=False)
    pd.set_option('display.width',250); pd.set_option('display.max_colwidth',200)
    print(T[['Item','a_primary','b_secondary_withGrok','c_primary_noPerplexity','pools_disagree','agree_primary7']+[c for c in T.columns if c.startswith('sens_')]].to_string())
    print(T[['Item']+[c for c in T.columns if c.startswith('scorer_')]].to_string())
    for k in ids: print(k,'| a:',allX['a_primary(7 runs)'][k],'\n   b:',allX['b_secondary(+Grok)'][k],'\n   c:',allX['c_primary_noPerplexity'][k],'\n   d:',allX['d_primary_noManus'][k])
    # reproduction check of Section 7 stated 2020 values from Part 1 pooled cells (sanity only)
    for kind in ['noGrok','withGrok']:
        q=P1[kind].assign(V=lambda d:d.C+d.K-d.S); m=q[q.Year==2020].groupby('Entity').V.mean().round(2).to_dict(); print('Part1 2020 M',kind,m)
