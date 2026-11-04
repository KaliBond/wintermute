# ADVERSARY-02 entity personality: two axes from pooled primitives. x = K - S (capacity minus stress), y = A * C (abstraction times coherence).
# Node-level: from the node's pooled mean primitives. Entity-year: mean over valid nodes of node-level values (>=6 valid nodes, else NA), the same validity rule as M.
# 2021-2024: primary pool of 7 runs (incl. Manus), pooled exactly as the verdict pool (a) (run_cells with n_eff rule, equal weight per run).
# 2010-2020: Part 1 pooled cells from Section 7 (section7/pooled_cells_noGrok.csv), spliced as they are. Different scorer sets and prompt era: the 2020/2021 seam is a method seam.
import importlib.util, pandas as pd, numpy as np
O='/workspace/evidence/runs/part2/analysis_v2_manus_2026-11-04/'
spec=importlib.util.spec_from_file_location('s','/workspace/evidence/runs/part2/analysis_v2_manus_2026-11-04/51_score.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
YM=3
cells={r:s.run_cells(r) for r in s.PRIMARY}
pool=s.Pool('a',cells).cells[['Entity','Year','Node','C','K','S','A','n_runs']]
p1=pd.read_csv(s.S7+'pooled_cells_noGrok.csv')[['Entity','Year','Node','C','K','S','A','n_sets']].rename(columns={'n_sets':'n_runs'})
def add_axes(d): d=d.dropna(subset=['C','K','S','A']).copy(); d['x_K_minus_S']=d.K-d.S; d['y_A_times_C']=d.A*d.C; return d
node=add_axes(pd.concat([p1.assign(Source='Part1_pooled_noGrok'),pool.assign(Source='Part2_pool_a_7runs')],ignore_index=True))
node.round(4).to_csv(O+'entity_personality_node_level.csv',index=False)
X0=0.0; Y0=5.5*5.5   # quadrant lines: K-S = 0; A*C = 30.25 (product of the scale midpoints 5.5 x 5.5)
def quad(x,y):
    if np.isnan(x): return 'n/a'
    return ('high AC' if y>=Y0 else 'low AC')+' / '+('high KS' if x>=X0 else 'low KS')
def ey(d):
    rows=[]
    for (e,y),g in d.groupby(['Entity','Year']):
        n=len(g); ok=n>=6; gg=g.sort_values('y_A_times_C')
        rows.append(dict(Entity=e,Year=y,n_nodes=n,x_K_minus_S=g.x_K_minus_S.mean() if ok else np.nan,y_A_times_C=g.y_A_times_C.mean() if ok else np.nan,
          strongest_node_AC=gg.Node.iloc[-1],weakest_node_AC=gg.Node.iloc[0],strongest_node_KS=g.sort_values('x_K_minus_S').Node.iloc[-1],weakest_node_KS=g.sort_values('x_K_minus_S').Node.iloc[0]))
    r=pd.DataFrame(rows); r['quadrant']=[quad(a,b) for a,b in zip(r.x_K_minus_S,r.y_A_times_C)]; return r
EY=ey(node); EY.insert(2,'Source',np.where(EY.Year<=2020,'Part1_pooled_noGrok','Part2_pool_a_7runs'))
EY.round(4).to_csv(O+'entity_personality_entity_year.csv',index=False)
# per scorer (own run cells, 2021-2024)
rows=[]
for r,c in cells.items():
    d=add_axes(c.assign(n_runs=1)); e=ey(d); e.insert(0,'Run',r); rows.append(e)
PS=pd.concat(rows,ignore_index=True); PS.round(4).to_csv(O+'entity_personality_per_scorer.csv',index=False)
# spread: per entity-year across the 7 primary runs
pool_e=EY[EY.Year>=2021].set_index(['Entity','Year'])
sp=[]
for (e,y),g in PS.groupby(['Entity','Year']):
    g=g.dropna(subset=['x_K_minus_S']); pe=pool_e.loc[(e,y)]
    sp.append(dict(Entity=e,Year=y,n_runs_scorable=len(g),pool_x=pe.x_K_minus_S,pool_y=pe.y_A_times_C,pool_quadrant=pe.quadrant,
      x_min=g.x_K_minus_S.min(),x_max=g.x_K_minus_S.max(),x_sd=g.x_K_minus_S.std(ddof=0),y_min=g.y_A_times_C.min(),y_max=g.y_A_times_C.max(),y_sd=g.y_A_times_C.std(ddof=0),
      runs_in_pool_quadrant=int((g.quadrant==pe.quadrant).sum())))
SP=pd.DataFrame(sp); SP.round(4).to_csv(O+'entity_personality_scorer_spread.csv',index=False)
# agreement on entity ordering: per run and year, Spearman of the 8 entity positions against the pool
ag=[]
for r,g in PS.groupby('Run'):
    for y in (2021,2024):
        a=g[g.Year==y].set_index('Entity'); b=pool_e.xs(y,level='Year')
        j=a.join(b,lsuffix='_run',rsuffix='_pool',how='inner').dropna(subset=['x_K_minus_S_run'])
        ag.append(dict(Run=r,Year=y,n_entities=len(j),spearman_x=j.x_K_minus_S_run.corr(j.x_K_minus_S_pool,method='spearman'),spearman_y=j.y_A_times_C_run.corr(j.y_A_times_C_pool,method='spearman'),
          same_quadrant_as_pool=int((j.quadrant_run==j.quadrant_pool).sum())))
AG=pd.DataFrame(ag); AG.round(4).to_csv(O+'entity_personality_scorer_agreement.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(EY[EY.Year.isin([2010,2015,2020,2021,2022,2023,2024])].round(2).to_string())
print(SP[SP.Year.isin([2021,2024])].round(2).to_string()); print(AG.round(2).to_string())
