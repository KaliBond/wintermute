import pandas as pd
R='/workspace/wintermute/'
src={'ENS':R+'cleaned_datasets/Australia_ENS_1875_2026_cleaned.csv','CAM5':R+'cleaned_datasets/Australia_cam5_1996_2026_cleaned.csv','Gemini':R+'Australian_Gem_Jan30.csv'}
out=[]
def gaps(df,name):
    for y,g in df.groupby('Year'):
        if not 2008<=y<=2025: continue
        c=g.groupby('Node')['Coherence'].mean()
        if g.Node.nunique()!=8 or len(g)!=8: print('WARN',name,y,len(g))
        D=c['Helm']-c.drop('Helm').mean()
        out.append(dict(series=name,year=y,D=D,I=abs(D),fires=abs(D)>=1.86-1e-12 and abs(D)>=1.86))
for n,p in src.items(): gaps(pd.read_csv(p),n)
k=pd.read_csv('/workspace/uploads/z1027/KSCORESv1_Australia_2015_2025_block0.csv')
for (r,b),g in k.groupby(['Run','Batch']): gaps(g,f'Kimi_{r}_{b}')
o=pd.DataFrame(out); o.to_csv('signed_gaps.csv',index=False,float_format='%.6f')
print(o.pivot(index='year',columns='series',values='D').round(3).to_string())
