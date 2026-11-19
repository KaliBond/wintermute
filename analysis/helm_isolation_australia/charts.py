import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o=pd.read_csv('signed_gaps.csv'); W,H=12,6.27
p=o[o.series.isin(['ENS','CAM5','Gemini'])].pivot(index='year',columns='series',values='I')
f,a=plt.subplots(figsize=(W,H),dpi=100)
for y in (2010,2018): a.axvspan(y-.4,y+.4,color='#f4d03f',alpha=.35)
for s,c in [('ENS','#1f4e79'),('CAM5','#2e86c1'),('Gemini','#c0392b')]: a.plot(p.index,p[s],'o-',lw=2.5,color=c,label=s)
a.axhline(1.86,ls='--',color='k'); a.text(2008,1.93,'Frozen threshold 1.86',fontsize=11)
a.set_title('Helm Isolation, Australia 2008–2025: all three series fire in 2010 and 2018',fontsize=15)
a.set_ylabel('|Helm coherence − mean of other seven|'); a.set_xticks(range(2008,2026)); a.legend(); a.spines[['top','right']].set_visible(False)
f.tight_layout(); f.savefig('charts/01_abs_gap_three_series.png')
k=pd.read_csv('/workspace/wintermute/Australian_Gem_Jan30.csv'); k=k[k.Year.between(2009,2013)]
h=k[k.Node=='Helm'].set_index('Year').Coherence; m=k[k.Node!='Helm'].groupby('Year').Coherence.mean()
f,a=plt.subplots(figsize=(W,H),dpi=100); a.plot(h.index,h,'o-',lw=3,color='#c0392b',label='Helm'); a.plot(m.index,m,'o-',lw=3,color='#1f4e79',label='Mean of other seven nodes')
for y in h.index: a.annotate(f'{h[y]-m[y]:+.2f}',(y,(h[y]+m[y])/2),ha='center',fontsize=12)
a.set_ylim(0,10); a.set_xticks(h.index); a.set_ylabel('Coherence'); a.legend(loc='lower left'); a.spines[['top','right']].set_visible(False)
a.set_title('Gemini series, 2009–2013: Helm drops away while the rest holds',fontsize=15); f.tight_layout(); f.savefig('charts/02_gemini_helm_vs_rest.png')
o['lab']=o.series.str.replace('_B[12]','',regex=True); g=o.groupby(['lab','year']).fires.max().unstack()
g=g.reindex(['ENS','CAM5','Gemini','Kimi_R1','Kimi_R2','Kimi_R3'])
f,a=plt.subplots(figsize=(W,H),dpi=100)
for i,s in enumerate(g.index):
  for y in g.columns:
    v=g.loc[s,y]; a.add_patch(plt.Rectangle((y-.45,i-.4),.9,.8,color='#eee' if pd.isna(v) else ('#c0392b' if v else '#d5dbdb')))
a.set_xlim(2007.5,2025.5); a.set_ylim(-.6,len(g)-.4); a.set_yticks(range(len(g))); a.set_yticklabels(g.index); a.set_xticks(range(2008,2026)); a.invert_yaxis()
a.set_title('Years the frozen rule fires (red). Blank = not scored.',fontsize=15); [a.spines[x].set_visible(False) for x in a.spines]
f.tight_layout(); f.savefig('charts/03_firing_grid.png')
