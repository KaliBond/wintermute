import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o=pd.read_csv('signed_gaps.csv'); BG='#fbfaf7'; TX='#222'
plt.rcParams.update({'font.size':13,'text.color':TX,'axes.labelcolor':TX,'xtick.color':TX,'ytick.color':TX,'axes.facecolor':BG})
f=plt.figure(figsize=(12,15),dpi=100,facecolor=BG)
gs=f.add_gridspec(3,1,height_ratios=[1.15,1,1],hspace=.45,left=.1,right=.96,top=.9,bottom=.08)
f.text(.5,.965,"Australia's political Helm became isolated.",ha='center',fontsize=26,weight='bold')
f.text(.5,.935,"The rest of the ship kept sailing.",ha='center',fontsize=22,color='#555')
C={'ENS':'#3b5b7a','CAM5':'#7fa3c0','Gemini':'#b5524a'}
a=f.add_subplot(gs[0]); p=o.pivot(index='year',columns='series',values='I')
for y in (2010,2018): a.axvspan(y-.4,y+.4,color='#e8d9a8',alpha=.6,lw=0)
for s in C: a.plot(p.index,p[s],'o-',lw=2.5,color=C[s],label=s)
a.axhline(1.86,ls='--',color='#444',lw=1.3); a.text(2025.4,1.95,'threshold 1.86',ha='right',fontsize=12,color='#444')
a.set_xticks(range(2008,2026)); a.tick_params(axis='x',labelsize=11); a.set_ylim(0,4.9); a.set_ylabel('Absolute Helm gap')
a.legend(loc='upper right',frameon=False,ncol=3); a.set_title('A. Absolute Helm gap, 2008–2025: all three series fire in 2010 and 2018',loc='left',fontsize=15,weight='bold')
b=f.add_subplot(gs[1]); g=pd.read_csv('/workspace/wintermute/Australian_Gem_Jan30.csv'); g=g[g.Year.between(2009,2013)]
h=g[g.Node=='Helm'].set_index('Year').Coherence; m=g[g.Node!='Helm'].groupby('Year').Coherence.mean()
d=o[(o.series=='Gemini')].set_index('year').D
b.plot(h.index,h,'o-',lw=3,color=C['Gemini'],label='Helm'); b.plot(m.index,m,'o-',lw=3,color=C['ENS'],label='Mean of other seven')
for y in h.index:
  b.vlines(y,min(h[y],m[y]),max(h[y],m[y]),color='#bbb',lw=1)
  b.text(y+.07 if y>2009 else y+.1,(h[y]+m[y])/2 if y>2009 else 6.6,f'{d[y]:+.2f}'.replace('-','−'),ha='left',fontsize=13,weight='bold' if y==2012 else 'normal',va='center')
b.set_ylim(0,10.5); b.set_xlim(2008.7,2013.5); b.set_xticks(h.index); b.set_ylabel('Coherence'); b.legend(loc='lower left',frameon=False)
b.set_title('B. Gemini 2009–2013: Helm falls away, gap peaks at −4.43 in 2012',loc='left',fontsize=15,weight='bold')
c=f.add_subplot(gs[2]); o['lab']=o.series.str.replace('_B[12]','',regex=True).str.replace('_',' ')
gr=o.groupby(['lab','year']).fires.max().unstack().reindex(['ENS','CAM5','Gemini','Kimi R1','Kimi R2','Kimi R3'])
for y in (2010,2018): c.axvspan(y-.5,y+.5,color='#e8d9a8',alpha=.6,lw=0)
for i,s in enumerate(gr.index):
  for y in gr.columns:
    v=gr.loc[s,y]; c.add_patch(plt.Rectangle((y-.4,i-.35),.8,.7,color='#f0eee9' if pd.isna(v) else ('#b5524a' if v else '#d9d6cf'),lw=0))
c.set_xlim(2007.5,2025.5); c.set_ylim(len(gr)-.4,-1.1); c.set_yticks(range(len(gr))); c.set_yticklabels(gr.index); c.set_xticks(range(2008,2026)); c.tick_params(axis='x',labelsize=11,length=0); c.tick_params(axis='y',length=0)
c.annotate('2019: all below',xy=(2019,-.45),xytext=(2020.6,-.8),ha='left',va='center',fontsize=11,color='#444',arrowprops=dict(arrowstyle='->',color='#777'))
c.add_patch(plt.Rectangle((2018.55,-.45),.9,len(gr)-.1,fill=False,ec='#777',ls=':',lw=1.2))
[c.spines[x].set_visible(False) for x in c.spines]
c.set_title('C. Firing years (red = fires, grey = below, pale = not scored)',loc='left',fontsize=15,weight='bold')
for ax in (a,b): ax.set_facecolor(BG); [ax.spines[x].set_visible(False) for x in ('top','right')]
f.text(.5,.025,'Frozen rule: |Helm − mean(other 7)| ≥ 1.86, set 20 Oct 2026. Recomputed from raw files. Neural Nations / CAMS',ha='center',fontsize=11.5,color='#555')
f.savefig('charts/dashboard_light.png',facecolor=BG)
