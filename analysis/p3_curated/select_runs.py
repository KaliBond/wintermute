"""Apply PREREG_P3c eligibility + selection (no outcomes)."""
import pandas as pd, numpy as np, os
R='/workspace/wm_pr'
I=pd.read_csv('run_inventory_prefreeze.csv')
TEMPLATE={'juno/Australia_CAMS_m5n_2000-2026_block2.csv','juno/China_CAMS_m5n_2000-2026_block2.csv','juno/Russia_CAMS_m5n_2000-2026_block2.csv',
 'juno/UK_CAMS_m5n_2000-2026_block2.csv','juno/USA_CAMS_m5n_2000-2026_block2.csv','juno/Belgium_CAMS_m5n_1900-2026_block2.csv',
 'juno/Morocco_CAMS_m5n_1900-2026_block2.csv','juno/Portugal_CAMS_m5n_1900-2026_block2.csv','juno/Sri_Lanka_CAMS_m5n_1900-2026_block2.csv',
 'juno/Japan_CAMS_m5n_1850-2026_block2.csv','cleaned_datasets/Japan_ENV_1850_2026_cleaned.csv'}
status=open('ENVELOPE_STATUS_446928c.md').read()
flag=[l.split('`')[1] for l in status.splitlines() if 'not an ensemble' in l and l.startswith('| `')]
I['n_1950_2022']=0
for i,r in I.iterrows():
    if r.annual==True:
        a=pd.read_csv(f'{R}/{r.ens}',encoding='utf-8-sig').rename(columns={'Entity':'Society'})
        I.loc[i,'n_1950_2022']=int(((a.Year>=1950)&(a.Year<=2022)).groupby(a.Year).first().sum())
reasons=[]
for r in I.itertuples():
    why=[]
    if r.annual!=True: why.append('not annual')
    if r.complete!=True: why.append('incomplete')
    if r.match!=True: why.append('envelope rows mismatch')
    if not (r.zero_sd_share<=0.10 and r.sd_missing<=0.05): why.append(f'degenerate envelope (zero-SD {r.zero_sd_share})')
    if r.env in TEMPLATE: why.append('template envelope')
    if 'test' in r.ens.lower() or r.ens in flag: why.append('test / not an ensemble')
    if r.n_1950_2022<20: why.append('<20 yrs in 1950-2022')
    reasons.append('; '.join(why))
I['exclusion']=reasons
I.to_csv('run_eligibility.csv',index=False)
E=I[I.exclusion==''].copy()
NORM={'United Kingdom':'UK','United States':'USA','Türkiye':'Turkey'}
E['society_n']=E.society.replace(NORM)
E=E.sort_values(['society_n','mean_sd','commit_date'],ascending=[True,True,False]).drop_duplicates('sha')
SEL=E.groupby('society_n').head(1)
SEL.to_csv('selected_runs.csv',index=False)
print(SEL[['society_n','ens','y0','y1','n_1950_2022','mean_sd','zero_sd_share','commit_date']].to_string(index=False))
print('\nExcluded summary:'); print(I[I.exclusion!=''][['society','ens','exclusion']].to_string(index=False))
rows=[]
for r in SEL.itertuples():
    a=pd.read_csv(f'{R}/{r.ens}',encoding='utf-8-sig').rename(columns={'Entity':'Society'})
    a['Society']=r.society_n; rows.append(a[['Society','Year','Node','Coherence','Capacity','Stress','Abstraction']])
C=pd.concat(rows); C['Year']=C.Year.astype(int); C.to_csv('CURATED_CAMS.csv',index=False)
print(len(C), C.Society.nunique())
