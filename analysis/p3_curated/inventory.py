"""Inventory candidate ensemble runs with matching envelopes (quality metrics only, no outcomes)."""
import os, re, glob, subprocess, pandas as pd, numpy as np
R='/workspace/wm_pr'; os.chdir(R)
NODES={'Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'}
pairs=[]
for d in ['cleaned_datasets','data/nations','data/nations/5yr']:
    for f in glob.glob(f'{d}/*ENS*.csv'):
        e=os.path.join(d,os.path.basename(f).replace('ENS','ENV',1))
        if os.path.exists(e): pairs.append(('ENS/ENV',f,e))
pairs.append(('ENS/ENV','cleaned_datasets/Argentina_CAMS5_ensemble_1950_2026_cleaned.csv','cleaned_datasets/Argentina_ENV_1950_2026_cleaned.csv'))
for f in glob.glob('juno/*_m5n_*_block1.csv'): pairs.append(('m5n',f,f.replace('block1','block2')))
for f in glob.glob('analysis/camnations5_*_[0-9]*.csv'):
    e=re.sub(r'_\d{4}_\d{4}\.csv$','_envelope.csv',f)
    if os.path.exists(e): pairs.append(('camnations5',f,e))
for f in glob.glob('juno/*CAMNATIONS5_test*.csv'):
    pairs.append(('test',f,f.replace('test1','raw_envelope1')))
rows=[]
for fam,f,e in pairs:
    if not os.path.exists(e): continue
    a=pd.read_csv(f,encoding='utf-8-sig'); b=pd.read_csv(e,encoding='utf-8-sig')
    ren={'Entity':'Society','SD_Coherence':'C_sd','SD_Capacity':'K_sd','SD_Stress':'S_sd','SD_Abstraction':'A_sd'}
    a=a.rename(columns=ren); b=b.rename(columns=ren)
    if 'Year' not in a or 'Year' not in b:
        rows.append(dict(family=fam,ens=f,env=e,society=str(a.Society.iloc[0]),annual=False,note='not annual (Quarter)')); continue
    k=lambda x:set(zip(x.Society.str.strip(),x.Year.astype(str),x.Node.str.strip()))
    match=k(a)==k(b); yrs=sorted(a.Year.unique())
    annual=all(isinstance(y,(int,np.integer)) or float(y).is_integer() for y in yrs) and (len(yrs)<2 or np.median(np.diff(np.array(yrs,float)))==1)
    cnt=a.groupby('Year').Node.nunique(); dims=a[['Coherence','Capacity','Stress','Abstraction']]
    complete=set(a.Node.unique())==NODES and (cnt==8).all() and dims.notna().all().all() and not a.duplicated(['Society','Year','Node']).any()
    sd=b[['C_sd','K_sd','S_sd','A_sd']].apply(pd.to_numeric,errors='coerce')
    date=subprocess.run(['git','log','-1','--format=%cI','origin/main','--',f],capture_output=True,text=True).stdout.strip()
    rows.append(dict(family=fam,ens=f,env=e,society=a.Society.iloc[0],y0=int(min(yrs)),y1=int(max(yrs)),n_years=len(yrs),
        annual=annual,complete=bool(complete),match=match,mean_sd=round(float(sd.mean().mean()),4),
        zero_sd_share=round(float((sd.fillna(0).sum(axis=1)==0).mean()),3),sd_missing=round(float(sd.isna().mean().mean()),3),
        yrs_1950_2025=int(sum(1950<=y<=2025 for y in yrs)),commit_date=date,sha=__import__('hashlib').sha256(open(f,'rb').read()).hexdigest()[:16]))
I=pd.DataFrame(rows).sort_values(['society','family','ens']); I.to_csv('/workspace/analysis/p3_curated/run_inventory.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(I.drop(columns=['env']).to_string(index=False))
