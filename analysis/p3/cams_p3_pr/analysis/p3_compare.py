"""P3: evaluate whether eight-node CAMS configuration adds held-out predictive value.

Input: long node-year CSV with Society, Year, Node, Coherence, Capacity,
Stress, Abstraction; independent events CSV with Society, Year, Event.
Event must be coded externally to CAMS. A positive event in (t, t+horizon]
is the binary target for society-year t. Observations at/after event are
not removed; recurrence is allowed. Do not treat this as prospective validation
if the historical CAMS scores were informed by the events.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NODES = ('Helm','Shield','Lore','Stewards','Archive','Craft','Hands','Flow')
DIMS = ('Coherence','Capacity','Stress','Abstraction')


def load_panel(path):
    df = pd.read_csv(path)
    needed = {'Society','Year','Node',*DIMS}
    if not needed.issubset(df):
        raise ValueError('Missing panel columns: '+str(sorted(needed-set(df))))
    df = df[list(needed)].copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    if df.duplicated(['Society','Year','Node']).any():
        raise ValueError('Duplicate society/year/node cells: provide one frozen ensemble score per cell')
    if not set(df.Node.unique()).issubset(set(NODES)):
        raise ValueError('Unrecognised nodes: '+str(sorted(set(df.Node)-set(NODES))))
    for d in DIMS:
        df[d] = pd.to_numeric(df[d],errors='raise')
    if not np.isfinite(df[list(DIMS)].to_numpy(dtype=float)).all():
        raise ValueError('NaN or infinite node scores')
    counts = df.groupby(['Society','Year']).Node.nunique()
    good = counts[counts==8].index
    df = df.set_index(['Society','Year']).loc[good].reset_index()
    if df.empty:
        raise ValueError('No complete eight-node society-years')
    wide = df.pivot(index=['Society','Year'],columns='Node',values=list(DIMS))
    wide.columns = [f'{dim}_{node}' for dim,node in wide.columns]
    wide=wide.reset_index()
    for dim in DIMS:
        wide[f'mean_{dim}'] = df.groupby(['Society','Year'])[dim].mean().reindex(
            pd.MultiIndex.from_frame(wide[['Society','Year']])).to_numpy()
    return wide


def make_target(wide, events_path, horizon):
    ev = pd.read_csv(events_path)
    if not {'Society','Year','Event'}.issubset(ev):
        raise ValueError('Events require Society, Year, Event columns')
    ev['Year']=pd.to_numeric(ev.Year,errors='raise').astype(int)
    ev['Event']=pd.to_numeric(ev.Event,errors='raise')
    if not ev.Event.isin([0,1]).all() or ev.duplicated(['Society','Year']).any():
        raise ValueError('Events must be unique society-years with Event=0/1')
    # Treat events as a complete annual register: absence of a row is NOT a known zero.
    evs = {(str(r.Society),int(r.Year)):int(r.Event) for r in ev.itertuples()}
    targets=[]; eligible=[]
    for r in wide.itertuples():
        sequence=[evs.get((str(r.Society),int(r.Year)+k)) for k in range(1,horizon+1)]
        eligible.append(all(x is not None for x in sequence))
        targets.append(int(any(x==1 for x in sequence)) if all(x is not None for x in sequence) else np.nan)
    wide=wide.copy()
    wide['target']=targets
    return wide.loc[eligible].reset_index(drop=True)


def fit_predict(train,test,cols):
    y=train.target.astype(int).to_numpy()
    if len(np.unique(y))<2:
        return np.full(len(test),float(y.mean()))
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                        LogisticRegression(C=0.1,max_iter=2000,class_weight=None))
    model.fit(train[cols],y)
    return model.predict_proba(test[cols])[:,1]


def evaluate(panel,events,horizon=3):
    wide=make_target(load_panel(panel),events,horizon)
    cols_basic=[f'mean_{d}' for d in DIMS]
    cols_full=[f'{d}_{n}' for d in DIMS for n in NODES]
    results=[]; predictions=[]
    for society in sorted(wide.Society.unique()):
        test=wide.loc[wide.Society==society]
        train=wide.loc[wide.Society!=society]
        if len(train)<15 or len(test)<2: continue
        y=test.target.astype(int).to_numpy()
        pred={'baseline':fit_predict(train,test,cols_basic),
              'configuration':fit_predict(train,test,cols_full),
              'base_rate':np.full(len(test),float(train.target.mean()))}
        record={'society':society,'n':len(test),'events_in_window':int(sum(y))}
        for name,p in pred.items():
            record[f'{name}_brier']=float(brier_score_loss(y,p))
            record[f'{name}_auc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
            for i,(yr,actual,prob) in enumerate(zip(test.Year,y,p)):
                predictions.append({'society':society,'year':int(yr),'actual':int(actual),
                                    'model':name,'probability':float(prob)})
        results.append(record)
    if not results: raise ValueError('No eligible LOSO folds')
    out={'protocol':'P3 v1; leave-one-society-out; independent complete annual outcome register',
         'horizon_years':horizon,'complete_society_years':len(wide),
         'folds':results,'unweighted_mean_brier':{},'unweighted_mean_auc_evaluable':{}}
    for model in ('base_rate','baseline','configuration'):
        out['unweighted_mean_brier'][model]=float(np.mean([r[f'{model}_brier'] for r in results]))
        auc=[r[f'{model}_auc'] for r in results if r[f'{model}_auc'] is not None]
        out['unweighted_mean_auc_evaluable'][model]=float(np.mean(auc)) if auc else None
    out['delta_brier_configuration_minus_baseline']=out['unweighted_mean_brier']['configuration']-out['unweighted_mean_brier']['baseline']
    out['interpretation']='Negative delta Brier favours eight-node model; no significance claim without precommitted uncertainty analysis.'
    return out,predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',required=True)
    p.add_argument('--events',required=True)
    p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--out',default='p3_result.json')
    args=p.parse_args()
    if args.horizon<1: p.error('horizon must be positive')
    out,preds=evaluate(args.panel,args.events,args.horizon)
    Path(args.out).write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame(preds).to_csv(str(Path(args.out).with_suffix(''))+'_predictions.csv',index=False)
    print(json.dumps({'folds':len(out['folds']),'delta_brier':out['delta_brier_configuration_minus_baseline'],'report':args.out},indent=2))

if __name__=='__main__': main()
