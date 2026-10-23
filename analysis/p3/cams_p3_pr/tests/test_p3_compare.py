import tempfile
from pathlib import Path
import pandas as pd
from analysis.p3_compare import evaluate, load_panel, NODES


def fixture(tmp):
    rows=[]; ev=[]
    for j,s in enumerate(('Alpha','Beta','Gamma','Delta')):
        for year in range(2000,2024):
            ev.append({'Society':s,'Year':year,'Event':int((year+j)%7==0)})
            for i,n in enumerate(NODES):
                rows.append({'Society':s,'Year':year,'Node':n,'Coherence':5+(i%3)/2,
                             'Capacity':5+(j%2),'Stress':2+(year%6)/3,
                             'Abstraction':4+(i%2)})
    p=tmp/'panel.csv'; e=tmp/'events.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    pd.DataFrame(ev).to_csv(e,index=False)
    return p,e


def test_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        p,e=fixture(Path(t))
        out,preds=evaluate(p,e,3)
        assert len(out['folds'])==4
        assert out['complete_society_years']==4*21 # 2000..2020; future 3 years all known
        assert len(preds)==4*21*3
        assert all(0<=x['probability']<=1 for x in preds)


def test_reject_duplicate():
    with tempfile.TemporaryDirectory() as t:
        p,e=fixture(Path(t))
        df=pd.read_csv(p)
        pd.concat([df,df.head(1)]).to_csv(p,index=False)
        try:load_panel(p)
        except ValueError as err:assert 'Duplicate' in str(err)
        else:raise AssertionError('duplicates accepted')
