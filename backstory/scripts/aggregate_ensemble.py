"""Aggregate N independent scorer CSVs per the CAMNATIONSM5N aggregation spec.

Per (Society, Node, Dimension): mean of non-NA values if n_eff_d >= 3 else NA; sample SD (ddof=1).
n_eff = min n_eff_d over the four dimensions. NA_rate = NA count / (n_scorers x 4).
V_i = C + K - S + 0.5 A per complete scorer; V_range/min/max need >= 3 complete scorers.
Node Value and Bond Strength from ensemble means (bond_n >= 4 partners required).
No overlap is used for single-anchor deep-time cases, so there is no SEAM line.

Usage: python aggregate_ensemble.py <out_dir> scorer1.csv scorer2.csv ...
"""
import sys, os, math
import pandas as pd

DIMS = ["Coherence","Capacity","Stress","Abstraction"]
ORDER = ["Helm","Shield","Lore","Stewards","Craft","Hands","Archive","Flow"]
out, files = sys.argv[1], sys.argv[2:]
frames = []
for i, f in enumerate(files):
    d = pd.read_csv(f, na_values=["NA"], keep_default_na=False)
    d["scorer"] = i
    frames.append(d)
allr = pd.concat(frames)
keys = ["Society","Year","Node"]
rows1, rows2 = [], []
for k, g in allr.groupby(keys, sort=False):
    rec1, rec2 = dict(zip(keys, k)), dict(zip(keys, k))
    neff = []
    for dmn in DIMS:
        v = g[dmn].dropna(); neff.append(len(v))
        rec1[dmn] = round(v.mean(), 1) if len(v) >= 3 else float("nan")
        rec2[{"Coherence":"C","Capacity":"K","Stress":"S","Abstraction":"A"}[dmn] + "_sd"] = round(v.std(ddof=1), 2) if len(v) >= 3 else float("nan")
    comp = g.dropna(subset=DIMS)
    V = comp.Coherence + comp.Capacity - comp.Stress + 0.5 * comp.Abstraction
    if len(V) >= 3:
        rec2.update(V_range=round(V.max() - V.min(), 1), V_min=round(V.min(), 1), V_max=round(V.max(), 1))
    else:
        rec2.update(V_range=float("nan"), V_min=float("nan"), V_max=float("nan"))
    rec2["n_eff"] = min(neff)
    rec2["NA_rate"] = round(g[DIMS].isna().sum().sum() / (len(files) * 4), 2)
    rows1.append(rec1); rows2.append(rec2)
b1, b2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
b1["Node Value"] = (b1.Coherence + b1.Capacity - b1.Stress + 0.5 * b1.Abstraction).round(1)
bs, bn = [], []
for _, r in b1.iterrows():
    if r[["Coherence","Abstraction","Stress"]].isna().any():
        bs.append(float("nan")); bn.append(0); continue
    grp = b1[(b1.Society == r.Society) & (b1.Year == r.Year) & (b1.Node != r.Node)].dropna(subset=["Coherence","Abstraction","Stress"])
    vals = [(0.6*r.Coherence*p.Coherence + 0.4*r.Abstraction*p.Abstraction) * math.exp(-(r.Stress + p.Stress)/20) for _, p in grp.iterrows()]
    bn.append(len(vals)); bs.append(round(sum(vals)/len(vals), 3) if len(vals) >= 4 else float("nan"))
b1["Bond Strength"] = bs; b2["bond_n"] = bn
os.makedirs(out, exist_ok=True)
b1.to_csv(os.path.join(out, "block1_ensemble_mean.csv"), index=False, na_rep="NA")
b2.to_csv(os.path.join(out, "block2_envelope.csv"), index=False, na_rep="NA")
# verification checks from the spec
assert len(b1) == len(b2)
assert b2.V_range.dropna().nunique() > 1, "V_range identical everywhere: computation error"
assert ((b2.n_eff < len(files)) == (b2.NA_rate > 0)).all(), "NA accounting drifted"
assert b1.loc[b1[DIMS].isna().any(axis=1), "Node Value"].isna().all()
print("verification passed:", len(b1), "rows")
