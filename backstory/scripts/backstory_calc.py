"""Backstory calc: Cognition, Energy, Node Value and Bond Strength from raw CAMS scores.

Formulas (CAMNATIONSM5N aggregation spec, JUNO v1.2 node/bond equations):
  Cognition  = A x C
  Energy     = K - S
  Node Value = C + K - S + 0.5 A
  B_ij       = [0.6 C_i C_j + 0.4 A_i A_j] x exp(-(S_i + S_j)/20)
  Bond Strength = mean over computable partners, NA if fewer than 4 partners.
NA is propagated, never coerced to zero.

Usage: python backstory_calc.py <raw_scores.csv> <out_dir>
"""
import sys, os, math
import pandas as pd

ORDER = ["Helm","Shield","Lore","Stewards","Craft","Hands","Archive","Flow"]

def calc(raw):
    d = raw.copy()
    for c in ["Coherence","Capacity","Stress","Abstraction"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Cognition"] = d.Abstraction * d.Coherence
    d["Energy"] = d.Capacity - d.Stress
    d["Node Value"] = d.Coherence + d.Capacity - d.Stress + 0.5 * d.Abstraction
    bonds, ns = [], []
    for _, r in d.iterrows():
        grp = d[(d.Society == r.Society) & (d.Year == r.Year) & (d.Node != r.Node)]
        if pd.isna(r[["Coherence","Abstraction","Stress"]]).any():
            bonds.append(float("nan")); ns.append(0); continue
        vals = []
        for _, p in grp.iterrows():
            if pd.isna(p[["Coherence","Abstraction","Stress"]]).any():
                continue
            vals.append((0.6*r.Coherence*p.Coherence + 0.4*r.Abstraction*p.Abstraction)
                        * math.exp(-(r.Stress + p.Stress)/20))
        ns.append(len(vals))
        bonds.append(round(sum(vals)/len(vals), 3) if len(vals) >= 4 else float("nan"))
    d["Bond Strength"] = bonds
    d["bond_n"] = ns
    d["Node Value"] = d["Node Value"].round(1)
    d["NodeOrd"] = d.Node.map({n:i for i,n in enumerate(ORDER)})
    return d.sort_values(["Society","NodeOrd"]).drop(columns="NodeOrd")

def system(d):
    g = d.groupby(["Society","Year"])
    s = pd.DataFrame({
        "nodes_scored": g.Coherence.count(),
        "NA_nodes": g.Coherence.apply(lambda x: x.isna().sum()),
        "mean_Cognition": g.Cognition.mean().round(2),
        "mean_Energy": g.Energy.mean().round(2),
        "mean_NodeValue": g["Node Value"].mean().round(2),
        "mean_Bond": g["Bond Strength"].mean().round(3),
        "min_Energy_node": g[["Node","Energy"]].apply(lambda x: x.loc[x.Energy.idxmin(), "Node"] if x.Energy.notna().any() else "NA"),
    }).reset_index()
    return s

if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1], na_values=["NA"], keep_default_na=False)
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    d = calc(raw)
    d.to_csv(os.path.join(out, "node_metrics.csv"), index=False, na_rep="NA")
    system(d).to_csv(os.path.join(out, "system_metrics.csv"), index=False, na_rep="NA")
    print(system(d).to_string(index=False))
