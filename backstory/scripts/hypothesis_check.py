"""Preliminary test of H1-H5 (HYPOTHESES.md) on one run's node_metrics.csv.

Usage: python hypothesis_check.py <node_metrics.csv> <out.md>
Operationalisations are fixed here; change them only with a dated note in RUNLOG.md.
"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1], na_values=["NA"], keep_default_na=False)
d["case"] = d.Society.str[:3]
nv = lambda case, node: d[(d.case == case) & (d.Node == node)]["Node Value"].mean()
RIVER, MARITIME, PASTORAL = ["B01","B02","B03","B04","B06"], ["B07","B11"], ["B12"]
STATE_STABLE, NONSTATE = ["B01","B02","B03","B04","B06","B07","B08"], ["B10","B11","B12"]
out = []
def say(s=""): out.append(s)

say("# Preliminary hypothesis check\n")
say(f"Input: `{sys.argv[1]}`. Single run; treat as a pilot reading, not a result.\n")

# H1
say("## H1 Setting shapes topology")
sa = d[d.Node.isin(["Stewards","Archive"])]
r = sa[sa.case.isin(RIVER)]["Node Value"].mean(); o = sa[~sa.case.isin(RIVER + ["B05","B09"])]["Node Value"].mean()
say(f"- (a) Stewards+Archive mean Node Value: river-valley {r:.2f} vs other non-collapse {o:.2f} -> {'supports' if r > o else 'against'}")
def flow_minus_helm(c):
    f, h = nv(c,"Flow"), nv(c,"Helm")
    return f - h if pd.notna(f) and pd.notna(h) else None
mar = {c: (round(float(v),2) if v is not None else "NA") for c in MARITIME for v in [flow_minus_helm(c)]}
oth = [flow_minus_helm(c) for c in d.case.unique() if c not in MARITIME + ["B05","B09"]]
oth = [x for x in oth if x is not None]
say(f"- (b) Flow minus Helm Node Value, maritime: {mar}; other mean {sum(oth)/len(oth):.2f}. A missing value means Helm was NA (the hidden-Helm pattern itself).")
x = d[d.case == "B12"].sort_values("Node Value", ascending=False).Node.tolist()
say(f"- (c) Xiongnu node ranking by Node Value: {x}. Shield rank {x.index('Shield')+1}, Flow rank {x.index('Flow')+1} of 8.\n")

# H2
say("## H2 Hysteresis (fall phase only)")
for a, b in [("B04","B05"),("B08","B09")]:
    A, B = d[d.case == a].set_index("Node"), d[d.case == b].set_index("Node")
    dE = (B.Energy - A.Energy).dropna(); dC = (B.Cognition - A.Cognition).dropna()
    both = ((B.Energy < A.Energy) & (B.Cognition < A.Cognition)).sum()
    say(f"- {a} -> {b}: mean dEnergy {dE.mean():+.2f}, mean dCognition {dC.mean():+.2f}; nodes where both fell: {both}/8")
say("- Recovery leg not testable: no post-collapse recovery windows scored yet.\n")

# H3
say("## H3 Helm-Lore coherence falls before breakdown")
say("- Not testable in this run: needs at least two windows before each breakdown (e.g. Egypt late 5th and 6th dynasty; Mycenaean c.1350 and c.1250).\n")

# H4
say("## H4 Relief devices hold Hands' stress down")
h = d[d.Node == "Hands"].set_index("case").Stress
others = h[[c for c in ["B02","B04","B08"] if c in h.index]]
say(f"- Hands Stress, Old Babylonian (misharum edicts): {h.get('B06')}; other palace/state cases {others.to_dict()} mean {others.mean():.2f} -> {'supports' if h.get('B06') < others.mean() else 'against'}")
say("- Caveat: H4 predicts 'lower for longer', which needs time series; a single window can only show level.\n")

# H5
say("## H5 No ladder")
g = d.groupby("case")[["Energy","Cognition"]].mean()
s, n = g.loc[STATE_STABLE].mean(), g.loc[NONSTATE].mean()
say(f"- Non-collapse state cases: mean Energy {s.Energy:.2f}, mean Cognition {s.Cognition:.2f}")
say(f"- Non-state cases: mean Energy {n.Energy:.2f}, mean Cognition {n.Cognition:.2f}")
say(f"- Counts against H5 only if non-state cases cluster at the low end: {'not the case' if n.Energy >= s.Energy * 0.8 else 'they do'}\n")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("\n".join(out))
