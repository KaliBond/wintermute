"""Series tests for H2 (recovery leg), H3 and H4 ('for longer'). Pre-registered 2026-10-02
before the run-3 windows were scored; hash recorded in RUNLOG.md.

Usage: python hypothesis_check_series.py <node_metrics.csv (all windows)> <out.md>
"""
import sys
import pandas as pd

SERIES = {
  "Egypt":  ["B04","B13","B14","B05","B15"],   # -2500 -2400 -2250 | -2150 breakdown | -1900 recovery
  "Aegean": ["B16","B08","B09","B17","B18"],   # -1350 -1250 | -1175 breakdown | -1000 -750 recovery
}
BREAK = {"Egypt": "B05", "Aegean": "B09"}
MESO = ["B19","B06","B20"]                      # -2050 -1760 -1650, relief edicts attested
COMPARATOR = ["B04","B13","B14","B15","B16","B08"]  # non-collapse palace/state windows without attested debt relief

d = pd.read_csv(sys.argv[1], na_values=["NA"], keep_default_na=False)
d["case"] = d.Society.str[:3]
yr = d.groupby("case").Year.first()
E = d.groupby("case").Energy.mean(); G = d.groupby("case").Cognition.mean()
def C(case, node):
    v = d[(d.case == case) & (d.Node == node)].Coherence
    return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else float("nan")
HL = lambda c: (C(c,"Helm") + C(c,"Lore")) / 2
out = ["# Series hypothesis check\n", f"Input: `{sys.argv[1]}`\n"]

out.append("## Trajectories (mean Energy, mean Cognition, Helm-Lore coherence)\n")
out.append("| Series | Window | Year | Energy | Cognition | Helm-Lore C |\n| --- | --- | --- | --- | --- | --- |")
for s, cs in SERIES.items():
    for c in cs:
        out.append(f"| {s} | {c} | {yr[c]} | {E[c]:.2f} | {G[c]:.1f} | {HL(c):.1f} |")

out.append("\n## H2 Hysteresis: recovery slower than fall")
for s, cs in SERIES.items():
    b = BREAK[s]; i = cs.index(b); pre, post = cs[i-1], cs[i+1]
    fallE = (E[b]-E[pre])/(yr[b]-yr[pre])*100; recE = (E[post]-E[b])/(yr[post]-yr[b])*100
    fallG = (G[b]-G[pre])/(yr[b]-yr[pre])*100; recG = (G[post]-G[b])/(yr[post]-yr[b])*100
    regain = E[cs[-1]] - E[pre]
    verdict = "supports" if abs(recE) < abs(fallE) and abs(recG) < abs(fallG) else "against"
    out.append(f"- {s}: Energy fall {fallE:+.2f}/century vs recovery {recE:+.2f}/century; Cognition fall {fallG:+.1f} vs recovery {recG:+.1f}/century -> {verdict}")
    out.append(f"  - Last window vs last pre-collapse window: Energy {regain:+.2f}, Cognition {G[cs[-1]]-G[pre]:+.1f}")

out.append("\n## H3 Helm-Lore coherence falls before breakdown")
for s, cs in SERIES.items():
    i = cs.index(BREAK[s]); pres = cs[:i]
    drop = HL(pres[-1]) - HL(pres[0])
    eneg = [c for c in pres if E[c] < 0]
    verdict = "not testable (Helm or Lore NA)" if pd.isna(drop) else ("supports" if drop <= -0.5 else "against")
    out.append(f"- {s}: Helm-Lore C {HL(pres[0]):.1f} ({pres[0]}) -> {HL(pres[-1]):.1f} ({pres[-1]}), change {drop:+.1f}; pre-collapse windows with negative Energy: {eneg or 'none'} -> {verdict}")

out.append("\n## H4 Relief devices hold Hands' stress lower for longer")
hs = d[d.Node == "Hands"].set_index("case").Stress
m, comp = hs[MESO], hs[COMPARATOR].mean()
below = int((m < comp).sum())
out.append(f"- Mesopotamian Hands Stress by window: {m.round(1).to_dict()}; comparator mean {comp:.2f}")
out.append(f"- Windows below comparator: {below} of {len(MESO)} -> {'supports' if below == len(MESO) else 'against' if below <= 1 else 'mixed'}")
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("\n".join(out))
