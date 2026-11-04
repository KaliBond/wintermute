"""ADVERSARY-02 aggregation: 5 scorer passes -> ensemble_mean.csv + envelope.csv.

Node Value (per user rule 6):  V = C + K - S
Bond Strength (from scorer_prompts.txt aggregation prompt), computed per entity-year
on ensemble means (unrounded):
    B_ij = [0.6*C_i*C_j + 0.4*A_i*A_j] * exp(-(S_i + S_j)/20)
    Bond Strength_i = mean of B_ij over partner nodes j != i with valid means
NA handling: a cell's means use non-NA scorers only (n_eff); Node Value and bond
participation require n_eff >= 3.
Supplementary: ensemble_mean_V_protocol.csv uses V = C + K - S + 0.5*A.
"""
import csv, math, statistics, sys
from pathlib import Path

base = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

ENTITIES = ["China", "Evergrande", "Vanke", "Longfor", "Huawei", "SMIC", "BYD", "CATL"]
NAT = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
CORP = ["Board", "Risk", "RD", "Finance", "Operations", "Workforce", "Brand", "Markets"]
DIMS = ["C", "K", "S", "A"]

def years(e):
    return range(2011 if e == "CATL" else 2010, 2021)

def nodes(e):
    return NAT if e == "China" else CORP

passes = []
problems = []
for n in range(1, 6):
    rows = {}
    with open(base / f"pass{n}" / f"pass{n}.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["Entity"].strip(), int(r["Year"]), r["Node"].strip())
            vals = []
            for d in DIMS:
                v = r[d].strip()
                vals.append(None if v.upper() == "NA" else int(v))
            if any(v is None for v in vals) and not all(v is None for v in vals):
                problems.append(f"pass{n} {key}: partial NA")
            for v in vals:
                if v is not None and not 1 <= v <= 10:
                    problems.append(f"pass{n} {key}: out of range {v}")
            if key in rows:
                problems.append(f"pass{n} {key}: duplicate")
            rows[key] = vals
    expected = {(e, y, nd) for e in ENTITIES for y in years(e) for nd in nodes(e)}
    for k in expected - rows.keys():
        problems.append(f"pass{n} missing {k}")
    for k in rows.keys() - expected:
        problems.append(f"pass{n} unexpected {k}")
    passes.append(rows)

def r1(x): return None if x is None else f"{x:.1f}"
def r3(x): return None if x is None else f"{x:.3f}"
def fmt(x): return "NA" if x is None else x

mean_rows, env_rows, protov_rows = [], [], []
for e in ENTITIES:
    for y in years(e):
        cells = {}
        for nd in nodes(e):
            scored = [p[(e, y, nd)] for p in passes
                      if (e, y, nd) in p and p[(e, y, nd)][0] is not None]
            n_eff = len(scored)
            means = [statistics.mean(s[i] for s in scored) if n_eff else None for i in range(4)]
            sds = [statistics.stdev(s[i] for s in scored) if n_eff >= 2 else None for i in range(4)]
            vs = [s[0] + s[1] - s[2] for s in scored]
            valid = n_eff >= 3
            V = means[0] + means[1] - means[2] if valid else None
            Vp = V + 0.5 * means[3] if valid else None
            cells[nd] = dict(n=n_eff, m=means, sd=sds, vs=vs, V=V, Vp=Vp, valid=valid)
        for nd in nodes(e):
            c = cells[nd]
            bond, bond_n = None, 0
            if c["valid"]:
                Ci, Ki, Si, Ai = c["m"]
                bs = []
                for o in nodes(e):
                    if o == nd or not cells[o]["valid"]:
                        continue
                    Cj, Kj, Sj, Aj = cells[o]["m"]
                    bs.append((0.6 * Ci * Cj + 0.4 * Ai * Aj) * math.exp(-(Si + Sj) / 20))
                bond_n = len(bs)
                bond = statistics.mean(bs) if bs else None
            m = c["m"] if c["valid"] else [None] * 4
            mean_rows.append([e, y, nd] + [fmt(r1(x)) for x in m] + [fmt(r1(c["V"])), fmt(r3(bond))])
            protov_rows.append([e, y, nd, fmt(r1(c["Vp"]))])
            vmin = min(c["vs"]) if c["vs"] else None
            vmax = max(c["vs"]) if c["vs"] else None
            vr = vmax - vmin if c["vs"] else None
            env_rows.append([e, y, nd] + [fmt(r3(x)) for x in c["sd"]] +
                            [fmt(vr), fmt(vmin), fmt(vmax), c["n"],
                             round(1 - c["n"] / 5, 2), bond_n])

with open(out / "ensemble_mean.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction", "Node Value", "Bond Strength"])
    w.writerows(mean_rows)
with open(out / "envelope.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "C_sd", "K_sd", "S_sd", "A_sd", "V_range", "V_min", "V_max", "n_eff", "NA_rate", "bond_n"])
    w.writerows(env_rows)
with open(out / "ensemble_mean_V_protocol.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "Node Value (C+K-S+0.5A)"])
    w.writerows(protov_rows)

vr = {r[7] for r in env_rows}
print("rows:", len(mean_rows), "| distinct V_range values:", len(vr))
print("problems:", len(problems))
for p in problems[:50]:
    print("  ", p)
