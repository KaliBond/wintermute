"""Add Bond_Strength_Bounded to a CAMS block1 CSV, leaving all existing columns untouched.

Bounded bond (cannot exceed 1):
    q_i  = (0.6*C_i + 0.4*A_i) / 10
    B_ij = sqrt(q_i*q_j) * 2**(-(S_i+S_j)/10)
Bond_Strength_Bounded = mean of B_ij over all node pairs (28 for 8 nodes) in a year,
written on every node row of that year. Leading '#' metadata lines are preserved.

Usage: python scripts/bounded_bond.py data/nations/USA_v12opt_hifi_block1.csv
"""
import csv, io, itertools, math, sys

def main(path):
    lines = open(path, newline="").read().splitlines(keepends=True)
    meta = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    rows = list(csv.reader(body))
    hdr, data = rows[0], [r for r in rows[1:] if r]
    if "Bond_Strength_Bounded" in hdr:
        col = hdr.index("Bond_Strength_Bounded"); data = [r[:col] + r[col+1:] for r in data]; hdr = hdr[:col] + hdr[col+1:]
    ix = {k: hdr.index(k) for k in ("Year", "Coherence", "Stress", "Abstraction")}
    years = {}
    for r in data:
        years.setdefault(r[ix["Year"]], []).append(r)
    bond = {}
    for y, rs in years.items():
        nodes = [((0.6*float(r[ix["Coherence"]]) + 0.4*float(r[ix["Abstraction"]]))/10, float(r[ix["Stress"]])) for r in rs]
        b = [math.sqrt(qi*qj) * 2**(-(si+sj)/10) for (qi, si), (qj, sj) in itertools.combinations(nodes, 2)]
        bond[y] = sum(b)/len(b)
    out = io.StringIO(); w = csv.writer(out, lineterminator="\n")
    w.writerow(hdr + ["Bond_Strength_Bounded"])
    for r in data:
        w.writerow(r + [f"{bond[r[ix['Year']]]:.4f}"])
    open(path, "w", newline="").write("".join(meta) + out.getvalue())
    print(f"{path}: {len(years)} years, bounded bond max {max(bond.values()):.4f}, min {min(bond.values()):.4f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
