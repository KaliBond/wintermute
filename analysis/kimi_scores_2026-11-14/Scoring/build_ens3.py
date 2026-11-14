# Scorer 3 of 3: build and validate CAMS raw score CSV
# Data drawn from scorer 3's own searches + framing calendar, 2026-11-14.

ENTITY = "Australian Government (Q30 comparison set)"
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# (segment, year): {node: (C, K, S, A)}
S = {}

# Rudd I (Dec 2007 - Jun 2010): 2008, 2009
S[("Rudd I", 2008)] = {
    "Helm": (7,7,3,6), "Shield": (5,5,4,5), "Lore": (7,6,3,7),
    "Stewards": (5,5,4,5), "Craft": (5,5,3,5), "Hands": (5,5,4,5),
    "Archive": (6,5,3,6), "Flow": (5,6,6,5)}
S[("Rudd I", 2009)] = {
    "Helm": (6,5,4,5), "Shield": (4,4,6,4), "Lore": (6,5,4,6),
    "Stewards": (5,5,5,5), "Craft": (5,5,5,5), "Hands": (6,5,5,5),
    "Archive": (6,5,4,6), "Flow": (7,7,5,6)}

# Gillard (Jun 2010 - Jun 2013): 2010-2013
S[("Gillard", 2010)] = {
    "Helm": (5,5,5,5), "Shield": (4,4,6,4), "Lore": (4,4,6,5),
    "Stewards": (5,5,5,5), "Craft": (5,5,5,5), "Hands": (5,5,4,5),
    "Archive": (6,5,4,6), "Flow": (6,6,4,6)}
S[("Gillard", 2011)] = {
    "Helm": (6,5,6,6), "Shield": (3,3,7,3), "Lore": (5,5,6,6),
    "Stewards": (5,5,5,6), "Craft": (6,5,5,6), "Hands": (4,5,5,5),
    "Archive": (6,5,4,6), "Flow": (6,6,5,6)}
S[("Gillard", 2012)] = {
    "Helm": (5,5,6,5), "Shield": (5,5,6,4), "Lore": (5,5,6,6),
    "Stewards": (6,5,5,6), "Craft": (6,6,5,6), "Hands": (5,5,5,5),
    "Archive": (5,5,5,5), "Flow": (6,6,4,6)}
S[("Gillard", 2013)] = {
    "Helm": (4,4,7,4), "Shield": (5,5,6,4), "Lore": (4,4,7,5),
    "Stewards": (6,5,5,6), "Craft": (6,6,5,6), "Hands": (4,4,6,4),
    "Archive": (5,5,5,5), "Flow": (5,6,5,6)}

# Rudd II (Jun-Sep 2013): 2013
S[("Rudd II", 2013)] = {
    "Helm": (4,4,7,4), "Shield": (6,5,6,4), "Lore": (4,4,7,4),
    "Stewards": (5,5,5,5), "Craft": (5,5,5,5), "Hands": (4,4,6,4),
    "Archive": (5,5,5,5), "Flow": (5,6,5,5)}

# Abbott (Sep 2013 - Sep 2015): 2013-2015
S[("Abbott", 2013)] = {
    "Helm": (6,5,4,4), "Shield": (6,6,5,4), "Lore": (5,4,5,4),
    "Stewards": (5,5,5,4), "Craft": (5,5,4,4), "Hands": (4,4,5,4),
    "Archive": (4,4,5,4), "Flow": (5,6,4,5)}
S[("Abbott", 2014)] = {
    "Helm": (4,4,6,4), "Shield": (7,6,5,4), "Lore": (4,4,6,4),
    "Stewards": (4,4,6,4), "Craft": (5,5,5,4), "Hands": (4,4,6,4),
    "Archive": (4,4,6,4), "Flow": (6,6,5,6)}
S[("Abbott", 2015)] = {
    "Helm": (3,3,7,3), "Shield": (7,6,5,4), "Lore": (3,3,6,3),
    "Stewards": (5,5,5,4), "Craft": (5,5,4,4), "Hands": (4,4,6,4),
    "Archive": (4,4,5,4), "Flow": (6,6,5,6)}

# Turnbull (Sep 2015 - Aug 2018): 2015-2018
S[("Turnbull", 2015)] = {
    "Helm": (6,5,4,5), "Shield": (6,5,5,4), "Lore": (6,5,4,5),
    "Stewards": (5,5,5,4), "Craft": (6,5,4,5), "Hands": (5,5,4,5),
    "Archive": (5,5,4,5), "Flow": (6,6,4,6)}
S[("Turnbull", 2016)] = {
    "Helm": (4,4,6,4), "Shield": (6,5,5,4), "Lore": (5,5,5,5),
    "Stewards": (3,3,7,3), "Craft": (4,4,6,4), "Hands": (5,5,5,5),
    "Archive": (4,4,6,4), "Flow": (6,6,4,6)}
S[("Turnbull", 2017)] = {
    "Helm": (4,4,6,4), "Shield": (6,6,5,4), "Lore": (5,5,6,5),
    "Stewards": (3,3,7,3), "Craft": (4,4,6,4), "Hands": (6,5,4,5),
    "Archive": (4,4,6,4), "Flow": (5,6,5,6)}
S[("Turnbull", 2018)] = {
    "Helm": (3,3,7,3), "Shield": (6,5,5,4), "Lore": (4,4,6,4),
    "Stewards": (3,3,7,3), "Craft": (4,4,6,4), "Hands": (6,5,4,5),
    "Archive": (4,4,6,4), "Flow": (5,6,5,6)}

# Morrison (Aug 2018 - May 2022): 2018-2022
S[("Morrison", 2018)] = {
    "Helm": (4,4,6,4), "Shield": (6,5,5,4), "Lore": (5,4,5,4),
    "Stewards": (3,3,7,3), "Craft": (5,4,5,4), "Hands": (6,5,4,5),
    "Archive": (4,4,6,4), "Flow": (5,6,5,5)}
S[("Morrison", 2019)] = {
    "Helm": (6,5,4,4), "Shield": (5,5,5,4), "Lore": (5,5,4,4),
    "Stewards": (5,5,7,4), "Craft": (5,5,4,4), "Hands": (6,5,5,5),
    "Archive": (3,4,7,3), "Flow": (6,6,5,6)}
S[("Morrison", 2020)] = {
    "Helm": (6,6,6,5), "Shield": (6,6,6,5), "Lore": (6,6,6,5),
    "Stewards": (7,7,6,6), "Craft": (7,7,6,6), "Hands": (5,6,7,5),
    "Archive": (3,4,7,3), "Flow": (6,6,7,6)}
S[("Morrison", 2021)] = {
    "Helm": (4,4,6,4), "Shield": (6,6,6,6), "Lore": (4,4,6,4),
    "Stewards": (6,6,5,5), "Craft": (5,5,6,5), "Hands": (6,6,5,5),
    "Archive": (3,4,8,3), "Flow": (5,6,7,6)}
S[("Morrison", 2022)] = {
    "Helm": (3,3,7,3), "Shield": (6,6,5,6), "Lore": (4,4,7,4),
    "Stewards": (5,5,5,4), "Craft": (4,4,6,4), "Hands": (6,6,6,5),
    "Archive": (3,3,8,3), "Flow": (5,5,7,5)}

# Albanese (May 2022 - ): 2022, 2023
S[("Albanese", 2022)] = {
    "Helm": (6,5,4,5), "Shield": (5,5,5,5), "Lore": (6,5,4,5),
    "Stewards": (6,5,4,5), "Craft": (6,5,5,5), "Hands": (6,5,5,6),
    "Archive": (6,5,5,6), "Flow": (5,5,7,5)}
S[("Albanese", 2023)] = {
    "Helm": (5,6,5,5), "Shield": (6,6,5,6), "Lore": (5,5,6,5),
    "Stewards": (5,5,5,5), "Craft": (5,5,5,5), "Hands": (6,5,5,6),
    "Archive": (6,6,5,6), "Flow": (6,6,6,6)}

EXPECTED = {
    "Rudd I": [2008, 2009],
    "Gillard": [2010, 2011, 2012, 2013],
    "Rudd II": [2013],
    "Abbott": [2013, 2014, 2015],
    "Turnbull": [2015, 2016, 2017, 2018],
    "Morrison": [2018, 2019, 2020, 2021, 2022],
    "Albanese": [2022, 2023],
}

def validate():
    errors = []
    # 1. expected segment-years present, exactly
    got = sorted(set(k for k in S))
    want = sorted(set((seg, y) for seg, ys in EXPECTED.items() for y in ys))
    if got != want:
        errors.append(f"segment-year mismatch: extra={set(got)-set(want)} missing={set(want)-set(got)}")
    # 2. each has all 8 nodes, integers 1-10
    for key, nodes in S.items():
        if sorted(nodes) != sorted(NODES):
            errors.append(f"{key}: node set mismatch")
        for n, vals in nodes.items():
            if len(vals) != 4 or any(not isinstance(v, int) or v < 1 or v > 10 for v in vals):
                errors.append(f"{key}/{n}: bad scores {vals}")
    # 3. within-segment adjacent-year step <= 2 per metric
    for seg, years in EXPECTED.items():
        for n in NODES:
            for a, b in zip(years, years[1:]):
                va, vb = S[(seg, a)][n], S[(seg, b)][n]
                for i, m in enumerate("CKSA"):
                    if abs(vb[i] - va[i]) > 2:
                        errors.append(f"step>2 {seg} {n} {m}: {a}={va[i]} {b}={vb[i]}")
    return errors

def write_csv(path):
    lines = ["Entity,Year,Segment,Node,Coherence,Capacity,Stress,Abstraction"]
    for (seg, y) in [(s, yy) for s in ["Rudd I","Gillard","Rudd II","Abbott","Turnbull","Morrison","Albanese"] for yy in EXPECTED[s]]:
        for n in NODES:
            c, k, ss, a = S[(seg, y)][n]
            lines.append(f"{ENTITY},{y},{seg},{n},{c},{k},{ss},{a}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 1

if __name__ == "__main__":
    errs = validate()
    for e in errs:
        print("ERROR:", e)
    if not errs:
        n = write_csv(r"C:\Users\julie\kimi_scores\Scoring\ens3_Q30_comparison_2008-2023_annual_raw.csv")
        print(f"OK: {n} data rows written; all constraints pass")
