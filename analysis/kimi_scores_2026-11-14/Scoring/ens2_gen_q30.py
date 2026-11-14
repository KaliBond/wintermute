# Scorer 2 of 3 - CAMS Universal Instruction Set v1.4, annual resolution
# Australian Government (Q30 comparison set), 2008-2023
# Generates and validates ens2_Q30_comparison_2008-2023_annual_raw.csv
# Scores: integers 1-10 per node per segment-year (C,K,S,A)
# Constraint: within-segment adjacent-year steps <= 2 per metric.

ENTITY = "Australian Government (Q30 comparison set)"
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# (segment, [years]) with per-node scores as dict node -> [(C,K,S,A) per year]
DATA = {
    "Rudd I": (["2008", "2009"], {
        "Helm":    [(7,8,6,7), (7,7,6,7)],
        "Shield":  [(6,6,6,6), (6,6,7,6)],
        "Lore":    [(6,7,5,6), (6,7,6,6)],
        "Stewards":[(7,7,6,6), (7,7,7,6)],
        "Craft":   [(6,6,6,6), (5,5,7,6)],
        "Hands":   [(7,7,7,6), (7,6,7,6)],
        "Archive": [(6,6,5,6), (6,6,6,6)],
        "Flow":    [(8,8,7,7), (8,8,8,8)],
    }),
    "Gillard": (["2010", "2011", "2012", "2013"], {
        "Helm":    [(5,6,7,6), (5,7,7,6), (5,6,7,6), (4,5,8,6)],
        "Shield":  [(5,5,7,5), (4,5,8,5), (4,5,9,5), (4,6,9,5)],
        "Lore":    [(5,6,6,6), (5,6,7,6), (4,6,7,6), (4,5,8,6)],
        "Stewards":[(6,6,6,7), (6,7,6,7), (6,7,6,7), (6,7,7,7)],
        "Craft":   [(6,6,6,6), (6,6,6,6), (5,5,7,6), (5,5,7,6)],
        "Hands":   [(6,7,6,6), (6,6,6,6), (6,6,6,6), (6,6,7,6)],
        "Archive": [(6,6,5,6), (6,6,6,6), (5,6,7,6), (5,6,7,6)],
        "Flow":    [(6,7,6,6), (6,7,6,6), (5,6,6,6), (5,6,7,6)],
    }),
    "Rudd II": (["2013"], {
        "Helm":    [(4,5,8,5)],
        "Shield":  [(5,5,8,5)],
        "Lore":    [(4,5,7,5)],
        "Stewards":[(6,6,7,7)],
        "Craft":   [(5,5,7,5)],
        "Hands":   [(6,6,7,6)],
        "Archive": [(5,6,7,6)],
        "Flow":    [(5,6,7,6)],
    }),
    "Abbott": (["2013", "2014", "2015"], {
        "Helm":    [(6,6,6,5), (4,5,8,4), (3,4,8,4)],
        "Shield":  [(7,7,6,6), (7,7,7,6), (7,7,7,6)],
        "Lore":    [(5,6,6,5), (4,5,7,4), (4,5,6,4)],
        "Stewards":[(5,6,6,5), (4,5,8,5), (5,5,7,5)],
        "Craft":   [(6,6,6,5), (6,6,6,5), (6,6,6,5)],
        "Hands":   [(6,6,7,5), (5,6,7,5), (6,6,6,5)],
        "Archive": [(6,6,5,6), (6,6,6,6), (5,6,7,6)],
        "Flow":    [(5,6,6,5), (4,5,8,4), (4,5,7,4)],
    }),
    "Turnbull": (["2015", "2016", "2017", "2018"], {
        "Helm":    [(6,6,6,6), (5,6,7,6), (4,5,7,6), (3,4,8,5)],
        "Shield":  [(7,7,6,6), (7,7,7,6), (6,7,7,6), (6,6,7,6)],
        "Lore":    [(5,6,6,6), (4,5,7,6), (4,5,7,6), (4,5,7,5)],
        "Stewards":[(6,6,6,6), (5,5,7,5), (4,5,8,5), (4,5,7,5)],
        "Craft":   [(6,6,6,6), (5,5,7,6), (5,5,7,6), (5,5,7,6)],
        "Hands":   [(6,6,7,6), (6,6,7,6), (5,6,7,6), (5,6,7,6)],
        "Archive": [(6,6,5,6), (6,6,6,6), (6,6,7,6), (5,6,7,6)],
        "Flow":    [(6,6,6,6), (6,6,6,6), (6,6,7,6), (5,5,8,5)],
    }),
    "Morrison": (["2018", "2019", "2020", "2021", "2022"], {
        "Helm":    [(4,4,8,4), (5,5,7,5), (6,7,9,6), (5,6,8,6), (4,4,8,4)],
        "Shield":  [(6,6,7,5), (6,6,6,5), (7,7,8,7), (7,7,7,8), (7,6,7,7)],
        "Lore":    [(4,5,7,5), (4,5,6,5), (5,5,8,6), (4,5,7,6), (4,5,7,5)],
        "Stewards":[(4,5,7,5), (4,5,7,5), (5,7,9,7), (5,7,8,6), (4,5,7,5)],
        "Craft":   [(5,5,7,5), (5,5,7,5), (5,7,9,7), (6,6,8,6), (5,5,8,6)],
        "Hands":   [(5,5,7,5), (5,5,7,5), (5,6,9,6), (5,6,7,6), (5,5,7,6)],
        "Archive": [(5,6,6,5), (5,6,6,5), (5,6,7,6), (5,6,6,6), (4,5,7,5)],
        "Flow":    [(5,5,7,5), (5,5,7,5), (6,7,9,6), (6,6,7,6), (5,5,8,5)],
    }),
    "Albanese": (["2022", "2023"], {
        "Helm":    [(6,6,7,6), (6,6,7,6)],
        "Shield":  [(6,6,7,7), (6,7,7,7)],
        "Lore":    [(5,6,6,6), (4,5,7,6)],
        "Stewards":[(5,6,7,6), (5,6,8,6)],
        "Craft":   [(5,6,8,6), (6,6,7,6)],
        "Hands":   [(6,6,7,6), (6,6,7,6)],
        "Archive": [(6,7,6,6), (7,7,6,7)],
        "Flow":    [(5,6,8,5), (6,6,8,6)],
    }),
}

OUT = r"C:\Users\julie\kimi_scores\Scoring\ens2_Q30_comparison_2008-2023_annual_raw.csv"

def main():
    rows = []
    for seg, (years, nodes) in DATA.items():
        assert set(nodes) == set(NODES), f"node set mismatch in {seg}"
        for node in NODES:
            vals = nodes[node]
            assert len(vals) == len(years), f"year count mismatch {seg}/{node}"
            for y, (c, k, s, a) in zip(years, vals):
                rows.append((ENTITY, y, seg, node, c, k, s, a))

    # ---- validation ----
    assert len(rows) == 168, f"row count {len(rows)} != 168"
    keys = [(r[2], r[1], r[3]) for r in rows]
    assert len(set(keys)) == 168, "duplicate (segment, year, node)"

    expected = []
    for seg, (years, _) in DATA.items():
        for y in years:
            for n in NODES:
                expected.append((seg, y, n))
    assert set(expected) == set(keys), "segment-year list mismatch"

    for r in rows:
        for v in r[4:]:
            assert isinstance(v, int) and 1 <= v <= 10, f"score out of range: {r}"

    for seg, (years, nodes) in DATA.items():
        for node in NODES:
            series = nodes[node]
            for i in range(1, len(series)):
                for m in range(4):
                    step = abs(series[i][m] - series[i-1][m])
                    assert step <= 2, f"step violation {seg}/{node} {years[i-1]}->{years[i]} metric{m}: {step}"

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("Entity,Year,Segment,Node,Coherence,Capacity,Stress,Abstraction\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"OK: {len(rows)} rows written to {OUT}")
    # Helm summary for the report
    for seg, (years, nodes) in DATA.items():
        for y, v in zip(years, nodes["Helm"]):
            print(seg, y, "Helm C,K =", v[0], v[1])

if __name__ == "__main__":
    main()
