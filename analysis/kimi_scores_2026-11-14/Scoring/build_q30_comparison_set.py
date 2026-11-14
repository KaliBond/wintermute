# Generator for the Q30 Australian Government comparison set (parent scorer, scorer 1 of 3).
# Governing document: ImplAddendum_Q30_ComparisonSet_2026-11-14.md (frozen before scoring,
# SHA-256 2e9a5b3b688bbdbe53322672b2d928946cd27a6fdb18f307346949ccfbfc1cfb).
# Annual resolution; 7 segments; handover years scored per government on its own period's
# evidence. Validator checks: integers 1-10; step <= 2 WITHIN each segment only
# (handover years are separate entities and may join discontinuously).
# Run: python build_q30_comparison_set.py

import csv

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# segment -> list of (year, (8 x (C,K,S,A) in NODES order))
SEGMENTS = {
    "Rudd I": [
        (2008, [(6,6,5,5),(5,5,5,4),(6,5,5,5),(5,5,5,4),(5,5,5,4),(6,6,5,4),(5,5,4,5),(6,6,6,5)]),
        (2009, [(5,6,6,5),(5,5,5,4),(5,5,5,4),(5,5,6,4),(5,4,7,4),(5,5,7,4),(5,5,4,5),(5,5,7,5)]),
    ],
    "Gillard": [
        (2010, [(4,5,6,4),(5,5,5,4),(4,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,4,5),(5,6,5,5)]),
        (2011, [(5,5,6,5),(5,5,6,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,4,5),(6,6,5,5)]),
        (2012, [(4,5,6,4),(5,5,6,4),(4,5,5,4),(5,5,6,4),(5,5,6,4),(5,5,5,4),(5,5,5,5),(5,5,6,5)]),
        (2013, [(3,4,7,4),(5,5,6,4),(4,5,6,4),(5,5,6,4),(5,5,5,4),(5,5,6,4),(5,5,5,5),(5,5,6,5)]),
    ],
    "Rudd II": [
        (2013, [(4,4,7,4),(5,5,7,4),(4,5,6,4),(5,5,6,4),(5,5,5,4),(5,5,6,4),(5,5,5,5),(5,5,7,4)]),
    ],
    "Abbott": [
        (2013, [(5,5,5,4),(6,6,5,4),(4,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,4,5),(5,5,5,4)]),
        (2014, [(3,4,7,4),(6,6,5,5),(4,5,6,4),(5,5,6,4),(4,5,6,4),(5,5,6,4),(4,4,5,4),(4,5,6,4)]),
        (2015, [(3,4,7,4),(6,6,5,5),(4,5,6,4),(4,5,6,4),(4,4,6,4),(5,5,6,4),(4,4,5,4),(5,5,6,4)]),
    ],
    "Turnbull": [
        (2015, [(5,5,6,4),(6,6,5,5),(4,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),(4,4,5,4),(5,5,5,4)]),
        (2016, [(5,5,6,5),(6,6,5,5),(5,5,5,5),(5,5,6,4),(4,4,6,4),(5,5,6,4),(5,4,5,4),(5,5,5,5)]),
        (2017, [(4,4,7,4),(6,6,6,5),(4,5,6,4),(5,5,6,4),(5,4,6,4),(5,5,6,4),(5,4,5,5),(5,5,6,4)]),
        (2018, [(3,4,7,4),(6,6,6,5),(4,5,7,4),(5,5,6,4),(5,4,6,4),(5,5,7,4),(5,5,5,4),(4,4,7,4)]),
    ],
    "Morrison": [
        (2018, [(4,4,7,4),(6,6,6,5),(4,5,6,4),(5,5,7,4),(4,5,6,4),(5,5,7,4),(5,5,6,5),(4,5,7,4)]),
        (2019, [(5,5,6,4),(6,6,5,5),(5,5,6,5),(5,5,7,4),(4,4,7,4),(5,5,7,4),(5,5,6,6),(5,5,6,5)]),
        (2020, [(5,5,8,5),(6,6,7,6),(6,6,7,6),(6,6,9,6),(5,6,8,5),(5,5,9,5),(5,4,7,5),(5,5,7,5)]),
        (2021, [(5,5,7,5),(6,6,6,6),(5,5,7,5),(5,5,8,5),(5,5,7,5),(5,5,8,5),(5,4,7,6),(5,5,8,5)]),
        (2022, [(5,5,7,4),(6,6,6,6),(5,5,6,5),(6,6,7,5),(4,4,7,4),(6,5,8,5),(5,5,6,5),(4,4,8,4)]),
    ],
    "Albanese": [
        (2022, [(6,6,6,6),(6,6,5,6),(6,6,6,6),(6,6,6,6),(5,5,6,5),(6,6,6,5),(5,5,6,6),(5,5,6,5)]),
        (2023, [(5,6,6,5),(5,5,6,6),(5,5,6,5),(6,6,7,5),(5,5,7,5),(6,6,6,5),(5,5,6,6),(5,6,7,5)]),
    ],
}

# status: N = New_Evidence year, R = Reaffirmed
STATUS = {
    "Rudd I":   {"2008": "N", "2009": "N"},
    "Gillard":  {"2010": "N", "2011": "R", "2012": "N", "2013": "N"},
    "Rudd II":  {"2013": "N"},
    "Abbott":   {"2013": "N", "2014": "N", "2015": "N"},
    "Turnbull": {"2015": "N", "2016": "R", "2017": "N", "2018": "N"},
    "Morrison": {"2018": "N", "2019": "N", "2020": "N", "2021": "N", "2022": "N"},
    "Albanese": {"2022": "N", "2023": "N"},
}

ENTITY = "Australian Government (Q30 comparison set)"

def validate():
    for seg, years in SEGMENTS.items():
        prev = None
        for y, nodes in years:
            assert len(nodes) == 8, (seg, y)
            for n, vals in zip(NODES, nodes):
                assert all(isinstance(v, int) and 1 <= v <= 10 for v in vals), (seg, y, n, vals)
            if prev is not None:
                for a, b, m in zip(prev, nodes, NODES):
                    for j, mm in enumerate(("C", "K", "S", "A")):
                        assert abs(a[j] - b[j]) <= 2, (seg, y, m, mm, a[j], b[j])
            prev = nodes
    print("validation passed:", sum(len(v) for v in SEGMENTS.values()), "segment-years")

def write():
    with open("Australia_Govt_Q30_comparison_set_2008-2023_annual_cams_v1.4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Year", "Segment", "Node", "Coherence", "Capacity", "Stress", "Abstraction"])
        for seg, years in SEGMENTS.items():
            for y, nodes in years:
                for n, (c, k, s, a) in zip(NODES, nodes):
                    w.writerow([ENTITY, y, seg, n, c, k, s, a])
    with open("Australia_Govt_Q30_comparison_set_2008-2023_annual_score-status.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Year", "Segment", "Node", "C_status", "K_status", "S_status", "A_status"])
        for seg, years in SEGMENTS.items():
            for y, nodes in years:
                st = "New_Evidence" if STATUS[seg][str(y)] == "N" else "Reaffirmed"
                for n in NODES:
                    w.writerow([ENTITY, y, seg, n, st, st, st, st])
    print("wrote comparison set CSVs")

if __name__ == "__main__":
    validate()
    write()
