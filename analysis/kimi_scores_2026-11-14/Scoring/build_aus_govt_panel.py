# Generator for the whole-of-government 8-node quarterly panel.
# Entity: Australian Government. Node order canonical: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.
# Scores (C,K,S,A) per quarter 2018Q1..2023Q4. One inner list of 4 ints per quarter, in node order.
# Run: python build_aus_govt_panel.py

import csv

ENTITY = "Australian Government"
QUARTERS = [f"{y}Q{q}" for y in range(2018, 2024) for q in range(1, 5)]
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# scores[node] = [ (C,K,S,A) x 24 ]
scores = {
    "Helm": [
        (5,5,6,4),(5,5,6,4),(3,4,7,4),(4,5,7,4),
        (5,5,6,4),(6,6,5,5),(6,6,5,5),(4,5,7,4),
        (6,6,8,6),(6,6,9,6),(5,5,8,5),(5,5,7,5),
        (5,5,7,5),(5,4,7,4),(6,5,8,6),(6,6,8,6),
        (4,4,8,4),(5,5,7,4),(4,5,6,4),(6,6,7,6),
        (6,6,6,6),(6,6,6,6),(5,6,7,5),(4,6,6,4),
    ],
    "Shield": [
        (6,6,4,5),(6,6,4,5),(6,6,4,5),(6,6,5,5),
        (6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,7,5),
        (6,6,7,6),(6,6,7,6),(6,6,7,6),(6,6,6,6),
        (6,6,7,5),(6,6,7,5),(6,6,8,6),(6,6,7,6),
        (6,6,7,5),(5,5,8,5),(6,5,8,5),(6,5,7,6),
        (6,5,7,6),(6,5,7,6),(6,5,7,6),(6,5,7,6),
    ],
    "Lore": [
        (5,6,5,4),(5,6,5,4),(5,6,5,4),(5,6,5,4),
        (5,6,6,4),(5,6,6,4),(5,6,6,4),(4,5,8,4),
        (6,6,8,5),(8,6,8,6),(6,6,7,5),(6,6,7,5),
        (5,5,7,4),(5,5,8,4),(4,5,8,4),(5,5,7,4),
        (5,6,6,5),(5,6,6,5),(5,6,6,5),(6,6,6,5),
        (5,6,6,4),(5,6,7,4),(5,6,8,4),(4,5,7,4),
    ],
    "Stewards": [
        (6,6,4,5),(6,6,4,5),(6,6,4,5),(6,6,4,5),
        (6,6,5,5),(6,6,5,5),(5,5,5,4),(5,5,6,4),
        (6,6,8,5),(6,6,9,6),(5,5,8,5),(5,5,7,5),
        (5,5,6,5),(5,5,6,5),(5,5,7,5),(5,5,7,5),
        (5,5,8,4),(5,5,8,4),(5,5,8,4),(6,6,7,5),
        (6,6,7,5),(6,6,6,5),(6,6,7,5),(6,6,7,5),
    ],
    "Craft": [
        (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),
        (5,5,5,4),(5,5,5,4),(4,5,6,4),(4,4,7,4),
        (5,5,8,4),(5,6,8,5),(5,6,7,5),(5,5,6,5),
        (5,5,7,4),(5,4,8,4),(5,5,7,5),(5,5,6,5),
        (5,5,7,5),(5,5,7,5),(4,4,7,4),(4,4,7,4),
        (4,5,7,4),(4,5,7,4),(4,5,6,4),(4,5,6,4),
    ],
    "Hands": [
        (6,6,5,4),(6,6,5,4),(6,6,5,4),(6,6,5,4),
        (6,6,5,4),(6,6,5,4),(6,5,6,4),(5,5,8,4),
        (6,5,8,5),(5,5,9,5),(5,5,8,5),(5,5,7,5),
        (5,5,7,5),(5,5,8,5),(5,5,8,5),(5,5,8,5),
        (5,5,8,5),(5,5,8,5),(6,5,8,5),(6,5,8,5),
        (6,5,8,5),(6,5,8,5),(6,5,7,5),(6,6,7,5),
    ],
    "Archive": [
        (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),
        (5,5,6,5),(4,5,6,5),(5,5,6,5),(5,5,6,6),
        (5,5,7,5),(5,4,7,5),(5,4,7,5),(5,4,7,6),
        (5,4,7,6),(5,4,6,6),(5,4,6,6),(5,4,6,6),
        (5,4,6,5),(5,4,6,5),(5,5,6,5),(5,5,6,5),
        (5,5,6,6),(5,5,6,6),(5,5,6,6),(5,5,6,6),
    ],
    "Flow": [
        (6,6,4,5),(6,6,4,5),(6,6,4,5),(6,6,4,5),
        (6,6,5,5),(6,6,5,5),(6,6,6,5),(5,5,7,4),
        (6,6,7,4),(6,6,8,5),(5,5,7,5),(5,5,7,5),
        (5,5,8,5),(5,5,8,5),(5,5,7,5),(5,5,7,5),
        (5,5,8,5),(5,5,9,5),(4,4,8,4),(5,5,6,5),
        (6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,5,5),
    ],
}

# per-node per-quarter status: N = New_Evidence, R = Reaffirmed (no Carried_Forward)
status = {
    "Helm": ["N","N","N","N","N","N","R","N","N","N","N","N","N","N","N","N","N","N","N","N","R","R","N","N"],
    "Shield": ["R","R","R","R","N","R","R","N","N","N","N","R","R","R","N","N","N","N","N","R","N","N","R","R"],
    "Lore":   ["R","R","R","R","R","R","R","N","N","N","N","R","N","R","N","R","N","N","N","R","R","R","N","N"],
    "Stewards":["R","R","R","R","N","N","R","N","N","N","N","N","R","N","R","N","N","N","N","N","R","N","N","R"],
    "Craft":  ["R","R","R","R","R","R","N","N","N","N","R","R","R","N","N","R","R","R","N","R","R","R","N","R"],
    "Hands":  ["R","R","R","R","R","R","N","N","N","N","N","R","R","N","N","N","N","N","N","N","R","R","N","N"],
    "Archive":["R","R","R","R","N","N","N","R","N","N","N","N","N","N","R","R","R","R","N","R","R","R","N","R"],
    "Flow":   ["R","R","R","R","R","R","R","N","N","N","R","N","N","R","R","R","N","N","N","R","N","R","N","N"],
}

def validate():
    assert set(scores) == set(NODES) and len(NODES) == 8
    for n in NODES:
        assert len(scores[n]) == 24, n
        assert len(status[n]) == 24, n
        prev = None
        for i, q in enumerate(QUARTERS):
            vals = scores[n][i]
            assert all(isinstance(v, int) and 1 <= v <= 10 for v in vals), (n, q, vals)
            if prev is not None:
                for j, m in enumerate(("C","K","S","A")):
                    assert abs(vals[j] - prev[j]) <= 2, (n, q, m, prev[j], vals[j])
            prev = vals
    print("validation passed")

def write():
    with open("Australia_Govt_2018Q1-2023Q4_cams_v1.4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity","Year","Node","Coherence","Capacity","Stress","Abstraction"])
        for i, q in enumerate(QUARTERS):
            for n in NODES:
                c,k,s,a = scores[n][i]
                w.writerow([ENTITY, q, n, c, k, s, a])
    with open("Australia_Govt_2018Q1-2023Q4_score-status.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity","Year","Node","C_status","K_status","S_status","A_status"])
        for i, q in enumerate(QUARTERS):
            for n in NODES:
                st = "New_Evidence" if status[n][i] == "N" else "Reaffirmed"
                w.writerow([ENTITY, q, n, st, st, st, st])
    print("wrote CSVs")

if __name__ == "__main__":
    validate()
    write()
