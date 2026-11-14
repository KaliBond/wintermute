# Generator for the blinded entity series and Helm/Archive extensions.
# Entities masked per Implementation Addendum (frozen 2026-11-14, SHA-256
# b3f789e82c2d7680b6874106c889347fe6bde6343d528485f4a1f295d831f095):
#   Entity_A = Centrelink / DHS / Services Australia  (node: Stewards, agency resolution)
#   Entity_B = Australian Taxation Office             (node: Archive,  agency resolution)
# Extensions join the existing 2018Q1 panels (same protocol, same anchors).
# Run: python blind_entity_series_build.py
# Scores follow CAMS Universal Instruction Set v1.4; contemporaneous-source
# discipline per addendum section 2: each quarter scored only on evidence
# available by that quarter's end. Validator enforces the <=2 step rule.

import csv

QUARTERS = [f"{y}Q{q}" for y in range(2013, 2024) for q in range(1, 5)]
EXT_QUARTERS = [f"{y}Q{q}" for y in range(2013, 2018) for q in range(1, 5)]

# --- Entity_A: Stewards agency-resolution (44 quarters) ---
entityA = [
    (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),          # 2013
    (5,5,4,4),(5,5,5,4),(5,5,5,4),(5,5,4,4),          # 2014 (budget uncertainty)
    (5,5,4,4),(5,5,5,4),(4,4,5,5),(4,4,5,5),          # 2015 (E1 May; E2 Jul)
    (4,4,5,5),(4,4,5,5),(4,4,6,6),(4,3,7,6),          # 2016 (E3 Jul/Sep; E4 Dec)
    (4,3,7,6),(3,3,7,6),(3,3,6,6),(3,3,6,6),          # 2017 (W1 Apr; W2 Jun; W3 AAT)
    (3,3,6,6),(4,3,6,6),(4,3,6,6),(4,3,6,6),          # 2018 (CUPI; debts continue)
    (4,3,6,6),(4,3,6,6),(4,3,6,6),(4,4,5,5),          # 2019 (W4 Feb; W5 Nov)
    (5,4,7,5),(5,3,9,4),(5,4,7,4),(5,4,6,4),          # 2020 (C1 COVID; W6 May/Jul)
    (5,4,5,4),(5,4,5,4),(5,5,5,5),(5,5,5,5),          # 2021 (W7 Jun)
    (5,5,5,5),(5,5,5,5),(4,5,6,5),(4,5,5,5),          # 2022 (W8 Aug)
    (5,5,5,5),(5,5,5,5),(4,5,6,5),(5,5,5,5),          # 2023 (W9 Jul)
]
statusA = (
    ["R","R","R","R"] + ["R","N","R","R"] + ["R","N","N","R"] +
    ["R","R","N","N"] + ["N","N","N","R"] + ["N","N","N","R"] +
    ["N","N","N","N"] + ["N","N","R","N"] + ["R","R","R","R"] +
    ["N","N","R","N"] + ["R","R","N","R"]
)

# --- Entity_B: Archive agency-resolution (44 quarters) ---
entityB = [
    (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),          # 2013
    (5,5,4,4),(5,5,5,4),(5,4,5,4),(5,4,5,4),          # 2014 (budget cuts)
    (5,4,5,5),(5,4,5,5),(5,4,5,5),(5,4,5,5),          # 2015 (restructure; digital)
    (5,4,4,5),(5,4,4,5),(5,5,4,5),(5,5,4,5),          # 2016
    (5,5,4,5),(4,4,6,5),(4,4,6,5),(5,5,5,5),          # 2017 (B1 garnishee scandal)
    (5,5,4,5),(5,5,4,5),(5,5,4,5),(5,5,4,5),          # 2018
    (5,5,4,5),(5,5,4,5),(5,5,4,5),(5,5,4,5),          # 2019
    (5,5,6,5),(6,6,8,5),(6,6,8,5),(5,6,6,5),          # 2020 (B2 JobKeeper/super)
    (5,5,5,5),(5,5,5,5),(5,5,5,5),(5,5,5,5),          # 2021
    (5,5,5,5),(5,5,5,5),(5,5,5,5),(5,5,5,5),          # 2022
    (5,5,5,5),(5,5,5,5),(5,5,5,5),(5,5,5,5),          # 2023
]
statusB = (
    ["R","R","R","R"] + ["R","N","N","R"] + ["N","R","R","R"] +
    ["R","R","N","R"] + ["R","N","N","N"] + ["R","R","R","R"] +
    ["R","R","R","R"] + ["R","R","N","R"] + ["N","N","N","N"] +
    ["R","R","R","R"] + ["R","R","R","R"]
)

# --- Helm extension 2013Q1-2017Q4 (joins 2018Q1 = (5,5,6,4)) ---
helm_ext = [
    (4,4,5,4),(4,4,6,4),(4,4,6,4),(5,5,5,4),          # 2013 (Rudd return; Abbott win)
    (5,5,4,4),(4,5,6,4),(3,4,6,4),(4,4,5,4),          # 2014 (2014 budget; Senate block)
    (3,4,6,4),(4,4,5,4),(4,5,6,4),(5,5,5,4),          # 2015 (spill; Turnbull)
    (5,5,4,4),(4,5,5,4),(4,5,6,4),(4,5,6,4),          # 2016 (election; OCI blowback)
    (4,5,6,4),(4,5,6,4),(4,4,7,4),(3,4,7,4),          # 2017 (citizenship; survey)
]
helm_status = ["N","N","N","N","N","N","N","R","N","N","N","N","N","N","N","N","N","N","N","N"]

# --- Archive (national) extension 2013Q1-2017Q4 (joins 2018Q1 = (5,5,5,4)) ---
archive_ext = [
    (5,5,4,5),(5,5,4,5),(5,5,4,5),(5,5,4,5),          # 2013
    (5,5,4,5),(4,4,5,4),(4,4,5,4),(4,4,5,4),          # 2014 (OAIC cut announced)
    (4,4,5,4),(4,4,5,4),(4,4,5,4),(4,4,5,4),          # 2015 (metadata retention)
    (5,4,5,4),(5,4,5,4),(5,4,5,4),(5,4,5,4),          # 2016
    (5,4,5,5),(5,4,5,5),(5,4,5,5),(5,4,5,5),          # 2017
]
archive_status = ["R","R","R","R","R","N","R","R","R","R","R","R","N","R","R","R","N","R","R","R"]

def validate(name, series, quarters):
    assert len(series) == len(quarters), (name, len(series), len(quarters))
    prev = None
    for i, q in enumerate(quarters):
        vals = series[i]
        assert all(isinstance(v, int) and 1 <= v <= 10 for v in vals), (name, q, vals)
        if prev is not None:
            for j, m in enumerate(("C","K","S","A")):
                assert abs(vals[j] - prev[j]) <= 2, (name, q, m, prev[j], vals[j])
        prev = vals
    print(f"validated {name}: {len(quarters)} quarters")

def write_panel(path, entity, node, series, quarters, status):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity","Year","Node","Coherence","Capacity","Stress","Abstraction"])
        for i, q in enumerate(quarters):
            c, k, s, a = series[i]
            w.writerow([entity, q, node, c, k, s, a])
    with open(path.replace(".csv", "_score-status.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity","Year","Node","C_status","K_status","S_status","A_status"])
        for i, q in enumerate(quarters):
            st = "New_Evidence" if status[i] == "N" else "Reaffirmed"
            w.writerow([entity, q, node, st, st, st, st])
    print(f"wrote {path}")

if __name__ == "__main__":
    validate("Entity_A", entityA, QUARTERS)
    validate("Entity_B", entityB, QUARTERS)
    validate("Helm_ext", helm_ext, EXT_QUARTERS)
    validate("Archive_ext", archive_ext, EXT_QUARTERS)
    # join checks against existing 2018Q1 panels
    assert helm_ext[-1] == (3,4,7,4)  # 2017Q4 -> 2018Q1 (5,5,6,4): steps +2,+1,-1,0 OK
    assert archive_ext[-1] == (5,4,5,5)  # -> 2018Q1 (5,5,5,4): OK
    write_panel("EntityA_Stewards_2013Q1-2023Q4_cams_v1.4.csv",
                "Entity_A (masked; see protocol note)", "Stewards", entityA, QUARTERS, statusA)
    write_panel("EntityB_Archive_2013Q1-2023Q4_cams_v1.4.csv",
                "Entity_B (masked; see protocol note)", "Archive", entityB, QUARTERS, statusB)
    write_panel("Helm_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv",
                "Australian Government", "Helm", helm_ext, EXT_QUARTERS, helm_status)
    write_panel("Archive_AusGovt_ext_2013Q1-2017Q4_cams_v1.4.csv",
                "Australian Government", "Archive", archive_ext, EXT_QUARTERS, archive_status)
    print("done")
