# Scorer 3 build script for CAMS ensemble run - Australian Government 2018Q1-2023Q4
# Generates the raw CSV from the scored grid and validates structure.
# Node order within each quarter: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow

QUARTERS = [f"{y}Q{q}" for y in range(2018, 2024) for q in range(1, 5)]
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# Grid[node] = list of 24 (C, K, S, A) in quarter order
G = {
#           18Q1 18Q2 18Q3 18Q4 | 19Q1 19Q2 19Q3 19Q4 | 20Q1 20Q2 20Q3 20Q4
#           21Q1 21Q2 21Q3 21Q4 | 22Q1 22Q2 22Q3 22Q4 | 23Q1 23Q2 23Q3 23Q4
"Helm": [
 (4,4,5,3),(4,4,5,3),(3,4,6,3),(3,4,6,3),   # 2018: Turnbull limping; Aug spills; minority gov
 (3,4,6,3),(4,5,6,3),(4,5,5,3),(4,5,5,3),   # 2019: May election restores majority
 (3,4,8,3),(6,6,8,5),(6,6,8,5),(6,6,8,5),   # 2020: bushfires then National Cabinet + JobKeeper
 (6,6,7,5),(5,5,7,4),(5,5,7,5),(5,5,7,5),   # 2021: rollout blame game, then reopening plan
 (4,4,7,4),(5,5,7,4),(5,5,6,4),(4,5,6,4),   # 2022: omicron, election, Bell report
 (5,5,6,4),(5,5,6,4),(5,5,6,5),(4,5,6,5),   # 2023: stable minority; referendum defeat Q4
],
"Shield": [
 (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),   # 2018: ADF respected but capability gaps known
 (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),   # 2019: steady state
 (5,5,5,4),(6,5,5,5),(6,5,5,5),(6,5,5,5),   # 2020: bushfire assist + border/coercion surge
 (5,5,5,5),(5,5,5,5),(6,5,6,5),(6,5,6,5),   # 2021: AUKUS pivot Sep
 (6,5,6,5),(6,5,6,5),(6,5,6,5),(6,5,6,5),   # 2022: Solomons-China pact Apr
 (6,5,6,6),(6,4,7,6),(6,4,7,6),(6,4,7,6),   # 2023: DSR "not fully fit for purpose"; austerity drive
],
"Lore": [
 (4,5,5,3),(4,5,5,3),(3,4,6,3),(4,4,5,3),   # 2018: spill week; Wentworth loss
 (4,4,5,3),(4,4,5,3),(4,4,5,3),(4,4,5,3),   # 2019: low trust (41% democracy dissatisfaction)
 (4,5,7,3),(5,5,6,4),(5,5,6,4),(5,5,6,4),   # 2020: bushfire trauma then rally-round-flag
 (5,5,6,4),(4,4,6,3),(4,4,6,3),(4,4,6,3),   # 2021: blame game, mandate polarisation
 (3,4,7,3),(4,4,7,3),(4,4,6,3),(4,4,6,3),   # 2022: convoy protests, election realignment
 (4,4,6,3),(4,4,6,3),(4,4,7,4),(3,3,7,3),   # 2023: referendum defeat 60-40
],
"Stewards": [
 (5,5,4,3),(5,5,4,3),(5,5,4,3),(5,5,4,3),   # 2018: surplus in reach; robodebt running
 (5,5,4,3),(5,5,4,3),(5,5,4,3),(4,5,4,3),   # 2019: robodebt unlawful operation continues
 (5,5,6,3),(6,6,8,4),(6,6,7,4),(6,6,7,4),   # 2020: $2b fire fund, then JobKeeper surge
 (6,6,6,4),(5,5,6,4),(5,5,6,4),(5,5,6,4),   # 2021: JobKeeper ends Mar
 (5,5,6,4),(5,5,6,4),(5,5,6,4),(5,5,7,4),   # 2022: cost-of-living onset
 (5,5,7,4),(5,5,7,4),(4,5,6,3),(4,5,6,4),   # 2023: robodebt RC; accepts 57 recommendations
],
"Craft": [
 (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),   # 2018: NBN build completing
 (5,5,4,4),(5,5,4,4),(5,5,4,4),(5,5,4,4),   # 2019: steady, mixed digital delivery
 (5,5,5,4),(5,5,5,3),(5,5,5,4),(5,5,5,4),   # 2020: COVIDsafe app fails; CSL AZ production works
 (5,4,6,4),(4,4,6,3),(4,4,6,4),(5,5,5,4),   # 2021: rollout logistics fail, then state/GP channels fix
 (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),   # 2022: recovery
 (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),   # 2023: shipbuilding delays; mRNA facility building
],
"Hands": [
 (6,5,4,4),(6,5,4,4),(6,5,4,4),(6,5,4,4),   # 2018: volunteer mobilisation (drought)
 (6,5,4,4),(6,5,4,4),(6,5,4,4),(6,5,4,4),   # 2019: steady
 (7,5,8,4),(7,6,7,5),(7,6,7,5),(7,6,7,5),   # 2020: Black Summer + lockdown compliance
 (7,6,7,5),(6,6,7,3),(6,6,7,4),(6,6,6,4),   # 2021: fatigue; repeated quarantine leaks
 (5,5,7,4),(5,5,6,4),(5,5,6,4),(5,5,6,4),   # 2022: omicron collapse, labour shortage
 (5,5,6,4),(5,5,6,4),(5,5,6,4),(5,5,6,4),   # 2023: shortages ease, migration resumed
],
"Archive": [
 (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),   # 2018: Thodey commissioned May
 (5,5,5,4),(4,4,5,3),(4,4,5,3),(4,4,5,3),   # 2019: Tune Review (Apr) + Thodey (Dec)
 (4,4,6,3),(4,4,6,3),(4,4,6,3),(4,4,6,3),   # 2020: RC flags unlearned lessons
 (4,4,6,3),(4,4,6,3),(4,4,6,3),(4,4,6,3),   # 2021: $67.7m rescue; APS Inc hollow-out
 (4,4,6,3),(4,4,6,3),(4,4,6,3),(3,3,6,3),   # 2022: secret ministries revealed; Bell report
 (4,4,5,3),(4,4,5,3),(5,4,6,4),(5,4,5,4),   # 2023: robodebt RC + notification Act = repair
],
"Flow": [
 (6,5,4,4),(6,5,4,4),(6,5,4,4),(6,5,4,4),   # 2018: open flows
 (6,5,4,4),(6,5,4,4),(6,5,4,4),(6,5,4,4),   # 2019: open flows
 (6,5,6,4),(5,5,6,4),(5,5,5,4),(5,5,5,4),   # 2020: China coercion + border halt
 (5,5,5,4),(5,5,5,4),(5,5,5,4),(5,5,5,4),   # 2021: coercion peak, rerouting works
 (5,5,5,3),(4,4,8,3),(4,5,7,4),(4,5,7,4),   # 2022: NEM suspension; inflation/rate shock
 (5,5,7,4),(5,5,6,4),(5,5,6,4),(5,5,6,4),   # 2023: China thaw; inflation easing
],
}

lines = ["Entity,Year,Node,Coherence,Capacity,Stress,Abstraction"]
for qi, q in enumerate(QUARTERS):
    for n in NODES:
        c, k, s, a = G[n][qi]
        lines.append(f"Australian Government,{q},{n},{c},{k},{s},{a}")

out = "C:/Users/julie/kimi_scores/Scoring/ens3_Australia_Govt_2018Q1-2023Q4_raw.csv"
with open(out, "w", newline="") as f:
    f.write("\n".join(lines) + "\n")

# ---- validation ----
import csv, sys
rows = list(csv.reader(open(out)))
assert rows[0] == ["Entity","Year","Node","Coherence","Capacity","Stress","Abstraction"], "bad header"
data = rows[1:]
assert len(data) == 192, f"row count {len(data)}"
i = 0
for q in QUARTERS:
    for n in NODES:
        r = data[i]
        assert r[0] == "Australian Government" and r[1] == q and r[2] == n, f"order broken at {i}: {r}"
        for v in r[3:]:
            assert v == "NA" or (v.isdigit() and 1 <= int(v) <= 10), f"bad value {v} at row {i}"
        i += 1
assert data[-1][1] == "2023Q4" and data[-1][2] == "Flow"
print("VALIDATION PASSED: 192 rows, header OK, quarter sequence OK, node order OK, all metrics integer 1-10 or NA")
