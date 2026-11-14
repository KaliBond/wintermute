# Validate CAMS panel pass CSVs (mechanical checks only; no scoring, no aggregation)
import csv, sys

BASE = r"C:\Users\julie\kimi_scores\Scoring"
ENTITIES = ["China", "Evergrande", "Vanke", "Longfor", "Huawei", "SMIC", "BYD", "CATL"]
NODES = {
    "China": ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"],
    "corp":  ["Board", "Risk", "RD", "Finance", "Operations", "Workforce", "Brand", "Markets"],
}
EXPECTED = [(e, y, n) for e in ENTITIES for y in (2021, 2022, 2023, 2024)
            for n in NODES["China" if e == "China" else "corp"]]

ok_all = True
for p in range(1, 6):
    path = f"{BASE}\\pass{p}.csv"
    problems = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["Entity", "Year", "Node", "C", "K", "S", "A"]:
        problems.append(f"header != canonical: {rows[0]}")
    data = rows[1:]
    if len(data) != 256:
        problems.append(f"data rows = {len(data)} (want 256)")
    keys = [(r[0], int(r[1]), r[2]) for r in data if len(r) >= 3]
    if keys != EXPECTED:
        bad = [i for i, (a, b) in enumerate(zip(keys, EXPECTED)) if a != b]
        problems.append(f"order mismatch at row idx {bad[:5]} (of {len(bad)} mismatches)")
    for i, r in enumerate(data):
        if len(r) != 7:
            problems.append(f"row {i+2}: {len(r)} cols"); continue
        vals = r[3:]
        if any(v == "NA" for v in vals) and not all(v == "NA" for v in vals):
            problems.append(f"row {i+2}: partial NA {vals}")
        for v in vals:
            if v == "NA":
                continue
            if not (v.isdigit() and 1 <= int(v) <= 10):
                problems.append(f"row {i+2}: bad value {v!r}")
    na_cells = sum(1 for r in data for v in r[3:] if v == "NA")
    china_na = sum(1 for r in data if r[0] == "China" for v in r[3:] if v == "NA")
    status = "OK" if not problems else "PROBLEMS"
    if problems:
        ok_all = False
    print(f"pass{p}.csv: {status} | data rows={len(data)} | NA cells={na_cells} | China NA={china_na}")
    for pr in problems[:10]:
        print("   -", pr)
print("ALL OK" if ok_all else "VALIDATION FAILED")
sys.exit(0 if ok_all else 1)
