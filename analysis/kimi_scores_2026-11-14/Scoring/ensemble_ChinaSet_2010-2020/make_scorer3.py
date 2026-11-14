# scorer3_raw.csv generator — CAMS Universal Instruction Set v1.4
# Scorer 3 of 5, ensemble run. Entity-year-node matrices as decided from
# scorer 3's own evidence draw (see scorer3_note.md). Values: int 1-10 or "NA".
# Node order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
YEARS = list(range(2010, 2021))
NA = "NA"

def row(c, k, s, a):
    return (c, k, s, a)

# ---- CHINA (national) ----
china = {
"Helm":     [row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(8,7,7,6),row(8,7,7,6),row(8,8,7,6),row(8,8,7,7),row(8,8,7,7),row(9,8,7,7),row(9,8,7,7),row(8,8,8,7)],
"Shield":   [row(6,7,6,5),row(6,7,6,5),row(6,7,6,5),row(6,7,6,5),row(6,7,7,5),row(7,7,7,6),row(8,8,7,7),row(8,8,6,7),row(8,8,7,7),row(8,8,7,7),row(8,8,7,7)],
"Lore":     [row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(7,6,6,5),row(7,7,6,5),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(8,7,7,6)],
"Stewards": [row(6,7,6,5),row(6,7,6,5),row(6,7,6,5),row(6,7,7,5),row(6,7,7,5),row(6,7,7,6),row(7,7,7,6),row(7,7,7,6),row(7,6,8,6),row(7,6,7,6),row(7,7,8,7)],
"Craft":    [row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,7,6),row(7,7,7,6),row(7,7,7,6),row(7,8,7,7),row(7,8,7,7),row(7,8,7,7),row(7,8,7,7)],
"Hands":    [row(6,8,6,5),row(6,8,6,5),row(6,8,6,5),row(6,7,7,5),row(6,7,7,6),row(6,7,7,6),row(6,7,7,6),row(6,7,7,6),row(6,7,7,6),row(6,7,7,6),row(7,7,8,7)],
"Archive":  [row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(6,7,5,5),row(6,7,5,6),row(6,7,5,6),row(6,7,5,6),row(6,7,5,6),row(6,7,5,6),row(6,7,6,6)],
"Flow":     [row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,8,6,6),row(7,8,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,7,6),row(8,8,7,6),row(8,8,7,7)],
}

# ---- EVERGRANDE ----
evergrande = {
"Helm":     [row(8,7,6,5),row(8,7,6,5),row(7,6,7,5),row(7,7,7,5),row(7,7,7,5),row(7,7,7,5),row(6,7,8,5),row(6,7,8,5),row(7,7,8,5),row(6,7,8,5),row(7,6,9,6)],
"Shield":   [row(6,6,6,5),row(5,6,7,5),row(5,6,7,5),row(6,6,6,5),row(6,6,6,5),row(6,7,6,5),row(6,7,6,5),row(7,7,6,5),row(7,7,6,5),row(6,7,7,5),row(5,6,8,5)],
"Lore":     [row(6,6,5,5),row(6,6,5,5),row(5,6,6,5),row(7,7,5,5),row(7,7,5,5),row(8,7,6,5),row(8,7,6,5),row(8,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,7,5)],
"Stewards": [row(6,6,6,5),row(6,6,7,5),row(5,6,7,5),row(6,6,7,5),row(6,6,7,5),row(6,7,7,5),row(6,7,8,5),row(6,7,8,5),row(6,7,8,5),row(6,6,8,5),row(6,6,9,6)],
"Craft":    [row(7,6,6,5),row(7,6,6,5),row(6,6,7,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,7,5),row(6,6,7,5),row(6,6,7,5),row(6,6,8,5)],
"Hands":    [row(7,7,6,5),row(7,7,7,5),row(6,6,7,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,7,5),row(7,7,7,5),row(7,7,7,5),row(7,7,7,5),row(7,7,8,5)],
"Archive":  [row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(6,6,5,5),row(6,6,6,5),row(6,6,7,5)],
"Flow":     [row(6,6,6,5),row(6,6,7,5),row(6,6,7,5),row(6,7,6,5),row(6,7,6,5),row(7,7,6,5),row(7,7,7,5),row(7,7,7,5),row(7,7,7,5),row(7,7,7,5),row(6,6,8,5)],
}

# ---- HUAWEI ----
huawei = {
"Helm":     [row(8,7,6,6),row(8,7,6,6),row(8,7,7,6),row(8,7,6,6),row(8,7,6,6),row(8,7,6,6),row(8,7,6,6),row(8,8,6,7),row(8,8,6,7),row(9,8,8,7),row(8,7,9,8)],
"Shield":   [row(6,6,5,5),row(6,6,5,5),row(5,6,7,5),row(5,6,7,5),row(5,6,7,5),row(5,6,7,5),row(5,6,6,5),row(6,6,6,5),row(6,6,7,5),row(6,7,8,6),row(6,7,8,6)],
"Lore":     [row(6,6,5,5),row(6,6,5,5),row(6,6,6,5),row(6,6,6,5),row(7,6,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(8,7,6,6),row(8,7,7,6),row(8,7,7,6)],
"Stewards": [row(7,7,5,6),row(7,7,5,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,8,6,6),row(7,8,6,6),row(7,7,7,7),row(7,7,8,7)],
"Craft":    [row(7,7,5,6),row(7,7,5,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,7),row(7,8,6,7),row(8,8,6,7),row(8,8,6,8),row(8,8,7,8),row(8,7,8,8)],
"Hands":    [row(7,7,5,5),row(7,7,5,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,8,6,5),row(8,7,8,6),row(7,7,8,6)],
"Archive":  [row(7,7,5,6),row(7,7,5,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,8,6,7),row(8,8,7,7),row(8,8,7,7)],
"Flow":     [row(8,7,5,6),row(8,7,5,6),row(8,7,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,6,6),row(8,8,6,7),row(7,7,8,7),row(6,6,9,7)],
}

# ---- BYD ----
byd = {
"Helm":     [row(8,6,7,5),row(7,6,8,5),row(7,6,8,5),row(7,6,7,5),row(8,7,7,6),row(8,7,7,6),row(8,7,6,6),row(7,7,6,6),row(7,7,7,6),row(7,7,6,6),row(8,7,6,7)],
"Shield":   [row(6,6,6,5),row(6,6,7,5),row(6,6,7,5),row(6,6,6,5),row(6,7,6,5),row(7,7,6,6),row(6,6,7,6),row(6,6,7,6),row(6,7,6,6),row(6,7,6,6),row(6,7,6,6)],
"Lore":     [row(6,5,6,5),row(6,5,6,5),row(5,5,7,5),row(5,5,6,5),row(6,5,6,5),row(7,6,6,5),row(7,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(7,6,5,6)],
"Stewards": [row(6,6,7,5),row(6,6,8,5),row(5,5,8,5),row(6,6,7,5),row(6,6,6,5),row(7,7,6,6),row(6,6,6,6),row(6,6,7,6),row(6,7,6,6),row(6,6,6,6),row(6,7,6,7)],
"Craft":    [row(7,7,6,6),row(7,7,6,6),row(6,6,7,6),row(7,6,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,7,6),row(7,7,6,6),row(7,7,6,7),row(8,8,5,8)],
"Hands":    [row(6,7,6,5),row(6,7,6,5),row(6,6,7,5),row(6,7,6,5),row(6,7,6,5),row(7,7,6,5),row(7,7,6,5),row(6,7,6,5),row(6,7,6,5),row(6,7,6,5),row(7,7,6,5)],
"Archive":  [row(6,6,5,5),row(6,6,5,5),row(6,6,6,5),row(6,6,5,5),row(6,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,6),row(7,7,5,7)],
"Flow":     [row(6,6,7,5),row(6,6,7,5),row(6,6,8,5),row(6,6,7,5),row(6,7,6,5),row(7,7,6,5),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6)],
}

# ---- CATL (founded 2011; 2010 = NA all nodes) ----
catl = {
"Helm":     [row(NA,NA,NA,NA),row(6,5,5,5),row(6,5,5,5),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(7,6,6,5),row(7,6,6,5),row(7,7,6,6),row(7,7,5,6),row(7,7,6,6),row(7,7,5,6)],
"Shield":   [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(7,6,6,5),row(7,7,6,6),row(7,7,6,6),row(7,7,5,6),row(7,6,6,6),row(7,6,6,6)],
"Lore":     [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(6,5,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5)],
"Stewards": [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(5,5,5,5),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(6,6,6,5),row(7,6,6,5),row(7,7,6,6),row(7,8,6,6),row(6,7,8,6),row(7,7,7,6)],
"Craft":    [row(NA,NA,NA,NA),row(6,5,5,5),row(6,6,5,5),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(7,6,6,6),row(7,7,6,6),row(8,8,6,7),row(8,8,6,7),row(8,8,7,7),row(8,8,6,8)],
"Hands":    [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA)],
"Archive":  [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(NA,NA,NA,NA)],
"Flow":     [row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(6,5,5,5),row(NA,NA,NA,NA),row(NA,NA,NA,NA),row(6,6,6,5),row(7,7,6,5),row(8,8,6,6),row(8,8,5,6),row(7,7,7,6),row(7,7,6,6)],
}

# ---- VANKE ----
vanke = {
"Helm":     [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(4,6,8,5),row(3,6,9,5),row(5,7,7,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6)],
"Shield":   [row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(5,6,7,5),row(5,6,8,5),row(8,7,6,5),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6)],
"Lore":     [row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(6,7,7,5),row(6,7,7,5),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,6,6)],
"Stewards": [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,6,6),row(6,6,8,6),row(6,6,8,6),row(7,7,6,6),row(7,7,6,7),row(7,7,6,7),row(7,7,6,7)],
"Craft":    [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,7,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6)],
"Hands":    [row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(6,7,6,5),row(6,6,7,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,6,5)],
"Archive":  [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(6,7,5,6),row(6,6,6,5),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6)],
"Flow":     [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(5,6,8,5),row(6,6,8,5),row(7,7,6,6),row(7,7,5,6),row(7,7,5,6),row(7,7,6,6)],
}

# ---- LONGFOR ----
longfor = {
"Helm":     [row(8,7,5,5),row(8,7,5,5),row(7,7,6,5),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6)],
"Shield":   [row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,6),row(7,7,6,6),row(7,7,6,6),row(7,7,6,6)],
"Lore":     [row(7,6,5,5),row(7,6,5,5),row(6,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,6,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5)],
"Stewards": [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6)],
"Craft":    [row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6),row(8,7,5,6)],
"Hands":    [row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5)],
"Archive":  [row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6)],
"Flow":     [row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,5),row(7,7,5,6),row(7,7,5,6),row(7,7,5,6)],
}

# ---- SMIC ----
smic = {
"Helm":     [row(5,6,7,5),row(4,6,8,5),row(6,6,7,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,7,6,5),row(6,7,6,6),row(6,7,6,6),row(7,7,6,6),row(5,7,8,6)],
"Shield":   [row(6,6,7,5),row(6,6,7,5),row(7,6,6,5),row(7,6,6,5),row(7,6,6,5),row(7,6,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(7,7,6,5),row(6,6,8,5)],
"Lore":     [row(5,5,6,4),row(5,5,6,4),row(6,5,6,4),row(6,5,5,4),row(6,5,5,4),row(6,5,5,4),row(6,5,5,4),row(6,6,5,5),row(6,6,5,5),row(7,6,5,5),row(7,6,6,5)],
"Stewards": [row(5,5,7,4),row(5,5,7,4),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,7,5,5),row(6,6,6,5),row(6,6,7,5)],
"Craft":    [row(6,6,7,4),row(6,6,7,4),row(6,6,6,4),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,6,6,5),row(6,7,6,6),row(7,7,6,7),row(7,7,7,7)],
"Hands":    [row(6,6,7,4),row(6,6,7,4),row(6,6,6,4),row(6,6,6,4),row(6,6,5,4),row(6,6,5,4),row(6,6,5,4),row(7,6,5,5),row(7,7,5,5),row(7,7,5,5),row(6,6,7,5)],
"Archive":  [row(5,6,7,4),row(5,6,7,4),row(6,6,6,4),row(6,6,6,4),row(6,6,6,4),row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(6,7,5,5),row(6,7,5,5),row(6,6,6,5)],
"Flow":     [row(6,6,7,4),row(6,6,7,4),row(6,6,6,5),row(6,6,5,5),row(6,6,5,5),row(6,6,5,5),row(6,7,5,5),row(7,7,5,5),row(7,7,5,5),row(6,7,6,5),row(6,6,8,5)],
}

ENTITIES = [
    ("China (national)", china),
    ("Evergrande", evergrande),
    ("Huawei", huawei),
    ("BYD", byd),
    ("CATL", catl),
    ("Vanke", vanke),
    ("Longfor", longfor),
    ("SMIC", smic),
]

import csv, os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorer3_raw.csv")
rows = 0
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"])
    for name, matrix in ENTITIES:
        for year in YEARS:
            for node in NODES:  # canonical order
                c, k, s, a = matrix[node][year - 2010]
                w.writerow([name, year, node, c, k, s, a])
                rows += 1
print("data rows:", rows)
assert rows == 704, "row count mismatch"
print("OK ->", out)
