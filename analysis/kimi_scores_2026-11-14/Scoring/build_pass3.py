# Build pass3.csv — CAMS pass 3 of 5. Row structure only; scores entered by scorer judgment.
# Entity order, year ascending 2021-2024, canonical node order per protocol section 7.

header = "Entity,Year,Node,C,K,S,A"

china_nodes = ["Helm","Shield","Lore","Stewards","Craft","Hands","Archive","Flow"]
corp_nodes = ["Board","Risk","RD","Finance","Operations","Workforce","Brand","Markets"]

# Scores[entity][year][node] = (C,K,S,A) or ("NA",)*4
S = {}

# ---------------- CHINA ----------------
S["China"] = {
 2021: {"Helm":(7,7,4,7),"Shield":(7,7,3,7),"Lore":(8,7,3,8),"Stewards":(6,7,5,6),
        "Craft":(5,6,6,6),"Hands":(5,5,6,5),"Archive":(6,6,4,7),"Flow":(6,7,6,6)},
 2022: {"Helm":(6,6,6,6),"Shield":(7,7,3,7),"Lore":(6,6,5,6),"Stewards":(6,7,5,6),
        "Craft":(6,7,5,6),"Hands":(5,5,8,5),"Archive":(6,6,5,6),"Flow":(5,6,7,5)},
 2023: {"Helm":(5,6,6,6),"Shield":(5,6,5,6),"Lore":(7,7,4,7),"Stewards":(6,7,4,6),
        "Craft":(6,7,4,7),"Hands":(5,5,7,5),"Archive":(6,6,5,6),"Flow":(5,5,7,5)},
 2024: {"Helm":(6,6,5,6),"Shield":(7,7,3,7),"Lore":(6,6,5,6),"Stewards":(6,7,4,6),
        "Craft":(6,8,5,7),"Hands":(5,5,6,5),"Archive":(6,6,4,6),"Flow":(5,6,6,5)},
}

# ---------------- EVERGRANDE ----------------
NA4 = ("NA","NA","NA","NA")
S["Evergrande"] = {
 2021: {"Board":(4,4,7,4),"Risk":(3,3,8,3),"RD":(5,4,6,4),"Finance":(2,2,9,2),
        "Operations":(4,4,7,3),"Workforce":NA4,"Brand":NA4,"Markets":(4,4,8,3)},
 2022: {"Board":(3,3,8,3),"Risk":(2,2,9,2),"RD":(4,3,7,4),"Finance":(2,2,9,2),
        "Operations":(4,4,7,3),"Workforce":(3,3,8,3),"Brand":(2,3,9,2),"Markets":(2,2,9,2)},
 2023: {"Board":(2,2,9,3),"Risk":(2,2,9,2),"RD":(3,2,8,3),"Finance":(2,2,9,2),
        "Operations":(3,3,7,3),"Workforce":NA4,"Brand":(2,2,8,2),"Markets":(2,2,8,2)},
 2024: {"Board":(2,2,8,2),"Risk":(2,2,9,2),"RD":(2,2,8,2),"Finance":(2,2,9,2),
        "Operations":(2,2,8,2),"Workforce":(2,2,8,2),"Brand":(2,2,8,2),"Markets":(2,2,8,2)},
}

# ---------------- VANKE ----------------
S["Vanke"] = {
 2021: {"Board":(7,7,4,6),"Risk":(7,7,4,6),"RD":(6,6,4,6),"Finance":(7,7,4,6),
        "Operations":(7,7,4,6),"Workforce":NA4,"Brand":(7,7,4,6),"Markets":(7,7,5,6)},
 2022: {"Board":(7,7,4,6),"Risk":(7,6,5,6),"RD":(6,7,4,6),"Finance":(7,6,5,6),
        "Operations":(7,6,5,6),"Workforce":NA4,"Brand":(7,6,5,6),"Markets":(6,6,6,6)},
 2023: {"Board":(6,6,5,5),"Risk":(6,6,6,5),"RD":(6,6,5,6),"Finance":(6,6,6,5),
        "Operations":(7,6,5,5),"Workforce":NA4,"Brand":(6,6,6,5),"Markets":(6,6,6,5)},
 2024: {"Board":(3,4,7,4),"Risk":(5,5,7,4),"RD":(6,6,5,6),"Finance":(5,5,7,4),
        "Operations":(6,5,6,5),"Workforce":(3,4,7,4),"Brand":(6,5,7,5),"Markets":(6,5,7,5)},
}

# ---------------- LONGFOR ----------------
S["Longfor"] = {
 2021: {"Board":(7,7,4,6),"Risk":(7,7,4,6),"RD":(7,7,4,6),"Finance":(8,7,3,6),
        "Operations":(7,7,4,6),"Workforce":NA4,"Brand":(8,7,3,6),"Markets":(7,7,4,6)},
 2022: {"Board":(6,6,5,5),"Risk":(6,7,5,6),"RD":(6,6,5,5),"Finance":(7,7,5,6),
        "Operations":(7,7,4,6),"Workforce":(6,6,6,5),"Brand":(7,7,4,6),"Markets":(6,6,6,5)},
 2023: {"Board":(6,6,4,5),"Risk":(6,6,6,5),"RD":(6,5,5,5),"Finance":(6,6,6,5),
        "Operations":(7,6,5,5),"Workforce":(5,5,6,5),"Brand":(6,6,5,5),"Markets":(6,6,5,5)},
 2024: {"Board":(6,6,5,5),"Risk":(5,5,7,4),"RD":(6,6,4,5),"Finance":(6,6,6,5),
        "Operations":(7,6,5,5),"Workforce":NA4,"Brand":(6,6,6,5),"Markets":(5,5,7,5)},
}

# ---------------- HUAWEI ----------------
S["Huawei"] = {
 2021: {"Board":(7,7,5,7),"Risk":(6,6,6,6),"RD":(8,7,6,8),"Finance":(7,7,5,6),
        "Operations":(6,6,7,6),"Workforce":(6,6,6,6),"Brand":(7,6,6,6),"Markets":(5,4,8,5)},
 2022: {"Board":(7,7,5,6),"Risk":(6,6,6,6),"RD":(8,7,6,8),"Finance":(6,6,6,6),
        "Operations":(6,6,7,6),"Workforce":(5,5,7,5),"Brand":(6,6,6,6),"Markets":(5,5,7,5)},
 2023: {"Board":(7,7,4,7),"Risk":(6,6,6,6),"RD":(8,8,5,8),"Finance":(7,7,4,6),
        "Operations":(7,6,5,7),"Workforce":NA4,"Brand":(7,7,4,7),"Markets":(7,6,5,7)},
 2024: {"Board":(7,7,5,7),"Risk":(5,6,7,6),"RD":(8,8,5,8),"Finance":(7,8,4,6),
        "Operations":(7,7,4,7),"Workforce":NA4,"Brand":(7,7,4,7),"Markets":(7,7,4,7)},
}

# ---------------- SMIC ----------------
S["SMIC"] = {
 2021: {"Board":(4,5,7,4),"Risk":(6,6,6,6),"RD":(6,6,6,6),"Finance":(7,7,4,6),
        "Operations":(7,7,4,6),"Workforce":(5,6,6,5),"Brand":(6,6,5,5),"Markets":(7,7,4,6)},
 2022: {"Board":(4,4,7,4),"Risk":(5,6,7,5),"RD":(6,6,6,6),"Finance":(7,7,5,6),
        "Operations":(7,7,5,6),"Workforce":(5,5,7,5),"Brand":(6,6,5,5),"Markets":(7,7,5,6)},
 2023: {"Board":(4,4,6,4),"Risk":(5,5,7,5),"RD":(6,7,5,7),"Finance":(6,6,6,5),
        "Operations":(6,6,7,5),"Workforce":(6,6,4,5),"Brand":(7,7,4,6),"Markets":(6,6,6,5)},
 2024: {"Board":NA4,"Risk":(4,5,8,5),"RD":(6,6,6,7),"Finance":(6,7,5,5),
        "Operations":(7,7,5,5),"Workforce":NA4,"Brand":(7,7,4,6),"Markets":(6,7,5,5)},
}

# ---------------- BYD ----------------
S["BYD"] = {
 2021: {"Board":(7,7,4,7),"Risk":(6,6,5,6),"RD":(8,7,4,8),"Finance":(7,7,4,6),
        "Operations":(7,7,5,6),"Workforce":NA4,"Brand":(7,6,5,6),"Markets":(7,7,4,6)},
 2022: {"Board":(7,7,4,7),"Risk":(6,6,6,6),"RD":(8,8,5,8),"Finance":(7,8,3,6),
        "Operations":(7,8,5,6),"Workforce":NA4,"Brand":(7,7,5,6),"Markets":(8,8,4,6)},
 2023: {"Board":(7,8,4,7),"Risk":(6,6,6,6),"RD":(8,8,4,8),"Finance":(8,8,3,6),
        "Operations":(8,8,4,6),"Workforce":(6,7,4,6),"Brand":(8,7,4,7),"Markets":(8,8,4,7)},
 2024: {"Board":(7,8,4,7),"Risk":(6,6,6,6),"RD":(8,8,4,8),"Finance":(7,8,4,6),
        "Operations":(8,8,5,6),"Workforce":(6,7,6,6),"Brand":(8,8,4,7),"Markets":(8,8,5,7)},
}

# ---------------- CATL ----------------
S["CATL"] = {
 2021: {"Board":(7,7,4,6),"Risk":(6,6,5,5),"RD":(8,7,4,8),"Finance":(7,7,4,6),
        "Operations":(7,8,5,6),"Workforce":(6,6,6,5),"Brand":(7,7,4,6),"Markets":(7,8,4,6)},
 2022: {"Board":(7,7,4,6),"Risk":(6,6,6,5),"RD":(8,8,4,8),"Finance":(7,7,5,6),
        "Operations":(7,8,5,6),"Workforce":(6,7,5,5),"Brand":(6,6,5,5),"Markets":(7,8,5,6)},
 2023: {"Board":(7,7,5,6),"Risk":(6,6,6,5),"RD":(8,8,4,8),"Finance":(7,7,5,6),
        "Operations":(7,8,6,6),"Workforce":(6,7,5,6),"Brand":(7,6,4,5),"Markets":(7,8,5,6)},
 2024: {"Board":(7,7,5,6),"Risk":(5,6,7,5),"RD":(8,8,4,8),"Finance":(7,7,5,6),
        "Operations":(7,8,5,6),"Workforce":NA4,"Brand":(7,7,4,6),"Markets":(7,8,5,6)},
}

entities = ["China","Evergrande","Vanke","Longfor","Huawei","SMIC","BYD","CATL"]
years = [2021,2022,2023,2024]

lines = [header]
for e in entities:
    nodes = china_nodes if e == "China" else corp_nodes
    for y in years:
        for n in nodes:
            c,k,s,a = S[e][y][n]
            lines.append(f"{e},{y},{n},{c},{k},{s},{a}")

out = "\n".join(lines) + "\n"
with open(r"C:\Users\julie\kimi_scores\Scoring\pass3.csv","w",newline="") as f:
    f.write(out)

data_rows = len(lines) - 1
na_rows = sum(1 for l in lines[1:] if l.endswith(",NA,NA,NA,NA"))
print(f"data rows: {data_rows}")
print(f"NA node-years: {na_rows} (cells: {na_rows*4})")
from collections import Counter
c = Counter()
for l in lines[1:]:
    if l.endswith(",NA,NA,NA,NA"):
        c[l.split(',')[0]] += 1
print(dict(c))
