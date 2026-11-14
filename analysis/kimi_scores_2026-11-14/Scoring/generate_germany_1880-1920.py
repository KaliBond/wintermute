# generate_germany_1880-1920.py
# CAMS Universal Instruction Set v1.4 — Germany 1880-1920 panel generator.
# Produces: Germany_1880-1920_cams_v1.4.csv (328 rows) and
#           Germany_1880-1920_score-status.csv (provenance per cell).
# Scoring is probe-based, not measurement: scores are integer 1-10 on each of
# the four CAMS metrics, canonical node order per year.
# Run: python generate_germany_1880-1920.py

import csv, sys
from pathlib import Path

OUT = Path(r"C:\Users\julie\kimi_scores\Scoring")
ENTITY = "Germany"
YEARS = list(range(1880, 1921))  # 41 annual conditions, 31 December each
NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

# ---------------------------------------------------------------------------
# Era constants. Each value is (C, K, S, A) for the years in its range,
# unless overridden by a year-specific entry below.
# Era structure: E1 1880-88 Bismarck; E2 1888-1900 early Wilhelm II;
# E3 1900-13 late empire; E4 1914-16; E5 1917-18; E6 1918-19 revolution/Weimar;
# E7 1920.
# ---------------------------------------------------------------------------
SERIES = {
    "Helm": {
        (1880, 1887): (7, 7, 4, 7),   # Bismarckian planned executive
        (1888, 1888): (6, 6, 6, 6),   # Year of Three Emperors
        (1889, 1889): (6, 6, 5, 6),
        (1890, 1890): (5, 6, 7, 6),   # Bismarck dismissed March 1890
        (1891, 1896): (6, 6, 5, 6),   # Caprivi / Hohenlohe
        (1897, 1907): (7, 7, 4, 6),   # Bulow/Tirpitz system
        (1908, 1908): (5, 7, 6, 6),   # Daily Telegraph affair
        (1909, 1909): (6, 7, 5, 6),   # Bethmann Hollweg
        (1910, 1911): (6, 6, 5, 6),
        (1912, 1912): (5, 6, 6, 6),   # SPD largest Reichstag party
        (1913, 1913): (4, 6, 7, 6),   # Zabern affair; Army Bill; deadlock
        (1914, 1914): (6, 7, 6, 5),   # Burgfrieden
        (1915, 1915): (6, 6, 7, 5),
        (1916, 1916): (5, 5, 8, 3),   # Hindenburg/Ludendorff silent dictatorship
        (1917, 1917): (4, 4, 8, 3),   # peace resolution ignored
        (1918, 1918): (3, 4, 9, 2),   # collapse; abdication; S9 rupture
        (1919, 1919): (4, 4, 8, 4),   # Ebert; National Assembly; Weimar constitution
        (1920, 1920): (4, 4, 7, 4),   # Kapp exposes thin consent
    },
    "Shield": {
        (1880, 1887): (7, 6, 4, 6),   # General Staff planning culture
        (1888, 1896): (7, 6, 5, 6),
        (1897, 1907): (7, 6, 4, 6),   # Tirpitz naval build begins 1898
        (1908, 1913): (7, 6, 4, 6),
        (1914, 1914): (7, 7, 6, 6),   # mobilisation executes; Marne
        (1915, 1915): (7, 7, 7, 6),   # blockade bite; Verdun preparation
        (1916, 1916): (7, 6, 8, 5),   # Verdun; the Somme
        (1917, 1917): (6, 6, 8, 5),
        (1918, 1918): (5, 4, 9, 4),   # spring offensive fails; collapse
        (1919, 1919): (4, 3, 8, 2),   # Freikorps ecology; demobilisation
        (1920, 1920): (4, 3, 7, 2),   # Seeckt: "Reichswehr does not fire on Reichswehr"
    },
    "Lore": {
        (1880, 1887): (5, 5, 4, 5),
        (1888, 1896): (6, 5, 5, 5),   # Wilhelm II cult of personality
        (1897, 1907): (6, 6, 4, 6),   # Navy League mass nationalism
        (1908, 1908): (5, 6, 6, 6),   # Telegraph affair dents Kaiser myth
        (1909, 1912): (6, 6, 5, 6),
        (1913, 1913): (5, 6, 6, 6),   # Zabern: nation vs. military
        (1914, 1914): (8, 7, 5, 6),   # Burgfrieden
        (1915, 1915): (7, 7, 6, 6),
        (1916, 1916): (6, 6, 7, 5),
        (1917, 1917): (5, 6, 8, 5),   # USPD split; Fatherland Party vs Peace Resolution
        (1918, 1918): (3, 5, 9, 4),   # symbolic rupture; stab-in-back myth forms
        (1919, 1919): (4, 5, 8, 4),   # Weimar founding; contested symbols
        (1920, 1920): (3, 5, 8, 4),   # Kapp flags vs red flags
    },
    "Stewards": {
        (1880, 1888): (6, 6, 4, 5),   # Reichsbank (1876), gold mark; insurance 1883/84
        (1889, 1899): (6, 6, 5, 5),   # 1889 old-age pensions
        (1900, 1913): (7, 7, 4, 6),   # mature fiscal state; 1911 Imperial Insurance Code
        (1914, 1914): (6, 6, 6, 5),   # war-bond finance (hostile-witness anchor)
        (1915, 1915): (6, 6, 7, 5),
        (1916, 1916): (5, 5, 8, 3),   # Turnip Winter; War Food Office 22 May 1916
        (1917, 1917): (4, 4, 8, 3),   # rations 1,560 -> 1,000 cal/day
        (1918, 1918): (3, 3, 9, 3),   # blockade; collapse of provisioning
        (1919, 1919): (4, 3, 8, 4),   # blockade to July 1919; demobilisation burden
        (1920, 1920): (4, 4, 7, 4),   # Kapp general strike; reparations pressure
    },
    "Craft": {
        (1880, 1888): (6, 6, 5, 6),   # dyes; synthetic chemistry
        (1889, 1899): (7, 7, 4, 7),   # science-industry fusion; electricals
        (1900, 1913): (7, 7, 4, 7),   # KWI 1911; Haber-Bosch 1913
        (1914, 1914): (7, 7, 6, 6),
        (1915, 1915): (7, 6, 7, 6),   # blockade-driven substitution chemistry
        (1916, 1917): (6, 5, 8, 5),   # Hindenburg Programme: skilled men to the front
        (1918, 1918): (5, 4, 8, 5),   # material exhaustion; January strike hits plants
        (1919, 1919): (5, 5, 7, 5),   # demobilisation; conversion struggle
        (1920, 1920): (6, 6, 6, 6),   # recovery; Kapp disruption brief
    },
    "Hands": {
        (1880, 1889): (5, 5, 5, 4),   # labour movement rising under Anti-Socialist Law
        (1890, 1909): (6, 6, 4, 5),   # Law lapses 1890; SPD/union subculture grows
        (1910, 1913): (6, 6, 5, 5),
        (1914, 1914): (7, 7, 5, 5),   # Burgfrieden; unions trade peace for recognition
        (1915, 1915): (7, 7, 6, 5),
        (1916, 1916): (6, 6, 7, 4),   # Auxiliary Service Law: compulsion enters
        (1917, 1917): (5, 6, 8, 4),   # April 1917 strikes
        (1918, 1918): (3, 6, 9, 4),   # January strike ~1m; Kiel mutiny
        (1919, 1919): (4, 5, 8, 4),   # demob ~6.5m; strike wave (figures medium authority)
        (1920, 1920): (4, 7, 7, 5),   # Kapp: ~12m general strike; mobilisation shown
    },
    "Archive": {
        (1880, 1913): (7, 6, 4, 7),   # statistical office; research universities
        (1914, 1914): (7, 6, 5, 7),
        (1915, 1915): (7, 6, 6, 7),
        (1916, 1918): (6, 6, 6, 6),   # science mobilised under strain
        (1919, 1920): (7, 6, 5, 6),   # constitution drafting expertise; system intact
    },
    "Flow": {
        (1880, 1888): (6, 6, 5, 5),   # dense rail; universal banks; cartels
        (1889, 1913): (7, 7, 4, 6),   # gold standard era circulation
        (1914, 1914): (6, 6, 7, 5),   # blockade; shipping insurance crisis
        (1915, 1915): (6, 5, 8, 5),
        (1916, 1916): (5, 4, 8, 4),   # Turnip Winter: circulation failure
        (1917, 1917): (4, 4, 8, 4),
        (1918, 1918): (3, 3, 9, 3),   # systemic circulation breakdown
        (1919, 1919): (4, 4, 8, 4),   # blockade lifted July 1919
        (1920, 1920): (5, 5, 6, 5),   # recovery; Ruhr fighting disrupts
    },
}

# Years where a named fact confirmed a held score (Reaffirmed) per node.
# Everything else that neither changes nor is listed here is Carried_Forward.
REAFFIRM = {
    "Shield": {1898, 1900, 1906, 1912},   # naval laws; 1912 army bill
    "Craft":  {1911, 1913},               # Kaiser Wilhelm Institutes; Haber-Bosch
    "Archive": {1911},                    # KWI founding
    "Stewards": {1889, 1911},             # old-age pensions; Imperial Insurance Code
    "Lore":   {1897},                     # Navy League founded
}

def series_for(node):
    out = {}
    for (a, b), vals in SERIES[node].items():
        for y in range(a, b + 1):
            assert y not in out, f"overlap {node} {y}"
            out[y] = vals
    missing = [y for y in YEARS if y not in out]
    assert not missing, f"{node} missing years {missing}"
    return out

# --- write main CSV ---------------------------------------------------------
main_path = OUT / "Germany_1880-1920_cams_v1.4.csv"
rows = []
for y in YEARS:
    for n in NODES:
        c, k, s, a = series_for(n)[y]
        rows.append([ENTITY, y, n, c, k, s, a])
with main_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"])
    w.writerows(rows)

# --- write score-status CSV -------------------------------------------------
status_path = OUT / "Germany_1880-1920_score-status.csv"
with status_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "C_status", "K_status", "S_status", "A_status"])
    for n in NODES:
        ser = series_for(n)
        for y in YEARS:
            st = []
            for i in range(4):
                if y == YEARS[0]:
                    st.append("New_Evidence")
                elif ser[y][i] != ser[y - 1][i]:
                    st.append("New_Evidence")
                elif y in REAFFIRM.get(n, set()):
                    st.append("Reaffirmed")
                else:
                    st.append("Carried_Forward")
            w.writerow([ENTITY, y, n, *st])

# --- validation -------------------------------------------------------------
errors = []
assert len(rows) == 41 * 8 == 328, len(rows)
for r in rows:
    for v in r[3:]:
        if not (isinstance(v, int) and 1 <= v <= 10):
            errors.append(f"non-integer/out-of-range {r}")
for y in YEARS:  # canonical order per year
    if [r[2] for r in rows if r[1] == y] != NODES:
        errors.append(f"node order broken {y}")

low, high, moves = [], [], []
for n in NODES:
    ser = series_for(n)
    for y in YEARS:
        c, k, s, a = ser[y]
        for i, metric in enumerate(["C", "K", "S", "A"]):
            v = (c, k, s, a)[i]
            if v < 5:
                low.append(f"{n} {y} {metric}{v}")
            if v > 8:
                high.append(f"{n} {y} {metric}{v}")
            if y > YEARS[0]:
                pv = ser[y - 1][i]
                if abs(v - pv) >= 2:
                    moves.append(f"{n} {metric} {y-1}->{y} {pv}->{v}")

carried = sum(1 for n in NODES for y in YEARS for i in range(4)
              if y != YEARS[0] and series_for(n)[y][i] == series_for(n)[y-1][i])
total_cf_slots = 40 * 8 * 4
print(f"rows={len(rows)}  validation_errors={len(errors)}")
print(f"scores <5 ({len(low)}):", "; ".join(low))
print(f"scores >8 ({len(high)}):", "; ".join(high) if high else "none")
print(f"moves >=2 ({len(moves)}):")
for m in moves:
    print("  ", m)
print(f"carried-forward cells: {carried}/{total_cf_slots} = {carried/total_cf_slots:.1%}")
if errors:
    print("ERRORS:", errors)
    sys.exit(1)
print("OK")
