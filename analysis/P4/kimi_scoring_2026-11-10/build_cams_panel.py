"""Expand CAMS era score tables into an annual panel (1990-2026).

CAMS Universal Instruction Set v1.4 — blind raw scoring, single scorer.
Each entity is defined by era segments: (start_year, end_year, {node: (C,K,S,A)}).
Scores are integers 1-10 or "NA". Nodes are emitted in canonical order:
Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.

Score_Status per cell:
  New_Evidence   — first year of series, or a year where a score moved on
                   evidence examined this run
  Reaffirmed     — year-specific evidence examined this run, score held
  Carried_Forward— no year-specific evidence examined this run, score held
  NA             — cell is NA (evidence too thin for that node-year)
"""

import csv
import os

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]

NA = "NA"

ERAS = {
    "Boeing": [
        (1990, 1996, {"Helm": (7,6,4,7), "Shield": (6,6,4,5), "Lore": (8,7,3,6), "Stewards": (7,7,4,6),
                      "Craft": (8,8,3,8), "Hands": (7,7,5,6), "Archive": (8,8,3,8), "Flow": (7,7,4,6)}),
        (1997, 1999, {"Helm": (5,5,6,5), "Shield": (6,6,5,5), "Lore": (6,6,4,5), "Stewards": (6,6,5,5),
                      "Craft": (6,6,6,6), "Hands": (6,6,6,5), "Archive": (7,7,4,7), "Flow": (5,5,7,5)}),
        (2000, 2002, {"Helm": (5,5,7,5), "Shield": (6,6,5,5), "Lore": (6,6,5,5), "Stewards": (5,5,7,5),
                      "Craft": (6,6,6,6), "Hands": (5,5,7,5), "Archive": (7,7,4,7), "Flow": (5,5,7,5)}),
        (2003, 2004, {"Helm": (4,4,7,4), "Shield": (4,4,6,4), "Lore": (4,5,6,4), "Stewards": (5,5,6,5),
                      "Craft": (5,6,6,5), "Hands": (5,5,6,5), "Archive": (6,6,4,6), "Flow": (5,5,6,5)}),
        (2005, 2007, {"Helm": (5,6,6,5), "Shield": (5,5,5,5), "Lore": (5,6,5,5), "Stewards": (6,6,6,6),
                      "Craft": (5,6,7,5), "Hands": (5,5,7,5), "Archive": (6,6,5,6), "Flow": (4,5,7,5)}),
        (2008, 2010, {"Helm": (6,6,7,6), "Shield": (5,5,5,5), "Lore": (5,6,6,5), "Stewards": (6,6,7,6),
                      "Craft": (5,6,7,5), "Hands": (4,5,8,5), "Archive": (5,5,6,6), "Flow": (4,5,8,5)}),
        (2011, 2012, {"Helm": (6,7,6,6), "Shield": (5,5,5,5), "Lore": (6,6,5,6), "Stewards": (6,7,6,6),
                      "Craft": (6,7,6,6), "Hands": (6,6,6,5), "Archive": (5,5,6,6), "Flow": (6,6,6,5)}),
        (2013, 2013, {"Helm": (6,7,6,6), "Shield": (5,5,7,5), "Lore": (5,6,6,5), "Stewards": (6,7,6,6),
                      "Craft": (5,6,7,5), "Hands": (6,6,6,5), "Archive": (5,5,6,6), "Flow": (6,6,6,5)}),
        (2014, 2016, {"Helm": (7,7,5,6), "Shield": (6,6,4,6), "Lore": (5,6,5,5), "Stewards": (7,7,5,5),
                      "Craft": (5,6,6,5), "Hands": (6,6,6,5), "Archive": (5,5,5,5), "Flow": (7,7,5,6)}),
        (2017, 2018, {"Helm": (7,7,5,6), "Shield": (6,6,5,6), "Lore": (4,5,5,4), "Stewards": (7,7,5,5),
                      "Craft": (4,5,7,4), "Hands": (6,6,6,5), "Archive": (4,4,5,4), "Flow": (7,7,6,6)}),
        (2019, 2019, {"Helm": (3,3,9,3), "Shield": (3,2,9,3), "Lore": (2,2,9,2), "Stewards": (3,2,9,3),
                      "Craft": (3,3,8,3), "Hands": (4,4,8,4), "Archive": (3,2,8,3), "Flow": (2,2,9,2)}),
        (2020, 2020, {"Helm": (4,4,9,4), "Shield": (3,2,8,3), "Lore": (3,3,8,3), "Stewards": (3,3,9,3),
                      "Craft": (3,4,8,3), "Hands": (4,4,8,4), "Archive": (3,3,7,3), "Flow": (3,3,9,3)}),
        (2021, 2022, {"Helm": (5,5,7,4), "Shield": (4,3,7,4), "Lore": (3,3,7,3), "Stewards": (4,3,8,4),
                      "Craft": (4,4,7,4), "Hands": (4,4,7,4), "Archive": (3,3,7,3), "Flow": (4,4,8,4)}),
        (2023, 2023, {"Helm": (5,5,7,4), "Shield": (4,4,6,4), "Lore": (4,4,6,4), "Stewards": (4,4,7,4),
                      "Craft": (4,5,6,4), "Hands": (5,5,6,4), "Archive": (4,4,6,4), "Flow": (5,5,7,4)}),
        (2024, 2024, {"Helm": (3,3,9,3), "Shield": (3,2,9,3), "Lore": (2,2,9,2), "Stewards": (3,2,9,3),
                      "Craft": (3,3,9,3), "Hands": (2,3,9,3), "Archive": (2,2,8,2), "Flow": (2,2,9,2)}),
        (2025, 2025, {"Helm": (5,5,7,5), "Shield": (4,4,6,4), "Lore": (4,4,6,4), "Stewards": (4,3,7,4),
                      "Craft": (4,4,7,4), "Hands": (4,4,7,4), "Archive": (3,3,6,3), "Flow": (4,4,7,4)}),
        (2026, 2026, {"Helm": (5,6,6,5), "Shield": (4,4,5,4), "Lore": (4,5,5,4), "Stewards": (4,4,6,4),
                      "Craft": (5,5,6,5), "Hands": (5,5,6,4), "Archive": (4,4,5,4), "Flow": (5,5,6,5)}),
    ],
    "Lockheed Martin": [
        (1995, 1997, {"Helm": (7,6,5,7), "Shield": (6,6,4,6), "Lore": (6,6,4,6), "Stewards": (6,6,5,6),
                      "Craft": (7,7,4,7), "Hands": (6,6,5,6), "Archive": (7,7,3,7), "Flow": (6,6,5,6)}),
        (1998, 2000, {"Helm": (6,5,6,6), "Shield": (6,6,5,6), "Lore": (6,6,4,6), "Stewards": (5,5,6,6),
                      "Craft": (7,7,5,7), "Hands": (6,6,5,6), "Archive": (7,7,3,7), "Flow": (6,6,6,6)}),
        (2001, 2005, {"Helm": (7,7,5,7), "Shield": (7,7,5,7), "Lore": (7,7,4,6), "Stewards": (7,7,5,7),
                      "Craft": (8,8,5,7), "Hands": (7,7,4,6), "Archive": (7,7,3,7), "Flow": (7,7,5,6)}),
        (2006, 2011, {"Helm": (7,7,5,7), "Shield": (7,7,5,7), "Lore": (7,7,4,6), "Stewards": (7,7,5,7),
                      "Craft": (8,8,5,7), "Hands": (7,7,4,6), "Archive": (8,8,3,8), "Flow": (7,7,5,7)}),
        (2012, 2015, {"Helm": (7,7,5,7), "Shield": (7,7,5,7), "Lore": (7,7,4,6), "Stewards": (7,7,6,7),
                      "Craft": (8,7,6,7), "Hands": (7,7,5,6), "Archive": (8,8,3,8), "Flow": (7,7,6,7)}),
        (2016, 2019, {"Helm": (7,7,5,7), "Shield": (7,7,5,7), "Lore": (7,7,4,6), "Stewards": (7,7,5,7),
                      "Craft": (8,8,5,8), "Hands": (7,7,4,6), "Archive": (8,8,3,8), "Flow": (7,7,5,7)}),
        (2020, 2022, {"Helm": (7,7,6,7), "Shield": (7,7,6,7), "Lore": (7,7,5,6), "Stewards": (7,7,6,7),
                      "Craft": (7,7,6,7), "Hands": (7,7,6,6), "Archive": (8,8,4,8), "Flow": (6,6,7,6)}),
        (2023, 2024, {"Helm": (7,7,6,7), "Shield": (7,7,6,7), "Lore": (7,7,5,6), "Stewards": (7,7,6,7),
                      "Craft": (7,6,7,7), "Hands": (7,7,6,6), "Archive": (8,8,4,8), "Flow": (6,6,7,6)}),
        (2025, 2026, {"Helm": (7,7,6,7), "Shield": (7,7,5,7), "Lore": (7,7,4,6), "Stewards": (7,7,6,7),
                      "Craft": (7,7,6,7), "Hands": (7,7,5,6), "Archive": (8,8,4,8), "Flow": (6,6,6,6)}),
    ],
    "Theranos": [
        (2003, 2006, {"Helm": (6,4,3,5), "Shield": (NA,NA,NA,NA), "Lore": (NA,NA,NA,NA), "Stewards": (5,4,3,4),
                      "Craft": (3,2,5,3), "Hands": (4,4,3,4), "Archive": (3,2,3,3), "Flow": (NA,NA,NA,NA)}),
        (2007, 2009, {"Helm": (7,4,4,4), "Shield": (5,4,4,4), "Lore": (5,4,3,4), "Stewards": (4,4,4,4),
                      "Craft": (3,2,5,3), "Hands": (4,4,4,4), "Archive": (2,2,3,2), "Flow": (3,3,3,3)}),
        (2010, 2013, {"Helm": (7,4,5,4), "Shield": (6,5,4,5), "Lore": (6,5,4,5), "Stewards": (3,3,5,3),
                      "Craft": (2,2,7,2), "Hands": (4,3,6,3), "Archive": (2,2,5,2), "Flow": (2,2,7,2)}),
        (2014, 2014, {"Helm": (7,4,4,4), "Shield": (6,4,4,5), "Lore": (8,5,3,5), "Stewards": (3,3,5,3),
                      "Craft": (2,2,7,2), "Hands": (4,3,6,3), "Archive": (2,2,4,2), "Flow": (2,2,7,2)}),
        (2015, 2015, {"Helm": (6,4,8,4), "Shield": (4,3,8,4), "Lore": (3,3,9,3), "Stewards": (3,2,8,3),
                      "Craft": (2,2,9,2), "Hands": (3,3,8,3), "Archive": (2,2,7,2), "Flow": (2,2,9,2)}),
        (2016, 2016, {"Helm": (4,3,9,3), "Shield": (2,2,9,2), "Lore": (2,2,9,2), "Stewards": (2,2,9,2),
                      "Craft": (1,1,9,1), "Hands": (2,2,8,2), "Archive": (2,2,8,2), "Flow": (1,1,9,1)}),
        (2017, 2017, {"Helm": (3,3,8,3), "Shield": (2,2,7,2), "Lore": (2,2,7,2), "Stewards": (2,2,8,2),
                      "Craft": (1,1,7,1), "Hands": (2,2,7,2), "Archive": (2,2,6,2), "Flow": (1,1,7,1)}),
        (2018, 2018, {"Helm": (1,1,9,1), "Shield": (1,1,9,1), "Lore": (1,1,9,1), "Stewards": (1,1,9,1),
                      "Craft": (1,1,9,1), "Hands": (1,1,9,1), "Archive": (1,1,9,1), "Flow": (1,1,9,1)}),
    ],
    "Illumina": [
        (1998, 2001, {"Helm": (6,4,4,5), "Shield": (5,4,3,4), "Lore": (NA,NA,NA,NA), "Stewards": (5,4,5,4),
                      "Craft": (5,4,4,5), "Hands": (NA,NA,NA,NA), "Archive": (NA,NA,NA,NA), "Flow": (4,4,4,4)}),
        (2002, 2006, {"Helm": (6,5,5,5), "Shield": (5,5,4,4), "Lore": (5,5,4,5), "Stewards": (5,5,6,5),
                      "Craft": (6,5,5,6), "Hands": (5,5,5,5), "Archive": (5,5,4,5), "Flow": (5,5,5,5)}),
        (2007, 2009, {"Helm": (7,6,5,6), "Shield": (6,5,4,5), "Lore": (6,6,4,6), "Stewards": (6,6,5,6),
                      "Craft": (7,7,5,7), "Hands": (6,6,5,5), "Archive": (6,6,4,6), "Flow": (6,6,5,6)}),
        (2010, 2012, {"Helm": (7,7,5,7), "Shield": (7,6,6,6), "Lore": (7,7,4,6), "Stewards": (7,7,5,7),
                      "Craft": (8,8,5,8), "Hands": (7,7,4,6), "Archive": (7,7,4,7), "Flow": (7,7,5,7)}),
        (2013, 2016, {"Helm": (7,7,5,7), "Shield": (7,6,4,6), "Lore": (7,7,4,7), "Stewards": (7,7,4,7),
                      "Craft": (9,9,4,8), "Hands": (7,7,4,6), "Archive": (7,7,4,7), "Flow": (8,8,4,7)}),
        (2017, 2019, {"Helm": (7,7,4,7), "Shield": (7,6,4,6), "Lore": (7,7,4,7), "Stewards": (7,7,4,7),
                      "Craft": (8,9,4,8), "Hands": (7,7,4,6), "Archive": (7,7,4,7), "Flow": (8,8,4,7)}),
        (2020, 2021, {"Helm": (5,5,7,5), "Shield": (4,4,7,4), "Lore": (6,6,5,6), "Stewards": (4,4,8,4),
                      "Craft": (8,8,5,8), "Hands": (6,6,5,6), "Archive": (7,7,4,7), "Flow": (7,7,5,7)}),
        (2022, 2023, {"Helm": (4,4,8,4), "Shield": (3,3,8,3), "Lore": (5,5,7,5), "Stewards": (4,3,8,4),
                      "Craft": (7,8,5,7), "Hands": (6,6,6,6), "Archive": (7,7,5,7), "Flow": (7,7,6,7)}),
        (2024, 2024, {"Helm": (6,6,6,6), "Shield": (5,5,5,5), "Lore": (6,6,5,6), "Stewards": (5,5,6,5),
                      "Craft": (7,8,5,8), "Hands": (6,6,5,6), "Archive": (7,7,4,7), "Flow": (7,7,5,7)}),
        (2025, 2025, {"Helm": (6,6,6,6), "Shield": (5,5,6,5), "Lore": (6,6,5,6), "Stewards": (5,5,7,5),
                      "Craft": (7,7,6,7), "Hands": (6,6,6,6), "Archive": (7,7,4,7), "Flow": (5,6,7,6)}),
        (2026, 2026, {"Helm": (6,6,6,6), "Shield": (5,5,5,5), "Lore": (6,6,5,6), "Stewards": (5,5,6,5),
                      "Craft": (7,7,6,7), "Hands": (6,6,6,6), "Archive": (7,7,4,7), "Flow": (6,6,6,6)}),
    ],
    "WorldCom": [
        (1995, 1996, {"Helm": (6,5,5,5), "Shield": (5,5,3,5), "Lore": (6,6,3,5), "Stewards": (5,5,5,4),
                      "Craft": (6,6,4,5), "Hands": (5,5,4,5), "Archive": (4,4,3,4), "Flow": (6,6,4,5)}),
        (1997, 1999, {"Helm": (7,6,4,5), "Shield": (5,5,4,5), "Lore": (7,6,4,6), "Stewards": (4,4,6,4),
                      "Craft": (6,6,4,5), "Hands": (5,5,5,5), "Archive": (4,3,4,4), "Flow": (7,7,4,6)}),
        (2000, 2000, {"Helm": (5,5,6,5), "Shield": (5,4,5,5), "Lore": (5,5,6,5), "Stewards": (3,3,8,3),
                      "Craft": (6,6,5,5), "Hands": (5,5,6,5), "Archive": (3,3,5,3), "Flow": (7,6,6,6)}),
        (2001, 2001, {"Helm": (4,4,7,4), "Shield": (4,4,6,4), "Lore": (4,4,6,4), "Stewards": (2,2,9,2),
                      "Craft": (6,5,6,5), "Hands": (4,4,7,4), "Archive": (2,2,6,2), "Flow": (6,6,7,5)}),
        (2002, 2002, {"Helm": (1,1,9,1), "Shield": (1,1,9,1), "Lore": (1,1,9,1), "Stewards": (1,1,10,1),
                      "Craft": (5,5,6,5), "Hands": (3,3,9,3), "Archive": (1,1,8,1), "Flow": (5,5,8,5)}),
    ],
    "Verizon": [
        (2000, 2001, {"Helm": (6,6,6,6), "Shield": (6,6,4,6), "Lore": (6,6,4,6), "Stewards": (6,6,6,6),
                      "Craft": (6,6,4,6), "Hands": (4,5,7,5), "Archive": (6,6,4,6), "Flow": (6,6,5,6)}),
        (2002, 2004, {"Helm": (7,7,5,7), "Shield": (6,6,4,6), "Lore": (6,6,4,6), "Stewards": (6,6,6,6),
                      "Craft": (6,6,5,6), "Hands": (6,6,5,5), "Archive": (6,6,4,6), "Flow": (6,6,5,6)}),
        (2005, 2007, {"Helm": (7,7,5,7), "Shield": (6,6,4,6), "Lore": (7,7,4,6), "Stewards": (6,6,6,6),
                      "Craft": (7,7,5,7), "Hands": (6,6,5,5), "Archive": (6,6,4,6), "Flow": (7,7,5,6)}),
        (2008, 2010, {"Helm": (7,7,5,7), "Shield": (6,6,4,6), "Lore": (7,7,4,6), "Stewards": (6,6,6,6),
                      "Craft": (7,7,5,7), "Hands": (6,6,5,5), "Archive": (6,6,4,6), "Flow": (7,7,5,7)}),
        (2011, 2013, {"Helm": (7,7,5,7), "Shield": (6,6,5,6), "Lore": (6,6,5,6), "Stewards": (6,6,6,6),
                      "Craft": (7,7,5,7), "Hands": (5,5,6,5), "Archive": (6,6,4,6), "Flow": (7,7,6,7)}),
        (2014, 2016, {"Helm": (7,6,5,6), "Shield": (6,6,5,6), "Lore": (6,6,5,6), "Stewards": (5,5,6,5),
                      "Craft": (7,7,5,7), "Hands": (4,5,8,5), "Archive": (6,6,4,6), "Flow": (7,7,5,7)}),
        (2017, 2019, {"Helm": (7,6,5,7), "Shield": (6,6,4,6), "Lore": (6,6,4,6), "Stewards": (4,4,6,5),
                      "Craft": (7,7,5,7), "Hands": (6,5,5,5), "Archive": (6,6,4,6), "Flow": (7,7,5,7)}),
        (2020, 2021, {"Helm": (7,7,6,7), "Shield": (6,6,5,6), "Lore": (6,6,5,6), "Stewards": (5,5,7,6),
                      "Craft": (7,7,6,7), "Hands": (6,6,6,6), "Archive": (6,6,4,6), "Flow": (7,7,7,7)}),
        (2022, 2022, {"Helm": (7,6,6,6), "Shield": (6,6,5,6), "Lore": (6,6,5,6), "Stewards": (5,5,7,5),
                      "Craft": (7,7,6,7), "Hands": (6,6,6,6), "Archive": (6,6,4,6), "Flow": (6,6,6,6)}),
        (2023, 2024, {"Helm": (7,6,6,6), "Shield": (5,5,6,5), "Lore": (6,6,5,6), "Stewards": (5,5,7,5),
                      "Craft": (7,7,6,7), "Hands": (6,5,6,6), "Archive": (6,6,4,6), "Flow": (6,6,6,6)}),
        (2025, 2025, {"Helm": (6,6,7,6), "Shield": (5,5,5,5), "Lore": (6,6,5,6), "Stewards": (5,5,7,5),
                      "Craft": (7,7,6,7), "Hands": (5,5,7,5), "Archive": (6,6,4,6), "Flow": (6,6,6,6)}),
        (2026, 2026, {"Helm": (6,6,7,6), "Shield": (5,5,5,5), "Lore": (6,6,5,6), "Stewards": (5,5,7,5),
                      "Craft": (7,7,6,7), "Hands": (5,4,7,5), "Archive": (6,6,4,6), "Flow": (6,6,6,6)}),
    ],
}

# Years (interior to an era) where year-specific evidence was examined this run
# and scores were held -> Reaffirmed.
REAFFIRMED = {
    "Boeing": {2005, 2012, 2022},
    "Lockheed Martin": {2011, 2014, 2019, 2022},
    "Theranos": {2013},
    "Illumina": {2012, 2021},
    "WorldCom": {1998},
    "Verizon": {2012, 2015, 2021},
}

METRICS = ["Coherence", "Capacity", "Stress", "Abstraction"]


def validate():
    """Check era table integrity before expansion."""
    for entity, eras in ERAS.items():
        eras_sorted = sorted(eras, key=lambda e: e[0])
        for i, (s, e, nodes) in enumerate(eras_sorted):
            assert s <= e, f"{entity}: era {s}-{e} inverted"
            if i + 1 < len(eras_sorted):
                nxt = eras_sorted[i + 1][0]
                assert e + 1 == nxt, f"{entity}: gap/overlap between {e} and {nxt}"
            for node in NODES:
                assert node in eras[i][2], f"{entity} {s}-{e}: missing node {node}"
                for v in eras[i][2][node]:
                    assert v == NA or (isinstance(v, int) and 1 <= v <= 10), (
                        f"{entity} {s}-{e} {node}: bad value {v}")


def expand():
    rows, status_rows = [], []
    for entity, eras in ERAS.items():
        years = {}
        for s, e, nodes in eras:
            for y in range(s, e + 1):
                years[y] = nodes
        era_starts = {s for s, _, _ in eras}
        first_year = min(years)
        prev = None
        for y in sorted(years):
            nodes = years[y]
            for node in NODES:
                vals = nodes[node]
                changed = prev is not None and prev[node] != vals
                if y == first_year or changed:
                    status = "New_Evidence"
                elif y in era_starts or y in REAFFIRMED.get(entity, set()):
                    status = "Reaffirmed"
                else:
                    status = "Carried_Forward"
                rows.append([entity, y, node] + [str(v) for v in vals])
                status_rows.append([entity, y, node] + [
                    NA if v == NA else status for v in vals])
            prev = nodes
    return rows, status_rows


def main():
    validate()
    rows, status_rows = expand()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    scores_path = os.path.join(out_dir, "cams_scores_1990-2026.csv")
    status_path = os.path.join(out_dir, "cams_score_status_1990-2026.csv")
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Year", "Node"] + METRICS)
        w.writerows(rows)
    with open(status_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Year", "Node",
                    "C_status", "K_status", "S_status", "A_status"])
        w.writerows(status_rows)
    # Summary for the console
    from collections import Counter
    print(f"rows: {len(rows)}  entity-years: {len(rows)//8}")
    print("entities:", sorted({r[0] for r in rows}))
    print("status counts:", Counter(r[3] for r in status_rows))
    na_cells = sum(1 for r in rows for v in r[3:] if v == NA)
    print("NA cells:", na_cells)


if __name__ == "__main__":
    main()
