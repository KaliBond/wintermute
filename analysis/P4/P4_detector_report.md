# P4 Pathology Detector Results
**Run date:** 2026-11-10  
**Rule:** crisis_typology_rule_v1 (sealed 2026-11-05)  
**V formula:** C + K − S (A excluded)  
**EPI cluster:** Helm, Lore  |  **ECO cluster:** Hands, Stewards, Flow  
**Gap = ECO_mean − EPI_mean | ≥+1.0 → epistemological | ≤−1.0 → economic**  

---

## verify_canonical

V recomputed from stored C/K/S before all calculations. Max delta reported per firm.

---

## Firing summary (years where |gap| ≥ 1.0)

| Entity | Role | Year | Classification | Gap |
|---|---|---|---|---|
| AT&T | control | 1996 | epistemological | 2.633 |
| AT&T | control | 1997 | epistemological | 2.900 |
| AT&T | control | 1999 | epistemological | 1.533 |
| AT&T | control | 2001 | epistemological | 2.133 |
| AT&T | control | 2002 | epistemological | 2.200 |
| Airbus | control | 2015 | epistemological | 1.767 |
| Airbus | control | 2016 | epistemological | 2.133 |
| Airbus | control | 2017 | epistemological | 1.867 |
| Airbus | control | 2018 | epistemological | 1.833 |
| Airbus | control | 2019 | epistemological | 2.733 |
| Airbus | control | 2020 | economic | -3.467 |
| Boeing | case | 2013 | epistemological | 1.633 |
| Boeing | case | 2014 | epistemological | 1.767 |
| Boeing | case | 2015 | epistemological | 2.467 |
| Boeing | case | 2016 | epistemological | 2.967 |
| Boeing | case | 2017 | epistemological | 3.833 |
| Boeing | case | 2018 | epistemological | 3.200 |
| Boeing | case | 2019 | epistemological | 2.600 |
| Boeing | case | 2020 | economic | -2.233 |
| Quest Diagnostics | control | 2008 | epistemological | 1.033 |
| Quest Diagnostics | control | 2010 | epistemological | 1.233 |
| Quest Diagnostics | control | 2011 | epistemological | 1.200 |
| Quest Diagnostics | control | 2012 | epistemological | 1.633 |
| Quest Diagnostics | control | 2014 | economic | -1.867 |
| Theranos | case | 2008 | economic | -1.733 |
| Theranos | case | 2009 | economic | -1.600 |
| Theranos | case | 2012 | epistemological | 1.200 |
| Theranos | case | 2013 | epistemological | 2.800 |
| Theranos | case | 2014 | epistemological | 3.433 |
| Theranos | case | 2015 | epistemological | 2.367 |
| WorldCom | case | 1998 | epistemological | 2.133 |
| WorldCom | case | 1999 | epistemological | 1.900 |
| WorldCom | case | 2000 | epistemological | 1.467 |
| WorldCom | case | 2001 | epistemological | 1.267 |

---

## Case vs control comparison

### Boeing (case) vs Airbus (control)

| Year | Boeing gap | Boeing class | Airbus gap | Airbus class |
|---|---|---|---|---|
| 2013 | 1.633 | epistemological | 0.033 | unclassified |
| 2014 | 1.767 | epistemological | 0.967 | unclassified |
| 2015 | 2.467 | epistemological | 1.767 | epistemological |
| 2016 | 2.967 | epistemological | 2.133 | epistemological |
| 2017 | 3.833 | epistemological | 1.867 | epistemological |
| 2018 | 3.200 | epistemological | 1.833 | epistemological |
| 2019 | 2.600 | epistemological | 2.733 | epistemological |
| 2020 | -2.233 | economic | -3.467 | economic |
| 2021 | -0.667 | unclassified | -0.267 | unclassified |
| 2022 | 0.000 | unclassified | -0.100 | unclassified |

### Theranos (case) vs Quest Diagnostics (control)

| Year | Theranos gap | Theranos class | Quest Diagnostics gap | Quest Diagnostics class |
|---|---|---|---|---|
| 2008 | -1.733 | economic | 1.033 | epistemological |
| 2009 | -1.600 | economic | 0.467 | unclassified |
| 2010 | -0.500 | unclassified | 1.233 | epistemological |
| 2011 | 0.367 | unclassified | 1.200 | epistemological |
| 2012 | 1.200 | epistemological | 1.633 | epistemological |
| 2013 | 2.800 | epistemological | -0.200 | unclassified |
| 2014 | 3.433 | epistemological | -1.867 | economic |
| 2015 | 2.367 | epistemological | -0.367 | unclassified |
| 2016 | 0.533 | unclassified | 0.433 | unclassified |
| 2017 | -0.500 | unclassified | -0.267 | unclassified |
| 2018 | 0.000 | unclassified | -0.667 | unclassified |

### WorldCom (case) vs AT&T (control)

| Year | WorldCom gap | WorldCom class | AT&T gap | AT&T class |
|---|---|---|---|---|
| 1996 | 0.700 | unclassified | 2.633 | epistemological |
| 1997 | 0.767 | unclassified | 2.900 | epistemological |
| 1998 | 2.133 | epistemological | 0.300 | unclassified |
| 1999 | 1.900 | epistemological | 1.533 | epistemological |
| 2000 | 1.467 | epistemological | 0.233 | unclassified |
| 2001 | 1.267 | epistemological | 2.133 | epistemological |
| 2002 | 0.967 | unclassified | 2.200 | epistemological |
| 2003 | -0.167 | unclassified | 0.133 | unclassified |
| 2004 | 0.300 | unclassified | -0.267 | unclassified |

---

## Notes

- Discovery register is separate from any prospective watchlist (protocol §P4)
- False positives in controls (if any) are reported, not suppressed
- Circular-shift null and simpler baselines (Stress-change-alone, first differences of V) to be run in next step

*Source files: Boeing_ENS.csv, Airbus_ENS.csv, Theranos_ENS.csv, QuestDiagnostics_ENS.csv, WorldCom_ENS.csv, ATT_ENS.csv*  