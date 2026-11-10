# P4 Pathology Detector — 9-Firm Results
**Run date:** 2026-11-10  
**Rule:** crisis_typology_rule_v1 (sealed 2026-11-05)  
**V formula:** C + K − S (A excluded)  
**EPI cluster:** Helm, Lore  |  **ECO cluster:** Hands, Stewards, Flow  
**Gap = ECO_mean − EPI_mean | ≥+1.0 → epistemological | ≤−1.0 → economic**  

**Secondary controls are a sensitivity test** — framed as side test given primary detector results (controls fire nearly as often as cases under |gap| ≥ 1.0 threshold).  

---

## 9-Firm Summary

| Entity | Role | Tier | Pair | Years | Fires | Fire% | Epi | Eco | Peak Gap | Peak Year |
|---|---|---|---|---|---|---|---|---|---|---|
| Boeing | case | primary | boeing | 10 | 8 | 80.0% | 7 | 1 | 3.833 | 2017 |
| Airbus | control-primary | primary | boeing | 10 | 6 | 60.0% | 5 | 1 | -3.467 | 2020 |
| Theranos | case | primary | theranos | 11 | 6 | 54.5% | 4 | 2 | 3.433 | 2014 |
| Quest Diagnostics | control-primary | primary | theranos | 11 | 5 | 45.5% | 4 | 1 | -1.867 | 2014 |
| WorldCom | case | primary | worldcom | 9 | 4 | 44.4% | 4 | 0 | 2.133 | 1998 |
| AT&T | control-primary | primary | worldcom | 9 | 5 | 55.6% | 5 | 0 | 2.900 | 1997 |
| Lockheed Martin | control-secondary | secondary | boeing | 10 | 6 | 60.0% | 0 | 6 | -1.500 | 2020 |
| Illumina | control-secondary | secondary | theranos | 11 | 6 | 54.5% | 4 | 2 | 1.167 | 2013 |
| Verizon | control-secondary | secondary | worldcom | 5 | 5 | 100.0% | 0 | 5 | -2.000 | 2000 |

---

## Firing Events (all firms)

| Entity | Role | Tier | Year | Classification | Gap |
|---|---|---|---|---|---|
| AT&T | control-primary | primary | 1996 | epistemological | 2.633 |
| AT&T | control-primary | primary | 1997 | epistemological | 2.900 |
| AT&T | control-primary | primary | 1999 | epistemological | 1.533 |
| AT&T | control-primary | primary | 2001 | epistemological | 2.133 |
| AT&T | control-primary | primary | 2002 | epistemological | 2.200 |
| Airbus | control-primary | primary | 2015 | epistemological | 1.767 |
| Airbus | control-primary | primary | 2016 | epistemological | 2.133 |
| Airbus | control-primary | primary | 2017 | epistemological | 1.867 |
| Airbus | control-primary | primary | 2018 | epistemological | 1.833 |
| Airbus | control-primary | primary | 2019 | epistemological | 2.733 |
| Airbus | control-primary | primary | 2020 | economic | -3.467 |
| Boeing | case | primary | 2013 | epistemological | 1.633 |
| Boeing | case | primary | 2014 | epistemological | 1.767 |
| Boeing | case | primary | 2015 | epistemological | 2.467 |
| Boeing | case | primary | 2016 | epistemological | 2.967 |
| Boeing | case | primary | 2017 | epistemological | 3.833 |
| Boeing | case | primary | 2018 | epistemological | 3.200 |
| Boeing | case | primary | 2019 | epistemological | 2.600 |
| Boeing | case | primary | 2020 | economic | -2.233 |
| Quest Diagnostics | control-primary | primary | 2008 | epistemological | 1.033 |
| Quest Diagnostics | control-primary | primary | 2010 | epistemological | 1.233 |
| Quest Diagnostics | control-primary | primary | 2011 | epistemological | 1.200 |
| Quest Diagnostics | control-primary | primary | 2012 | epistemological | 1.633 |
| Quest Diagnostics | control-primary | primary | 2014 | economic | -1.867 |
| Theranos | case | primary | 2008 | economic | -1.733 |
| Theranos | case | primary | 2009 | economic | -1.600 |
| Theranos | case | primary | 2012 | epistemological | 1.200 |
| Theranos | case | primary | 2013 | epistemological | 2.800 |
| Theranos | case | primary | 2014 | epistemological | 3.433 |
| Theranos | case | primary | 2015 | epistemological | 2.367 |
| WorldCom | case | primary | 1998 | epistemological | 2.133 |
| WorldCom | case | primary | 1999 | epistemological | 1.900 |
| WorldCom | case | primary | 2000 | epistemological | 1.467 |
| WorldCom | case | primary | 2001 | epistemological | 1.267 |
| Illumina | control-secondary | secondary | 2008 | economic | -1.000 |
| Illumina | control-secondary | secondary | 2009 | economic | -1.000 |
| Illumina | control-secondary | secondary | 2013 | epistemological | 1.167 |
| Illumina | control-secondary | secondary | 2014 | epistemological | 1.167 |
| Illumina | control-secondary | secondary | 2015 | epistemological | 1.167 |
| Illumina | control-secondary | secondary | 2016 | epistemological | 1.167 |
| Lockheed Martin | control-secondary | secondary | 2013 | economic | -1.167 |
| Lockheed Martin | control-secondary | secondary | 2014 | economic | -1.167 |
| Lockheed Martin | control-secondary | secondary | 2015 | economic | -1.167 |
| Lockheed Martin | control-secondary | secondary | 2020 | economic | -1.500 |
| Lockheed Martin | control-secondary | secondary | 2021 | economic | -1.500 |
| Lockheed Martin | control-secondary | secondary | 2022 | economic | -1.500 |
| Verizon | control-secondary | secondary | 2000 | economic | -2.000 |
| Verizon | control-secondary | secondary | 2001 | economic | -2.000 |
| Verizon | control-secondary | secondary | 2002 | economic | -1.833 |
| Verizon | control-secondary | secondary | 2003 | economic | -1.833 |
| Verizon | control-secondary | secondary | 2004 | economic | -1.833 |

---

## Case vs Control Comparison by Pair

### Boeing (case) | Airbus (primary ctrl) | Lockheed Martin (secondary ctrl)

| Year | Boeing gap | Boeing class | Airbus gap | Airbus class | Lockheed Martin gap | Lockheed Martin class |
|---|---|---|---|---|---|---|
| 2013 | 1.633 | epistemological | 0.033 | unclassified | -1.167 | economic |
| 2014 | 1.767 | epistemological | 0.967 | unclassified | -1.167 | economic |
| 2015 | 2.467 | epistemological | 1.767 | epistemological | -1.167 | economic |
| 2016 | 2.967 | epistemological | 2.133 | epistemological | -0.167 | unclassified |
| 2017 | 3.833 | epistemological | 1.867 | epistemological | -0.167 | unclassified |
| 2018 | 3.200 | epistemological | 1.833 | epistemological | -0.167 | unclassified |
| 2019 | 2.600 | epistemological | 2.733 | epistemological | -0.167 | unclassified |
| 2020 | -2.233 | economic | -3.467 | economic | -1.500 | economic |
| 2021 | -0.667 | unclassified | -0.267 | unclassified | -1.500 | economic |
| 2022 | 0.000 | unclassified | -0.100 | unclassified | -1.500 | economic |

### Theranos (case) | Quest Diagnostics (primary ctrl) | Illumina (secondary ctrl)

| Year | Theranos gap | Theranos class | Quest Diagnostics gap | Quest Diagnostics class | Illumina gap | Illumina class |
|---|---|---|---|---|---|---|
| 2008 | -1.733 | economic | 1.033 | epistemological | -1.000 | economic |
| 2009 | -1.600 | economic | 0.467 | unclassified | -1.000 | economic |
| 2010 | -0.500 | unclassified | 1.233 | epistemological | -0.167 | unclassified |
| 2011 | 0.367 | unclassified | 1.200 | epistemological | -0.167 | unclassified |
| 2012 | 1.200 | epistemological | 1.633 | epistemological | -0.167 | unclassified |
| 2013 | 2.800 | epistemological | -0.200 | unclassified | 1.167 | epistemological |
| 2014 | 3.433 | epistemological | -1.867 | economic | 1.167 | epistemological |
| 2015 | 2.367 | epistemological | -0.367 | unclassified | 1.167 | epistemological |
| 2016 | 0.533 | unclassified | 0.433 | unclassified | 1.167 | epistemological |
| 2017 | -0.500 | unclassified | -0.267 | unclassified | 0.667 | unclassified |
| 2018 | 0.000 | unclassified | -0.667 | unclassified | 0.667 | unclassified |

### WorldCom (case) | AT&T (primary ctrl) | Verizon (secondary ctrl)

| Year | WorldCom gap | WorldCom class | AT&T gap | AT&T class | Verizon gap | Verizon class |
|---|---|---|---|---|---|---|
| 1996 | 0.700 | unclassified | 2.633 | epistemological | — | — |
| 1997 | 0.767 | unclassified | 2.900 | epistemological | — | — |
| 1998 | 2.133 | epistemological | 0.300 | unclassified | — | — |
| 1999 | 1.900 | epistemological | 1.533 | epistemological | — | — |
| 2000 | 1.467 | epistemological | 0.233 | unclassified | -2.000 | economic |
| 2001 | 1.267 | epistemological | 2.133 | epistemological | -2.000 | economic |
| 2002 | 0.967 | unclassified | 2.200 | epistemological | -1.833 | economic |
| 2003 | -0.167 | unclassified | 0.133 | unclassified | -1.833 | economic |
| 2004 | 0.300 | unclassified | -0.267 | unclassified | -1.833 | economic |

---

## Notes

- verify_canonical run: V recomputed from C/K/S before all calculations; stored Node Value column uses C+K−S+0.5A (expected delta)
- Secondary controls scored from uploaded `cams_scores_1990-2026.csv`; single-scorer (ENV SDs = 0)
- Verizon window is 2000–2004 (WorldCom pair is 1996–2004; Verizon data available from 2000 only)
- Circular-shift null and simpler baselines (Stress-change-alone, first differences of V) remain pending

*Source: crisis_typology_rule_v1.md (sealed 2026-11-05) | P4 protocol frozen 2026-11-10*  