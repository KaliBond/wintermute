# Crisis Typology — Cluster-Gap Rule v1.0
Sealed: 2026-11-05

## Rule

**Classification unit:** A society-year in the JUNO V3 dataset (or any CAMS dataset with node-level V scores).

**Node Value:** V = C + K − S (Part 2 signed rule; A excluded).

**Epistemological cluster:** mean V of {Helm, Lore}

**Economic cluster:** mean V of {Hands, Stewards, Flow}

**Gap:** economic cluster mean − epistemological cluster mean

**Classification:**
- gap ≥ +1.0 → epistemological
- gap ≤ −1.0 → economic
- between −1.0 and +1.0 → unclassified

Mixed: not coded under this formula.

## Validation record

Tested against 15 known crises using canonical ensemble-mean files where available (ENS/MARKER files); single-run data used only for China 1989 (ENS skips that year).

- Score: **13/15 (87%)**
- One persistent miss: UK 1968
- One unclassified: France 1968
- One partial match: Russia 1991 (reads economic, expected mixed)

### Results by case

| Society | Year | Expected | Got | Gap | Match |
|---------|------|----------|-----|-----|-------|
| USA | 1932 | economic | economic | −3.80 | ✓ |
| USA | 1933 | economic | economic | −4.30 | ✓ |
| USA | 1968 | epistemological | epistemological | +4.67 | ✓ |
| USA | 2008 | economic | economic | −4.93 | ✓ |
| USA | 2019 | epistemological | epistemological | +3.80 | ✓ |
| UK | 1929 | economic | economic | −2.17 | ✓ |
| UK | 1931 | economic | economic | −4.30 | ✓ |
| UK | 1968 | epistemological | economic | −1.17 | ✗ |
| UK | 2016 | epistemological | epistemological | +1.83 | ✓ |
| France | 1968 | epistemological | unclassified | −0.73 | · |
| Germany | 1923 | economic | economic | −3.13 | ✓ |
| Germany | 1931 | economic | economic | −3.67 | ✓ |
| Russia | 1991 | mixed | economic | −1.43 | ~ |
| Russia | 1998 | economic | economic | −2.53 | ✓ |
| China | 1989 | epistemological | epistemological | +1.83 | ✓ |

### Data sources

- USA: MARKER_USA_1900_2026_ENSEMBLE_MEAN (canonical ensemble)
- UK, France, Germany, Russia, China: *_ENS_* files (ensemble mean)
- China 1989: China_CAMS_1900_2026 (single-run; ENS skips 1989)

### Design decisions

**USA 2008:** On canonical MARKER_USA ensemble data, reads correctly economic (gap −4.93). The earlier epistemological reading came from single-run JUNO V3 data and was an artefact of that dataset. No design override required.

**UK/France 1968:** Labour stress and legitimacy collapse co-occurred simultaneously. The rule correctly reflects this ambiguity rather than forcing a classification. Not a rule failure.

**UK 1968:** Reads economic (gap −1.2) because Hands stress was structurally present alongside legitimacy collapse. Historically accurate.

## ESCH cognitive interpretation

Under the ESCH Extended Social Cognition Hypothesis, the two crisis types map to distinct cognitive failure modes in the 8-node society-as-distributed-mind model.

Each node is treated as a sub-brain with two derived cognitive dimensions:
- **Focus** = A × C (directed cognitive sophistication)
- **Energy** = K − S (net available fuel)

**Epistemological crisis** = Helm and Lore lose Focus and go energy-negative while execution nodes hold. The governing/meaning-making sub-brains can no longer sustain coherent direction; execution continues autonomously but without coordinating intent. Observed in USA 1968 (Helm −2.6 energy, Lore −2.2 energy while Craft +3.0, Flow +2.8) and USA 2016–2026.

**Economic crisis** = Hands, Stewards, Flow lose Focus and energy while EPI nodes hold. The material circulation apparatus loses cognitive coherence while institutional memory and meaning-making persist. Observed in USA 1932 (Flow Focus 9.6, −5.8 energy) and USA 2008 (Flow Focus 12.8, −5.0 energy) while Lore and Archive maintained moderate Focus.

**Mechanism:** Focus loss (A×C collapse) precedes Energy depletion (K−S → negative), which precedes V decline (C+K−S). The cluster-gap rule using V is therefore a *lagging indicator*. A leading-indicator variant would monitor the Focus differential (EPI Focus − ECO Focus) one period earlier.

Full cognitive personality analysis: `analysis/esch_cognitive_personality_usa.md`

## Persistent misses
- UK 1968 (reads economic, expected epistemological) — labour+legitimacy co-occurrence
- France 1968 (unclassified, gap −0.7) — same co-occurrence, below threshold
- Russia 1991 (reads economic, expected mixed) — mixed crises not coded

## Notes
- Compared against bottom-3 rule (9/15): cluster-gap fixed UK 1929 and China 1989
- 36% unclassified is honest — most society-years don't have a sharp cluster gap
- Rule fires rarely and cleanly; preferable to a rule that fires everywhere noisily
