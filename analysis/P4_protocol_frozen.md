# Phase 4 Protocol — Frozen Control List
## Retrospective Validation: Organisational Pathology Detector

**Frozen:** 2026-11-10  
**Baseline commit:** 58fcaaf  
**Status:** LOCKED — do not modify after hash is recorded

---

## Case firms and scoring windows

| Firm | Type | Window |
|---|---|---|
| Boeing | Case — aerospace/defense | ~2018–2020 (737 MAX crisis window) |
| Theranos | Case — blood diagnostics / lab tech | ~2003–2018 (rise and collapse) |
| WorldCom | Case — telecom | ~1999–2002 (accounting fraud and bankruptcy) |

## Matched controls (PRIMARY — required)

| Case firm | Primary control | Matching rationale |
|---|---|---|
| Boeing | Airbus | Direct commercial-aviation peer of comparable scale and complexity |
| Theranos | Quest Diagnostics | Established clinical-laboratory peer; no fraud collapse in window |
| WorldCom | AT&T | Contemporaneous large US telecom; no equivalent accounting fraud |

## Matched controls (SECONDARY — optional, for sensitivity)

| Case firm | Secondary control | Matching rationale |
|---|---|---|
| Boeing | Lockheed Martin | US defense peer; no commercial aviation safety/governance crisis |
| Theranos | Illumina | High-ambition diagnostics/genomics firm; technological ambition without fraud collapse |
| WorldCom | Verizon | Contemporaneous large US telecom; no equivalent accounting fraud |

## Matching criteria (must be preserved if substitutions are made)

1. Same industry and approximate scale/era of operation as the case firm
2. Absence of a documented organisational pathology of the **same signature family** in the relevant window
3. Control list must be locked **before any scoring begins** — no post-hoc substitution

Acceptable substitutes: Northrop Grumman (for Lockheed), Labcorp (for Quest), Sprint (for AT&T/Verizon), or non-US peers — provided criteria 1 and 2 are preserved and the substitution is recorded here before scoring.

## Node mapping (corporate, frozen at 58fcaaf)

| CAMS Node | Corporate carrier |
|---|---|
| Helm | Board and executive |
| Shield | Compliance, legal, discipline |
| Lore | Culture and narratives |
| Stewards | Ownership and capital |
| Flow | Supply chain and market |
| Craft | Engineering and expertise |
| Hands | Workforce |
| Archive | Records, audit, institutional memory |

## Scoring rules

- Score all six firms (three cases + three primary controls) over the **same pre-specified years**
- Use repeated independent AI passes under the corporate node mapping above
- Run the national detector **unchanged** — report every firing, including false positives in controls
- Keep the discovery register separate from any prospective watchlist
- Do not expand the signature pack before this test
- Compare against: circular-shift null; Stress-change-alone baseline; first differences of V

## Constraints (from thesis protocol §0A and sealed rules)

- No retrospective adjustment of nodes, scoring criteria, thresholds, batches, or interpretation rules after observing results
- FAIL is FAIL — a registered fail fires its pre-committed consequence and is written as FAIL
- Do not change, regenerate, or rehash this document after scoring begins
- Where the protocol is silent, stop and ask — do not improvise
- verify_canonical MUST run against stored V and Bond columns before any quantity is computed
- Red-line classifications remain in the evaluator's dataset, not the scorers' instructions

---

*This file is the frozen control list for P4. Hash recorded separately in P4_protocol_hash.txt.*
