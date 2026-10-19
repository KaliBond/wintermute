# CAMS: Organisational Pathologies as Recurring Functional Topologies

Working proposition, October 2026. Source: an external (GPT) assessment of the draft thesis, revised by Lab Chief on 20 October 2026 with figures corrected to the frozen signature run. This is a statement of a hypothesis and a test plan, not a finding.

## 1. The proposition

Organisational pathologies recur as identifiable patterns of changing relationships among functional components. These patterns may persist across organisational scales, even where institutional forms, historical circumstances and the people involved differ.

That organisations suffer similar troubles is familiar from organisational sociology, political science and complexity theory. What CAMS adds is the claim that the troubles take recognisable *shapes* in how functions move relative to one another, and that those shapes can be read off scored data.

## 2. Three claims to keep apart

1. **Functional recurrence.** Households, firms and nations face analogous coordination problems. Plausible, and largely accepted.
2. **Topological recurrence.** Dysfunction appears through comparable configurations of change among functions. An important empirical hypothesis, partly supported within national data (section 6).
3. **Quantitative scale invariance.** The same measurement rules and decision thresholds identify these configurations at every scale. A much stronger result, and not yet established.

We use the language *scale-recurring* or *cross-scale invariant candidate* until the cross-scale test in section 8 is complete.

## 3. What the data are, and what a signature is

Node readings are primary observations. Each scorer rates Coherence (C), Capacity (K), Stress (S) and Abstraction (A) for each of the eight JUNO v1.2 nodes (Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow) in each year, with node value V = C + K − S + 0.5A. No scorer ever scores "Helm isolation" or "Archive amnesia". Signatures are regularities inferred afterwards from how the scored topology moves across many society-years. A society that fires constantly tells us about how that series' scores move (step-coding, scale resolution, flat runs), not about a pattern imposed during scoring.

### Pathology is a relationship, not necessarily a low score

Two societies can have the same mean V. In one, declining Helm capacity is offset by a strong Archive, effective Stewards and adaptive Flow. In the other, Helm becomes isolated while Shield increasingly enforces its decisions and Lore legitimises them. The averages match; the configurations do not. The question is about the arrangement of relationships, not the level of any single node.

This also follows from our recent concern about over-weighting absolute levels against changes. The temporal structure is made explicit:

- X_{i,t} = (C_{i,t}, K_{i,t}, S_{i,t}, A_{i,t})
- ΔX_{i,t} = X_{i,t} − X_{i,t−1}
- D_{ij,t} = ΔV_{i,t} − ΔV_{j,t}

D_{ij,t} measures divergence in node-value changes. It is not a transmission pathway.

## 4. Operational topology

A signature should specify: which nodes change; the direction and timing of the changes; which relationships weaken or strengthen; and whether the change persists. Edges must be operational, defined from the scores (for example lagged co-movement, gap from the mean of other nodes, or rank inversion), with the rule written down before it is run. A diagram of hypothesised decoupling or coercive redirection is a conceptual illustration, not a measured CAMS network, and should be labelled that way.

### Three levels of claim

- **Level A, description:** a signature names a configuration in the scored record.
- **Level B, regularity:** the configuration recurs beyond chance under a structure-preserving null, within source and outside flagged series.
- **Level C, mechanism:** the configuration reflects a pathway through which strain is transmitted, absorbed, amplified or concealed.

The current evidence reaches Level A for all six signatures and partial Level B for some. Nothing yet reaches Level C. Measurement (a pattern is present in the scores) is kept separate from mechanism (why it is present).

## 5. Why the patterns might recur

The draft explained recurrence through human traits such as cheating, deception, hierarchy and status-seeking. That risks reducing pathology to defects of character. Humans also cooperate, share knowledge, enforce reciprocity, build trust and create institutions to restrain dominant individuals, and failures need not begin with deception or incompetence.

The better framing: organisations recurrently face **coordination dilemmas under bounded information, contested resources, asymmetric power and changing environments**. Slow information flows, mismatched incentives, incompatible time horizons, external shocks and once-successful adaptations that become maladaptive can all produce similar configurations. This makes recurrence an expectation of the structure of the problem, not a verdict on the participants.

## 6. Current evidence from JUNO (frozen run, 20 October 2026)

Data: `JUNO_Unified_Dataset_v3_2026-10-19.csv` (41,914 rows, 42 societies). Exclusions fixed in advance: Denmark, India, Turkey, Israel before 2015, Canada and Thailand, leaving 37 societies. Thresholds were set from the data using only years in which the profile changed, frozen in prereg commit `1b9b9f1` before outcomes, with results in commit `58fcaaf`. Null: circular shift of each society's firing series, 10,000 reps, seed 20261020. p-values below are unadjusted.

| Signature | Firing years | In claimed windows (obs / chance, p) | Before a big fall in mean V (obs / chance, p) |
|---|---|---|---|
| Archive amnesia | 22 | 15 / 0.75, 0.0001 | 5 / 4.65, 0.49 |
| Idiots in charge | 25 | 4 / 0.18, 0.022 | 14 / 9.25, 0.074 |
| Helm isolation | 225 | 17 / 8.41, 0.004 | 57 / 55.94, 0.47 |
| Shield inversion | 268 | 24 / 10.37, 0.006 | 67 / 63.52, 0.36 |
| Late abstraction collapse | 104 | no windows | 62 / 38.56, 0.0002 |
| Narrative–material gap | 52 | no windows | 28 / 20.43, 0.072 |

How to read this:

- **Window concentration is partly circular.** The claimed windows were taken from where the draft rules had already fired. Recovering them under frozen thresholds shows the signatures are stable, not that they were discovered independently.
- **Prediction is mostly absent.** Only late abstraction collapse precedes big falls more than chance, and its rule partly contains the fall: a Coherence drop lowers V directly.
- **Archive amnesia** fires in 22 years, 15 inside the claimed windows. Five of eight windows survive: Argentina 1976–78, Chile 1975–77, Cambodia 1975–78, Greece 1967–71, Hong Kong 2020–24. Laos 1975, Germany 1931 and Hong Kong 1997 drop out. Australia 1976–80, which the draft flagged, no longer fires: it rested on a single step down in Archive in 1976 followed by a flat run. Cambodia's hits rest on few changing years and carry less weight. The honest reading is that Archive amnesia is a precise label for episodes we can already name, not a predictor.
- **Constant firers** (for example Venezuela, Poland, Russia, Saudi Arabia and Sweden for Helm isolation; Sweden and South Africa for late abstraction collapse) are properties of how those series move, and are reported separately.
- **One directional regularity.** A separate exploratory analysis (19 October, not preregistered) found that within a node a rise in Stress is followed next year by a fall in Capacity (r = −0.072) but not the reverse (r = +0.004), in 33 of 37 societies. V has no time order in it, so this is not built in, though a scorer-timing habit cannot be ruled out from inside the record.
- **Unverified draft claim.** The draft referred to "nine-nation and five-nation chains". These do not appear in our analysis files and are not part of the evidence until they are specified and run.

## 7. Two registers, and benign signatures

- **Discovery register:** everything above. Rules were written after looking at the data, so results are hypothesis-generating.
- **Prospective register:** frozen rules applied to years and cases not yet scored or not yet seen, with every firing logged in advance of outcomes. Only this register can move a signature to firm Level B.

Optional documentary anchors (contemporaneous records of an episode) can help readers interpret a firing, but they are not a validation requirement: the node readings are the observations.

**Benign and adaptive signatures.** Some configurations that look pathological may be adaptive: a Shield rise under genuine external threat, or a deliberate Helm reorganisation. Each signature should carry a stated benign reading, and the false-positive rate should be reported alongside the hits.

**Repair catalogue.** If damaging configurations recur, repair configurations may recur too: Archive recovery, Helm recoupling, Shield restraint, and renewed co-movement between Lore and the material nodes (Craft, Hands, Flow). These should be defined and tested with the same discipline.

## 8. The priority experiment: P4, cross-scale transfer (design proposal, not a completed test)

1. Freeze all national rules, thresholds, timing conventions and exclusions as of commit `58fcaaf`.
2. Write a corporate scoring rubric mapping the eight functions to firms before scoring any firm.
3. Keep scorers unaware of the signature definitions and, as far as practicable, of the hypotheses.
4. Score Boeing, Theranos and WorldCom, plus matched corporate controls without a known crisis, across pre-specified observation years.
5. Run the national detector unchanged and report every firing, including misses and false positives.
6. Compare against the circular-shift null and simpler alternatives: overall Stress change, or a single declining node.

The key counterfactual: **does topology add diagnostic or predictive information beyond absolute CAMS levels and their first differences?** If not, the topology is a complicated description of deterioration already visible in simpler measures. If it does, on held-out cases and across scales, the thesis has something substantial.

We would not expand the signature pack before this test.

## 9. Judgement

The idea worth protecting is that organisations may fail through a limited repertoire of relational configurations, not only through exhausted resources or poor leadership. CAMS already has a vocabulary of functions and an instrument that makes them comparable; the thesis adds a grammar of how their relationships change over time. It does not yet establish causal pathways, universal laws or scale invariance.

The purpose of CAMS is not to judge organisations by their apparent virtues or vices, but to understand the architecture through which they maintain, lose and sometimes recover the capacity to cooperate. The most consequential question is whether the relationships that make an organisation vulnerable also reveal the routes by which it recovers.
