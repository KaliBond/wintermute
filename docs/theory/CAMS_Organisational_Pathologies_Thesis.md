# CAMS: Organisational Pathologies as Recurring Functional Topologies

Working thesis, October 2026. This redraft (20 October 2026) restates the earlier working proposition, itself an external assessment revised by Lab Chief, in our framing. Every figure is reconciled against the frozen signature run (prereg commit `1b9b9f1`, results commit `58fcaaf`) and `juno/v3/CAMS_Signature_Pack_v1.md`.

## 1. The proposition

We treat societies and organisations as complex adaptive systems built from eight functions:
- Helm: direction.
- Shield: coercion and protection.
- Lore: meaning.
- Stewards: property and resources.
- Flow: exchange.
- Craft: skill.
- Hands: labour.
- Archive: memory.

Our proposition is that organisational pathologies are characteristic paths that stress takes through these functions. They are patterns of changing relationships, not just low scores. Because the functional topology is the same wherever humans organise, the same paths recur and can be identified quantitatively, even where institutional forms, eras and people differ.

## 2. Warrant: repeatability

Node readings are our primary observations. Independent AI scoring passes each rate Coherence (C), Capacity (K), Stress (S) and Abstraction (A) for each of the eight nodes in each year. They keep recovering the same structure in a society's history, and we take that repeatability as the warrant, not conformity with external indexes.

The method is closer to reconstructing climate history from monastic and archival records. We trust no single chronicler, but when independent records agree on the hard winters and the good harvests, we take the pattern as real.

No scoring pass ever scores "Helm isolation" or "Archive amnesia". Signatures are regularities in topology change, inferred afterwards across many society-years. They are not something imposed while scoring. The metrics are the signal.

## 3. Pathology is a relationship

Two societies can have the same mean V. In one, declining Helm capacity is offset by a strong Archive, effective Stewards and adaptive Flow. In the other, Helm becomes isolated while Shield increasingly enforces its decisions and Lore legitimises them. The averages match, but the configurations do not.

We make the temporal structure explicit:
- X_{i,t} = (C_{i,t}, K_{i,t}, S_{i,t}, A_{i,t}), and V_{i,t} = C + K + A/2 − S.
- ΔX_{i,t} = X_{i,t} − X_{i,t−1}.
- D_{ij,t} = ΔV_{i,t} − ΔV_{j,t}.

D_{ij,t} measures relative change between two functions. It is not a proven transmission edge.

A signature specifies:
- which nodes change;
- the direction and timing of the changes;
- which relationships weaken or strengthen;
- whether the change persists.

Edges are defined operationally from the scores (gap from the mean of the other nodes, rank inversion, lagged co-movement), and each rule is written down before it is run.

## 4. Why the patterns recur: two layers

**The coordination problem.** Organised humans face the same dilemmas again and again: bounded information, contested resources, asymmetric power and changing environments. The eight functions are standing answers to that problem set, needed by a household as much as by an empire. Slow information, mismatched incentives, incompatible time horizons, shocks and once-successful adaptations that go stale can all produce similar configurations.

**The primate who runs it.** Every scale is run by cheating, lying, status-seeking primates. Coercion gets misused. Delusion is joined to authority. Groups keep holding beliefs that are no longer true. Memory is stripped after a shock. The same primates also cooperate, enforce reciprocity and build institutions to restrain the dominant. Both layers matter: the first explains why the functions exist, the second why the same failures recur at every scale.

**Functional case by case.** Some pathologies look maladaptive and are still functional in context. Helm Incompetence can be tolerable when the other nodes carry the load. A Shield rise can be the right response to a real external threat. Signatures are diagnoses of how an organisation is arranged, not verdicts on it, and each carries a stated benign reading.

## 5. The signature pack (frozen rules)

Thresholds are pooled percentiles over changing years, frozen before outcomes were counted.

| Signature | Rule |
|---|---|
| Archive amnesia | 5-yr ΔV(Archive) ≤ −5.77 (P5), while Helm V or Shield V rises over the same 5 years |
| Helm Incompetence | Over 3 yrs: Helm ΔK ≤ −1.0 (P10), Helm ΔS ≥ +2.0 (P90), Helm ΔA ≥ 0; mean K(Craft, Flow) at or above its trailing 10-yr mean. Informally, "idiots in charge": a capable society with an incapable helm |
| Helm isolation | \|C(Helm) − mean C(other 7)\| ≥ 1.86 (P95) |
| Shield inversion | Shield is the highest-V node and V(Flow) − mean V ≤ −1.18 (P25) |
| Late abstraction collapse | Some node has A ≥ 7.8 (P90) and that node's 2-yr ΔC ≤ −1.2 (P5) |
| Narrative–material gap | Mean A(Lore, Archive) ≥ 8.0 (P90) and K(Flow, Craft, Hands) minus its trailing 10-yr mean ≤ −0.9 (P10) |

Lore–Helm divergence was defined in the draft but is not in the frozen run.

## 6. Evidence from JUNO v3

**37 or 42 societies.** The dataset (`juno/v3/JUNO_Unified_Dataset_v3_2026-10-19.csv`) holds 41,914 rows for 42 societies. The frozen run used 37 of them. Five whole series were excluded in advance because of measurement problems:
- Denmark: Stress recorded as zero throughout.
- India and Turkey: 26–31 sampled years each.
- Thailand: 5-yearly scoring.
- Canada: 33 years.

Israel stays in the 37, but only from 2015, because the earlier series is flat. All the counts and p-values below are for those 37 societies.

The draft-threshold counts in the Signature Pack (`sig.py`) had different coverage. They excluded only Denmark and pre-2015 Israel. Their annual-only rules skipped the non-annual series, while the Hands-floor and Shield-inversion rows still list Canada, India, Thailand and Turkey.

**Method.** We compared firing years against a shuffle null: each society's firing series is circularly shifted, 10,000 reps, seed 20261020. The p-values are unadjusted, and the prereg specifies Holm correction within each null family.

| Signature | Firing years | In named windows (obs / chance, p) | Before a big fall in mean V (obs / chance, p) |
|---|---|---|---|
| Archive amnesia | 22 | 15 / 0.75, 0.0001 | 5 / 4.65, 0.49 |
| Helm Incompetence | 25 | 4 / 0.18, 0.022 | 14 / 9.25, 0.074 |
| Helm isolation | 225 | 17 / 8.41, 0.004 | 57 / 55.94, 0.47 |
| Shield inversion | 268 | 24 / 10.37, 0.006 | 67 / 63.52, 0.36 |
| Late abstraction collapse | 104 | no windows | 62 / 38.56, 0.0002 |
| Narrative–material gap | 52 | no windows | 28 / 20.43, 0.072 |

Total: 696 firing years. A big fall is an annual drop of 2.0 or more in mean V, in years t to t+3.

**Window concentration.** All four signatures with named windows concentrate in them well beyond chance. After Holm correction across the four, all stay under 0.05.

Twelve of the eighteen named windows survived the frozen thresholds:
- Archive amnesia: Argentina 1976–78, Chile 1975–77, Cambodia 1975–78, Greece 1967–71 and Hong Kong 2020–24.
- Helm Incompetence: the UK 2016–19.
- Helm isolation: Venezuela 1999–2013 and Australia 2010–18.
- Shield inversion since 2015: Iran, Russia and Venezuela, plus Finland 2023–25.

Six dropped out: Laos 1975, Germany 1931 and Hong Kong 1997 (Archive amnesia); Germany 1939 and Pakistan 1968–70 (Helm Incompetence); Iraq 2015–25 (Shield inversion). Australia 1976–80, a check window rather than a claimed hit, does not fire.

This concentration is partly circular, because the windows were named from where the draft rules had already fired. What it shows is that the patterns survive stricter thresholds that were fixed in advance. It does not show they were found independently.

**Before big falls.** Late abstraction collapse is the one signature that precedes big falls clearly beyond chance (62 against 38.56 expected, p = 0.0002; about 0.001 after Holm correction across six). Its rule partly contains the fall, because a drop in Coherence lowers V directly. Helm Incompetence (p = 0.074) and the narrative–material gap (p = 0.072) lean the same way. Archive amnesia, Helm isolation and Shield inversion show no excess before falls; they mark episodes rather than forecast them.

**Observed co-occurrences.** These are frozen hits in the same nation, with one signature beginning within five years of another:
- Helm isolation and Shield inversion follow each other in both directions, each in 9 nations.
- Helm Incompetence is followed by Helm isolation in 5 (including the UK 2016→2018 and Venezuela 2008→2009).
- The narrative–material gap is followed by Late abstraction collapse in 5.
- Late abstraction collapse is followed by Helm isolation in 5 and by Shield inversion in 5.
- Archive amnesia is followed by Shield inversion in 4.

These are observed co-occurrences, not established chains. They need comparison against base rates, and signatures that share nodes can co-fire by construction.

**Constant firers.** We report these with their event rates, as properties of how the series move:
- Helm isolation: Venezuela 0.48, Russia 0.22, Poland 0.20, Sweden 0.20 and Saudi Arabia 0.19, against a pooled 0.046.
- Shield inversion: Russia 0.44, Venezuela 0.25, Argentina 0.21 and Japan 0.19, against a pooled 0.054.
- Late abstraction collapse: Sweden 0.27 and South Africa 0.15, against a pooled 0.021.
- Narrative–material gap: Sweden 0.13, Finland 0.11 and Brazil 0.07, against a pooled 0.011.

**A directional regularity** (exploratory, 19 October, not preregistered). Across 38,704 node-years in 37 societies, a rise in Stress is followed the next year by a fall in Capacity (r = −0.072), but not the reverse (r = +0.004). The society-cluster bootstrap 95% CI of the difference is [−0.103, −0.052], and the asymmetry holds in 33 of 37 societies. V has no time order built in, so this is not an artefact of the formula.

## 7. Three levels of cross-scale claim

- **A. Structural transfer.** The same eight functions and kinds of pathology appear at every scale.
- **B. Rule transfer.** The frozen national rules, unchanged, pick out the same pathologies in firms and institutions.
- **C. Calibrated transfer.** Thresholds and outcome rates carry across scales quantitatively.

We currently claim scale-recurring topology (level A), with window concentration and one before-falls result inside national data. Level B is the next test.

## 8. The next test: P4, rule transfer to corporations

We will score firms with this corporate node rubric, written before any firm is scored:

| Node | Corporate function |
|---|---|
| Helm | Board and executive |
| Shield | Compliance, legal, discipline |
| Lore | Culture and narratives |
| Stewards | Ownership and capital |
| Flow | Supply chain and market |
| Craft | Engineering and expertise |
| Hands | Workforce |
| Archive | Records, audit, institutional memory |

The test runs in six steps:
1. Freeze national rules, thresholds, timing conventions and exclusions as of `58fcaaf`.
2. Score Boeing, Theranos and WorldCom, plus matched control firms without a known crisis, over pre-specified years, using repeated independent AI scoring passes.
3. Run the national detector unchanged. Report every firing, including misses and false positives in the controls.
4. Compare against the circular-shift null and simpler baselines: Stress change alone, and first differences of V.
5. Keep the discovery register (everything in section 6) separate from the prospective watchlist, which logs firings before outcomes are known.
6. Do not expand the signature pack before this test.

The question P4 answers is whether topology adds information beyond absolute levels and their first differences.

## 9. The recovery catalogue

If damaging configurations recur, recovery configurations should recur too. We will define them with node rules, freeze them, and search for them across the same corpus with the same discipline:
- Archive recovery.
- Helm recoupling.
- Shield restraint.
- Lore–material reconnection: renewed co-movement between Lore and Craft, Hands and Flow.

## 10. Purpose

Our purpose is to understand the architecture through which organisations maintain, lose and recover their capacity to cooperate.
