# CAMS Thesis: Organisational Pathologies as Recurring Functional Topologies

*Revised thesis, 20 October 2026. Data: JUNO Unified Dataset v3 (42 societies, 41,914 rows).*

## 1. Central claim

We treat societies and organisations as complex adaptive systems built from eight functions:
- Helm: direction.
- Shield: coercion and protection.
- Lore: meaning.
- Stewards: property and resources.
- Flow: exchange.
- Craft: skill.
- Hands: labour.
- Archive: memory.

Every organised human group has to perform these functions in some form. How well it performs them, and how they relate to one another, gives it a shape. We call that shape its functional topology.

Our claim is that organisational pathologies are characteristic paths that stress takes through these functions. A pathology is a pattern of changing relationships, not just a low score. Because the topology is the same wherever humans organise, the same paths can be found again and recognised quantitatively.

A simple illustration shows why configuration matters more than level. Take two societies with the same mean node value. In the first, Helm, Shield and Archive sit close together and move together. In the second, Shield leads, Flow has sunk below the mean, and Helm's coherence has come loose from the rest. The averages are the same, but the organisations are not. The second is showing a recognisable pathology, and the average alone would hide it.

## 2. Mechanism

We read the mechanism in two layers.

**Why the functions exist.** Organised humans face the same coordination problems again and again: bounded information, contested resources, asymmetric power and changing environments. Each of the eight functions is a standing answer to part of that problem set. Direction, protection, meaning, resources, exchange, skill, labour and memory are needed by a household as much as by an empire.

**Why the failures recur.** The same failures recur because every scale is run by cheating, lying, status-seeking primates. Coercion gets misused. Delusion is joined to authority. Groups keep holding beliefs that are no longer true. Memory is stripped after a shock. These are the failure modes people recognise at home, at work, in corporations and in nations.

The same primates also cooperate and build constraints on themselves, so failure is not the only story, and not every failure is a vice. Pathologies also come from scarcity, from external shocks, and from adaptations that were sound once and have gone stale.

Some pathologies are functional case by case. Helm Incompetence can look maladaptive and still be tolerable, even useful, when the other nodes carry the load. Our signatures are diagnoses of how an organisation is arranged, not verdicts on it.

## 3. Measurement and warrant

Each node is scored each year on four metrics: coherence (C), capacity (K), stress (S) and abstraction (A). The node value is

V = C + K + A/2 − S.

We work with both levels and changes. The change in any quantity X for node i in year t is

ΔX_{i,t} = X_{i,t} − X_{i,t−1},

and the pairwise divergence between two nodes is

D_{ij,t} = ΔV_i − ΔV_j.

Divergence measures relative change between two functions. It is not a proven transmission edge: a divergence between Lore and Helm says the two moved apart, not that one pushed the other.

The warrant for the scores is repeatability. Independent AI scoring passes, each rating the same eight nodes on the same four metrics, keep recovering the same structure in a society's history. We take that agreement as the warrant, not conformity with external indexes.

Our method is closer to reconstructing climate history from monastic and archival records. We trust no single chronicler, but when independent records agree on the hard winters and the good harvests, we take the pattern as real. The node scores are our monks' paperwork, and the metrics are the signal.

## 4. The signature pack

These signatures are defined as rules on node scores. Thresholds are the frozen values from the pre-registered run, set as pooled percentiles over changing years before any outcome was counted.

- **Archive amnesia.** The 5-year change in Archive V is ≤ −5.77 (P5), while Helm V or Shield V rises over the same 5 years. Memory is stripped as the command nodes gain.
- **Helm Incompetence.** Over 3 years, Helm K falls by at least 1.0 (P10) and Helm S rises by at least 2.0 (P90), while Helm A holds or rises. Meanwhile mean K of Craft and Flow stays at or above its trailing 10-year mean. In short, a capable society with an incapable helm.
- **Helm isolation.** The gap between Helm C and the mean C of the other seven nodes is ≥ 1.86 (P95).
- **Shield inversion.** Shield is the highest-valued node, and Flow V minus the mean V is ≤ −1.18 (P25). This is the pattern of the 1930s and the 2020s.
- **Late abstraction collapse.** Some node has A ≥ 7.8 (P90) while its 2-year change in C is ≤ −1.2 (P5). Meaning has decoupled from organisation.
- **Narrative–material gap.** Mean A of Lore and Archive is ≥ 8.0 (P90), while mean K of Flow, Craft and Hands falls 0.9 or more below its trailing 10-year mean (P10).
- **Lore–Helm divergence** (draft only). The 5-year change in the Lore–Helm V gap is at least 2 in either direction. At that threshold it fires in about 30% of years, so we treat it as a background rate rather than a signature.

## 5. Evidence

The frozen rules produced 696 firing years across JUNO v3. Six series were excluded: Denmark (Stress recorded as zero), Israel before 2015 (a flat series), India and Turkey (sampled years only), Thailand (5-yearly) and Canada (too short). Step-coded series are labelled.

To test enrichment, we compared each signature's firing years with a shuffle null: each society's firing series is circularly shifted, 10,000 times.

**Named windows.** All four signatures tested against named windows fire inside them far more often than chance:

| Signature | Firings in windows | Expected by chance | p |
|---|---|---|---|
| Archive amnesia | 15 | 0.75 | 0.0001 |
| Helm isolation | 17 | 8.41 | 0.0043 |
| Shield inversion | 24 | 10.37 | 0.006 |
| Helm Incompetence | 4 | 0.18 | 0.0223 |

Twelve of the eighteen named windows survived the move from draft to frozen thresholds:
- Archive amnesia: Argentina 1976–78, Chile 1975–77, Cambodia 1975–78, Greece 1967–71 and Hong Kong 2020–24.
- Helm Incompetence: the UK 2016–19.
- Helm isolation: Venezuela 1999–2013 and Australia 2010–18.
- Shield inversion since 2015: Iran, Russia and Venezuela, plus Finland 2023–25.

Six did not survive: Laos 1975, Germany 1931 and Hong Kong 1997 (Archive amnesia), Germany 1939 and Pakistan 1968–70 (Helm Incompetence), and Iraq 2015–25 (Shield inversion).

**Before big falls.** A big fall is an annual drop of 2.0 or more in mean V, within the firing year or the three years after it.
- Late abstraction collapse is followed by a big fall in 62 cases against 38.56 expected (p = 0.0002).
- Helm Incompetence (14 against 9.25, p = 0.074) and the narrative–material gap (28 against 20.43, p = 0.072) lean the same way.

**Chains.** We counted how often one signature begins within five years of another in the same nation:
- Helm isolation and Shield inversion follow each other in both directions, each in 9 nations.
- Helm Incompetence is followed by Helm isolation in 5 nations, including the UK 2016→2018 and Venezuela 2008→2009.
- The narrative–material gap is followed by Late abstraction collapse in 5.
- Late abstraction collapse is followed by Helm isolation in 5 and by Shield inversion in 5.
- Archive amnesia is followed by Shield inversion in 4.

We state the limits plainly.
- **The windows are partly circular.** They were named from draft firings, so their enrichment shows that the pattern survives a stricter, pre-set threshold, not that it was found blind.
- **The chains need a base-rate comparison.** Signatures that share nodes (Helm in isolation and Incompetence; Lore and Archive abstraction in collapse and the narrative gap) can co-fire by construction.
- **Some series fire constantly.** We report them with their event rates rather than drop them:
  - Helm isolation: Venezuela 0.48 and Russia 0.22, against a pooled 0.046; Poland, Sweden and Saudi Arabia are also flagged.
  - Shield inversion: Russia 0.44, Venezuela 0.25, Argentina 0.21 and Japan 0.19, against a pooled 0.054.
  - Late abstraction collapse: Sweden 0.27 and South Africa 0.15, against a pooled 0.021.
  - Narrative–material gap: Sweden 0.13, Finland 0.11 and Brazil 0.07, against a pooled 0.011.
  
  A constant firer tells us how that series moves, and we read it that way.

## 6. Three levels of cross-scale claim

Our premise is that human organisation has scale-invariant properties. We separate three strengths of that claim:
- **A. Structural transfer.** The same eight functions and the same kinds of pathology appear at every scale.
- **B. Rule transfer.** The frozen national rules, applied unchanged, pick out the same pathologies in firms and institutions.
- **C. Calibrated transfer.** The thresholds and outcome rates carry across scales quantitatively.

At present we claim scale-recurring topology, which is level A. Level B is the next test.

## 7. The next test (P4): corporations

We will score firms on the same eight nodes using this rubric:

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

The cases are Boeing, Theranos and WorldCom, each with matched control firms. We will run the frozen national rules unchanged and report every firing, including false positives in the controls. We will also compare against simpler baselines: stress change alone, and first differences of V.

We keep two registers apart. The discovery register holds patterns found in data already seen. The prospective watchlist holds firings recorded before outcomes are known.

## 8. The recovery catalogue

Pathology is half the programme; recovery is the other half. Alongside the signature pack we will catalogue recovery topologies:
- Archive recovery: memory rebuilt after amnesia.
- Helm recoupling: the helm's coherence rejoining the other nodes.
- Shield restraint: coercion stepping back from the lead.
- Lore–material reconnection: narrative coming back into contact with Flow, Craft and Hands.

Each will be defined by node rules in the same way and searched for across the same corpus.

## 9. Purpose

Our purpose is to understand the architecture through which organisations maintain, lose and recover their capacity to cooperate.
