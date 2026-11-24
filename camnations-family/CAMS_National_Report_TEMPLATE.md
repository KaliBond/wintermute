# [Society] | CAMS national report, [first year]–[last year]

**Territorial and institutional scope** — [Define the polity, territory, and institutions covered. Record changes in coverage or borders that affect comparability.]  
**Assessment** — [Name the CAMNATIONS run and scoring protocol. Record the pass count, dates, data status, and source CSV links pinned to a version or commit.]  
**Prepared by** — [Author or team.]

## National overview

[Summarise the institutional configuration at the start and end of the period. Identify the most consequential changes. Give the years, nodes, metric values, and source IDs for major claims.]

## Institutional structure

| Node | Function | Institutions and changes in responsibility | Source IDs |
| --- | --- | --- | --- |
| Helm | Executive direction, governance, behavioural selection | […] | […] |
| Shield | Defence, security, threat response | […] | […] |
| Lore | Education, knowledge, shared models and meaning | […] | […] |
| Stewards | Ownership, land, stored wealth and asset deployment | […] | […] |
| Craft | Skilled production, engineering, manufacturing | […] | […] |
| Hands | Labour, workers, material throughput | […] | […] |
| Archive | Memory, records, law, cultural continuity | […] | […] |
| Flow | Trade, finance, logistics and circulation | […] | […] |

[Explain who performed each function in practice, where organisations shared functions, and how responsibility changed between periods.]

## Historical periods and turning points

| Year or period | Event or institutional shift | Nodes affected | Source IDs |
| --- | --- | --- | --- |
| […] | […] | […] | […] |

[Describe the baseline and sequence of changes. Distinguish temporary shocks, chronic strain, and structural breaks. Link historical interpretations to evidence and identify plausible competing explanations.]

## Plots and change over time

| Figure | Plot specification | Caption prompt |
| --- | --- | --- |
| 1. Node Value | Eight node time series on a shared scale. Label selected turning points. For ensembles, show the observed per-pass minimum and maximum where available. | Which nodes changed, when, and by how much? |
| 2. Bond Strength | Eight node time series on a shared scale, with the scoring run and bond operator identified. | Where did coupling change, and what institutional evidence bears on the change? |
| 3. Scoring dimensions | Node-by-year heatmaps for Coherence, Capacity, Stress, and Abstraction on a common 1–10 scale. | Which dimensions account for the largest changes in Node Value? |
| 4. Annual change in Node Value | Node-by-year heatmap with a diverging scale centred on zero. | Which changes coincide with documented turning points? |

[Insert plots and captions here. Keep node order and metric scales consistent across comparable reports. Mark missing observations as gaps. Define annual delta as the current year minus the preceding calendar year's value for the same node. Leave the first year and any comparison with a missing endpoint blank. For longer comparisons, cite both endpoints.]

| Comparison years | Node | Metric | Earlier value | Later value | Signed change | Historical interpretation and source IDs |
| --- | --- | --- | ---: | ---: | ---: | --- |
| […] | […] | […] | […] | […] | […] | […] |

## Methods and evidence

- **Coherence (C)** measures internal alignment, coordination, trust, and consistency.
- **Capacity (K)** measures the ability to act, mobilise resources, and sustain operations.
- **Stress (S)** measures internally experienced load, friction, and volatility.
- **Abstraction (A)** measures symbolic complexity, planning horizon, and the use of models.
- **Node Value** is `C + K − S + 0.5A`. Record score precision and how missing values were handled.
- **Bond Strength.** Identify the exact operator and version used in the source run. For the normalised CAMS calculator, `qᵢ = (0.6Cᵢ + 0.4Aᵢ)/10`, and `Bᵢⱼ = √(qᵢqⱼ) × 2^(−(Sᵢ + Sⱼ)/10)`. Each node's Bond Strength is the mean of its seven pairwise bonds. Check the operator before comparing across scoring runs or with a JUNO series.
- **Scorer disagreement.** For ensemble runs, link the companion envelope and report the sample standard deviations for C, K, S, and A where relevant. Describe the Node Value band as the observed spread across passes. For single-pass reports, omit this entry.
- **Evidence coverage.** Record material interpolation, weak coverage, and missing or contested evidence. Treat scores as evidence-based estimates.

| Source ID | Citation or archival reference | Locator | Coverage and limitations |
| --- | --- | --- | --- |
| […] | […] | […] | […] |

## Corpus record

[Save a copy for the society and assessment period. Keep the report with its score CSV, optional ensemble envelope, institutional profiles, event timeline, source register, delta table, and figures. Use stable source IDs across the tables and report. Index the society, coverage years, scoring method, and report path.]

See the [CAMNATIONS scoring run index](README.md) for the status and provenance of the repository's source runs.
