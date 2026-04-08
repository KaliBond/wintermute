# CAMS-CAN Master Reference
## Common Adaptive Model of Society – Catch-All Network

**Repository Master File | Single Source of Truth**

- **Originator**: Kari Freyr McKern  
- **Origin Date**: 27 September 2024  
- **First Public Appearance**: *Pearls and Irritations*, October 2024  
- **Current Version**: 1.0-RC1 (Release Candidate)  
- **Last Updated**: 8 April 2026  
- **License**: All rights reserved – Kari Freyr McKern  
- **Status**: Active development; Bond Strength denominator pending final confirmation

---

## Table of Contents

1. [Framework Identity](#1-framework-identity)
2. [Canonical Principles](#2-canonical-principles)
3. [Node Architecture](#3-node-architecture)
4. [Canonical Metrics](#4-canonical-metrics)
5. [Core Formulas](#5-core-formulas)
6. [Thermodynamic Formalization](#6-thermodynamic-formalization)
7. [System-Level Dynamics](#7-system-level-dynamics)
8. [Phase Space & Bifurcation Thresholds](#8-phase-space--bifurcation-thresholds)
9. [Node-Level Metrics & Ranges](#9-node-level-metrics--ranges)
10. [Critical Numerical Thresholds](#10-critical-numerical-thresholds)
11. [Scoring Rules & Methodology](#11-scoring-rules--methodology)
12. [Output Standards](#12-output-standards)
13. [Canonical Workflow](#13-canonical-workflow)
14. [Known Limitations & Open Questions](#14-known-limitations--open-questions)
15. [Version History & Changelog](#15-version-history--changelog)
16. [Decision Register](#16-decision-register)
17. [Repository Rules](#17-repository-rules)

---

## 1. Framework Identity

### Official Definition

**CAMS-CAN** (Common Adaptive Model of Society – Catch-All Network) is a complex adaptive systems framework for analyzing nations, polities, institutions, and other organized social systems through a consistent node-based architecture.

CAMS-CAN treats societies as dissipative thermodynamic systems with eight invariant functional nodes and four measurement metrics, creating a 32-dimensional measurement space. The framework is *Keplerian, not Newtonian*: empirically calibrated rather than derived from first principles.

### Authoritative Status

**MASTER PRINCIPLE**: If any repository file, issue, branch, paper draft, scoring sheet, script, website page, or external publication conflicts with this file, **this file takes precedence**.

### Historical Context

CAMS originated in **mid-2024** through collaboration with AI as a forcing function for logical rigor, emerging from epistemological frustration with the fact-value distinction in social science and inspired by Socratic realism.

Key publications positioning CAMS as the third major unification attempt in social science (after Comte and cliodynamics) appeared in *Pearls and Irritations* (Oct 2024–Mar 2024), with geopolitical prescience validation published in Kari's 2022–2024 essay series.

---

## 2. Canonical Principles

### Principle 1: Platform Independence
CAMS-CAN is platform-independent. Any AI system, script, spreadsheet, data visualization, or human workflow must use this file as the reference standard.

### Principle 2: Canonical Beats Historical
Older files may be analytically valuable and preserve project history, but they are not automatically authoritative. This file supersedes all prior versions.

### Principle 3: Explicit Definitions Required
Every metric, formula, node label, scale, threshold, and workflow must be stated here in plain language. Vague or contextual definitions are not permitted in canonical reference.

### Principle 4: Mandatory Versioning
Any future change to formulas, node definitions, metrics, scales, threshold values, or canonical workflow must be logged in the changelog **before** implementation.

### Principle 5: No Silent Drift
If uncertainty or ambiguity arises during implementation, flag it in the Decision Register rather than silently improvising.

---

## 3. Node Architecture

### Canonical Eight-Node Model

The current canonical model uses **eight primary functional nodes**:

| # | Canonical Node | Functional Definition | Historical/Analogous Labels |
|---|---|---|---|
| 1 | **Executive** | Governance, leadership, executive decision-making, state apparatus | Helm, Government, Leadership, Monarchy, Oligarchy |
| 2 | **Army** | Military, armed forces, security apparatus, coercive power | Shield, Military, Defense, Warriors, Armed Forces |
| 3 | **Knowledge** | Education, research, priesthood, symbolic authority, intellectual systems | Lore, Priests, Clergy, Academy, Knowledge Class |
| 4 | **Property** | Asset owners, landowners, capitalists, resource controllers | Stewards, Landlords, Merchant-princes, Owners |
| 5 | **Trades** | Manufacturing, crafts, production, artisans, industrial system | Craft, Professions, Guild, Factory Workers, Producers |
| 6 | **Labor** | Proletariat, working class, services, manual labor | Hands, Workers, Servants, Laborers, Commons |
| 7 | **Memory** | Archives, cultural memory, tradition, institutional record, history | Archive, State Memory, Keepers, Historians, Traditions |
| 8 | **Commerce** | Finance, trade, merchants, markets, economic flows | Flow, Merchants, Commerce, Banking, Markets |

### Node Definition Rules

1. **These are functional categories, not culture-bound labels.**  
   Local and historical forms vary widely. Byzantine court structure differs from parliamentary democracy, which differs from feudal hierarchy—but all map to the same functional nodes.

2. **Scoring must map back to canonical types.**  
   If analyzing a society with unfamiliar social structures, identify the functional equivalent and document the mapping in analysis notes.

3. **One-to-one mapping required.**  
   Each empirical actor or institution maps to exactly one node. If a structure spans multiple nodes (e.g., a religious military order), decompose it into constituent parts.

4. **External Environment is contextual, not primary.**  
   Environmental factors (climate, geography, pandemic, war, resources) are recorded as exogenous context and explanatory fields. They influence node behavior but are not themselves scored as primary nodes.

---

## 4. Canonical Metrics

Each of the eight nodes is scored independently on four core metrics. All metrics are ordinal or interval scales; no cardinality assumption is made beyond the scale endpoints.

### 4.1 Coherence (C)

**Definition**: Internal unity, alignment, legitimacy, and organizational consistency within the node.

**Scale**: 1 to 10 (integer)

**Interpretation**:
- **1–3 (Low)**: Fragmented, divided, incoherent, internally contested, fractious, legitimacy disputed
- **4–6 (Medium)**: Mixed internal alignment, some coordination, contested authority
- **7–9 (High)**: Unified, disciplined, mutually aligned, functionally integrated, strong legitimacy
- **10 (Maximum)**: Perfect alignment, no internal divisions, complete disciplinary coordination

**Scoring Notes**:
- Coherence reflects *internal* structure, not external power.
- A weak army (low Capacity) can have high Coherence (e.g., disciplined guerrilla force).
- Fragmented elites lower Executive Coherence even if institutional continuity is maintained.
- Do not confuse legitimacy with popular support; coherence measures internal functional alignment.

---

### 4.2 Capacity (K)

**Definition**: Material, organizational, financial, operational, and human ability to act effectively.

**Scale**: 1 to 10 (integer)

**Interpretation**:
- **1–3 (Low)**: Weak resources, poor infrastructure, limited reach, low execution capability
- **4–6 (Medium)**: Moderate resources, functional infrastructure, adequate execution
- **7–9 (High)**: Strong resources, effective deployment, high competence, reach and coordination
- **10 (Maximum)**: Abundant resources, peak organizational efficiency, highest-order execution

**Scoring Notes**:
- Capacity is material and organizational, not aspirational.
- Record actual resources deployed, not theoretical maximum.
- Labor node Capacity = available workforce, infrastructure, productive assets.
- Commerce node Capacity = financial volume, market access, trading networks.
- Stress and Capacity are linked: high Stress depletes Capacity over time (see dynamics below).

---

### 4.3 Stress (S)

**Definition**: Pressure, strain, crisis load, or destabilizing burden acting on the node.

**Scale**: 1 to 10 (integer, always positive; no negative stress in current formalization)

**New Formalization (November 2025 correction)**:
- Stress is always positive (1–10).
- Stress increases when the node experiences pressure, demand overload, resource constraint, or existential threat.
- Resilience under pressure is captured separately through **system-level dynamics** (bond strength, abstraction growth), not via negative stress encoding.

**Interpretation**:
- **1 (Minimal)**: No appreciable pressure or constraint
- **4–6 (Elevated)**: Sustained pressure, strain building, resources strained but not yet critical
- **7–8 (High)**: Severe pressure, crisis conditions, resource exhaustion imminent
- **9–10 (Extreme)**: System under existential threat, resource depletion, organizational collapse risk

**Scoring Notes**:
- Stress must never be assigned mechanically; it requires explicit justification and evidence.
- External Stress (warfare, embargo, pandemic) is recorded but attributed to the affected nodes.
- Do not assume lower Stress = resilience. Resilience is system behavior (bond strength, abstraction adaptation).

---

### 4.4 Abstraction (A)

**Definition**: Strategic sophistication, symbolic reach, intellectual depth, planning horizon, and adaptive complexity.

**Scale**: 1 to 10 (integer)

**Interpretation**:
- **1–3 (Low/Concrete)**: Reactive, immediate, tactical, literal-minded, narrow planning horizon, unable to model alternatives
- **4–6 (Mixed)**: Some strategic planning, mixed reactive and forward-thinking, limited symbolic modeling
- **7–9 (High/Sophisticated)**: Strategic, conceptual, adaptive, system-aware, long planning horizon, symbolic innovation
- **10 (Maximal)**: Peak abstract reasoning, institutional sophistication, world-modeling, civilizational-scope planning

**Scoring Notes**:
- **Critical**: Do not inflate Abstraction without evidence of genuine strategic or symbolic sophistication.
- Sophisticated rhetoric is not sophistication; look for evidence of adaptive strategy, institutional innovation, or long-horizon planning.
- High Abstraction + Low Coherence = "Brittle Sophistication" signature (Phase 2 warning).
- Executive node Abstraction = quality of state planning and policy design.
- Knowledge node Abstraction = depth and systemicity of intellectual production.

---

## 5. Core Formulas

### 5.1 Node Value (NV)

**Canonical Formula**:

```
NV_i = C_i + K_i - S_i + (0.5 × A_i)
```

**Components**:
- `C_i` = Coherence of node i (1–10)
- `K_i` = Capacity of node i (1–10)
- `S_i` = Stress of node i (1–10, always positive)
- `A_i` = Abstraction of node i (1–10)

**Interpretation**:
- Coherence and Capacity contribute equally and directly to node value.
- Stress is subtracted: higher stress reduces node value.
- Abstraction contributes at half weight (0.5×), reflecting its secondary role relative to direct capability.

**Range**:
- Minimum: 1 + 1 - 10 + (0.5 × 1) = **–7.5** (theoretical floor)
- Maximum: 10 + 10 - 1 + (0.5 × 10) = **28.5** (theoretical ceiling)
- Observed range across 32+ societies: **3 to 21**

**Example Calculation**:
```
Executive node: C=7, K=8, S=3, A=6
NV = 7 + 8 - 3 + (0.5 × 6) = 15 + 3 = 18
```

---

### 5.2 Bond Strength (B_ij) – Pairwise

**Canonical Intent** (Status: Under final confirmation):

Bond Strength measures the functional synergy and information flow between two nodes, accounting for:
- **Coherence alignment** (60% weight): Both nodes must be internally unified to cooperate.
- **Abstraction complementarity** (40% weight): Strategic sophistication enables joint problem-solving.
- **Stress penalty**: Nodes under extreme stress decouple, reducing bond strength.

**Provisional Canonical Form**:

```
B_ij = [0.6 × (C_i × C_j) + 0.4 × (A_i × A_j)] × stress_penalty
```

**Stress Penalty** (provisional):
```
stress_penalty = exp(–λ × (S_i + S_j) / 20)
where λ ≈ 1.0 (calibrated)
```

**Range**: Observed [1.6, 4.3] across datasets

**CRITICAL OPEN ISSUE**: The exact denominator normalization and stress-penalty convention must be confirmed before v1.0 freeze. All implementations must document which Bond Strength variant they are using.

---

### 5.3 System Bond Density (B̄)

**Definition**: Average bond strength across all node pairs.

```
B̄ = (1 / 28) × Σ B_ij  for all i ≠ j
```

(28 = number of unique pairs in 8-node system)

**Interpretation**:
- `B̄ > 0.95` (relative to baseline): Full network integrity, high integration
- `0.6 < B̄ < 0.95`: Network thinning, isolated subsystems emerging
- `B̄ < 0.70`: Severe fragmentation, near-complete network collapse

---

## 6. Thermodynamic Formalization

### 6.1 System-Level Metrics

The following scalar metrics describe the overall state of the system, derived from node-level measurements and bond structure.

#### [τ] Affective Tone (Surplus Capacity Ratio)

**Definition**: The ratio of mean system Capacity to mean Stress.

```
τ = mean(K) / mean(S)
```

**Interpretation**:
- τ > 1.5: Healthy surplus, system can absorb perturbations
- 1.2 < τ < 1.5: Strained but viable; reserves being drawn
- τ ≈ 1.0: Critical stress; minimal reserves
- τ < 0.7: Catastrophic; system at metabolic minimum
- τ < 0.6: Historical collapse point (observed in Rome, France 1789, XXU)

**Physics**: As τ declines, the system loses capacity to respond to shocks.

---

#### [ε] Cognitive Style Ratio (Abstraction-to-Reality Coupling)

**Definition**: The ratio of mean system Abstraction to Affective Tone.

```
ε = mean(A) / τ
```

**Interpretation**:
- 0.9 < ε < 1.1: Balanced cognition, abstraction matched to material capacity
- ε > 1.15: Cognitive overreach; abstractions exceed material grounding
- ε > 1.3: Critical overreach; system developing fragile sophisticated solutions
- ε > 1.4: Hollow or collapsed cognition; abstract systems no longer functional

**Physics**: When material capacity (τ) declines but abstraction (A) continues to rise, the system develops "brittle sophistication"—elaborate institutions that fail under stress.

---

#### [R] Mode Ratio (Deliberative vs. Reactive Balance)

**Definition**: The ratio of system-level deliberative (forward-planning) vs. reactive (crisis-driven) modes.

```
R = reactive_mode / deliberative_mode
```

**Interpretation**:
- R < 1: Deliberative dominant; system planning ahead
- 1 < R < 3: Mixed modes; some crisis response but planning maintained
- R > 3–4: Reactive dominant; system responding to immediate crises
- R > 5: Nearly pure reaction; no long-horizon planning capacity

**Physics**: High Stress and declining Capacity force reactive mode. Once R > 3, bifurcation approaches.

---

### 6.2 Decay Index (D)

**Definition**: Unified measure of system decay and fragmentation risk.

```
D(t) = w₁·O + w₂·T + w₃·L + w₄·max(0, R - 1)
```

**Components**:

```
[O] Overreach Penalty:
O = max(0, ε - 1)
Returns: scalar ∈ [0, 1]
Interpretation: Cognitive gap severity

[T] Strain Penalty:
T = 1 - τ = 1 - (mean(K) / mean(S))
Returns: scalar ∈ [0, 1]
Interpretation: Metabolic deficit severity

[L] Connectivity Loss:
L = 1 - (B̄ / B̄₀)
where: B̄ = current system bond density
       B̄₀ = baseline bond density (reference period)
Returns: scalar ∈ [0, 1]
Interpretation: Network fragmentation
L = 0: full network integrity
L = 1: total network collapse

[Mode Imbalance]:
max(0, R - 1) = penalty for reactive overweight
```

**Weights** (example, calibration ongoing):
```
w₁ = 1.0 (Overreach)
w₂ = 1.0 (Strain)
w₃ = 1.0 (Connectivity loss)
w₄ = 0.5 (Mode imbalance)
```

**Normalized Output**:
```
D(t) ∈ [0, 10] (normalized)
```

**Phase Interpretation**:
- D ≤ 1: Healthy integration (Phase 1)
- 2 < D < 4: Brittle sophistication (Phase 2)
- 4 ≤ D ≤ 6: Critical slowing, bifurcation onset (Phase 3)
- D > 7: Fragmentation imminent (Phase 4)
- **D ≥ 6: Bifurcation threshold crossed; reversibility questionable**

---

### 6.3 Dynamical Updates (Discrete Time)

The following equations describe how system state evolves under stress.

#### [ε evolution] Abstraction Growth Under Strain

```
ε_{t+1} = ε_t + a·(1 - τ_t)·(1 - B̄_t)
```

**Parameters**:
- a = learning/adaptation rate (unspecified; typically 0.01–0.1)
- (1 - τ_t) = strain magnitude
- (1 - B̄_t) = isolation magnitude

**Physics**: When material capacity declines (τ↓) and nodes are isolated (B̄↓), nodes adopt more abstract, internal solutions. This increases ε (cognitive overreach) as a compensatory strategy.

**Phenomenon**: Explains the "brittle sophistication" phase—elaborate institutional solutions developed under stress that fail when pressure increases further.

---

#### [τ evolution] Surplus Capacity Decay

```
τ_{t+1} = τ_t · [1 - ξ·(ε_t - 1)]
```

**Parameters**:
- ξ = erosion rate (unspecified; typically 0.01–0.1)
- (ε_t - 1) = overreach magnitude

**Physics**: Cognitive overreach erodes capacity. The more the system over-abstracts relative to material grounding, the faster it depletes available reserves through maladapted strategies.

**Feedback**: As τ declines → ε rises (via ε evolution equation) → τ declines faster (via τ evolution equation). Nonlinear feedback loop.

---

#### [Stochastic Perturbations]

```
Crisis amplitude ∝ D(t)
```

**Physics**: Higher decay index amplifies the impact of exogenous shocks (war, pandemic, trade disruption). Systems with D > 6 cannot absorb normal perturbations without bifurcating.

**Hysteresis**: Once D crosses ~6, system cannot recover even if shock subsides.

---

## 7. System-Level Dynamics

### 7.1 Causal Pathways to Collapse

#### **Path 1: Stress → Surplus Depletion → Decay**

```
S ↑ → τ ↓ → D ↑ → R ↑ (reactive mode)
```

Rising stress exhausts capacity. As reserves deplete, the system shifts from deliberative to reactive mode, losing ability to plan ahead. Decay index rises, bifurcation risk increases.

#### **Path 2: Isolation → Cognitive Overreach → Decay**

```
B̄ ↓ → ε ↑ → τ ↓ (via overreach erosion) → D ↑
```

Network fragmentation forces nodes into isolation. Isolated nodes develop internal abstract solutions. These solutions erode actual capacity. System becomes brittle.

#### **Path 3: Network Collapse Feedback (Nonlinear Amplification)**

```
D ↑ → (ε_{t+1}, τ_{t+1} feedback) → D ↑↑
```

Decay index rises faster as thresholds are crossed. The system enters a regime where small shocks produce large responses.

---

### 7.2 Recovery Impossibility Boundary

**Irreversibility Threshold**:

```
IF: (τ < 0.6) AND (Bond loss > 40%)
THEN: D > 6 (irreversibility threshold crossed)
       Recovery probability → 0 even with energy influx
```

**Historical Examples**:
- Rome (late 4th century): τ < 0.6, >40% bond loss
- France (1789): τ ≈ 0.55, structural fragmentation
- XXU (end-state): τ < 0.5, complete network dissolution

---

## 8. Phase Space & Bifurcation Thresholds

### 8.1 Phase 1: High Integration (Stable Equilibrium)

**Conditions**:
```
ε ∈ [0.9, 1.1] AND τ > 1.4 AND B̄/B̄₀ > 0.95
D ≤ 1
```

**Characteristics**:
- Coherent planning (deliberative mode dominant)
- Strong surplus capacity
- Intact network integration
- Low bifurcation risk
- System absorbs perturbations without phase change

---

### 8.2 Phase 2: Brittle Sophistication (Unstable Equilibrium)

**Conditions**:
```
ε > 1.15 (rising) AND τ ∈ [1.2, 1.4] (declining) AND B̄: –5% to –15% loss
2 < D < 4
```

**Characteristics**:
- Cognitive overreach begins
- Institutional elaboration increases
- Surplus capacity declining but still present
- Network beginning to thin
- **Early warning signature**: D > 2

**Risk Profile**: System is stable but fragile. Small perturbations remain manageable, but margin for error shrinks.

---

### 8.3 Phase 3: Critical Slowing (Bifurcation Onset)

**Conditions**:
```
ε > 1.3 (peak) AND τ ≈ 1.0 AND R rising sharply AND recovery time variance increasing
4 < D < 6
```

**Characteristics**:
- Approaching metabolic minimum (τ ≈ 1.0)
- Cognitive overreach at peak
- Loss of adaptability; recovery from shocks takes longer
- Mode ratio rising sharply; reactive mode dominant
- **Bifurcation point**: D ≥ 6 (reversibility lost)

**Risk Profile**: System is in regime of critical slowing. Shocks produce disproportionate responses. Recovery trajectory becomes unpredictable.

---

### 8.4 Phase 4: Fragmentation / Dissociative Equilibrium (Breakdown)

**Conditions**:
```
[ε > 1.4 (hollow) OR ε < 1.0 (collapsed)] AND τ < 0.7 AND B̄/B̄₀ < 0.7
D > 7
```

**Characteristics**:
- Network disintegration; nodes operate independently
- Abstraction either hollow (detached from reality) or collapsed (unable to function)
- Severe metabolic constraint (τ < 0.7)
- **Irreversibility**: >40% bond loss means recovery impossible

**Historical Signature**: Rome (late 4th century), France (1789–1795), XXU (terminal phase)

---

## 9. Node-Level Metrics & Ranges

### 9.1 Coherence (C_i)

```
Range: [1, 10] integers
Crisis behavior: Drops 1–3 points during Phase 3→4 transition
Baseline: Most stable nodes fluctuate 6–8 during Phase 1–2
Collapse signature: C ≤ 3 across multiple nodes
```

---

### 9.2 Capacity (K_i)

```
Range: [1, 10] integers
Relationship to τ: τ = mean(K) / mean(S)
Phase 1: mean(K) > 8
Phase 2: mean(K) ∈ [6, 8]
Phase 3: mean(K) ∈ [4, 6]
Phase 4: mean(K) < 4
```

---

### 9.3 Stress (S_i)

```
Range: [1, 10] integers (always positive after November 2025 correction)
Scale: 1 = unstressed, 9–10 = extreme crisis
Crisis escalation: S typically rises 2–3 points in acute phase (1–3 years)
Chronic stress: S can remain at 7–8 for years in Phase 2–3
```

---

### 9.4 Abstraction (A_i)

```
Range: [1, 10] integers
Brittle signature: High A (7–9) with low C (4–6) = Phase 2 instability indicator
Overreach signature: A ≥ 8 while τ < 1.3 = cognitive-reality gap
Collapse signature: A peaks then collapses (ε > 1.4 then ε < 1.0)
```

---

### 9.5 Node Value (NV_i)

```
Formula: NV_i = C_i + K_i - S_i + (0.5 × A_i)
Observed range: [3, 21] across 32+ datasets
Typical range: [8, 18]
Crisis: NV drops 2–4 points during Phase 3 crisis
```

---

### 9.6 Bond Strength (B_i) – Per-Node Aggregate

```
B_i = (1/7) × Σ B_ij  for all j ≠ i

Interpretation: Average outbound bond strength from node i
Observed range: [1.6, 4.3]
Phase 1: B_i > 3.5 for most nodes
Phase 2: B_i ∈ [2.5, 3.5]
Phase 3: B_i ∈ [1.5, 2.5]
Phase 4: B_i < 1.8 (severe fragmentation)
```

---

## 10. Critical Numerical Thresholds

### Master Threshold Table

| Metric | Value Range | Interpretation | Phase(s) | Historical Case(s) |
|--------|-------------|---|---|---|
| **τ** (Affective Tone) | | | | |
| | > 1.5 | Healthy surplus | Phase 1 | Sweden 1950–2000 |
| | 1.2–1.4 | Strained but viable | Phase 2 | USA 1990–2020 |
| | ≈ 1.0 | Critical stress | Phase 3 | Russia 1990–1995 |
| | 0.7–1.0 | Severe constraint | Phase 3–4 | Germany 1923 |
| | < 0.6 | Collapse point | Phase 4 | Rome 380–410, France 1788–1794 |
| **ε** (Cognitive Style) | | | | |
| | 0.9–1.1 | Balanced cognition | Phase 1 | Norway (sustained) |
| | 1.15–1.3 | Overreach begins | Phase 2 | USA 2000–2008 |
| | > 1.3 | Critical overreach | Phase 3 | Weimar 1930–1933 |
| | > 1.4 or < 1.0 | Hollow/collapsed | Phase 4 | XXU 1985–1991 |
| **B̄/B̄₀** (Bond Density) | | | | |
| | > 0.95 | Full integrity | Phase 1 | Singapore (sustained) |
| | 0.60–0.95 | Network thinning | Phase 2–3 | Germany 1900–1945 |
| | < 0.70 | Severe fragmentation | Phase 4 | Rome (late) |
| **Bond Loss** | | | | |
| | < 40% loss | Recoverable | Phase 2–3 early | Most Phase 2 systems |
| | > 40% loss | Irreversible | Phase 3–4 | Rome, France 1789 |
| **R** (Mode Ratio) | | | | |
| | < 1.0 | Deliberative dominant | Phase 1 | Sweden, Norway |
| | 1–3 | Mixed | Phase 2 | Most transitioning systems |
| | 3–4 | Reactive rising | Phase 3 | USA crisis periods |
| | > 5 | Nearly pure reaction | Phase 4 | Collapse regimes |
| **D** (Decay Index) | | | | |
| | 0–1 | Healthy | Phase 1 | Stable societies |
| | 2–4 | Brittle | Phase 2 | Early warning |
| | 4–6 | Critical (D ≥ 6 = bifurcation) | Phase 3 | Approaching breakdown |
| | > 7 | Imminent collapse | Phase 4 | Irreversible regime |

---

## 11. Scoring Rules & Methodology

### Rule 1: Consistency Across Nodes
Use the same criteria for all nodes. Coherence of Executive = Coherence of Army = Coherence of Knowledge, using identical interpretation.

### Rule 2: No Abstraction Inflation
Do not inflate Abstraction without evidence of genuine strategic sophistication. Look for:
- Evidence of adaptive strategy (long-horizon planning)
- Institutional innovation
- System-aware decision-making
- Symbolic or intellectual depth

Sophisticated rhetoric ≠ sophistication.

### Rule 3: Explicit Stress Justification
Negative Stress (in older formulations) must be justified by demonstrated resilience, not assumed. Under the new formalization, Stress is always positive; resilience is captured in system dynamics (bond strength, abstraction adaptation).

### Rule 4: Historical Context Matters
Score systems relative to their historical period, not presentist standards. Agrarian societies ≠ industrial societies ≠ information societies. Abstraction expectations differ.

### Rule 5: Explainability
Every score should be explainable in 1–2 sentences with reference to evidence.

### Rule 6: Uncertainty Documentation
When high uncertainty exists, record confidence level:
- **High**: Multiple independent sources; consistent evidence
- **Medium**: Limited sources; some ambiguity
- **Low**: Single source; significant uncertainty

---

## 12. Output Standards

### 12.1 Canonical Scoring Record

Each scored record (one node, one year) should include, at minimum:

| Field | Format | Example |
|---|---|---|
| System / Nation | Text | "Rome" or "United States" |
| Year / Period | Integer or date range | 2020 or "1990–2000" |
| Node | Canonical node name | "Executive", "Army", "Commerce" |
| Coherence | Integer 1–10 | 7 |
| Capacity | Integer 1–10 | 8 |
| Stress | Integer 1–10 | 3 |
| Abstraction | Integer 1–10 | 6 |
| Node Value | Calculated | 18 |
| Bond Strength | Pairwise B_ij (8×8 matrix) | See Bond Strength matrix |
| Notes | Text | "High military capacity reflects post-WWII buildup; stress from Vietnam deployment" |
| Evidence / Source Trail | Text or citation | "Historical GDP data (Maddison), Military Records Archive" |
| Scorer | Text | "Kari McKern" or initials |
| Version Tag | Semantic version | "v1.0-RC1" |
| Confidence | High / Medium / Low | "High" |

### 12.2 Preferred Data Format

**CSV Header**:
```
System,Year,Node,Coherence,Capacity,Stress,Abstraction,NodeValue,BondStrength_Helm,BondStrength_Army,BondStrength_Knowledge,BondStrength_Property,BondStrength_Trades,BondStrength_Labor,BondStrength_Memory,BondStrength_Commerce,Notes,Evidence,Scorer,Version,Confidence
```

**Example Row**:
```
Rome,380,Executive,6,7,5,7,16.5,[bond values],Internal factionalism; Valens weakness,Gibbon Historical Decline,Kari McKern,v1.0-RC1,Medium
```

### 12.3 Summary Metrics (System Level)

For each system-year combination, also compute and record:

| Metric | Formula | Example |
|---|---|---|
| mean(C) | Average Coherence across 8 nodes | 6.4 |
| mean(K) | Average Capacity across 8 nodes | 6.8 |
| mean(S) | Average Stress across 8 nodes | 4.2 |
| mean(A) | Average Abstraction across 8 nodes | 5.9 |
| τ (Affective Tone) | mean(K) / mean(S) | 1.62 |
| ε (Cognitive Style) | mean(A) / τ | 3.64 |
| B̄ (Bond Density) | Average pairwise bond strength | 2.8 |
| D (Decay Index) | As per formula | 1.4 |
| Phase | Determined by D and thresholds | Phase 1 |

---

## 13. Canonical Workflow

### Step 1: Ingestion
Start from this file. No exceptions.

### Step 2: Node Mapping
Identify empirical actors and institutions in the target system. Map each to exactly one canonical node using the functional definitions in Section 3.

**Documentation**: Record the mapping explicitly (e.g., "Byzantine Court = Executive node"; "Imperial Legions = Army node").

### Step 3: Node Scoring
For each node in the target system, score on four metrics (C, K, S, A) using criteria in Section 4 and scoring rules in Section 11.

**Documentation**: Record evidence and sources for each score.

### Step 4: Calculate Node Value
For each node, apply formula in Section 5.1:
```
NV_i = C_i + K_i - S_i + (0.5 × A_i)
```

### Step 5: Calculate Bond Strength
Compute B_ij for all node pairs using the currently authorized formula variant (Section 5.2). Document which variant you are using.

### Step 6: Calculate System-Level Metrics
From node-level data, calculate:
- τ (mean(K) / mean(S))
- ε (mean(A) / τ)
- B̄ (average bond density)
- D (decay index)
- Phase assignment

### Step 7: Documentation
Record all results with notes, evidence trail, scorer identity, version tag, and confidence level.

### Step 8: Version Control
Save results with semantic version stamp (v0.1-draft, v1.0-RC1, v1.0, etc.). If a method changes, update the changelog **before** further scoring.

---

## 14. Known Limitations & Open Questions

### Open Formula Questions

| Item | Status | Impact |
|---|---|---|
| **Bond Strength denominator** | Pending confirmation | All implementations must document which variant used |
| **Stress-penalty function** | Provisional (exponential form) | Affects bond strength calculations; calibration ongoing |
| **Weights in Decay Index** | Provisional (w1–w4 example given) | May require empirical recalibration |
| **Learning/adaptation rates** | Unspecified (a, ξ) | Affect temporal dynamics; require calibration |

### Open Methodological Questions

| Question | Status | Path Forward |
|---|---|---|
| Should External Environment be a primary scored node? | Currently "contextual only" | Pending review; may upgrade to Node 9 in v1.1 |
| Should confidence scores be mandatory in all records? | Recommended but optional | Proposal: mandatory in v1.0-final |
| Should ontology mappings be standardized in separate file? | Not yet | Proposal: create CAMS-Ontology.md for node mapping examples |
| Can Phase assignments be automated? | Unclear | Automated assignment possible but requires validation |

### Data Limitations

1. **Historical Data Gaps**: Pre-1800 data for most societies is sparse and estimated.
2. **Metric Inference**: Some Abstraction and Bond Strength calculations are inferred from limited evidence.
3. **Cultural Context**: Societies with unfamiliar structures may not map neatly to canonical nodes.
4. **Temporal Resolution**: Most scoring is annual or decadal; finer temporal resolution would require data expansion.

---

## 15. Version History & Changelog

### v1.0-RC1 (Release Candidate) — 8 April 2026

**Status**: Final refinement before v1.0 release.

**Key Changes from Previous**:
- Confirmed eight-node canonical architecture (supersedes earlier ten-node variant)
- Confirmed four canonical metrics: Coherence, Capacity, Stress, Abstraction
- Confirmed Node Value formula: `NV = C + K - S + 0.5A`
- Finalized thermodynamic formalization (Affective Tone τ, Cognitive Style ε, Mode Ratio R)
- Finalized Decay Index formula (D) with phase thresholds
- Confirmed dynamical equations for τ and ε evolution
- Marked Bond Strength algebra for final confirmation (denominator pending)
- Reserved External Environment as contextual (pending explicit re-authorization)
- Documented all observed numerical thresholds from 32+ dataset validation
- Established irreversibility boundary: D ≥ 6, Bond Loss > 40%

**Known Issues**:
- Bond Strength denominator formula not finalized; all implementations document which variant used
- Stress penalty function in Bond Strength provisional (exponential form; alternative forms under review)
- Weights in Decay Index provisional (w1–w4 given as example; calibration ongoing)
- Learning and adaptation rates (a, ξ) not yet calibrated

**Pending Before v1.0-Final**:
- Final confirmation of Bond Strength equation
- Empirical recalibration of Decay Index weights
- Decision on External Environment node status
- Finalization of confidence score requirements

---

### v0.9 — January 2026

- Thermodynamic formalization completed
- Critical Slowing and Bifurcation thresholds documented
- 32+ society validation dataset completed
- Michael Hudson debt-militarization analysis completed

---

### v0.8 — November 2025

- **Critical Correction**: Stress metric sign convention clarified
  - Stress is always positive (1–10 range)
  - Resilience is captured in system dynamics, not negative stress encoding
  - Fixes thermodynamic inconsistency identified in October review

---

### v0.7 — September 2025

- Thermodynamic formalization initiated (τ, ε, R introduced)
- CAMS-CAN terminology standardized (eliminated inconsistencies with "Helm/Executive", etc.)

---

### v0.6 — August 2025

- CAMS-CAN applications developed for corporate foresight:
  - Boeing (supply chain fragmentation analysis)
  - GM (labor ecosystem stress modeling)
  - BYD (capacity planning under geopolitical constraint)
  - Tesla (supply chain and labor cohesion analysis)

---

### v0.5 — July 2025

- State space formalization begun
- Eight-node canonical architecture confirmed

---

### v0.1-draft — September 2024

- Framework origin; foundational definitions established
- Node architecture proposed (eight primary nodes)
- Four canonical metrics defined
- Node Value formula introduced
- Initial scoring on 8 test societies

---

### Pre-History

- **Origin**: 27 September 2024, collaboration with AI as forcing function for logical rigor
- **First Public Appearance**: *Pearls and Irritations*, October 2024
- **Intellectual Ancestry**: Comte's positivism, cliodynamics tradition, complexity science

---

## 16. Decision Register

### Open Decisions (Pending Resolution)

#### **OD-001: Final Bond Strength Equation**

**Status**: Under final confirmation

**Options**:
1. Finalize current provisional form: `B_ij = [0.6×(C_i×C_j) + 0.4×(A_i×A_j)] × exp(–λ·(S_i+S_j)/20)`
2. Explore alternative normalization (Dirichlet form, ratio form)
3. Calibrate stress penalty function (currently exponential; consider alternatives)

**Decision Required By**: Before v1.0-final release

**Owner**: Kari McKern

---

#### **OD-002: External Environment Node Status**

**Status**: Currently marked as "contextual only"

**Options**:
1. Keep as exogenous context layer (current)
2. Elevate to Node 9 in v1.1 (requires formula extensions)
3. Create separate "Environment Metrics" subsystem

**Rationale**: Environmental factors clearly influence all nodes, but the question is whether they warrant primary node status or remain explanatory fields.

**Decision Required By**: Before v1.1 planning

**Owner**: Kari McKern

---

#### **OD-003: Confidence Score Requirement**

**Status**: Recommended but currently optional

**Options**:
1. Make confidence scoring mandatory in all records (v1.0-final requirement)
2. Keep optional, encourage best practice
3. Create tiered confidence framework (High / Medium / Low / Very Low)

**Rationale**: Transparent uncertainty strengthens credibility; making it mandatory ensures consistency.

**Decision Required By**: Before v1.0-final release

**Owner**: Kari McKern

---

#### **OD-004: Ontology Standardization**

**Status**: Not yet formalized

**Options**:
1. Create separate CAMS-Ontology.md file with node mapping examples for 50+ societies
2. Embed ontology examples in main reference (Section 3 expansion)
3. Maintain separate examples folder in repository

**Rationale**: Standardized node mappings prevent drift and aid onboarding for collaborators.

**Decision Required By**: Before v1.0-final release

**Owner**: Kari McKern

---

### Resolved Decisions (Historical Record)

#### **RD-001: Stress Metric Sign Convention (November 2025)**

**Decision**: Stress is always positive (1–10 range). Resilience is captured through system-level dynamics (bond strength, abstraction evolution), not via negative stress encoding.

**Rationale**: Eliminates thermodynamic inconsistency; aligns with physical interpretation of stress as load/pressure.

**Implementation**: All scoring post-November 2025 uses positive stress only.

---

#### **RD-002: Eight-Node Canonical Model (July 2025)**

**Decision**: Adopt eight functional nodes as canonical (superseding earlier ten-node variant).

**Nodes**: Executive, Army, Knowledge, Property, Trades, Labor, Memory, Commerce

**Rationale**: Parsimony + functional completeness; maps to historical evidence across 32+ societies.

---

## 17. Repository Rules

### Rule 1: This File is Authoritative
Any conflict between this file and any other repository content is resolved in favor of this file.

### Rule 2: Mandatory Versioning
All changes to formulas, node definitions, scoring rules, or methodology must:
1. Be documented in the Version History section
2. Include rationale and implementation date
3. Receive sign-off from Kari McKern

### Rule 3: Branching Protocol
- `main`: Always contains stable, production-ready version (current v1.0-RC1)
- `develop`: Active development; experimental formulas documented with disclaimer
- `feature/*`: Topic branches for specific formula explorations; merged via pull request with documentation

### Rule 4: Pull Request Standards
- All PRs updating formulas must include: formula statement, rationale, validation against ≥1 test dataset, changelog entry
- All PRs updating node definitions must include: canonical statement, mapping examples, historical justification
- Trivial documentation fixes may bypass formula validation

### Rule 5: Data Integrity
- All scoring CSVs must include version tag and scorer identity
- No retroactive score changes without explicit note documenting original vs. revised values
- Historical data preserved in archive/ folder

### Rule 6: Issue Tracking
- Open questions (formulas, methodology) tracked in Issues with `[OPEN DECISION]` label
- Bug reports use `[BUG]` label
- Enhancement proposals use `[ENHANCEMENT]` label

### Rule 7: Silence is Not Consent
If uncertain about a formula, threshold, or mapping, flag it in an Issue rather than silently improvising.

---

## Metadata & Repository Information

**Repository URL**: [github.com/your-org/CAMS](https://github.com)  
**License**: All rights reserved – Kari Freyr McKern  
**Maintainer**: Kari McKern  
**Contact**: [Email / contact method]  

**Recommended Citation**:
```
McKern, Kari Freyr. (2026). "CAMS-CAN: Common Adaptive Model of Society – Catch-All Network. 
Master Reference." v1.0-RC1. [Repository URL].
```

**Companion Publications**:
- *Pearls and Irritations* essays (October 2024–March 2024)
- *The Architecture of Civilisation* (First Edition, 2026)
- CAMS-CAN Dataset Repository (32+ societies, 1800–2025)

---

## Document Notes

**This file was generated**: 8 April 2026

**Last reviewed**: 8 April 2026

**Next scheduled review**: 15 May 2026 (pre-v1.0-final)

**Feedback welcome**: Issues in repository; email to maintainer

---

**END OF MASTER REFERENCE**
