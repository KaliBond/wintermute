# THE JUNO FORMULATION — CAMS Project Instructions

*Paste this file into a Claude project's "Project instructions" field to enable full CAMS/JUNO scoring and calculation.*

---

## What This Is

**CAMS** (Complex Adaptive Meta-System) is a framework for measuring how well any coordinating society — a nation, corporation, or city — acquires, processes, stores, and acts on information under real conditions. It models the society as an **eight-node bipartite network** where each node is an institutional function. The **JUNO v1.2-Final** operators then compute structural health from raw scores.

This project enables Claude to:
- Score any entity (nation / corporation / city) across time
- Compute Node Values and Bond Strengths from those scores
- Derive system-level health metrics (S/K ratio, praetorian index, cognitive gap, Robson Gauge)
- Classify the entity's attractor basin
- Produce ready-to-analyse CSV output

---

## Part 1: The Schema — Primitives

### The Eight Nodes

Nodes are split into two functional loops:

**Slow Loop** (reflective, teal): Helm, Lore, Archive, Stewards
**Fast Loop** (reactive, amber): Shield, Craft, Hands, Flow

| Node | Institutional Function | Key Question |
|---|---|---|
| **Helm** | Strategic direction and executive coordination | Can leadership formulate goals, align institutions, and execute decisions under pressure? |
| **Shield** | Security, order, and coercive force | Can the system defend itself, maintain internal order, and project force when required? |
| **Lore** | Knowledge synthesis and cultural legitimation | Can knowledge institutions generate, preserve, and legitimate shared understanding? |
| **Stewards** | Resource ownership and capital allocation | Can asset-controlling actors allocate resources productively to system needs? |
| **Craft** | Skilled production and professional coordination | Can specialised workers and professionals deliver sophisticated functions reliably? |
| **Hands** | Labour execution and basic throughput | Can the labouring base mobilise and perform large-scale practical work? |
| **Archive** | Institutional memory and information storage | Can critical information be preserved, retrieved, and transmitted across time? |
| **Flow** | Commerce, trade, and circulation | Can goods, services, money, and logistics move effectively through the system? |

### The Four Dimensions

Score each node on all four dimensions. All scores are **integers 1–10**.

| Dim | Label | What It Measures |
|---|---|---|
| **C** | Coherence | Internal consistency of the node; how well its parts work in concert |
| **K** | Capacity | Demonstrated performance; what the node actually delivers under current conditions |
| **S** | Stress | Entropy production rate; observed breakdown, fragmentation, or disorder being generated |
| **A** | Abstraction | Operational sophistication; the quality of formal models, rules, and planning the node deploys |

**Critical distinctions:**
- Score **operational truth**, not formal structure. A ministry may exist on paper but score K=2 if it delivers nothing.
- **Stress** is disorder *production*, not external pressure. A node under enormous pressure but functioning coherently may score S=3.
- High **Abstraction** is not automatically good. Bureaucratic abstraction with no delivery still produces stress.
- Score each node **independently**. Do not smooth scores to make the system look balanced.

### Scoring Anchors

| Score | C / K / A | S |
|---|---|---|
| 1–2 | Collapsed, absent, or catastrophically impaired | Near-total breakdown, entropy production near maximum |
| 3–4 | Severely degraded; partial function only | High fragmentation; significant disorder production |
| 5–6 | Moderate function; notable gaps or inefficiencies | Moderate stress; manageable but real disorder |
| 7–8 | Strong, reliable performance with minor weaknesses | Low stress; minor disorder production |
| 9–10 | Exceptional; best-practice operation | Near-zero disorder; functions regenerating coherence |

---

## Part 2: The JUNO v1.2-Final Calculations

These are computed **after** raw C/K/S/A scores are assigned. Never anticipate them during scoring.

### Node Value

```
V_i = C_i + K_i − S_i + 0.5 × A_i
```

Range: typically −8 to +25. Negative V indicates a node is generating more disorder than it contributes capacity. This is structurally serious.

### Bond Strength

For each node *i*, compute pairwise bond strength with every other node *j*, then average the seven values.

**Step 1 — quality coefficient** for each node:
```
q_i = (0.6 × C_i + 0.4 × A_i) / 10
```

**Step 2 — pairwise bond** between nodes i and j:
```
B_ij = sqrt(q_i × q_j) × 2^(−(S_i + S_j) / 10)
```

**Step 3 — node Bond Strength** (mean of 7 pairwise values):
```
BS_i = mean(B_ij for all j ≠ i)
```

Bond Strength ranges 0–1. High stress suppresses bonds exponentially — this is the mechanism by which local node failure propagates system-wide fragmentation.

### Python Implementation

```python
import math

NODES = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow']

def juno_calc(scores):
    """
    scores: dict {node: {'C':int,'K':int,'S':int,'A':int}}
    returns: dict {node: {'V':float, 'BS':float}}
    """
    n = NODES
    q = {nd: (0.6*scores[nd]['C'] + 0.4*scores[nd]['A']) / 10.0 for nd in n}
    
    results = {}
    for i, nd in enumerate(n):
        V = scores[nd]['C'] + scores[nd]['K'] - scores[nd]['S'] + 0.5*scores[nd]['A']
        bonds = []
        for j, nd2 in enumerate(n):
            if i != j:
                b = math.sqrt(q[nd]*q[nd2]) * (2.0 ** (-(scores[nd]['S']+scores[nd2]['S'])/10.0))
                bonds.append(b)
        results[nd] = {'V': round(V, 3), 'BS': round(sum(bonds)/len(bonds), 5)}
    return results
```

### System-Level Derived Metrics

From the eight node scores per year, compute:

```
S/K ratio (sk)     = mean(S across 8 nodes) / mean(K across 8 nodes)
praetorian_index   = V(Shield) − V(Helm)      # positive = military over-reach
cognitive_gap      = mean(V for slow nodes) − mean(V for fast nodes)
                     # negative = fast loop outpacing slow = cognitive capture risk
dominant_node      = node with highest V
weakest_node       = node with lowest V
system_bond (Λ)    = mean(BS across 8 nodes)  # overall coupling coherence
```

**Threat interpretation:**

| S/K ratio | System State |
|---|---|
| < 0.8 | Healthy — capacity well exceeds stress |
| 0.8–1.0 | Moderate — stress approaching capacity |
| 1.0–1.4 | Elevated — stress eroding capacity; watch for cascade |
| > 1.4 | Critical — stress exceeding capacity; collapse risk |

---

## Part 3: Output Formats

### Block 1 — Scored Dataset (standard CSV)

```csv
Society,Year,Node,Coherence,Capacity,Stress,Abstraction,Node Value,Bond Strength
```

Eight rows per year. Node Value and Bond Strength are computed values, not raw inputs.

### Block 2 — Ensemble Envelope (multi-scorer only)

```csv
Society,Year,Node,C_sd,K_sd,S_sd,A_sd,V_range,V_min,V_max
```

Only produced when running a five-scorer ensemble (CAMNATIONS5 protocol). SD computed with ddof=1. V_range and V_min/V_max computed from per-scorer Node Values, not from averaged scores.

### System Metrics CSV (optional companion)

```csv
Society,Year,sk,praetorian,cog_gap,dominant,weakest,system_bond
```

---

## Part 4: Basin Classification

Classify each year using **sk**, **praetorian_index**, **cognitive_gap**, and **system_bond (Λ)**:

| Basin | Condition | Meaning |
|---|---|---|
| **Sovereign Optimum** | sk<0.8, Λ>0.45, praetorian near 0 | Peak coordination; all nodes coupled and functional |
| **Managed Tension** | sk 0.8–1.0, Λ>0.35 | Stress present but system absorbing it through intact bonds |
| **Praetorian Drift** | praetorian>2.0 | Shield significantly outvaluing Helm; executive authority eroding |
| **Cognitive Capture** | cog_gap<−3.0 | Fast loop (reactive) outpacing slow loop (reflective); legitimacy at risk |
| **Bond Decay** | Λ<0.25, sk<1.2 | Coupling failing without acute stress; institutional disconnection |
| **Fragmentation** | sk>1.2, Λ<0.30 | Stress exceeding capacity and bonds failing simultaneously |
| **Collapse Cascade** | sk>1.4, multiple nodes V<0 | Multiple nodes net-negative; systemic failure in progress |

---

## Part 5: The Robson Gauge (η_soil)

A specialised metric for **long-run cultural grounding** — how well slow-loop nodes (Lore and Archive) can anchor the system against fast-loop volatility.

```
η_soil = (BS_Lore × BS_Archive × C_Hands) / (S_Hands × σ_V + ε)
```

Where:
- `BS_Lore` = Bond Strength of Lore node
- `BS_Archive` = Bond Strength of Archive node
- `C_Hands` = Coherence score of Hands node
- `S_Hands` = Stress score of Hands node
- `σ_V` = standard deviation of all eight Node Values (ddof=0)
- `ε` = 2.0 (prevents division by zero)

### Path Classification

| Path Type | Condition | Meaning |
|---|---|---|
| **Ruptured** | η < 0.005 | Cultural grounding collapsed; system floating without anchor |
| **Failed Transplant** | η < 0.02 and σ_V > 3.5 | Institutional forms imported but not embedded in practice |
| **Standard Transplant** | η < 0.08 and σ_V > 2.0 | Moderate grounding with high variance between nodes |
| **Intermediate / Uncertain** | η 0.08–0.15 | Transitional; interpretation requires trajectory context |
| **Young Ancient Forest** | 0.08 ≤ η ≤ 0.15 and σ_V < 2.0 | Emerging grounding; coordination tightening |
| **Deep Ancient Forest** | η > 0.15 and σ_V < 2.0 | Strong cultural grounding with tight node cohesion |
| **Greenhouse Garden** | η > 0.15 and σ_V ≥ 2.0 | Strong grounding but heterogeneous node performance |

---

## Part 6: Adapting to Entity Types

### For Nations

Score institutional nodes at the national level. Use primary sources: GDP/capita for K proxies on Stewards/Craft/Hands; press freedom and academic output for Lore; military spending and conflict data for Shield; trade volume for Flow; archival and legal system quality for Archive.

Crisis anchors:
- Wars, coups, famines → Helm S↑, Shield S varies, Flow S↑, Hands S↑
- Financial crises → Stewards S↑↑, Flow S↑↑, Helm S↑
- Legitimacy crises → Lore S↑, Archive S↑, Helm S↑

### For Corporations

Map the eight nodes onto corporate functions:

| CAMS Node | Corporate Equivalent |
|---|---|
| Helm | Board / C-suite strategic direction |
| Shield | Risk management, compliance, legal defence |
| Lore | R&D, IP, brand narrative, organisational culture |
| Stewards | Finance, capital allocation, investor relations |
| Craft | Operations, product development, professional workforce |
| Hands | Front-line labour, manufacturing, service delivery |
| Archive | Data systems, institutional memory, knowledge management |
| Flow | Sales, distribution, supply chain, market access |

Corporate stress markers:
- Regulatory investigation → Shield S↑, Archive S↑
- Supply chain disruption → Flow S↑, Hands S↑, Craft S↑
- Leadership crisis → Helm S↑↑, Lore S↑
- Credit downgrade / cash crisis → Stewards S↑↑, Flow S↑

Score based on **fiscal year data**, analyst reports, workforce surveys, and regulatory filings. Coherence = internal integration; Capacity = output delivery; Stress = disorder production (turnover, failures, investigations, write-downs); Abstraction = planning and formalisation quality.

### For Cities

Map nodes onto municipal functions:

| CAMS Node | City / Municipal Equivalent |
|---|---|
| Helm | Mayor / council / planning authority |
| Shield | Police, fire, emergency management |
| Lore | Education system, cultural institutions, civic identity |
| Stewards | Property, municipal finance, development authorities |
| Craft | Infrastructure, utilities, technical services |
| Hands | Municipal workforce, sanitation, basic services |
| Archive | Records management, permits, planning memory |
| Flow | Local commerce, transport, economic circulation |

City stress markers:
- Fiscal crisis → Stewards S↑↑, Hands S↑, Flow S↑
- Crime surge → Shield S↑, Hands S↑, Lore S↑
- Infrastructure failure → Craft S↑↑, Hands S↑
- Rapid population change → Helm S↑, Stewards S↑, Archive S↑

Use census data, crime statistics, fiscal reports, infrastructure audits, and planning records as evidence base.

---

## Part 7: Workflow

### Single-Pass (Quick Analysis)

1. State the entity and year range
2. Score each year: 8 nodes × 4 dimensions = 32 integers per year
3. Compute Node Value and Bond Strength via JUNO formulas
4. Output Block 1 CSV
5. Compute derived metrics (sk, praetorian, cog_gap, Λ)
6. Classify basin per year
7. Optionally compute η_soil (Robson Gauge)

### Five-Scorer Ensemble (Publication-Quality)

Run five independent scoring passes. No narrative summary before or between passes — this collapses variance. Passes use different interpretive stances:

- **Pass 1**: Balanced / institutional reference scorer
- **Pass 2**: Governance-skeptic (Helm and Archive scored pessimistically; S+1 when S≥5)
- **Pass 3**: Economic-optimist (Stewards, Craft, Flow K+1, S-1 in non-crisis years)
- **Pass 4**: Social-pessimist (Lore C-1, S+1; Archive S+1 in high-stress years)
- **Pass 5**: Security-realist (Shield C+1, K+1; Shield S-1; Lore C+1 when C≤3)

Average C/K/S/A across passes (ddof=1 SD for envelope). Run JUNO on the means.

### Output to CSV

Request Claude output as a CSV code block for easy copy-paste into a spreadsheet or Python script.

---

## Part 8: Prompting Templates

### Score a Nation
```
Using the CAMS framework in your project instructions, score [NATION] from [START_YEAR] to [END_YEAR] in 5-year intervals. For each year, produce 8 rows (one per node) with columns: Society, Year, Node, Coherence, Capacity, Stress, Abstraction. Then compute Node Value and Bond Strength using the JUNO v1.2-Final formulas. Output as a single CSV code block.
```

### Score a Corporation
```
Using the CAMS framework adapted for corporations (per your project instructions), score [COMPANY] from [START_YEAR] to [END_YEAR] annually. Use corporate equivalents for each node. Output as a CSV code block with columns: Society, Year, Node, Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength.
```

### Score a City
```
Using the CAMS framework adapted for municipalities (per your project instructions), score [CITY] from [START_YEAR] to [END_YEAR]. Map nodes to municipal functions as specified. Output Block 1 CSV, then compute the S/K ratio and basin classification for each year.
```

### Run the Robson Gauge
```
Given the CAMS Block 1 CSV below, compute the Robson Gauge (η_soil) for each year using: η = (BS_Lore × BS_Archive × C_Hands) / (S_Hands × σ_V + 2.0), where σ_V = std of 8 Node Values (ddof=0). Classify each year by path type. Output: Year, eta_soil, sigma_V, path_type.

[paste CSV here]
```

---

## Part 9: Five Projects Strategy for Free Claude Accounts

The free tier allows five projects. A recommended allocation:

| Project | Purpose | What to paste |
|---|---|---|
| **CAMS-Nations** | Historical and current national scoring | This full file |
| **CAMS-Corp** | Corporate scoring for specific companies | This file + company-specific context |
| **CAMS-Cities** | Municipal / urban scoring | This file + city data sources |
| **CAMS-Calc** | Pure calculation engine | Part 2 + Part 5 only (lightweight) |
| **CAMS-Analysis** | Interpretation and visualisation | Part 4 + Part 5 + specific datasets |

Each project retains conversation history, so scoring runs accumulate over time within a project.

---

## Reference: CAMS Node Groupings

**Slow Loop** (Coherence-dominant, longer institutional timescales):
Helm, Lore, Archive, Stewards

**Fast Loop** (Capacity-dominant, shorter operational timescales):
Shield, Craft, Hands, Flow

A healthy system has both loops tightly coupled (high system_bond Λ) with the slow loop maintaining the cognitive frame within which the fast loop operates. Cognitive capture occurs when the fast loop consistently outperforms the slow loop over multiple years — the system loses its reflective capacity.

**Bipartite structure**: Slow and fast loops form a bipartite coupling graph. Cross-loop bonds (e.g., Helm–Craft, Lore–Flow) are typically stronger than within-loop bonds because they bridge different timescales. Collapse often appears first as weakening cross-loop bonds.

---

*JUNO v1.2-Final operators. CAMS Scoring Protocol v2.1. For the full mathematical derivation, see: appendix-eight-node-optimum.md and CAMS_Paper3_EES_Coordination_Formalism.pdf in the wintermute repository.*
