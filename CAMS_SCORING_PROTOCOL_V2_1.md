# CAMS Scoring Protocol v2.1

## Civilizational Analysis and Measurement System

### Scoring Layer Only

---

## Purpose

This protocol is for the **scoring stage only** of CAMS analysis.

Its purpose is to guide an evaluator — human or LLM — in assigning evidence-based scores to the **eight CAMS nodes** on four dimensions:

- **Coherence**
- **Capacity**
- **Stress**
- **Abstraction**

This protocol does **not** include downstream calculation.

Do **not** calculate:

- Node Value
- Bond Strength
- system-level averages
- any derived metric

Your task ends when you have produced a complete **8-node scoring snippet** in the required format.

---

## Philosophy & Approach

### Core Principle

You are analysing civilizational cognition through **institutional coordination capacity**.

Think thermodynamically: societies are metabolic systems that acquire, process, store, coordinate, and act on information. Your task is to assess how well each core function is operating under real historical conditions.

### Validator Stance: Evidence-Based Realism

When scoring, adopt the following position:

- **Capacity**: score based on **demonstrated performance**, not theoretical potential
- **Abstraction**: score based on **operational sophistication**, not aspirational complexity
- **Coherence**: score based on **actual coordination**, not formal structure
- **Stress**: score based on **entropy production** — that is, observed breakdown rate — not subjective discomfort

### Key Distinctions

- A node can be **structurally sophisticated** but **functionally impaired**
- A node can be under **heavy pressure** without being highly stressed, if it continues to function coherently
- A node can face **limited pressure** but still show high stress if it is fragmenting or failing

### Scoring Rule

**Score what is operationally true, not what ought to be true, and not what might be possible under better conditions.**

### Independence Rule

Score each node on the evidence for that node.

Do **not** adjust scores to make the total system appear balanced, elegant, symmetrical, or mathematically plausible.
Do **not** anticipate downstream formulas.

---

## Step 1: Define the Eight Nodes

| Node | Function | What to Assess |
|---|---|---|
| **Helm** | Strategic coordination and executive decision-making | Can leadership formulate direction, align institutions, and execute decisions? |
| **Shield** | Security provision and force projection | Can the system maintain order, defend itself, and project coercive force when required? |
| **Lore** | Knowledge synthesis and cultural legitimation | Can knowledge institutions generate, preserve, interpret, and legitimate shared understanding? |
| **Stewards** | Resource ownership and capital allocation | Can wealth-holding or asset-controlling actors allocate resources productively to system needs? |
| **Craft** | Skilled production and professional coordination | Can specialised workers and professionals deliver sophisticated functions reliably? |
| **Hands** | Labour execution and basic production | Can the labouring base mobilise and perform large-scale practical work? |
| **Archive** | Institutional memory and information storage | Can critical information be preserved, retrieved, and transmitted across time? |
| **Flow** | Commerce, trade, and circulation | Can goods, services, money, and logistical throughput move effectively through the system? |

---

## Step 2: Assign Scores on the Four Dimensions

All scores are integers from **1 to 10**.

---

## Step 3: Output Rules

Return exactly **8 rows**, one for each node:

- Helm
- Shield
- Lore
- Stewards
- Craft
- Hands
- Archive
- Flow

### Required fields

- Society
- Year
- Node
- Coherence
- Capacity
- Stress
- Abstraction

### Rules

- Scores must be integers from 1 to 10
- Each node must appear exactly once
- Return raw scores only
- Do not include notes
- Do not include explanations
- Do not calculate Node Value
- Do not calculate Bond Strength
- Do not calculate averages, totals, or derived metrics
- Do not add narrative commentary unless explicitly requested

### Preferred Output Format

```csv
Society,Year,Node,Coherence,Capacity,Stress,Abstraction
Markerx,1897,Helm,7,6,4,7
Markerx,1897,Shield,4,5,3,5
Markerx,1897,Lore,8,7,4,8
Markerx,1897,Stewards,5,4,7,6
Markerx,1897,Craft,6,6,4,7
Markerx,1897,Hands,4,4,8,5
Markerx,1897,Archive,8,7,3,7
Markerx,1897,Flow,4,6,6,6
```
