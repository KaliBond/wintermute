# CAMS Vision-Affect-Mythic National Report Skill

Generate sophisticated, literature-grade CAMS national diagnostic reports integrating systems metrics, vision-affect analysis, and mythopoetic language.

## What This Skill Does

Creates **full-length, structured CAMS diagnostic reports** that combine:
1. **Systems metrics** (Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength)
2. **Vision-Affect analysis** (Vision = Abstraction × Coherence; Affect = Capacity − Stress; Sigma = Vision × Affect)
3. **Mythopoetic translation** (governing metaphors, civilisational personas, gift/wound tensions, mythic attractors)
4. **Historical phase portraits** (comparing current state to 50–100+ year trends)
5. **Structural dynamics analysis** (decouplings, anchors, cascades, contradict ions)
6. **Risk and opportunity assessment** (forward-facing strategic implications)

## Triggers

Use this skill whenever you need to:
- Generate a CAMS national report for any society
- Analyse vision-affect states (clarifying which nodes are lucid vs. depleted)
- Translate systems metrics into lived experience and civilisational meaning
- Create mythopoetic language that is grounded in structural data
- Compare societies using the vision-affect framework
- Identify pathological signatures (decoupling, cascade, ossification, etc.)

Explicit triggers:
- "Generate a CAMS report for {{country}}"
- "Create a vision-affect analysis of {{society}}"
- "Write a CAMS national mood report"
- "Translate {{metrics}} into civilisational meaning"
- "What is the mythic attractor for {{node}}?"
- "Compare {{countries}} using vision-affect framework"

## Input Required

The skill requires:

1. **Dataset**: CSV or structured data with 8 nodes × 4 dimensions (Coherence, Capacity, Stress, Abstraction), ideally spanning 50+ years
   - Can optionally include calculated fields: Node Value, Bond Strength, Stress Resilience
   - Must have clear node labels: Helm, Shield, Archive, Lore, Stewards, Craft, Hands, Flow

2. **Current snapshot**: Most recent available year or rolling average (typically last 2–3 years)

3. **Society context**: 
   - Name and date/period
   - Rough historical periodisation (key eras, inflection points, regime transitions)
   - Known challenges or strengths

4. **Optional customisation**:
   - Stress convention (normal vs. inverted)
   - Alternative node vocabulary (e.g., Executive instead of Helm)
   - Mythic attractor preferences (optional — can use default Library/King/Temple/etc.)

## Output Structure

The skill generates an 12-section report:

1. **Executive Summary Phrase** — One-sentence governing image
2. **Executive Summary** — 4-5 paragraph overview of structure, mood, mythology
3. **Data Source & Method Notes** — Transparency on data and interpretation
4. **System Snapshot** — Table of key metrics and interpretation
5. **System Scalar Operators** — M(t), Y(t), Headroom, Λ, Θ, Ψ, Φ, χ with interpretation
6. **Current Node Table** — 8 nodes × (Coherence, Capacity, Stress, Abstraction, Node Value, Bond Strength, Vision, Affect, Sigma) with short readings
7. **Vision–Affect Reading** — Analysis of which nodes are lucid, depleted, strained, or functional
8. **Structural Dynamics** — 5 subsections:
   - Primary anomaly (main decoupling or overload)
   - Dominant anchor (which node is holding system together)
   - Main weakness (lowest-value or most depleted node)
   - Archive–Lore relation (memory vs. meaning)
   - Fast-loop/slow-loop balance
9. **Mythopoetic Layer** — 7 subsections:
   - National mood texture (phenomenological description)
   - Governing metaphor (one controlling image)
   - Civilisational persona (if system were a character)
   - Gift and wound (deepest strength and vulnerability)
   - Mythic tension (core contradiction)
   - Mythic alignment by node (using archetypal attractors)
   - One-line closing (earned, compressed image)
10. **Historical Phase Portrait** — Table showing M(t), Y(t), Headroom, Λ(t), Θ across 5–8 key eras
11. **Comparative Position** — Table comparing subject society to 3–4 peer societies
12. **Implications, Risk, and Opportunity** — Forward-looking analysis with 3 risks, strengths, and 1 main opportunity

## Key Concepts

### Vision-Affect Framework

- **Vision (V)** = Abstraction × Coherence
  - What the node can see/understand
  - High vision = clear sight; low vision = confusion
  
- **Affect (F)** = Capacity − Stress
  - Emotional/energetic surplus or deficit
  - Positive affect = has reserves; negative affect = depleted
  
- **Sigma (σ)** = Vision × Affect
  - Constructive potential of the node
  - σ > 100: Lucid and generative
  - σ 50–100: Functional
  - σ 0–50: Strained
  - σ < 0: Depleted or actively wounded

### Mythic Attractors

Standard mapping (customizable):
- **Helm** (governance) ↔ **King** (command, direction, will)
- **Shield** (defence/security) ↔ **Warrior** (protection, vigilance, order)
- **Archive** (institutional memory) ↔ **Library** (preservation, knowledge, law)
- **Lore** (narrative/meaning) ↔ **Temple** (belief, ritual, significance)
- **Stewards** (welfare/care) ↔ **Manor** (hospitality, obligation, stewardship)
- **Craft** (technical capacity) ↔ **Workshop** (mastery, creation, precision)
- **Hands** (labour) ↔ **Harvesters** (dignity, work, material substrate)
- **Flow** (circulation/trade) ↔ **Agora** (exchange, movement, commerce)

### Pathological Signatures

The skill identifies common structural failures:

- **Helm Collapse**: Helm σ < 0 while system continues; governance authority detached
- **Museum Effect**: Archive strong, Lore fragmenting; knowledge without meaning-making
- **Hands Dissolution**: Hands σ < 20; labour systems failing; dignity eroding
- **Lore Ossification**: Y(t) declining; inherited narrative not generating new meaning
- **Headroom Deficit**: K − S < 0; system running on reserve; shock absorption exhausted
- **Reactive Dominance**: Θ > 1.0; fast-loop overrides slow-loop; crisis mode persists
- **Mythic-Material Decoupling**: Large gap between Lore abstraction and Hands coherence

## Methodology Notes

### Vision-Affect Calculation

```
For each node i:
  Vision_i = Abstraction_i × Coherence_i
  Affect_i = Capacity_i − Stress_i
  Sigma_i = Vision_i × Affect_i
  
System-level:
  Mean Vision = mean(A_i × C_i across all nodes)
  System Affect = mean(K_i − S_i across all nodes)
  System Sigma = mean Vision × mean Affect (or sum of all Sigma_i / 8)
```

### Mythic Interpretation Rules

1. **Every image must be earned by metrics** — no poetry detached from data
2. **Metaphors must compress structure** — the image captures the relation between nodes
3. **Personas must reflect dominant configuration** — derived from node states, not invented
4. **Tensions must name real contradictions** — abstract from specific node patterns
5. **Mythopoetic close must translate, not embellish** — one sentence that says what the metrics mean

### Comparative Framing

When comparing societies, always:
- Place subject society first in tables
- Note distinctive advantages and vulnerabilities
- Avoid assuming all paths converge
- Flag unique structural positions (e.g., "only Western nation with negative headroom")
- Identify what peer societies can teach and what they cannot

## Example Outputs

Three completed examples are provided in this skill's directory:

1. **FRANCE_CAMS_Vision_Affect_Mythic_Report_2026.md**
   - Case: System Sigma negative; Headroom zero; Archive strong, Helm collapsed
   - Diagnostic: Lucid paralysis; Museum effect; reactive dominance
   - Phrase: "The Library remembers perfectly; the Temple has lost its congregation; the King cannot be heard."

2. **NORWAY_CAMS_Vision_Affect_Mythic_Report_2026.md**
   - Case: System Sigma highly positive; Headroom excellent; Flow dominates; Helm decoupled
   - Diagnostic: Material transcendence, narrative thinness; complacent excellence
   - Phrase: "The Library, Workshop, and Agora are luminous; the King is silent; the Temple whispers to itself."

3. **CAMS_National_Mood_Report_TEMPLATE.md**
   - Blank template for adapting to any society
   - All sections with placeholder structure

## Workflow

1. **Data intake**: Load dataset; check completeness and range
2. **Snapshot extraction**: Calculate metrics for current period (last 2–3 years averaged)
3. **Derivation**: Compute Vision, Affect, Sigma for all nodes and system totals
4. **Historical review**: Build phase portrait across 50–100+ years if data permits
5. **Anomaly detection**: Identify primary decoupling, dominant anchor, main weakness
6. **Structural diagnosis**: Analyse Archive–Lore relation, fast/slow-loop balance
7. **Mythopoetic translation**: Create mood texture, metaphor, persona, tension, mythic alignment
8. **Historical interpretation**: Narrate long-term trajectory and inflection points
9. **Comparative positioning**: Place society relative to peer benchmark
10. **Risk-opportunity assessment**: Identify 3 major risks and 1 major renewal pathway
11. **Formatting & integration**: Assemble into final report structure
12. **Verification**: Check that all mythopoetic elements map back to metrics

## Voice and Tone

- **Analytical first**, mythopoetic second
- **Formal and lucid**, not colloquial
- **Conditional language** ("suggests," "implies," "appears") rather than prophecy
- **Grounded in structure** — every claim traceable to data
- **Respectful complexity** — acknowledge trade-offs and contradictions
- **Literature-grade prose** — the writing should be worth reading for its own sake

## Limitations and Caveats

- **Snapshot bias**: Current data may not reflect long-term patterns; always reference historical context
- **Scorer assumptions**: CAMS scores reflect assessor's interpretation; note conventions (stress polarity, node vocabulary)
- **Incomplete data**: Some societies have shorter datasets; historical phase portrait may be limited
- **Comparative relativity**: Scores are meaningful within a dataset; cross-dataset comparison requires careful calibration
- **Futures uncertainty**: Risk assessment is structural tendency, not prediction; external shocks can override patterns

## Extensions and Variants

This skill can be extended to:
- **Sub-national analysis**: Compare regions within a country
- **Sector-specific reports**: Decompose nodes into sub-sectors (e.g., Craft into Agriculture/Manufacturing/Services)
- **Scenario analysis**: Model what-if futures ("if Helm recovered coherence, what would happen?")
- **Longitudinal tracking**: Automated dashboard showing M(t), Y(t), Headroom evolution
- **Multi-society synthesis**: Comparative study across 5–10 nations on single question
- **Early-warning systems**: Monitor critical thresholds (headroom < 0, Helm σ < −50, Y(t) declining)

## References and Further Reading

- **Vision-Affect framework**: Developed in this analytical practice; derives from affective neuroscience and systems thinking
- **CAMS nodes**: Based on institutional anthropology and complexity theory; see parent literature on civilisational analysis
- **Mythopoetic translation**: Drawing on systems thinking (Meadows, Senge), narrative analysis (McCloskey, White), and mythic archetypes (Campbell, Jung)
- **Historical phase portraits**: Technique from dynamical systems analysis; applied to civilisational time-scales

## Credits and Attribution

- **Template design**: Neural Nations Collaborative, CAMS v3.2-R.1-VA
- **Vision-Affect framework**: Integrative synthesis for this analytical ensemble
- **Mythopoetic methodology**: Grounded-theory approach to translating metrics into meaning
- **Three example reports**: Generated April 2026; Gemini assessor; publicly available datasets

---

## Quick Start

### Minimal Input (for rapid prototyping)

If you have only:
- 1 snapshot of recent CAMS data (8 nodes × 4 dimensions)
- Society name and rough period
- No historical data

The skill can still generate:
- Snapshot analysis (current node table, vision-affect reading, structural dynamics, mythopoetic interpretation)
- Comparative positioning (if you name 2–3 peer societies)
- Risk outline (based on current state only)

Deliverable: 6–8 sections instead of full 12.

### Full Input (for comprehensive analysis)

If you have:
- 50–100+ years of CAMS data
- Clear historical periodisation
- Known societal contexts and challenges
- Optional: pre-computed Node Value and Bond Strength

The skill delivers the full 12-section report.

## Invocation Examples

```
Invoke: "Generate a CAMS vision-affect report for Germany, April 2026, using the attached dataset (1900–2025)."

Invoke: "Create a mythopoetic analysis of Australia's current Lore and Helm states; compare to Norway."

Invoke: "Translate these CAMS metrics into civilisational meaning: Archive σ=540, Helm σ=−56, System Sigma=270."

Invoke: "Identify the dominant mythic attractor and the core tension for France 2026."

Invoke: "Build a comparative table of Vision-Affect states for France, Germany, Norway, and the USA."
```

---

**Version**: 1.0 (April 2026)
**Status**: Tested on three complete national datasets
**Readiness**: Production-ready for national-scale analysis; extensible to sub-national or sector-specific variants
