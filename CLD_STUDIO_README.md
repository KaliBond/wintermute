# CLD Studio Documentation

**Version:** 1.0
**Date:** 2026-02-20

## Overview

CLD Studio is an interactive system for generating Causal Loop Diagrams (CLDs) using the **Aha! Paradox framework** and the **8-node CAMS lattice** as a canonical foundation.

### Key Features

- **Two-stage generation**: Aha! Paradox → Analyst CLD → Audience Skins
- **Base 8-node CAMS lattice**: Canonical institutional framework
- **Multiple audience skins**: Policy brief, Town/kids, Boardroom
- **Perplexity API integration**: Automated generation using sonar-pro
- **CLDai visualization engine**: Interactive network diagrams with Vis.js
- **Export formats**: JSON, standalone HTML, PNG

## Architecture

### Component Structure

```
wintermute/
├── cld_studio.py                    # Main Streamlit UI
├── src/
│   ├── cld_architect.py             # Core CLD generation logic
│   ├── cld_skins.py                 # Audience translation layer
│   ├── cams_dyad_cld.py             # M-Y dyad analysis
│   └── ddig_analysis.py             # dDIG influence metrics
├── templates/
│   └── cldai_renderer.html          # Vis.js visualization engine
└── CLD_STUDIO_README.md             # This file
```

### Module Responsibilities

#### 1. `cld_architect.py` - CLD Generation Core

**Purpose**: Generates analyst-grade CLDs using Perplexity API and CAMS lattice

**Key Classes**:
- `CLDArchitect`: Main generation orchestrator
- `CLDNode`: Node specification (stocks/auxiliaries)
- `CLDEdge`: Causal edge with polarity, lag, width
- `CLDLoop`: Feedback loop (reinforcing/balancing)
- `CLDModel`: Complete CLD in CLDai format

**CAMS Base Lattice (8 canonical nodes)**:
```python
1. Lore (mythic) - Legitimacy, ideology, shared narrative
2. Shield (metabolic) - Coercive capacity, security apparatus
3. Archive (mythic) - Institutional memory, records, law
4. Helm (executive) - Executive steering and coordination
5. Hands (metabolic) - Household/workforce capacity
6. Craft (productive) - Productive/technical capability
7. Flow (metabolic) - Circulation systems (trade, finance, energy)
8. Stewards (mythic) - Capital/property stewardship
```

**Aha! Paradox Framework (7 sections)**:
1. **Anchor** (1 sentence): Conventional wisdom
2. **Default** (2-3 sentences): How mechanism is supposed to work
3. **Bottleneck** (1-2 sentences): Hidden constraint
4. **Collision** (2-3 sentences): What breaks
5. **Reversal** (1 sentence): Paradoxical outcome
6. **Commitment Filter** (1 sentence): Why system persists
7. **Kinetic Result** (2-3 sentences): Tangible consequences

**Methods**:
- `generate_aha_paradox(topic, narrative, must_include)` → Dict
- `generate_analyst_cld(aha_paradox, collision_domain)` → CLDModel
- `generate_full_cld(...)` → (aha_dict, CLDModel)

**CLD Architect Specification**:
- **Nodes**: 8-12 total (use CAMS base + 0-4 auxiliaries)
  - Stocks: teal boxes, type="stock"
  - Auxiliaries: blue ellipses, type="auxiliary"
- **Edges**: Polarity (+/-), width (2-7), lag (0-5), style (solid/dashed)
- **Loops**: 5-8 feedback loops, polarity (R/B), archetype
- **Archetypes**: Limits to Growth, Shifting the Burden, etc.

#### 2. `cld_skins.py` - Audience Translation

**Purpose**: Translates analyst CLDs into audience-specific formats while keeping structure intact

**Key Classes**:
- `CLDSkinTranslator`: Translation orchestrator
- `SkinParameters`: Audience configuration

**Skin Types**:
1. **Policy Brief**
   - Target: Policy makers, staffers, agency leads
   - Parameters: jurisdiction, time_horizon, audience, reading_level
   - Example: "Lore" → "Public Trust", "Shield" → "Law Enforcement"

2. **Town Hall / Kids**
   - Target: General public, community meetings, students
   - Parameters: audience, reading_level (6-12)
   - Example: "Lore" → "Community Trust", "Flow" → "Money Moving"

3. **Boardroom**
   - Target: Executives, board members, strategic planners
   - Parameters: time_horizon, audience, reading_level
   - Example: "Lore" → "Brand Equity", "Stewards" → "Investment Strategy"

**Preset Skins**:
- `federal_policy_brief`: Congressional staffers (grade 14)
- `state_policy_brief`: State legislators (grade 13)
- `municipal_policy_brief`: City council (grade 12)
- `town_hall`: General public (grade 8)
- `kids`: Middle school students (grade 6)
- `boardroom`: Corporate executives (grade 14)

**Methods**:
- `apply_skin(analyst_cld, params)` → CLDModel
- `generate_multiple_skins(analyst_cld, skin_params_list)` → Dict[str, CLDModel]

#### 3. `templates/cldai_renderer.html` - Visualization Engine

**Purpose**: Interactive network diagram renderer using Vis.js

**Features**:
- Color-coded nodes: Teal (stocks), Blue (auxiliaries)
- Color-coded edges: Green (+), Red (-)
- Feedback loop annotations (R/B)
- Interactive hover tooltips
- Zoom, pan, drag nodes
- Physics simulation (Barnes-Hut)
- Export to PNG

**Layout**:
- Left: Network visualization canvas
- Right: Sidebar with metadata, loops, archetypes, Aha! Paradox

**Controls**:
- Fit to View
- Reset Layout
- Toggle Physics
- Export PNG

#### 4. `cld_studio.py` - Main Streamlit UI

**Purpose**: Interactive web interface for CLD generation workflow

**Tab Structure**:

**Tab 1 - Builder**:
- Input fields: Topic, Narrative, Must-include
- Generate Aha! Paradox button
- Generate Analyst CLD button
- Display generated paradox sections

**Tab 2 - Analyst View**:
- Metrics: Node/Edge/Loop counts
- Tables: Nodes, Edges
- Feedback loops with archetype labels
- CLDai visualization (embedded iframe)

**Tab 3 - Audience Skins**:
- Skin configuration form (jurisdiction, time horizon, audience, reading level)
- Generate Skin button
- Preset skin quick-apply
- Comparison table (analyst vs skinned labels)
- CLDai visualization of selected skin

**Tab 4 - Export**:
- Download JSON
- Download standalone HTML (portable)
- Preview JSON

**Sidebar**:
- API key input (secure)
- Model selection (sonar-pro, sonar, sonar-reasoning)
- Collision domain selection (all, metabolic, mythic, executive, productive)
- CAMS lattice reference (expandable)

## Usage

### Prerequisites

1. **Install dependencies**:
```bash
pip install streamlit pandas numpy plotly requests
```

2. **Get Perplexity API key**:
   - Sign up at https://perplexity.ai
   - Generate API key from dashboard
   - Set as environment variable: `export PERPLEXITY_API_KEY=your_key_here`

### Running CLD Studio

```bash
cd C:\Users\julie\wintermute
streamlit run cld_studio.py
```

Browser will open at `http://localhost:8501`

### Workflow

#### Step 1: Configure API
1. Enter Perplexity API key in sidebar (or set `PERPLEXITY_API_KEY` env var)
2. Select model (sonar-pro recommended)
3. Choose collision domain (start with "all")

#### Step 2: Build Aha! Paradox
1. Go to **Builder** tab
2. Enter **Topic**: e.g., "Market Efficiency and Financial Instability"
3. Enter **Narrative**: 2-4 sentences describing the conventional wisdom and paradoxical outcome
4. Enter **Must Include**: Key concepts (one per line), e.g.:
   ```
   Price discovery
   Liquidity constraints
   Cascade failures
   ```
5. Click **Generate Aha! Paradox**
6. Review 7-section paradox output

#### Step 3: Generate Analyst CLD
1. Click **Generate Analyst CLD**
2. Wait ~20-60 seconds (depends on model)
3. CLD will use CAMS lattice nodes as foundation + 0-4 auxiliaries

#### Step 4: View Analyst Model
1. Go to **Analyst View** tab
2. Review metrics, loops, archetypes
3. Interact with visualization (zoom, pan, hover for tooltips)
4. Click nodes for details

#### Step 5: Generate Audience Skins
1. Go to **Audience Skins** tab
2. Configure skin:
   - **Policy Brief**: Set jurisdiction (Federal/State/Municipal), time horizon, audience
   - **Town Hall**: Set reading level (6-12)
   - **Boardroom**: Set time horizon, audience
3. Click **Generate Skin**
4. Compare analyst labels vs skinned labels
5. View translated visualization

#### Step 6: Export
1. Go to **Export** tab
2. Select model to export (analyst or specific skin)
3. Download:
   - **JSON**: For programmatic use, archiving
   - **HTML**: Standalone file (open in any browser, no server needed)

### Example Session

**Topic**: "Healthcare Access Paradox"

**Narrative**: "Increased healthcare spending should improve population health outcomes. The U.S. spends more per capita than any other developed nation. Yet life expectancy has plateaued and maternal mortality is rising."

**Must Include**:
```
Insurance coverage expansion
Administrative costs
Preventive care access
Price transparency
```

**Generated Aha! Sections** (example):
- **Anchor**: "More healthcare spending improves population health."
- **Default**: "Insurance coverage removes financial barriers. Patients access needed care. Health outcomes improve."
- **Bottleneck**: "Administrative complexity consumes 30% of spending. Price opacity prevents informed decisions."
- **Collision**: "Coverage expands but navigational friction delays care. Costs rise faster than utilization. Preventive care remains inaccessible."
- **Reversal**: "Higher spending produces stagnant or declining health outcomes."
- **Commitment Filter**: "Insurers and providers profit from complexity; oppose price transparency."
- **Kinetic Result**: "Maternal mortality up 50% (2000-2020). Medical bankruptcy #1 cause. Trust in system erodes."

**Generated CLD** (8 nodes from CAMS lattice):
- **Lore** (Narrative Legitimacy) → **Hands** (Workforce Capacity)
- **Shield** (Regulatory Enforcement) → **Flow** (Payment Systems)
- **Stewards** (Insurance Capital) → **Flow** (+)
- **Craft** (Medical Technology) → **Hands** (-)
- Loop 1 (R): Stewards → Flow → Craft → Stewards (cost spiral)
- Loop 2 (B): Shield → Hands → Lore → Shield (trust erosion)

**Policy Brief Skin**:
- "Lore" → "Public Trust in Healthcare System"
- "Shield" → "Regulatory Oversight & Enforcement"
- "Stewards" → "Insurance & Provider Capital Allocation"
- "Flow" → "Healthcare Payment & Reimbursement Networks"

**Town Hall Skin**:
- "Lore" → "Trust in Doctors and Hospitals"
- "Shield" → "Health Regulations"
- "Stewards" → "Insurance Companies"
- "Flow" → "How Money Moves in Healthcare"

## API Reference

### Environment Variables

```bash
PERPLEXITY_API_KEY=pplx-xxxxx   # Required for generation
```

### CLDArchitect API

```python
from src.cld_architect import CLDArchitect

architect = CLDArchitect(api_key="pplx-xxxxx")

# Generate Aha! Paradox
aha = architect.generate_aha_paradox(
    topic="Your Topic",
    narrative="Context in 2-4 sentences",
    must_include=["Concept 1", "Concept 2"],
    model="sonar-pro"  # or "sonar", "sonar-reasoning"
)

# Generate Analyst CLD
cld = architect.generate_analyst_cld(
    aha_paradox=aha,
    collision_domain="all",  # or "metabolic", "mythic", "executive", "productive"
    model="sonar-pro"
)

# Full pipeline
aha, cld = architect.generate_full_cld(
    topic="Your Topic",
    narrative="Context",
    must_include=["Concept 1"],
    collision_domain="all",
    model="sonar-pro"
)

# Export to JSON
from dataclasses import asdict
cld_json = {
    "metadata": cld.metadata,
    "nodes": [asdict(n) for n in cld.nodes],
    "edges": [asdict(e) for e in cld.edges],
    "loops": [asdict(l) for l in cld.loops],
    "archetypes": cld.archetypes,
    "aha_paradox": cld.aha_paradox
}
```

### CLDSkinTranslator API

```python
from src.cld_skins import CLDSkinTranslator, SkinParameters, PRESET_SKINS

translator = CLDSkinTranslator(api_key="pplx-xxxxx")

# Custom skin
params = SkinParameters(
    audience_type="policy_brief",
    jurisdiction="Federal",
    time_horizon="2026-2030",
    audience="Congressional staffers",
    reading_level=14
)
skinned_cld = translator.apply_skin(analyst_cld, params, model="sonar-pro")

# Preset skin
skinned_cld = translator.apply_skin(
    analyst_cld,
    PRESET_SKINS["federal_policy_brief"],
    model="sonar-pro"
)

# Multiple skins
skins = translator.generate_multiple_skins(
    analyst_cld,
    [PRESET_SKINS["federal_policy_brief"], PRESET_SKINS["town_hall"]],
    model="sonar-pro"
)
```

## File Formats

### CLD JSON Schema

```json
{
  "metadata": {
    "topic": "Title of CLD",
    "collision_domain": "all",
    "timestamp": "2026-02-20",
    "skin": {
      "type": "policy_brief",
      "jurisdiction": "Federal",
      "time_horizon": "2026-2030",
      "audience": "Congressional staffers",
      "reading_level": 14
    }
  },
  "nodes": [
    {
      "id": "n1",
      "label": "Lore",
      "description": "Legitimacy, ideology, shared narrative",
      "type": "stock",
      "domain": "mythic",
      "x": null,
      "y": null
    }
  ],
  "edges": [
    {
      "from_node": "n1",
      "to_node": "n2",
      "polarity": "+",
      "description": "Causal mechanism explanation",
      "width": 3,
      "lag": 0,
      "style": "solid"
    }
  ],
  "loops": [
    {
      "loop_id": 1,
      "nodes": ["n1", "n2", "n3", "n1"],
      "polarity": "R",
      "label": "Reinforcing cycle description",
      "archetype": "Limits to Growth"
    }
  ],
  "archetypes": ["Limits to Growth", "Shifting the Burden"],
  "aha_paradox": {
    "anchor": "Conventional wisdom statement",
    "default": "How it should work",
    "bottleneck": "Hidden constraint",
    "collision": "What breaks",
    "reversal": "Paradoxical outcome",
    "commitment_filter": "Why it persists",
    "kinetic_result": "Tangible consequences"
  }
}
```

## Troubleshooting

### "API key required for generation"
- Set `PERPLEXITY_API_KEY` environment variable
- Or enter API key in sidebar

### "Generation failed: 401 Unauthorized"
- Check API key is valid
- Verify API key has credits remaining
- Check https://perplexity.ai dashboard

### "No results generated"
- Ensure topic and narrative are substantive (2+ sentences)
- Check must_include list has 2-5 items
- Try simpler topic first to verify API connection

### "Bin edges must be unique" (in dDIG analysis)
- Dataset has insufficient variation in Bond Strength
- This is expected for some datasets
- Analysis will gracefully handle with fallback regime

### Visualization not rendering
- Check browser console for errors
- Ensure Vis.js CDN is accessible (requires internet)
- Try exporting standalone HTML and opening in browser

## Performance

### Generation Times (approximate)
- **Aha! Paradox**: 10-30 seconds (sonar-pro)
- **Analyst CLD**: 20-60 seconds (sonar-pro)
- **Skin Translation**: 15-40 seconds (sonar-pro)
- **Total pipeline**: 45-130 seconds for full workflow

### Model Comparison
| Model | Speed | Quality | Cost | Recommended For |
|-------|-------|---------|------|-----------------|
| sonar-pro | Slowest | Highest | High | Production CLDs |
| sonar | Medium | Good | Medium | Testing, iteration |
| sonar-reasoning | Slow | Highest | Highest | Complex paradoxes |

### Cost Estimates (Perplexity API)
- ~$0.10-0.50 per CLD (sonar-pro)
- ~$0.05-0.20 per skin translation
- Check https://docs.perplexity.ai/docs/pricing for current rates

## Roadmap

### Planned Features
- [ ] Batch generation (CSV input → multiple CLDs)
- [ ] CLD comparison tool (diff two models)
- [ ] Historical versioning (track edits)
- [ ] Manual node/edge editing UI
- [ ] Import existing CLDs (JSON upload)
- [ ] Integration with CAMS dDIG analysis (auto-suggest nodes from data)
- [ ] Export to System Dynamics tools (Vensim, Stella)
- [ ] Collaborative editing (multi-user sessions)
- [ ] Template library (common paradoxes)
- [ ] Natural language search (find similar CLDs)

### Known Limitations
- Requires Perplexity API (no offline mode)
- CAMS lattice assumes 8-node institutional model
- Skin translation may occasionally miss context
- No loop validation (user must verify feedback accuracy)
- PNG export resolution limited to viewport size

## Support

**Documentation**: See this file
**Source code**: `C:\Users\julie\wintermute\`
**GitHub**: https://github.com/KaliBond/wintermute
**Issues**: Create GitHub issue with reproduction steps

## License

Part of the CAMS Framework v2.1
Developed by Julie Bond, 2026
Powered by Perplexity AI and Claude Code

---

**Last Updated**: 2026-02-20
**Version**: 1.0
**Author**: Claude Sonnet 4.5 + Julie Bond
