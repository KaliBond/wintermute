"""
CLD Architect: Generates Causal Loop Diagrams using Aha! Paradox framework
and base 8-node CAMS lattice as canonical foundation.

Two-stage process:
1. Generate analyst-grade CLD with CAMS nodes as foundation
2. Apply audience skins for translation
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Base 8-node CAMS lattice (canonical foundation)
CAMS_LATTICE = [
    {
        "id": "n1",
        "label": "Lore",
        "description": "Legitimacy, ideology, shared narrative—the meaning-making that sanctions power and bonds.",
        "type": "stock",
        "domain": "mythic"
    },
    {
        "id": "n2",
        "label": "Shield",
        "description": "Coercive capacity, security apparatus—force projection and boundary control.",
        "type": "stock",
        "domain": "metabolic"
    },
    {
        "id": "n3",
        "label": "Archive",
        "description": "Institutional memory, records, law—the inscribed heritage that stabilizes and checks.",
        "type": "stock",
        "domain": "mythic"
    },
    {
        "id": "n4",
        "label": "Helm",
        "description": "Executive steering and coordination—the decision nexus that integrates signals and commands.",
        "type": "stock",
        "domain": "executive"
    },
    {
        "id": "n5",
        "label": "Hands",
        "description": "Household/workforce capacity—the labor pool that executes tasks.",
        "type": "stock",
        "domain": "metabolic"
    },
    {
        "id": "n6",
        "label": "Craft",
        "description": "Productive/technical capability—tools, knowledge, organized production chains.",
        "type": "stock",
        "domain": "productive"
    },
    {
        "id": "n7",
        "label": "Flow",
        "description": "Circulation systems (trade, finance, energy)—the networks that move resources.",
        "type": "stock",
        "domain": "metabolic"
    },
    {
        "id": "n8",
        "label": "Stewards",
        "description": "Capital/property stewardship—the guardians of wealth, land, and investment logic.",
        "type": "stock",
        "domain": "mythic"
    }
]

@dataclass
class AhaSectionPrompt:
    """Aha! Paradox framework section specification"""
    section: str
    length: str
    focus: str
    example: str

# Aha! Paradox 7-section framework
AHA_FRAMEWORK = [
    AhaSectionPrompt(
        section="Anchor",
        length="1 sentence",
        focus="Conventional wisdom/default assumption",
        example="Everyone assumes efficient markets self-correct via price signals."
    ),
    AhaSectionPrompt(
        section="Default",
        length="2-3 sentences",
        focus="How the conventional mechanism is supposed to work",
        example="When prices rise, supply increases and demand falls, restoring equilibrium. Rational actors arbitrage away distortions. The invisible hand allocates resources efficiently."
    ),
    AhaSectionPrompt(
        section="Bottleneck",
        length="1-2 sentences",
        focus="The hidden constraint or friction",
        example="But price discovery requires liquidity and transparency. In thin or opaque markets, signals are delayed or distorted."
    ),
    AhaSectionPrompt(
        section="Collision",
        length="2-3 sentences",
        focus="What breaks when the bottleneck interacts with the default",
        example="Distorted signals trigger misallocations (malinvestment). Investors chase past returns, amplifying bubbles. Eventually, sudden repricing causes cascading failures."
    ),
    AhaSectionPrompt(
        section="Reversal",
        length="1 sentence",
        focus="The paradoxical outcome (opposite of anchor)",
        example="Markets designed for efficiency produce systematic instability."
    ),
    AhaSectionPrompt(
        section="Commitment Filter",
        length="1 sentence",
        focus="Why the system persists despite dysfunction",
        example="Financial elites profit from volatility and oppose transparency reforms."
    ),
    AhaSectionPrompt(
        section="Kinetic Result",
        length="2-3 sentences",
        focus="Tangible consequences in the real world",
        example="2008 subprime crisis: $10T wealth destruction, 8.7M jobs lost. Retail investors lose savings while institutions get bailouts. Inequality ratchets up, eroding social trust."
    )
]

@dataclass
class CLDNode:
    """CLD node specification"""
    id: str
    label: str
    description: str
    type: str  # "stock" (teal box) or "auxiliary" (blue ellipse)
    domain: Optional[str] = None  # metabolic, mythic, executive, productive
    x: Optional[int] = None
    y: Optional[int] = None

@dataclass
class CLDEdge:
    """CLD edge specification"""
    from_node: str
    to_node: str
    polarity: str  # "+" or "-"
    description: str
    width: int = 2  # 2-7
    lag: int = 0  # 0-5
    style: str = "solid"  # or "dashed"

@dataclass
class CLDLoop:
    """CLD feedback loop"""
    loop_id: int
    nodes: List[str]  # ordered list of node IDs
    polarity: str  # "R" (reinforcing) or "B" (balancing)
    label: str  # brief description
    archetype: Optional[str] = None

@dataclass
class CLDModel:
    """Complete CLD model in CLDai format"""
    metadata: Dict
    nodes: List[CLDNode]
    edges: List[CLDEdge]
    loops: List[CLDLoop]
    archetypes: List[str]
    aha_paradox: Optional[Dict] = None

class CLDArchitect:
    """Generates analyst-grade CLDs using Perplexity API and CAMS lattice"""

    def __init__(self, perplexity_api_key: Optional[str] = None):
        self.api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"

    def build_aha_paradox_prompt(self, topic: str, narrative: str, must_include: List[str]) -> str:
        """Construct prompt for Aha! Paradox generation"""
        sections_spec = "\n".join([
            f"**{s.section}** ({s.length}): {s.focus}\n  Example: \"{s.example}\""
            for s in AHA_FRAMEWORK
        ])

        must_include_str = "\n".join([f"- {item}" for item in must_include]) if must_include else "None"

        return f"""Generate an Aha! Paradox insight for the following topic using the 7-section framework.

**Topic**: {topic}

**Narrative Context**: {narrative}

**Must Include**:
{must_include_str}

**Framework**:
{sections_spec}

Return ONLY a JSON object with this structure:
{{
  "anchor": "...",
  "default": "...",
  "bottleneck": "...",
  "collision": "...",
  "reversal": "...",
  "commitment_filter": "...",
  "kinetic_result": "..."
}}

Each field should be a string matching the length/focus specifications above."""

    def build_cld_architect_prompt(self, aha_paradox: Dict, collision_domain: str) -> str:
        """Construct prompt for CLD generation from Aha! Paradox"""

        # Filter CAMS nodes by domain if specified
        if collision_domain == "all":
            base_nodes = CAMS_LATTICE
        else:
            base_nodes = [n for n in CAMS_LATTICE if n.get("domain") == collision_domain]

        base_nodes_json = json.dumps(base_nodes, indent=2)

        return f"""Generate a Causal Loop Diagram (CLD) that explains this paradox.

**Aha! Paradox**:
{json.dumps(aha_paradox, indent=2)}

**Collision Domain**: {collision_domain}

**Base CAMS Nodes (use as foundation)**:
{base_nodes_json}

**CLD Architect Specification**:
1. **Nodes** (8-12 total):
   - Use provided CAMS base nodes as foundation
   - Add 0-4 auxiliary nodes if needed (blue ellipses, type="auxiliary")
   - Stock nodes are teal boxes (type="stock")
   - Each node needs: id, label, description, type, domain

2. **Edges** (polarity matters):
   - Polarity: "+" (same direction) or "-" (opposite)
   - Width: 2-7 (indicates strength)
   - Lag: 0-5 timesteps
   - Style: "solid" or "dashed"
   - Include description explaining the causal mechanism

3. **Loops** (5-8 feedback loops):
   - Each loop: loop_id, nodes (ordered list), polarity (R/B), label
   - R = reinforcing (positive feedback)
   - B = balancing (negative feedback)
   - Identify archetype if applicable (Limits to Growth, Shifting the Burden, etc.)

4. **Archetypes**: List any system archetypes present

Return ONLY a JSON object matching this structure:
{{
  "metadata": {{
    "topic": "{aha_paradox.get('reversal', 'Unknown')}",
    "collision_domain": "{collision_domain}",
    "timestamp": "2026-02-20"
  }},
  "nodes": [
    {{"id": "n1", "label": "Lore", "description": "...", "type": "stock", "domain": "mythic"}},
    ...
  ],
  "edges": [
    {{"from_node": "n1", "to_node": "n2", "polarity": "+", "description": "...", "width": 3, "lag": 0, "style": "solid"}},
    ...
  ],
  "loops": [
    {{"loop_id": 1, "nodes": ["n1", "n2", "n3", "n1"], "polarity": "R", "label": "...", "archetype": null}},
    ...
  ],
  "archetypes": ["Limits to Growth", ...]
}}"""

    def call_perplexity(self, prompt: str, model: str = "sonar-pro") -> Dict:
        """Call Perplexity API with prompt"""
        if not self.api_key:
            raise ValueError("Perplexity API key not set. Set PERPLEXITY_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in systems thinking, causal loop diagrams, and institutional dynamics. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    def generate_aha_paradox(self, topic: str, narrative: str, must_include: List[str],
                           model: str = "sonar-pro") -> Dict:
        """Generate Aha! Paradox using Perplexity API"""
        prompt = self.build_aha_paradox_prompt(topic, narrative, must_include)
        return self.call_perplexity(prompt, model)

    def generate_analyst_cld(self, aha_paradox: Dict, collision_domain: str = "all",
                           model: str = "sonar-pro") -> CLDModel:
        """Generate analyst-grade CLD from Aha! Paradox"""
        prompt = self.build_cld_architect_prompt(aha_paradox, collision_domain)
        cld_json = self.call_perplexity(prompt, model)

        # Convert to CLDModel
        nodes = [CLDNode(**n) for n in cld_json["nodes"]]
        edges = [CLDEdge(**e) for e in cld_json["edges"]]
        loops = [CLDLoop(**l) for l in cld_json["loops"]]

        return CLDModel(
            metadata=cld_json["metadata"],
            nodes=nodes,
            edges=edges,
            loops=loops,
            archetypes=cld_json.get("archetypes", []),
            aha_paradox=aha_paradox
        )

    def cld_to_json(self, cld: CLDModel) -> Dict:
        """Convert CLDModel to CLDai JSON format"""
        return {
            "metadata": cld.metadata,
            "nodes": [asdict(n) for n in cld.nodes],
            "edges": [asdict(e) for e in cld.edges],
            "loops": [asdict(l) for l in cld.loops],
            "archetypes": cld.archetypes,
            "aha_paradox": cld.aha_paradox
        }

    def generate_full_cld(self, topic: str, narrative: str, must_include: List[str],
                         collision_domain: str = "all", model: str = "sonar-pro") -> Tuple[Dict, CLDModel]:
        """
        Full two-stage generation:
        1. Generate Aha! Paradox
        2. Generate analyst-grade CLD

        Returns: (aha_paradox_dict, CLDModel)
        """
        # Stage 1: Aha! Paradox
        aha = self.generate_aha_paradox(topic, narrative, must_include, model)

        # Stage 2: Analyst CLD
        cld = self.generate_analyst_cld(aha, collision_domain, model)

        return aha, cld


# Example usage and testing
if __name__ == "__main__":
    # Test with example
    architect = CLDArchitect()

    # Example: Market efficiency paradox
    topic = "Market Efficiency and Financial Instability"
    narrative = "Modern financial markets are designed to be efficient, with price discovery mechanisms that should prevent major disruptions. Yet we observe recurring crises."
    must_include = [
        "Price discovery mechanism",
        "Liquidity constraints",
        "Cascade failures"
    ]

    print("Testing CLD Architect...")
    print(f"\nTopic: {topic}")
    print(f"Narrative: {narrative}")
    print(f"Must include: {must_include}")

    # Note: This will fail without PERPLEXITY_API_KEY
    try:
        aha, cld = architect.generate_full_cld(topic, narrative, must_include, collision_domain="metabolic")
        print("\n✓ Aha! Paradox generated:")
        print(json.dumps(aha, indent=2))
        print("\n✓ CLD generated:")
        print(json.dumps(architect.cld_to_json(cld), indent=2))
    except Exception as e:
        print(f"\n✗ Generation failed (expected without API key): {e}")
        print("\nTo use: Set PERPLEXITY_API_KEY environment variable")
