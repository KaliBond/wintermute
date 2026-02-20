"""
CLD Skins: Translates analyst-grade CLDs into audience-specific formats

Keeps canonical model structure stable but changes labels, descriptions,
and metadata for different audiences:
- Policy brief (jurisdiction, time horizon, audience parameters)
- Town/kids (simplified language)
- Boardroom (executive focus)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, replace
import copy
import os
import requests
import json

from cld_architect import CLDModel, CLDNode, CLDEdge, CLDLoop

@dataclass
class SkinParameters:
    """Parameters for audience skin generation"""
    audience_type: str  # "policy_brief", "town", "boardroom"
    jurisdiction: Optional[str] = None  # e.g., "Federal", "California", "Municipal"
    time_horizon: Optional[str] = None  # e.g., "2026-2030", "Next decade"
    audience: Optional[str] = None  # e.g., "Congressional staffers", "General public"
    reading_level: Optional[int] = None  # Flesch-Kincaid grade level (e.g., 8, 12, 16)

class CLDSkinTranslator:
    """Translates analyst CLDs into audience-specific formats"""

    def __init__(self, perplexity_api_key: Optional[str] = None):
        self.api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"

    def build_skin_prompt(self, analyst_cld: CLDModel, params: SkinParameters) -> str:
        """Build prompt for skin translation"""

        if params.audience_type == "policy_brief":
            return self._build_policy_brief_prompt(analyst_cld, params)
        elif params.audience_type == "town":
            return self._build_town_prompt(analyst_cld, params)
        elif params.audience_type == "boardroom":
            return self._build_boardroom_prompt(analyst_cld, params)
        else:
            raise ValueError(f"Unknown audience type: {params.audience_type}")

    def _build_policy_brief_prompt(self, analyst_cld: CLDModel, params: SkinParameters) -> str:
        """Build prompt for policy brief translation"""

        analyst_json = self._cld_to_dict(analyst_cld)

        return f"""Translate this analyst-grade Causal Loop Diagram into a policy brief format.

**Original Analyst CLD**:
{json.dumps(analyst_json, indent=2)}

**Target Audience**: {params.audience or 'Policy makers'}
**Jurisdiction**: {params.jurisdiction or 'Federal'}
**Time Horizon**: {params.time_horizon or '2026-2030'}
**Reading Level**: Grade {params.reading_level or 12}

**Translation Rules**:
1. **Keep structure intact**: Same nodes, edges, loops (IDs must match)
2. **Translate labels**: Convert CAMS terminology to policy language
   - "Lore" → "Public Trust", "Narrative Legitimacy", etc.
   - "Shield" → "Security Apparatus", "Law Enforcement", etc.
   - "Helm" → "Executive Leadership", "Coordination Capacity", etc.
3. **Rewrite descriptions**: Focus on policy implications and jurisdictional context
4. **Edge descriptions**: Explain mechanisms in policy terms
5. **Loop labels**: Frame as policy dynamics
6. **Metadata**: Add policy brief summary

**Example Translations**:
- Analyst: "Lore (Legitimacy, ideology, shared narrative)"
  Policy: "Public Trust (Citizen confidence in government institutions and policy narratives)"

- Analyst: "Shield → Hands (Coercive pressure reduces labor capacity)"
  Policy: "Law Enforcement Intensity → Workforce Availability (Incarceration and policing reduce available labor pool)"

Return ONLY a JSON object with the same structure as the input, but with translated labels and descriptions.

Important: Maintain all IDs, node types, edge polarities, loop structures exactly as given."""

    def _build_town_prompt(self, analyst_cld: CLDModel, params: SkinParameters) -> str:
        """Build prompt for town hall / general public translation"""

        analyst_json = self._cld_to_dict(analyst_cld)

        return f"""Translate this analyst-grade Causal Loop Diagram into simple, accessible language for a general audience.

**Original Analyst CLD**:
{json.dumps(analyst_json, indent=2)}

**Target Audience**: General public (town hall meeting, community education)
**Reading Level**: Grade {params.reading_level or 8}

**Translation Rules**:
1. **Keep structure intact**: Same nodes, edges, loops (IDs must match)
2. **Simplify labels**: Use everyday language
   - "Lore" → "Community Trust", "Shared Beliefs"
   - "Shield" → "Security", "Protection"
   - "Flow" → "Money and Goods", "Trade"
3. **Plain language descriptions**: Avoid jargon, use concrete examples
4. **Edge descriptions**: Explain cause-effect in simple terms
5. **Loop labels**: Describe feedback in accessible way

**Example Translations**:
- Analyst: "Lore (Legitimacy, ideology, shared narrative)"
  Town: "Community Trust (What people believe about how things should work)"

- Analyst: "Craft → Flow (Production capacity enables circulation)"
  Town: "Skills & Factories → Money Moving (When we make things, it creates jobs and trade)"

Return ONLY a JSON object with the same structure as the input, but with simplified labels and descriptions."""

    def _build_boardroom_prompt(self, analyst_cld: CLDModel, params: SkinParameters) -> str:
        """Build prompt for executive/boardroom translation"""

        analyst_json = self._cld_to_dict(analyst_cld)

        return f"""Translate this analyst-grade Causal Loop Diagram into executive/boardroom language.

**Original Analyst CLD**:
{json.dumps(analyst_json, indent=2)}

**Target Audience**: Corporate executives, board members, strategic planners
**Time Horizon**: {params.time_horizon or 'Strategic (3-5 years)'}

**Translation Rules**:
1. **Keep structure intact**: Same nodes, edges, loops (IDs must match)
2. **Executive terminology**: Use business strategy language
   - "Lore" → "Brand Equity", "Organizational Culture"
   - "Shield" → "Risk Management", "Compliance"
   - "Helm" → "Executive Leadership", "Governance"
   - "Flow" → "Capital Flow", "Supply Chain"
3. **Strategic framing**: Focus on competitive dynamics, ROI, risk
4. **Edge descriptions**: Explain in terms of business drivers
5. **Loop labels**: Frame as strategic cycles

**Example Translations**:
- Analyst: "Lore (Legitimacy, ideology, shared narrative)"
  Boardroom: "Brand Equity (Market reputation and stakeholder confidence)"

- Analyst: "Stewards → Flow (Capital allocation enables circulation)"
  Boardroom: "Investment Strategy → Liquidity (Capital deployment drives cash flow velocity)"

Return ONLY a JSON object with the same structure as the input, but with business-focused labels and descriptions."""

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
                    "content": "You are an expert translator of technical systems models into audience-specific formats. Preserve structure while adapting language. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,  # Lower temperature for consistent translation
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

    def apply_skin(self, analyst_cld: CLDModel, params: SkinParameters,
                   model: str = "sonar-pro") -> CLDModel:
        """
        Apply audience skin to analyst CLD

        Returns new CLDModel with translated labels/descriptions
        """
        prompt = self.build_skin_prompt(analyst_cld, params)
        skinned_json = self.call_perplexity(prompt, model)

        # Convert back to CLDModel
        nodes = [CLDNode(**n) for n in skinned_json["nodes"]]
        edges = [CLDEdge(**e) for e in skinned_json["edges"]]
        loops = [CLDLoop(**l) for l in skinned_json["loops"]]

        # Add skin metadata
        metadata = skinned_json["metadata"]
        metadata["skin"] = {
            "type": params.audience_type,
            "jurisdiction": params.jurisdiction,
            "time_horizon": params.time_horizon,
            "audience": params.audience,
            "reading_level": params.reading_level
        }

        return CLDModel(
            metadata=metadata,
            nodes=nodes,
            edges=edges,
            loops=loops,
            archetypes=skinned_json.get("archetypes", analyst_cld.archetypes),
            aha_paradox=analyst_cld.aha_paradox  # Preserve original paradox
        )

    def _cld_to_dict(self, cld: CLDModel) -> Dict:
        """Convert CLDModel to dict for JSON serialization"""
        from dataclasses import asdict
        return {
            "metadata": cld.metadata,
            "nodes": [asdict(n) for n in cld.nodes],
            "edges": [asdict(e) for e in cld.edges],
            "loops": [asdict(l) for l in cld.loops],
            "archetypes": cld.archetypes
        }

    def generate_multiple_skins(self, analyst_cld: CLDModel, skin_params: List[SkinParameters],
                              model: str = "sonar-pro") -> Dict[str, CLDModel]:
        """
        Generate multiple audience skins from single analyst CLD

        Returns: Dict mapping skin_id to CLDModel
        """
        skins = {}
        for i, params in enumerate(skin_params):
            skin_id = f"{params.audience_type}_{i+1}"
            skins[skin_id] = self.apply_skin(analyst_cld, params, model)
        return skins


# Preset skin configurations
PRESET_SKINS = {
    "federal_policy_brief": SkinParameters(
        audience_type="policy_brief",
        jurisdiction="Federal",
        time_horizon="2026-2030",
        audience="Congressional staffers and federal agency leads",
        reading_level=14
    ),
    "state_policy_brief": SkinParameters(
        audience_type="policy_brief",
        jurisdiction="State",
        time_horizon="2026-2028",
        audience="State legislators and gubernatorial staff",
        reading_level=13
    ),
    "municipal_policy_brief": SkinParameters(
        audience_type="policy_brief",
        jurisdiction="Municipal",
        time_horizon="2026-2027",
        audience="City council and municipal administration",
        reading_level=12
    ),
    "town_hall": SkinParameters(
        audience_type="town",
        audience="General public at community meeting",
        reading_level=8
    ),
    "kids": SkinParameters(
        audience_type="town",
        audience="Middle school students",
        reading_level=6
    ),
    "boardroom": SkinParameters(
        audience_type="boardroom",
        time_horizon="Strategic (3-5 years)",
        audience="Corporate executives and board members",
        reading_level=14
    )
}


# Example usage
if __name__ == "__main__":
    from cld_architect import CLDArchitect

    print("Testing CLD Skin Translator...")

    # Note: Requires analyst CLD and API key
    try:
        architect = CLDArchitect()
        translator = CLDSkinTranslator()

        # Generate analyst CLD
        topic = "Healthcare Access Paradox"
        narrative = "Increased healthcare spending should improve population health, but outcomes plateau or decline."
        must_include = ["Insurance coverage", "Cost burden", "Preventive care"]

        aha, analyst_cld = architect.generate_full_cld(topic, narrative, must_include)

        # Apply policy brief skin
        policy_skin = translator.apply_skin(analyst_cld, PRESET_SKINS["federal_policy_brief"])
        print("\n✓ Policy brief skin generated")

        # Apply town hall skin
        town_skin = translator.apply_skin(analyst_cld, PRESET_SKINS["town_hall"])
        print("✓ Town hall skin generated")

        print(f"\nOriginal node: {analyst_cld.nodes[0].label}")
        print(f"Policy translation: {policy_skin.nodes[0].label}")
        print(f"Town translation: {town_skin.nodes[0].label}")

    except Exception as e:
        print(f"\n✗ Skin generation failed (expected without API key): {e}")
        print("\nTo use: Set PERPLEXITY_API_KEY environment variable")
        print("\nPreset skins available:")
        for skin_id, params in PRESET_SKINS.items():
            print(f"  - {skin_id}: {params.audience_type} ({params.audience})")
