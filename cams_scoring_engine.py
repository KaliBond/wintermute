"""CAMS Scoring Engine v3.0.

Testable implementation of CAMS Instruction Set
(v2.0 + v3.0 Emergent Protocol), with AI scoring lookup support.
"""

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# CAMS standard node names (fixed order for reproducibility)
CAMS_NODES = [
    "Helm",
    "Shield",
    "Lore",
    "Stewards",
    "Craft",
    "Hands",
    "Archive",
    "Flow",
]

# Weighting constants for bond formula (v2.0 standard)
COHERENCE_WEIGHT = 0.6
ABSTRACTION_WEIGHT = 0.4

# Divergence threshold for emergent vs formulaic bond
DIVERGENCE_THRESHOLD = 3

# Protocol reference for AI evidence-based scoring
SCORING_PROTOCOL_PATH = Path(__file__).with_name("CAMS_SCORING_PROTOCOL_V2_1.md")


class Column(list):
    """Simple column wrapper with pandas-like tolist()."""

    def tolist(self):
        return list(self)


class CAMSDataFrame:
    """Lightweight DataFrame-like object with CSV export support."""

    def __init__(self, rows: List[dict]):
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, column: str) -> Column:
        return Column([row[column] for row in self._rows])

    def to_csv(self, path: str, index: bool = False) -> None:
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self._rows)

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self._rows)


# -------------------------------------------------------------------
# AI LOOKUP SUPPORT (SCORING LAYER)
# -------------------------------------------------------------------


def load_scoring_protocol() -> str:
    """Load CAMS Scoring Protocol v2.1 for AI-based node scoring prompts."""
    return SCORING_PROTOCOL_PATH.read_text(encoding="utf-8")



def build_ai_lookup_prompt(society: str, year: int, evidence: str) -> str:
    """Build an LLM prompt for scoring-only CAMS lookup.

    Returns a complete prompt containing protocol + evidence request.
    """
    protocol = load_scoring_protocol()
    return (
        f"{protocol}\n\n"
        "---\n"
        "## Lookup Request\n"
        f"Society: {society}\n"
        f"Year: {year}\n\n"
        "Use the protocol above and score only the eight CAMS nodes.\n"
        "Return only the CSV snippet with the required fields and exactly 8 rows.\n"
        "Do not compute Node Value, Bond Strength, averages, or any derived metric.\n\n"
        "### Evidence Context\n"
        f"{evidence.strip()}\n"
    )


def run_ai_lookup(
    society: str,
    year: int,
    evidence: str,
    model: str = "gpt-4.1-mini",
) -> str:
    """Call OpenAI Responses API and return raw CSV snippet from the model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    prompt = build_ai_lookup_prompt(society, year, evidence)
    payload = {
        "model": model,
        "input": prompt,
    }

    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["output"][0]["content"][0]["text"]


# -------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------


@dataclass
class NodeMetrics:
    """Stores CAMS metrics for a single node."""

    coherence: int
    capacity: int
    stress: int
    abstraction: int
    bs_native: Optional[float] = None  # Optional emergent bond strength (v3.0)

    def validate(self) -> None:
        """Ensure metrics are within 1–10 bounds."""
        for field, value in asdict(self).items():
            if field != "bs_native" and not (1 <= value <= 10):
                raise ValueError(f"{field} must be between 1 and 10")


# -------------------------------------------------------------------
# CORE CALCULATIONS
# -------------------------------------------------------------------


def compute_node_value(node: NodeMetrics) -> float:
    """v2.0 Node Value formula.

    NV = Coherence + Capacity - Stress + (Abstraction * 0.5)
    """
    return node.coherence + node.capacity - node.stress + (node.abstraction * 0.5)



def compute_pairwise_bond(node_a: NodeMetrics, node_b: NodeMetrics) -> float:
    """v2.0 Pairwise Bond Strength formula.

    Bond = [(C_a + C_b)*0.6 + (A_a + A_b)*0.4] / (1 + avg_stress)
    """
    avg_stress = (node_a.stress + node_b.stress) / 2

    numerator = (
        (node_a.coherence + node_b.coherence) * COHERENCE_WEIGHT
        + (node_a.abstraction + node_b.abstraction) * ABSTRACTION_WEIGHT
    )

    return numerator / (1 + avg_stress)



def compute_node_bond_strength(node_name: str, nodes: Dict[str, NodeMetrics]) -> float:
    """Average bond strength for a node across all 7 pairwise connections."""
    bonds = []
    for other_name, other_node in nodes.items():
        if other_name == node_name:
            continue
        bonds.append(compute_pairwise_bond(nodes[node_name], other_node))

    return sum(bonds) / len(bonds)



def compute_standard_bs(node: NodeMetrics) -> float:
    """v3.0 audit formula for standardized bond strength.

    BS_Std = C + K + A - 2S
    """
    return node.coherence + node.capacity + node.abstraction - 2 * node.stress


# -------------------------------------------------------------------
# EMERGENT PROTOCOL SUPPORT
# -------------------------------------------------------------------


def check_divergence(bs_native: Optional[float], bs_std: float) -> bool:
    """Detect significant divergence between emergent and formulaic bond strength."""
    if bs_native is None:
        return False
    return abs(bs_native - bs_std) > DIVERGENCE_THRESHOLD


# -------------------------------------------------------------------
# SCORING PIPELINE
# -------------------------------------------------------------------


def score_society(society: str, year: int, node_data: Dict[str, NodeMetrics]) -> CAMSDataFrame:
    """Main scoring function.

    Returns a table object ready for CSV export.
    """

    # Validate node coverage
    if set(node_data.keys()) != set(CAMS_NODES):
        raise ValueError("All 8 CAMS nodes must be provided")

    # Validate metric ranges
    for node in node_data.values():
        node.validate()

    rows = []

    for node_name in CAMS_NODES:
        metrics = node_data[node_name]
        nv = compute_node_value(metrics)
        bs_node = compute_node_bond_strength(node_name, node_data)
        bs_std = compute_standard_bs(metrics)
        divergence_flag = check_divergence(metrics.bs_native, bs_std)

        rows.append(
            {
                "Society": society,
                "Year": year,
                "Node": node_name,
                "Coherence": metrics.coherence,
                "Capacity": metrics.capacity,
                "Stress": metrics.stress,
                "Abstraction": metrics.abstraction,
                "Node Value": round(nv, 1),
                "Bond Strength": round(bs_node, 3),
                "BS_Std": bs_std,
                "BS_Native": metrics.bs_native,
                "Divergence": divergence_flag,
            }
        )

    return CAMSDataFrame(rows)


if __name__ == "__main__":
    # EXAMPLE TEST CASE: USA 1861 (Demonstration)
    usa_1861 = {
        "Helm": NodeMetrics(4, 5, 9, 5, bs_native=2),
        "Shield": NodeMetrics(3, 4, 10, 4, bs_native=1),
        "Lore": NodeMetrics(7, 7, 4, 8, bs_native=6),
        "Stewards": NodeMetrics(5, 6, 7, 5, bs_native=4),
        "Craft": NodeMetrics(6, 6, 5, 7, bs_native=5),
        "Hands": NodeMetrics(4, 5, 8, 4, bs_native=2),
        "Archive": NodeMetrics(7, 7, 4, 8, bs_native=6),
        "Flow": NodeMetrics(7, 7, 5, 7, bs_native=6),
    }

    df = score_society("USA", 1861, usa_1861)
    print(df)
    # df.to_csv("USA_1861_CAMS.csv", index=False)
