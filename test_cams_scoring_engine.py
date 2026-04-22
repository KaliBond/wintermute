from pathlib import Path
import json

import pytest

from cams_scoring_engine import (
    CAMS_NODES,
    SCORING_PROTOCOL_PATH,
    NodeMetrics,
    build_ai_lookup_prompt,
    check_divergence,
    compute_node_value,
    compute_pairwise_bond,
    compute_standard_bs,
    load_scoring_protocol,
    run_ai_lookup,
    score_society,
)


def sample_nodes():
    return {
        "Helm": NodeMetrics(4, 5, 9, 5, bs_native=2),
        "Shield": NodeMetrics(3, 4, 10, 4, bs_native=1),
        "Lore": NodeMetrics(7, 7, 4, 8, bs_native=6),
        "Stewards": NodeMetrics(5, 6, 7, 5, bs_native=4),
        "Craft": NodeMetrics(6, 6, 5, 7, bs_native=5),
        "Hands": NodeMetrics(4, 5, 8, 4, bs_native=2),
        "Archive": NodeMetrics(7, 7, 4, 8, bs_native=6),
        "Flow": NodeMetrics(7, 7, 5, 7, bs_native=6),
    }


def test_protocol_file_exists():
    assert Path(SCORING_PROTOCOL_PATH).exists()


def test_load_scoring_protocol_contains_required_section():
    protocol = load_scoring_protocol()
    assert "# CAMS Scoring Protocol v2.1" in protocol
    assert "Do **not** calculate:" in protocol


def test_build_ai_lookup_prompt_includes_context_and_constraints():
    prompt = build_ai_lookup_prompt("USA", 1861, "Civil war escalation and rail mobilization.")
    assert "Society: USA" in prompt
    assert "Year: 1861" in prompt
    assert "Return only the CSV snippet" in prompt
    assert "Do not compute Node Value" in prompt


def test_run_ai_lookup_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        run_ai_lookup("USA", 1861, "evidence")


def test_run_ai_lookup_calls_openai(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "output": [
                    {
                        "content": [
                            {
                                "text": "Society,Year,Node,Coherence,Capacity,Stress,Abstraction",
                            }
                        ]
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cams_scoring_engine.request.urlopen", lambda _: FakeResponse())
    result = run_ai_lookup("USA", 1861, "evidence", model="gpt-4.1-mini")
    assert result.startswith("Society,Year,Node")


def test_node_value_formula():
    node = NodeMetrics(coherence=6, capacity=5, stress=4, abstraction=8)
    assert compute_node_value(node) == 11


def test_pairwise_bond_formula():
    a = NodeMetrics(6, 6, 4, 8)
    b = NodeMetrics(4, 5, 6, 5)
    # numerator=((6+4)*0.6)+((8+5)*0.4)=11.2 ; avg_stress=5 ; bond=11.2/6
    assert compute_pairwise_bond(a, b) == pytest.approx(11.2 / 6)


def test_standard_bs_formula():
    node = NodeMetrics(7, 6, 4, 5)
    assert compute_standard_bs(node) == 10


def test_check_divergence():
    assert check_divergence(2, 10) is True
    assert check_divergence(8, 10) is False
    assert check_divergence(None, 10) is False


def test_score_society_output_shape_and_columns():
    df = score_society("USA", 1861, sample_nodes())
    assert len(df) == 8
    assert df["Node"].tolist() == CAMS_NODES
    assert {
        "Society",
        "Year",
        "Node",
        "Coherence",
        "Capacity",
        "Stress",
        "Abstraction",
        "Node Value",
        "Bond Strength",
        "BS_Std",
        "BS_Native",
        "Divergence",
    }.issubset(df.columns)


def test_score_society_requires_all_nodes():
    nodes = sample_nodes()
    nodes.pop("Flow")
    with pytest.raises(ValueError, match="All 8 CAMS nodes must be provided"):
        score_society("USA", 1861, nodes)


def test_validate_enforces_bounds():
    bad_nodes = sample_nodes()
    bad_nodes["Helm"] = NodeMetrics(11, 5, 5, 5)
    with pytest.raises(ValueError, match="coherence must be between 1 and 10"):
        score_society("USA", 1861, bad_nodes)
