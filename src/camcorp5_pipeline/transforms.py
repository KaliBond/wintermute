"""Core CAMCORP5 transformations."""

from __future__ import annotations

import pandas as pd

from .schema import KEY_COLUMNS


def merge_mean_with_envelope(mean: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    merged = mean.merge(envelope, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    expected = len(mean)
    if len(merged) != expected:
        missing = expected - len(merged)
        raise ValueError(f"Envelope merge dropped {missing} mean rows; check Entity/Year/Node coverage")

    merged = merged.sort_values(KEY_COLUMNS).reset_index(drop=True)
    merged["Stress Capacity Ratio"] = merged["Stress"] / merged["Capacity"].replace(0, pd.NA)
    merged["Resilience Index"] = (
        merged["Coherence"] + merged["Capacity"] + merged["Abstraction"] - merged["Stress"]
    ) / 3.0
    merged["Envelope Midpoint"] = (merged["V_min"] + merged["V_max"]) / 2.0
    merged["Envelope Delta"] = merged["Node Value"] - merged["Envelope Midpoint"]
    return merged


def summarize_by_year(node_panel: pd.DataFrame) -> pd.DataFrame:
    grouped = node_panel.groupby(["Entity", "Year"], as_index=False)
    summary = grouped.agg(
        node_count=("Node", "nunique"),
        mean_coherence=("Coherence", "mean"),
        mean_capacity=("Capacity", "mean"),
        mean_stress=("Stress", "mean"),
        mean_abstraction=("Abstraction", "mean"),
        mean_node_value=("Node Value", "mean"),
        mean_bond_strength=("Bond Strength", "mean"),
        mean_resilience_index=("Resilience Index", "mean"),
        mean_stress_capacity_ratio=("Stress Capacity Ratio", "mean"),
        mean_v_range=("V_range", "mean"),
        max_v_range=("V_range", "max"),
    )
    return summary.round(4)


def node_rankings(node_panel: pd.DataFrame) -> pd.DataFrame:
    ranked = node_panel.copy()
    ranked["node_value_rank"] = ranked.groupby(["Entity", "Year"])["Node Value"].rank(
        method="dense", ascending=False
    )
    ranked["stress_rank"] = ranked.groupby(["Entity", "Year"])["Stress"].rank(
        method="dense", ascending=False
    )
    return ranked.sort_values(["Entity", "Year", "node_value_rank", "Node"]).reset_index(drop=True)
