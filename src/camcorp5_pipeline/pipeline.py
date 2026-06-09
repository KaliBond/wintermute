"""Composable CAMCORP5 pipeline entry point."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .corp_v02 import compute_v02_outputs
from .io import load_ensemble_mean, load_envelope, write_csv
from .transforms import merge_mean_with_envelope, node_rankings, summarize_by_year


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    mean = load_ensemble_mean(config.ensemble_mean_csv)
    envelope = load_envelope(config.envelope_csv)

    if config.entity:
        mean = mean.loc[mean["Entity"].eq(config.entity)].copy()
        envelope = envelope.loc[envelope["Entity"].eq(config.entity)].copy()
        if mean.empty:
            raise ValueError(f"No ensemble mean rows found for entity: {config.entity}")

    node_panel = merge_mean_with_envelope(mean, envelope)
    yearly_summary = summarize_by_year(node_panel)
    rankings = node_rankings(node_panel)

    output_dir = config.output_dir
    outputs = {
        "node_panel": write_csv(node_panel, output_dir / "camcorp5_node_panel.csv"),
        "yearly_summary": write_csv(yearly_summary, output_dir / "camcorp5_yearly_summary.csv"),
        "node_rankings": write_csv(rankings, output_dir / "camcorp5_node_rankings.csv"),
    }

    for name, frame in compute_v02_outputs(node_panel).items():
        outputs[name] = write_csv(frame, output_dir / f"{name}.csv")

    return outputs
