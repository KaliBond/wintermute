"""Command-line interface for the CAMCORP5 analysis pipeline."""

from __future__ import annotations

import argparse

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CAMCORP5 analysis pipeline")
    parser.add_argument(
        "--config",
        default="configs/camcorp5.json",
        help="Path to a JSON pipeline config file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_json(args.config)
    outputs = run_pipeline(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
