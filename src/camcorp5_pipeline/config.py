"""Configuration helpers for the CAMCORP5 analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class PipelineConfig:
    """File locations and export settings for one pipeline run."""

    ensemble_mean_csv: Path
    envelope_csv: Path
    output_dir: Path
    entity: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        base_dir = config_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

        return cls(
            ensemble_mean_csv=resolve(data["ensemble_mean_csv"]),
            envelope_csv=resolve(data["envelope_csv"]),
            output_dir=resolve(data.get("output_dir", "../exports/camcorp5_pipeline")),
            entity=data.get("entity"),
        )

