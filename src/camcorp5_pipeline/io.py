"""Input and output helpers for CAMCORP5 analysis artifacts."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .schema import (
    ENVELOPE_COLUMNS,
    MEAN_COLUMNS,
    NUMERIC_ENVELOPE_COLUMNS,
    NUMERIC_MEAN_COLUMNS,
    coerce_numeric,
    require_columns,
    validate_unique_keys,
)


def load_ensemble_mean(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, MEAN_COLUMNS, "ensemble mean CSV")
    frame = frame[MEAN_COLUMNS].copy()
    frame["Entity"] = frame["Entity"].astype(str).str.strip()
    frame["Node"] = frame["Node"].astype(str).str.strip()
    frame = coerce_numeric(frame, NUMERIC_MEAN_COLUMNS, "ensemble mean CSV")
    frame["Year"] = frame["Year"].astype(int)
    validate_unique_keys(frame, "ensemble mean CSV")
    return frame


def load_envelope(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, ENVELOPE_COLUMNS, "envelope CSV")
    frame = frame[ENVELOPE_COLUMNS].copy()
    frame["Entity"] = frame["Entity"].astype(str).str.strip()
    frame["Node"] = frame["Node"].astype(str).str.strip()
    frame = coerce_numeric(frame, NUMERIC_ENVELOPE_COLUMNS, "envelope CSV")
    frame["Year"] = frame["Year"].astype(int)
    validate_unique_keys(frame, "envelope CSV")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
