"""Schema definitions and validation for CAMCORP5 inputs."""

from __future__ import annotations

import pandas as pd

KEY_COLUMNS = ["Entity", "Year", "Node"]
MEAN_COLUMNS = [
    *KEY_COLUMNS,
    "Coherence",
    "Capacity",
    "Stress",
    "Abstraction",
    "Node Value",
    "Bond Strength",
]
ENVELOPE_COLUMNS = [
    *KEY_COLUMNS,
    "C_sd",
    "K_sd",
    "S_sd",
    "A_sd",
    "V_range",
    "V_min",
    "V_max",
]
NUMERIC_MEAN_COLUMNS = [col for col in MEAN_COLUMNS if col not in {"Entity", "Node"}]
NUMERIC_ENVELOPE_COLUMNS = [col for col in ENVELOPE_COLUMNS if col not in {"Entity", "Node"}]


def require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def coerce_numeric(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid = cleaned[columns].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        details = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise ValueError(f"{label} contains non-numeric or blank values: {details}")
    return cleaned


def validate_unique_keys(frame: pd.DataFrame, label: str) -> None:
    duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate Entity/Year/Node rows")
