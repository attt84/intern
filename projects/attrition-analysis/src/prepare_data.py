from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "Attrition",
    "JobRole",
    "JobLevel",
    "OverTime",
    "YearsSinceLastPromotion",
]


def load_attrition_csv(path: str | Path) -> pd.DataFrame:
    """Load a public attrition dataset CSV."""
    return pd.read_csv(path)


def available_columns(frame: pd.DataFrame) -> list[str]:
    """Return the columns that match the portfolio analysis plan."""
    return [column for column in DEFAULT_COLUMNS if column in frame.columns]


def encode_attrition(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a binary target column when Attrition is Yes/No style."""
    encoded = frame.copy()
    if "Attrition" in encoded.columns:
        encoded["AttritionBinary"] = encoded["Attrition"].map({"Yes": 1, "No": 0})
    return encoded


def quick_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Return a small summary for notebook preview."""
    summary: dict[str, float] = {}
    if "Attrition" in frame.columns:
        attrition_rate = (frame["Attrition"] == "Yes").mean()
        summary["attrition_rate"] = float(attrition_rate)
    if "JobLevel" in frame.columns:
        summary["joblevel_unique"] = float(frame["JobLevel"].nunique())
    return summary
