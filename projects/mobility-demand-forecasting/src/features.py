from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_timeseries_csv(path: str | Path, date_column: str = "dteday") -> pd.DataFrame:
    """Load a public mobility demand CSV and parse the date column."""
    frame = pd.read_csv(path)
    if date_column in frame.columns:
        frame[date_column] = pd.to_datetime(frame[date_column])
    return frame


def add_calendar_features(frame: pd.DataFrame, date_column: str = "dteday") -> pd.DataFrame:
    """Add month, weekday, and day-of-year features."""
    enriched = frame.copy()
    if date_column in enriched.columns:
        enriched["month"] = enriched[date_column].dt.month
        enriched["weekday_name"] = enriched[date_column].dt.day_name()
        enriched["dayofyear"] = enriched[date_column].dt.dayofyear
    return enriched


def add_lag_features(frame: pd.DataFrame, target_column: str, lags: tuple[int, ...] = (1, 7)) -> pd.DataFrame:
    """Add simple lag features used in the portfolio notebook."""
    enriched = frame.copy()
    for lag in lags:
        enriched[f"{target_column}_lag_{lag}"] = enriched[target_column].shift(lag)
    enriched[f"{target_column}_rolling_7"] = enriched[target_column].rolling(7).mean()
    return enriched


def split_by_time(frame: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the frame in chronological order."""
    train_end = int(len(frame) * train_ratio)
    valid_end = int(len(frame) * (train_ratio + valid_ratio))
    return frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]
