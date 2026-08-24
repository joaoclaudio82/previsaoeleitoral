from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def as_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def assert_not_after_as_of(frame: pd.DataFrame, column: str, as_of_date: str | date) -> None:
    if frame.empty or column not in frame.columns:
        return
    cutoff = pd.Timestamp(as_date(as_of_date))
    observed = pd.to_datetime(frame[column], errors="coerce")
    invalid = observed.isna() | (observed > cutoff)
    if invalid.any():
        rows = frame.index[invalid].tolist()[:10]
        raise ValueError(f"Temporal leakage in {column}; invalid rows: {rows}")


def available_as_of(
    frame: pd.DataFrame,
    as_of_date: str | date,
    release_column: str = "release_date",
) -> pd.DataFrame:
    if release_column not in frame.columns:
        raise ValueError(f"Missing release column: {release_column}")
    cutoff = pd.Timestamp(as_date(as_of_date))
    release = pd.to_datetime(frame[release_column], errors="coerce")
    return frame.loc[release.notna() & (release <= cutoff)].copy()
