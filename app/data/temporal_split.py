from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train_index: list[int]
    test_index: list[int]
    cutoff: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    date_column: str,
    min_train_periods: int = 3,
) -> list[TemporalFold]:
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    periods = sorted(pd.Series(dates.dt.normalize().unique()).tolist())
    if len(periods) <= min_train_periods:
        return []
    folds: list[TemporalFold] = []
    for position in range(min_train_periods, len(periods)):
        cutoff = pd.Timestamp(periods[position - 1])
        test_day = pd.Timestamp(periods[position])
        train_mask = dates.dt.normalize() <= cutoff
        test_mask = dates.dt.normalize() == test_day
        folds.append(
            TemporalFold(
                train_index=frame.index[train_mask].tolist(),
                test_index=frame.index[test_mask].tolist(),
                cutoff=cutoff,
            )
        )
    return folds
