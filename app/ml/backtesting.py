from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from app.ml.scoring import (
    binary_log_loss,
    brier_score,
    expected_calibration_error,
    interval_coverage,
    mean_absolute_error,
)


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    observations: int
    brier: float
    log_loss: float
    calibration_error: float
    vote_share_mae: float | None = None
    interval_coverage: float | None = None


def evaluate_binary_forecasts(frame: pd.DataFrame) -> BacktestMetrics:
    required = {"outcome", "win_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    y = frame["outcome"].to_numpy(dtype=float)
    p = frame["win_probability"].to_numpy(dtype=float)
    vote_mae: float | None = None
    coverage: float | None = None
    if {"actual_share", "predicted_share"}.issubset(frame.columns):
        vote_mae = mean_absolute_error(frame["actual_share"].to_numpy(), frame["predicted_share"].to_numpy())
    if {"actual_share", "lower", "upper"}.issubset(frame.columns):
        coverage = interval_coverage(
            frame["actual_share"].to_numpy(),
            frame["lower"].to_numpy(),
            frame["upper"].to_numpy(),
        )
    return BacktestMetrics(
        observations=len(frame),
        brier=brier_score(y, p),
        log_loss=binary_log_loss(y, p),
        calibration_error=expected_calibration_error(y, p),
        vote_share_mae=vote_mae,
        interval_coverage=coverage,
    )


def summarize_by_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_column, dropna=False):
        metrics = evaluate_binary_forecasts(group)
        rows.append({group_column: key, **asdict(metrics)})
    return pd.DataFrame(rows)
