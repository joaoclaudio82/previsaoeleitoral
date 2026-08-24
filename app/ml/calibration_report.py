from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.calibration import calibration_table
from app.ml.scoring import brier_score, expected_calibration_error, interval_coverage, mean_absolute_error


def calibration_by_group(forecasts: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    required = {"win_probability", "outcome", "predicted_share", "actual_share", "lower", "upper", *group_columns}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast frame missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, group in forecasts.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        p = group["win_probability"].to_numpy(dtype=float)
        y = group["outcome"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "observations": len(group),
                "brier": brier_score(y, p),
                "ece": expected_calibration_error(y, p, bins=min(10, max(2, len(group) // 4))),
                "vote_share_mae": mean_absolute_error(group["actual_share"].to_numpy(), group["predicted_share"].to_numpy()),
                "interval_coverage": interval_coverage(group["actual_share"].to_numpy(), group["lower"].to_numpy(), group["upper"].to_numpy()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(forecasts: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = calibration_table(
        forecasts["win_probability"].to_numpy(dtype=float),
        forecasts["outcome"].to_numpy(dtype=float),
        bins=bins,
    )
    return pd.DataFrame(rows)


def calibration_slope_intercept(forecasts: pd.DataFrame) -> dict[str, float]:
    p = np.clip(forecasts["win_probability"].to_numpy(dtype=float), 1e-5, 1 - 1e-5)
    y = forecasts["outcome"].to_numpy(dtype=float)
    x = np.log(p / (1 - p))
    design = np.column_stack([np.ones_like(x), x])
    weights = np.clip(p * (1 - p), 1e-4, None)
    target = x + (y - p) / weights
    coef = np.linalg.pinv(design.T @ (weights[:, None] * design)) @ design.T @ (weights * target)
    return {"intercept": float(coef[0]), "slope": float(coef[1])}
