from __future__ import annotations

import numpy as np
import pandas as pd


def robust_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return 0.67448975 * (numeric - median) / mad


def flag_poll_outliers(
    frame: pd.DataFrame,
    value_column: str = "share",
    group_columns: tuple[str, ...] = ("candidate_id", "uf"),
    threshold: float = 3.5,
) -> pd.DataFrame:
    if value_column not in frame.columns:
        raise ValueError(f"Missing value column: {value_column}")
    existing_groups = [column for column in group_columns if column in frame.columns]
    output = frame.copy()
    if existing_groups:
        output["robust_z"] = output.groupby(existing_groups, dropna=False)[value_column].transform(robust_zscore)
    else:
        output["robust_z"] = robust_zscore(output[value_column])
    output["is_outlier"] = output["robust_z"].abs() > threshold
    return output
