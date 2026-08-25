from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BaselineForecast:
    method: str
    candidate_ids: list[str]
    candidate_names: list[str]
    draws: np.ndarray


def _poll_matrix(polls: pd.DataFrame, as_of_date: date) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    frame = polls.copy()
    frame["field_date"] = pd.to_datetime(frame["field_date"], errors="coerce").dt.date
    frame = frame[frame["field_date"].notna() & (frame["field_date"] <= as_of_date)].copy()
    if frame.empty:
        raise ValueError("No polls available at the requested cutoff")
    candidates = frame[["candidate_id", "candidate_name"]].drop_duplicates().sort_values("candidate_id")
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    candidate_names = candidates["candidate_name"].astype(str).tolist()
    pivot = frame.pivot_table(index="poll_id", columns="candidate_id", values="share", aggfunc="mean")
    pivot = pivot.reindex(columns=candidate_ids).dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No complete polls remain after candidate alignment")
    meta = (
        frame.groupby("poll_id", as_index=True)
        .agg(field_date=("field_date", "max"), sample_size=("sample_size", "max"))
        .reindex(pivot.index)
    )
    return pivot, meta, candidate_ids, candidate_names


def _weights(meta: pd.DataFrame, as_of_date: date, method: str, half_life_days: float) -> np.ndarray:
    age = np.array([(as_of_date - value).days for value in meta["field_date"]], dtype=float)
    recency = np.exp(-math.log(2.0) * np.maximum(age, 0.0) / half_life_days)
    sample = np.sqrt(np.clip(pd.to_numeric(meta["sample_size"], errors="coerce").fillna(1000).to_numpy(dtype=float), 100, None) / 1000.0)
    if method == "simple_mean":
        return np.ones(len(meta), dtype=float)
    if method == "recency_weighted":
        return recency
    if method == "sample_recency_weighted":
        return recency * sample
    if method == "latest_poll":
        latest = max(meta["field_date"])
        return np.array([1.0 if value == latest else 0.0 for value in meta["field_date"]], dtype=float)
    raise ValueError(f"Unknown baseline method: {method}")


def fit_polling_baseline(
    polls: pd.DataFrame,
    as_of_date: date,
    *,
    method: str,
    n_draws: int = 4000,
    seed: int = 42,
    half_life_days: float = 24.0,
) -> BaselineForecast:
    pivot, meta, candidate_ids, candidate_names = _poll_matrix(polls, as_of_date)
    values = np.clip(pivot.to_numpy(dtype=float), 0.01, None)
    values = values / values.sum(axis=1, keepdims=True)
    weights = _weights(meta, as_of_date, method, half_life_days)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    center = np.average(values, axis=0, weights=weights)

    centered = values - center
    if len(values) > 1:
        covariance = (centered * weights[:, None]).T @ centered
    else:
        covariance = np.eye(values.shape[1]) * 0.0004
    sampling_floor = np.diag(np.maximum(center * (1.0 - center) / max(float(meta["sample_size"].median()), 500.0), 1e-6))
    covariance = 0.75 * covariance + 0.25 * sampling_floor + np.eye(values.shape[1]) * 1e-7

    rng = np.random.default_rng(seed)
    raw = rng.multivariate_normal(center, covariance, size=n_draws)
    raw = np.clip(raw, 1e-6, None)
    draws = raw / raw.sum(axis=1, keepdims=True)
    return BaselineForecast(method, candidate_ids, candidate_names, draws)
