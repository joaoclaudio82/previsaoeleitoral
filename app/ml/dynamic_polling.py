from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DynamicPollingForecast:
    candidate_ids: list[str]
    candidate_names: list[str]
    draws: np.ndarray
    process_sd_per_day: float


def _inverse_alr(values: np.ndarray) -> np.ndarray:
    exp_values = np.exp(np.clip(values, -20.0, 20.0))
    denom = 1.0 + exp_values.sum(axis=-1, keepdims=True)
    return np.concatenate([exp_values / denom, 1.0 / denom], axis=-1)


def _prepare(polls: pd.DataFrame, as_of_date: date) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
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
        .agg(field_date=("field_date", "max"), margin_error=("margin_error", "max"))
        .reindex(pivot.index)
    )
    order = np.argsort(np.array(meta["field_date"], dtype="datetime64[D]"))
    return pivot.iloc[order], meta.iloc[order], candidate_ids, candidate_names


def fit_dynamic_polling_baseline(
    polls: pd.DataFrame,
    as_of_date: date,
    *,
    forecast_date: date | None = None,
    n_draws: int = 4000,
    seed: int = 42,
    process_sd_per_day: float = 0.025,
) -> DynamicPollingForecast:
    """Local-level state-space baseline in additive log-ratio coordinates.

    The latent national support vector follows a Gaussian random walk. Polls update
    the latent state sequentially, using only observations available at ``as_of_date``.
    Forecast uncertainty grows with the number of days between the cutoff and the
    requested forecast date. The process scale is fixed ex ante and can be subjected
    to sensitivity analysis; it is never estimated from the held-out election result.
    """
    target = forecast_date or as_of_date
    if target < as_of_date:
        raise ValueError("forecast_date cannot be earlier than as_of_date")
    pivot, meta, candidate_ids, candidate_names = _prepare(polls, as_of_date)
    shares = np.clip(pivot.to_numpy(dtype=float), 1e-4, None)
    shares = shares / shares.sum(axis=1, keepdims=True)
    observations = np.log(shares[:, :-1] / shares[:, [-1]])
    dimension = observations.shape[1]

    mean = observations[0].copy()
    covariance = np.eye(dimension) * 0.35
    last_date = meta.iloc[0]["field_date"]
    q = float(process_sd_per_day) ** 2

    for index, observation in enumerate(observations):
        current_date = meta.iloc[index]["field_date"]
        elapsed = max((current_date - last_date).days, 0)
        covariance = covariance + np.eye(dimension) * q * elapsed

        margin = pd.to_numeric(pd.Series([meta.iloc[index]["margin_error"]]), errors="coerce").iloc[0]
        margin = float(margin) if np.isfinite(margin) else 2.5
        share_sigma = max(margin / 100.0 / 1.96, 0.008)
        # Delta-method approximation on ALR coordinates, with a conservative floor.
        obs_sd = max(0.075, 2.5 * share_sigma)
        observation_covariance = np.eye(dimension) * obs_sd**2

        innovation_covariance = covariance + observation_covariance
        gain = covariance @ np.linalg.inv(innovation_covariance)
        mean = mean + gain @ (observation - mean)
        covariance = (np.eye(dimension) - gain) @ covariance
        covariance = (covariance + covariance.T) / 2.0
        last_date = current_date

    future_days = max((target - last_date).days, 0)
    covariance = covariance + np.eye(dimension) * q * future_days
    covariance = covariance + np.eye(dimension) * 1e-8

    rng = np.random.default_rng(seed)
    latent_draws = rng.multivariate_normal(mean, covariance, size=n_draws)
    draws = _inverse_alr(latent_draws)
    return DynamicPollingForecast(candidate_ids, candidate_names, draws, float(process_sd_per_day))
