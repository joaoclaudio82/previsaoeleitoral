from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClusterBootstrapResult:
    mean_difference: float
    lower: float
    upper: float
    probability_improvement: float
    draws: int


def paired_difference_by_cluster(
    frame: pd.DataFrame,
    *,
    cluster: str,
    treatment: str,
    control: str,
    value: str,
    model_column: str = "model",
) -> pd.DataFrame:
    """Return treatment-control differences after averaging within clusters."""
    required = {cluster, model_column, value}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    grouped = frame.groupby([cluster, model_column], as_index=False)[value].mean()
    pivot = grouped.pivot(index=cluster, columns=model_column, values=value)
    if treatment not in pivot or control not in pivot:
        raise ValueError("treatment/control models are not both present")
    paired = pivot[[treatment, control]].dropna().copy()
    paired["difference"] = paired[treatment] - paired[control]
    return paired.reset_index()


def cluster_bootstrap_difference(
    paired: pd.DataFrame,
    *,
    difference_column: str = "difference",
    draws: int = 20_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> ClusterBootstrapResult:
    """Bootstrap a paired treatment-control difference at the cluster level.

    Negative differences indicate improvement when the scored quantity is an error.
    With very few clusters the interval is necessarily coarse; callers should report
    the number of independent clusters alongside the result.
    """
    values = pd.to_numeric(paired[difference_column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two independent clusters are required")
    if draws < 100:
        raise ValueError("draws must be >= 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    return ClusterBootstrapResult(
        mean_difference=float(values.mean()),
        lower=float(np.quantile(sampled, alpha / 2)),
        upper=float(np.quantile(sampled, 1 - alpha / 2)),
        probability_improvement=float(np.mean(sampled < 0)),
        draws=int(draws),
    )


def select_strength_leave_one_election_out(
    sensitivity: pd.DataFrame,
    *,
    holdout_year: int,
    metric: str = "vote_share_mae",
) -> tuple[float, pd.DataFrame]:
    """Choose prior strength using every election except the held-out cycle."""
    required = {"election_year", "prior_strength", metric}
    missing = required.difference(sensitivity.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    train = sensitivity[sensitivity["election_year"] != holdout_year].copy()
    if train.empty:
        raise ValueError("no training elections remain after holdout")
    summary = (
        train.groupby("prior_strength", as_index=False)[metric]
        .mean()
        .sort_values([metric, "prior_strength"])
        .reset_index(drop=True)
    )
    return float(summary.iloc[0]["prior_strength"]), summary
