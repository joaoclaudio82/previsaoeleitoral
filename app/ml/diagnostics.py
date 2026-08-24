from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def variance_decomposition(components: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Approximate each independent component's share of total variance."""
    if not components:
        return {}
    variances = {name: float(np.var(np.asarray(values, dtype=float), ddof=1)) for name, values in components.items()}
    total = sum(max(value, 0.0) for value in variances.values())
    if total <= 0:
        return {name: 0.0 for name in variances}
    return {name: value / total for name, value in variances.items()}


def posterior_interval(draws: np.ndarray, level: float = 0.8) -> tuple[float, float, float]:
    values = np.asarray(draws, dtype=float)
    if values.size == 0:
        raise ValueError("draws cannot be empty")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    alpha = (1.0 - level) / 2.0
    low, median, high = np.quantile(values, [alpha, 0.5, 1.0 - alpha])
    return float(low), float(median), float(high)


def effective_sample_size_proxy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0):
        raise ValueError("weights cannot be negative")
    total = values.sum()
    if total <= 0:
        return 0.0
    normalized = values / total
    return float(1.0 / np.sum(normalized**2))
