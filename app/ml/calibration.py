from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(slots=True)
class ProbabilityCalibrator:
    """Monotonic out-of-sample probability calibrator."""

    out_of_bounds: str = "clip"
    _model: IsotonicRegression = field(init=False, repr=False)
    _fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)

    def fit(self, probability: np.ndarray, outcome: np.ndarray) -> "ProbabilityCalibrator":
        p = np.asarray(probability, dtype=float)
        y = np.asarray(outcome, dtype=float)
        if p.shape != y.shape:
            raise ValueError("probability and outcome must have the same shape")
        if len(np.unique(y)) < 2:
            raise ValueError("calibration requires both outcome classes")
        self._model.fit(p, y)
        self._fitted = True
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("calibrator is not fitted")
        values = np.asarray(probability, dtype=float)
        return np.clip(self._model.predict(values), 0.0, 1.0)


def calibration_table(
    probability: np.ndarray,
    outcome: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        mask = (p >= edges[idx]) & ((p < edges[idx + 1]) if idx < bins - 1 else (p <= edges[idx + 1]))
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": idx,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return rows
