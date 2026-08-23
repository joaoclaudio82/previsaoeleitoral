from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "historical_turnout", "abstention_trend", "registration_growth",
    "mobility_index", "weather_severity", "competitiveness",
]


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


@dataclass
class TurnoutNowcaster:
    beta: np.ndarray
    covariance: np.ndarray
    residual_sd: float
    means: dict[str, float]
    scales: dict[str, float]
    uf_levels: list[str]
    version: str = "0.2.0"

    @staticmethod
    def _design(
        frame: pd.DataFrame,
        means: dict[str, float],
        scales: dict[str, float],
        uf_levels: list[str],
    ) -> np.ndarray:
        columns = [np.ones(len(frame), dtype=float)]
        for feature in NUMERIC_FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(means[feature]).to_numpy(dtype=float)
            columns.append((values - means[feature]) / scales[feature])
        ufs = frame["uf"].astype(str).to_numpy()
        for uf in uf_levels:
            columns.append((ufs == uf).astype(float))
        return np.column_stack(columns)

    @classmethod
    def fit(cls, frame: pd.DataFrame, prior_sd: float = 1.25) -> "TurnoutNowcaster":
        required = set(NUMERIC_FEATURES + ["uf", "turnout"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no treino de comparecimento: {sorted(missing)}")
        means = {feature: float(pd.to_numeric(frame[feature], errors="coerce").mean()) for feature in NUMERIC_FEATURES}
        scales = {
            feature: float(max(pd.to_numeric(frame[feature], errors="coerce").std(ddof=0), 1e-6))
            for feature in NUMERIC_FEATURES
        }
        uf_levels = sorted(frame["uf"].astype(str).unique())
        x = cls._design(frame, means, scales, uf_levels)
        y = _logit(pd.to_numeric(frame["turnout"], errors="coerce").to_numpy(dtype=float))
        prior_precision = np.eye(x.shape[1]) / (prior_sd ** 2)
        prior_precision[0, 0] = 1 / 25.0
        precision = x.T @ x + prior_precision
        covariance = np.linalg.pinv(precision)
        beta = covariance @ x.T @ y
        residual = y - x @ beta
        residual_sd = float(max(np.sqrt(np.mean(np.square(residual))), 0.03))
        return cls(beta, covariance, residual_sd, means, scales, uf_levels)

    def predict_draws(self, frame: pd.DataFrame, n_draws: int, seed: int) -> np.ndarray:
        x = self._design(frame, self.means, self.scales, self.uf_levels)
        rng = np.random.default_rng(seed)
        beta_draws = rng.multivariate_normal(self.beta, self.covariance, size=n_draws)
        latent = beta_draws @ x.T
        latent += rng.normal(0, self.residual_sd, size=latent.shape)
        return np.clip(_sigmoid(latent), 0.35, 0.95)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TurnoutNowcaster":
        return joblib.load(path)
