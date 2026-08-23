from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "distance_advantage_a", "rejection_advantage_a", "same_bloc_advantage_a",
    "incumbency_advantage_a", "source_rejection", "finalist_distance_sum",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class BayesianBinomialLogit:
    beta: np.ndarray
    covariance: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        successes: np.ndarray,
        totals: np.ndarray,
        prior_sd: float = 1.0,
        iterations: int = 80,
    ) -> "BayesianBinomialLogit":
        x = np.asarray(x, dtype=float)
        successes = np.asarray(successes, dtype=float)
        totals = np.asarray(totals, dtype=float)
        beta = np.zeros(x.shape[1], dtype=float)
        prior_precision = np.eye(x.shape[1]) / (prior_sd ** 2)
        prior_precision[0, 0] = 1 / 25.0
        for _ in range(iterations):
            p = _sigmoid(x @ beta)
            weights = np.maximum(totals * p * (1 - p), 1e-6)
            gradient = x.T @ (successes - totals * p) - prior_precision @ beta
            hessian_precision = (x.T * weights) @ x + prior_precision
            step = np.linalg.solve(hessian_precision, gradient)
            beta_next = beta + step
            if np.max(np.abs(step)) < 1e-8:
                beta = beta_next
                break
            beta = beta_next
        p = _sigmoid(x @ beta)
        weights = np.maximum(totals * p * (1 - p), 1e-6)
        covariance = np.linalg.pinv((x.T * weights) @ x + prior_precision)
        return cls(beta, covariance)


@dataclass
class TransferModel:
    conditional_model: BayesianBinomialLogit
    abstention_model: BayesianBinomialLogit
    means: dict[str, float]
    scales: dict[str, float]
    version: str = "0.2.0"

    @staticmethod
    def _standardize(frame: pd.DataFrame, means: dict[str, float], scales: dict[str, float]) -> np.ndarray:
        columns = [np.ones(len(frame), dtype=float)]
        for feature in FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(means[feature]).to_numpy(dtype=float)
            columns.append((values - means[feature]) / scales[feature])
        return np.column_stack(columns)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TransferModel":
        required = set(FEATURES + ["to_a", "to_b", "abstain"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no treino de transferência: {sorted(missing)}")
        means = {feature: float(frame[feature].mean()) for feature in FEATURES}
        scales = {feature: float(max(frame[feature].std(ddof=0), 1.0)) for feature in FEATURES}
        x = cls._standardize(frame, means, scales)
        to_a = frame["to_a"].to_numpy(dtype=float)
        to_b = frame["to_b"].to_numpy(dtype=float)
        abstain = frame["abstain"].to_numpy(dtype=float)
        conditional = BayesianBinomialLogit.fit(x, to_a, np.maximum(to_a + to_b, 1.0))
        abstention = BayesianBinomialLogit.fit(x, abstain, np.maximum(to_a + to_b + abstain, 1.0))
        return cls(conditional, abstention, means, scales)

    @staticmethod
    def pair_features(source: pd.Series, finalist_a: pd.Series, finalist_b: pd.Series) -> dict[str, float]:
        source_ideology = float(source.get("ideology_score", 0.0))
        a_ideology = float(finalist_a.get("ideology_score", 0.0))
        b_ideology = float(finalist_b.get("ideology_score", 0.0))
        distance_a = abs(source_ideology - a_ideology)
        distance_b = abs(source_ideology - b_ideology)
        same_a = float(str(source.get("bloc", "")) == str(finalist_a.get("bloc", "")))
        same_b = float(str(source.get("bloc", "")) == str(finalist_b.get("bloc", "")))
        return {
            "distance_advantage_a": distance_b - distance_a,
            "rejection_advantage_a": float(finalist_b.get("rejection", 50)) - float(finalist_a.get("rejection", 50)),
            "same_bloc_advantage_a": same_a - same_b,
            "incumbency_advantage_a": float(finalist_a.get("incumbency", 0)) - float(finalist_b.get("incumbency", 0)),
            "source_rejection": float(source.get("rejection", 50)),
            "finalist_distance_sum": distance_a + distance_b,
        }

    def precompute_pair_draws(
        self,
        fundamentals: pd.DataFrame,
        n_draws: int,
        seed: int,
    ) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
        indexed = fundamentals.set_index("candidate_id", drop=False)
        ids = indexed.index.astype(str).tolist()
        rng = np.random.default_rng(seed)
        beta_cond = rng.multivariate_normal(self.conditional_model.beta, self.conditional_model.covariance, size=n_draws)
        beta_abs = rng.multivariate_normal(self.abstention_model.beta, self.abstention_model.covariance, size=n_draws)
        result: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for source_id in ids:
            for a_id in ids:
                for b_id in ids:
                    if len({source_id, a_id, b_id}) < 3:
                        continue
                    features = pd.DataFrame([
                        self.pair_features(indexed.loc[source_id], indexed.loc[a_id], indexed.loc[b_id])
                    ])
                    x = self._standardize(features, self.means, self.scales)[0]
                    p_a = _sigmoid(beta_cond @ x)
                    p_abstain = np.clip(_sigmoid(beta_abs @ x), 0.0, 0.65)
                    result[(source_id, a_id, b_id)] = (p_a, p_abstain)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransferModel":
        return joblib.load(path)
