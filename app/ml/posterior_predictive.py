from __future__ import annotations

import numpy as np


def _softmax_alr(alr: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*alr.shape[:-1], 1), dtype=float)
    logits = np.concatenate([alr, zeros], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True) * 100.0


def _to_alr(shares: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(shares, dtype=float) / 100.0, 1e-6, None)
    values = values / values.sum(axis=-1, keepdims=True)
    return np.log(values[..., :-1] / values[..., [-1]])


def add_correlated_predictive_error(
    national_draws: np.ndarray,
    state_draws: np.ndarray,
    residual_covariance: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert parameter draws into posterior-predictive election forecasts.

    The matrix-normal fit yields uncertainty about the latent mean support. A forecast
    for a future realized election also needs an irreducible polling/model-error term.
    We draw one correlated ALR innovation per posterior draw from the residual
    covariance estimated using only polls available at the cutoff. The same innovation
    is shared across geographies, representing a national polling miss while preserving
    state-specific uncertainty supplied by the hierarchical prior.

    No future election outcome is used to set the innovation scale.
    """
    national = np.asarray(national_draws, dtype=float)
    states = np.asarray(state_draws, dtype=float)
    covariance = np.atleast_2d(np.asarray(residual_covariance, dtype=float))
    dimensions = national.shape[-1] - 1
    if covariance.shape != (dimensions, dimensions):
        raise ValueError(
            f"residual_covariance must have shape {(dimensions, dimensions)}, got {covariance.shape}"
        )
    if states.shape[0] != national.shape[0] or states.shape[-1] != national.shape[-1]:
        raise ValueError("state_draws and national_draws have incompatible shapes")

    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    values = np.clip(values, 1e-8, None)
    root = vectors @ np.diag(np.sqrt(values))
    rng = np.random.default_rng(seed)
    innovation = rng.normal(size=(len(national), dimensions)) @ root.T

    national_alr = _to_alr(national) + innovation
    state_alr = _to_alr(states) + innovation[:, None, :]
    return _softmax_alr(national_alr), _softmax_alr(state_alr)
