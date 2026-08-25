from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class GeographicBaselineForecast:
    state_ids: list[str]
    candidate_ids: list[str]
    candidate_names: list[str]
    state_draws: np.ndarray


def apply_national_swing_state_lean(
    national_draws: np.ndarray,
    candidate_ids: list[str],
    candidate_names: list[str],
    state_priors: pd.DataFrame,
    *,
    seed: int = 42,
    concentration: float | None = None,
) -> GeographicBaselineForecast:
    """Apply previous-election state-vs-national leans to current national draws.

    This is deliberately simpler than ElectionAI's hierarchical model. It provides a
    transparent benchmark for the key geographic claim: whether historical state leans
    add value beyond a current national polling distribution. The baseline changes only
    geography; it does not estimate pollster, mode, population, or UF regression effects.
    """
    if state_priors is None or state_priors.empty:
        raise ValueError("state_priors are required")
    draws = np.asarray(national_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(candidate_ids):
        raise ValueError("national_draws must have shape (draws, candidates)")
    if draws.max() > 1.5:
        draws = draws / 100.0
    draws = np.clip(draws, 1e-9, None)
    draws = draws / draws.sum(axis=1, keepdims=True)

    frame = state_priors.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    state_ids = sorted(frame["uf"].astype(str).unique())
    rng = np.random.default_rng(seed)
    all_states: list[np.ndarray] = []

    for uf in state_ids:
        indexed = frame[frame["uf"].astype(str) == uf].set_index("candidate_id").reindex(candidate_ids)
        if indexed["prior_share"].isna().any() or indexed["national_prior_share"].isna().any():
            raise ValueError(f"Incomplete state prior for {uf}")
        state_prior = np.clip(indexed["prior_share"].to_numpy(dtype=float), 1e-6, None)
        national_prior = np.clip(indexed["national_prior_share"].to_numpy(dtype=float), 1e-6, None)
        state_prior = state_prior / state_prior.sum()
        national_prior = national_prior / national_prior.sum()

        if concentration is None:
            if "prior_concentration" in indexed.columns:
                local_concentration = float(pd.to_numeric(indexed["prior_concentration"], errors="coerce").median())
            else:
                local_concentration = 50.0
        else:
            local_concentration = float(concentration)
        local_concentration = max(local_concentration, 4.0)

        alpha = np.clip(state_prior * local_concentration, 0.05, None)
        sampled_state_prior = rng.dirichlet(alpha, size=len(draws))
        lean = sampled_state_prior / national_prior[None, :]
        projected = draws * lean
        projected = np.clip(projected, 1e-9, None)
        projected = projected / projected.sum(axis=1, keepdims=True)
        all_states.append(projected)

    return GeographicBaselineForecast(
        state_ids=state_ids,
        candidate_ids=list(candidate_ids),
        candidate_names=list(candidate_names),
        state_draws=np.stack(all_states, axis=1),
    )
