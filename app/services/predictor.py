from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from app.ml.model import WinnerModel
from app.ml.pollster import PollsterCalibration
from app.ml.transfer import TransferModel
from app.ml.turnout import TurnoutNowcaster
from app.services.digital_signal_guard import guard_digital_signals
from app.services.hierarchical_polls import fit_hierarchical_poll_model
from app.services.monte_carlo import simulate_election


@dataclass
class PredictionBundle:
    candidates: pd.DataFrame
    states: pd.DataFrame
    institute_reliability: pd.DataFrame
    diagnostics: dict
    warnings: list[str]
    model_version: str

    def __iter__(self):
        # Backward-compatible unpacking: result, version = predict(...)
        yield self.candidates
        yield self.model_version


def build_feature_frame(poll_summary: pd.DataFrame, fundamentals: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    guarded, warnings, digital_diagnostics = guard_digital_signals(fundamentals)
    merged = poll_summary.merge(
        guarded,
        on=["candidate_id", "candidate_name"],
        how="left",
        validate="one_to_one",
    )
    if merged.isna().any().any():
        missing = merged.loc[merged.isna().any(axis=1), "candidate_id"].tolist()
        raise ValueError(f"Dados fundamentais incompletos para: {missing}")
    return merged, warnings, digital_diagnostics


def _load_optional(path: str | Path | None, loader):
    if path is None:
        return None
    target = Path(path)
    return loader(target) if target.exists() else None


def _turnout_inputs(
    state_ids: list[str],
    state_priors: pd.DataFrame | None,
    turnout: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if state_priors is not None and not state_priors.empty:
        electorate = (
            state_priors.groupby("uf", as_index=False)["registered_voters"].max()
            .set_index("uf")
            .reindex(state_ids)
        )
    else:
        electorate = pd.DataFrame({"registered_voters": [1_000_000] * len(state_ids)}, index=state_ids)
    electorate["registered_voters"] = electorate["registered_voters"].fillna(1_000_000)

    if turnout is None or turnout.empty:
        current = pd.DataFrame({"uf": state_ids})
    else:
        current = turnout.copy()
        current["uf"] = current["uf"].astype(str)
        current = current.drop_duplicates("uf").set_index("uf").reindex(state_ids).reset_index()
    defaults = {
        "historical_turnout": 0.79,
        "abstention_trend": 0.0,
        "registration_growth": 0.0,
        "mobility_index": 0.0,
        "weather_severity": 0.0,
        "competitiveness": 0.5,
    }
    for column, default in defaults.items():
        if column not in current:
            current[column] = default
        current[column] = pd.to_numeric(current[column], errors="coerce").fillna(default)
    current["registered_voters"] = electorate["registered_voters"].to_numpy(dtype=float)
    return current, electorate["registered_voters"].to_numpy(dtype=float)


def predict(
    polls: pd.DataFrame,
    fundamentals: pd.DataFrame,
    as_of_date: date,
    model_path: str | Path,
    n_simulations: int,
    seed: int,
    *,
    state_priors: pd.DataFrame | None = None,
    turnout: pd.DataFrame | None = None,
    pollster_calibration_path: str | Path | None = None,
    turnout_model_path: str | Path | None = None,
    transfer_model_path: str | Path | None = None,
    posterior_draws: int = 8_000,
) -> PredictionBundle:
    calibration = _load_optional(pollster_calibration_path, PollsterCalibration.load)
    posterior = fit_hierarchical_poll_model(
        polls,
        as_of_date,
        state_priors=state_priors,
        calibration=calibration,
        n_draws=min(max(1_000, posterior_draws), max(n_simulations, 1_000)),
        seed=seed,
    )
    features, warnings, digital_diagnostics = build_feature_frame(posterior.national_summary, fundamentals)
    winner_model = WinnerModel.load(model_path)
    features["ml_probability"] = winner_model.predict_normalized(features)

    current_turnout, registered_voters = _turnout_inputs(posterior.state_ids, state_priors, turnout)
    turnout_model = _load_optional(turnout_model_path, TurnoutNowcaster.load)
    if turnout_model is not None:
        turnout_draws = turnout_model.predict_draws(current_turnout, n_simulations, seed + 101)
        turnout_mode = "bayesian_state_nowcast"
    else:
        rng = np.random.default_rng(seed + 101)
        means = current_turnout["historical_turnout"].to_numpy(dtype=float)
        turnout_draws = np.clip(rng.normal(means, 0.025, size=(n_simulations, len(means))), 0.35, 0.95)
        turnout_mode = "fallback_historical_turnout"
        warnings.append("Modelo treinado de comparecimento não encontrado; foi usado fallback histórico.")

    simulation_fundamentals = features.copy()
    transfer_model = _load_optional(transfer_model_path, TransferModel.load)
    if transfer_model is None:
        warnings.append("Modelo treinado de transferência não encontrado; foi usado fallback ideologia/rejeição.")
    simulation = simulate_election(
        posterior=posterior,
        fundamentals=simulation_fundamentals,
        turnout_draws=turnout_draws,
        registered_voters=registered_voters,
        n_simulations=n_simulations,
        seed=seed,
        transfer_model=transfer_model,
    )
    diagnostics = {
        **posterior.diagnostics,
        **simulation.diagnostics,
        "turnout_mode": turnout_mode,
        "digital_signals": digital_diagnostics,
        "residual_correlation": posterior.residual_correlation.round(5).tolist(),
        "pollster_calibration_loaded": calibration is not None,
    }
    return PredictionBundle(
        candidates=simulation.candidates,
        states=simulation.states,
        institute_reliability=posterior.institute_reliability,
        diagnostics=diagnostics,
        warnings=warnings,
        model_version=winner_model.version,
    )
