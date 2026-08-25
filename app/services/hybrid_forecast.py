from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from app.agents.schemas import AgentScenario
from app.services.predictor import PredictionBundle, predict


@dataclass
class HybridForecastComparison:
    """Side-by-side baseline and experimental agent scenario outputs."""

    baseline: PredictionBundle
    experimental: PredictionBundle
    candidate_deltas: pd.DataFrame
    event_id: str
    strength: float


def compare_hybrid_forecast(
    *,
    polls: pd.DataFrame,
    fundamentals: pd.DataFrame,
    as_of_date: date,
    model_path: str | Path,
    n_simulations: int,
    seed: int,
    scenario: AgentScenario,
    strength: float = 0.35,
    state_priors: pd.DataFrame | None = None,
    turnout: pd.DataFrame | None = None,
    pollster_calibration_path: str | Path | None = None,
    turnout_model_path: str | Path | None = None,
    transfer_model_path: str | Path | None = None,
    posterior_draws: int = 8_000,
) -> HybridForecastComparison:
    """Run the exact same forecast twice and isolate the multiagent delta."""
    common = dict(
        polls=polls,
        fundamentals=fundamentals,
        as_of_date=as_of_date,
        model_path=model_path,
        n_simulations=n_simulations,
        seed=seed,
        state_priors=state_priors,
        turnout=turnout,
        pollster_calibration_path=pollster_calibration_path,
        turnout_model_path=turnout_model_path,
        transfer_model_path=transfer_model_path,
        posterior_draws=posterior_draws,
    )
    baseline = predict(**common)
    experimental = predict(
        **common,
        agent_scenario=scenario,
        agent_scenario_strength=strength,
    )

    base = baseline.candidates.set_index("candidate_id")
    hybrid = experimental.candidates.set_index("candidate_id")
    if set(base.index) != set(hybrid.index):
        raise RuntimeError("Baseline e cenário produziram conjuntos diferentes de candidatos")

    deltas = pd.DataFrame(
        {
            "candidate_name": hybrid["candidate_name"],
            "baseline_first_round": base["expected_first_round_share"],
            "scenario_first_round": hybrid["expected_first_round_share"],
            "delta_first_round": hybrid["expected_first_round_share"] - base["expected_first_round_share"],
            "baseline_win_probability": base["win_probability"],
            "scenario_win_probability": hybrid["win_probability"],
            "delta_win_probability": hybrid["win_probability"] - base["win_probability"],
        }
    ).reset_index()

    return HybridForecastComparison(
        baseline=baseline,
        experimental=experimental,
        candidate_deltas=deltas,
        event_id=scenario.event_id,
        strength=strength,
    )
