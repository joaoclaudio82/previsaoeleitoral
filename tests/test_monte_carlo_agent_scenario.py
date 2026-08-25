import numpy as np
import pandas as pd

from app.agents.schemas import AgentScenario, CandidateShock
from app.services.hierarchical_polls import PollPosterior
from app.services.monte_carlo import simulate_election


def _posterior() -> PollPosterior:
    state_draws = np.tile(np.array([[[40.0, 35.0, 25.0]]]), (500, 1, 1))
    national_summary = pd.DataFrame(
        [
            {"candidate_id": "A", "poll_mean": 40.0, "poll_lower": 38.0, "poll_upper": 42.0, "poll_uncertainty": 2.0},
            {"candidate_id": "B", "poll_mean": 35.0, "poll_lower": 33.0, "poll_upper": 37.0, "poll_uncertainty": 2.0},
            {"candidate_id": "C", "poll_mean": 25.0, "poll_lower": 23.0, "poll_upper": 27.0, "poll_uncertainty": 2.0},
        ]
    )
    return PollPosterior(
        candidate_ids=["A", "B", "C"],
        candidate_names=["Alice", "Bruno", "Carla"],
        state_ids=["CE"],
        national_draws=state_draws[:, 0, :],
        state_draws=state_draws,
        undecided_state_draws=np.zeros((500, 1)),
        national_summary=national_summary,
        state_summary=pd.DataFrame(),
        institute_reliability=pd.DataFrame(),
        residual_covariance=np.eye(2),
        residual_correlation=np.eye(2),
        diagnostics={},
    )


def _fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate_id": "A", "ml_probability": 1 / 3, "rejection": 35.0, "ideology_score": -0.5, "late_decider_score": 0.0},
            {"candidate_id": "B", "ml_probability": 1 / 3, "rejection": 35.0, "ideology_score": 0.0, "late_decider_score": 0.0},
            {"candidate_id": "C", "ml_probability": 1 / 3, "rejection": 35.0, "ideology_score": 0.5, "late_decider_score": 0.0},
        ]
    )


def test_agent_vote_shock_changes_draws_without_mutating_baseline():
    kwargs = dict(
        posterior=_posterior(),
        fundamentals=_fundamentals(),
        turnout_draws=np.full((1_000, 1), 0.8),
        registered_voters=np.array([1_000_000.0]),
        n_simulations=1_000,
        seed=123,
    )
    baseline = simulate_election(**kwargs)
    scenario = AgentScenario(
        event_id="debate",
        title="Debate",
        candidate_shocks=[
            CandidateShock(candidate_id="A", uf="CE", vote_shift_mean=5.0, vote_shift_sd=0.0, confidence=1.0)
        ],
    )
    hybrid = simulate_election(**kwargs, agent_scenario=scenario)

    base_share = baseline.candidates.set_index("candidate_id").loc["A", "expected_first_round_share"]
    hybrid_share = hybrid.candidates.set_index("candidate_id").loc["A", "expected_first_round_share"]
    assert hybrid_share > base_share + 2.0
    assert baseline.diagnostics["agent_layer_enabled"] is False
    assert hybrid.diagnostics["agent_layer_enabled"] is True
    assert hybrid.diagnostics["agent_scenario"]["event_id"] == "debate"
