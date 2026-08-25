import pytest

from app.agents.schemas import AgentScenario, CandidateShock, StateShock
from app.services.agent_scenarios import validate_agent_scenario


def test_confidence_weighting_shrinks_means_but_preserves_uncertainty():
    scenario = AgentScenario(
        event_id="debate-1",
        title="Debate",
        candidate_shocks=[
            CandidateShock(
                candidate_id="A",
                uf="ce",
                vote_shift_mean=2.0,
                vote_shift_sd=0.8,
                confidence=0.5,
            )
        ],
        state_shocks=[
            StateShock(
                uf="CE",
                turnout_shift_mean=0.02,
                turnout_shift_sd=0.01,
                undecided_shift_mean=-2.0,
                undecided_shift_sd=0.5,
                confidence=0.5,
            )
        ],
    )

    weighted = scenario.confidence_weighted(0.4)

    assert weighted.candidate_shocks[0].vote_shift_mean == pytest.approx(0.4)
    assert weighted.candidate_shocks[0].vote_shift_sd == pytest.approx(0.8)
    assert weighted.state_shocks[0].turnout_shift_mean == pytest.approx(0.004)
    assert weighted.state_shocks[0].undecided_shift_mean == pytest.approx(-0.4)


def test_scenario_rejects_duplicate_candidate_effects():
    with pytest.raises(ValueError, match="duplicados"):
        AgentScenario(
            event_id="x",
            title="X",
            candidate_shocks=[
                CandidateShock(candidate_id="A", uf="CE", vote_shift_mean=1.0),
                CandidateShock(candidate_id="A", uf="CE", vote_shift_mean=0.5),
            ],
        )


def test_validate_agent_scenario_rejects_unknown_entities():
    scenario = AgentScenario(
        event_id="x",
        title="X",
        candidate_shocks=[CandidateShock(candidate_id="Z", uf="CE", vote_shift_mean=1.0)],
    )
    with pytest.raises(ValueError, match="Candidatos desconhecidos"):
        validate_agent_scenario(scenario, candidate_ids=["A", "B"], state_ids=["CE"])
