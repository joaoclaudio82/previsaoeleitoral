from __future__ import annotations

import json
from pathlib import Path

from app.adapters.mirofish import parse_agent_scenario
from app.agents.schemas import AgentScenario


def load_agent_scenario(path: str | Path) -> AgentScenario:
    target = Path(path)
    return parse_agent_scenario(target.read_text(encoding="utf-8"))


def save_agent_scenario(scenario: AgentScenario, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def validate_agent_scenario(
    scenario: AgentScenario,
    *,
    candidate_ids: list[str],
    state_ids: list[str],
) -> None:
    candidates = {str(value) for value in candidate_ids}
    states = {str(value).upper() for value in state_ids}

    unknown_candidates = sorted(
        {shock.candidate_id for shock in scenario.candidate_shocks if shock.candidate_id not in candidates}
    )
    unknown_states = sorted(
        {
            shock.uf
            for shock in [*scenario.candidate_shocks, *scenario.state_shocks]
            if shock.uf not in states
        }
    )
    if unknown_candidates:
        raise ValueError(f"Candidatos desconhecidos no cenário multiagente: {unknown_candidates}")
    if unknown_states:
        raise ValueError(f"UFs desconhecidas no cenário multiagente: {unknown_states}")


def scenario_diagnostics(scenario: AgentScenario) -> dict:
    candidate_abs = [abs(shock.vote_shift_mean) for shock in scenario.candidate_shocks]
    turnout_abs = [abs(shock.turnout_shift_mean) for shock in scenario.state_shocks]
    undecided_abs = [abs(shock.undecided_shift_mean) for shock in scenario.state_shocks]
    return {
        "event_id": scenario.event_id,
        "source": scenario.source,
        "experimental": scenario.experimental,
        "simulation_runs": scenario.simulation_runs,
        "candidate_shock_count": len(scenario.candidate_shocks),
        "state_shock_count": len(scenario.state_shocks),
        "max_abs_vote_shift_mean": max(candidate_abs, default=0.0),
        "max_abs_turnout_shift_mean": max(turnout_abs, default=0.0),
        "max_abs_undecided_shift_mean": max(undecided_abs, default=0.0),
    }
