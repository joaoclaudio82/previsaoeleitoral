from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CandidateShock(BaseModel):
    """Candidate-specific vote-share shock produced by a social simulation."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    uf: str = Field(min_length=2, max_length=2)
    vote_shift_mean: float = Field(ge=-10.0, le=10.0)
    vote_shift_sd: float = Field(default=0.0, ge=0.0, le=5.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("uf")
    @classmethod
    def normalize_uf(cls, value: str) -> str:
        return value.strip().upper()


class StateShock(BaseModel):
    """State-level turnout and undecided-voter shock."""

    model_config = ConfigDict(extra="forbid")

    uf: str = Field(min_length=2, max_length=2)
    turnout_shift_mean: float = Field(default=0.0, ge=-0.10, le=0.10)
    turnout_shift_sd: float = Field(default=0.0, ge=0.0, le=0.05)
    undecided_shift_mean: float = Field(default=0.0, ge=-10.0, le=10.0)
    undecided_shift_sd: float = Field(default=0.0, ge=0.0, le=5.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("uf")
    @classmethod
    def normalize_uf(cls, value: str) -> str:
        return value.strip().upper()


class AgentScenario(BaseModel):
    """Validated exchange contract between MiroFish and ElectionAI.

    Agent outputs are deliberately modeled as uncertain scenario shocks, never as
    direct forecasts. `experimental` is fixed to True so this layer cannot be
    confused with the empirically calibrated Bayesian baseline.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = ""
    source: Literal["mirofish", "manual", "historical_replay"] = "mirofish"
    experimental: Literal[True] = True
    simulation_runs: int = Field(default=1, ge=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidate_shocks: list[CandidateShock] = Field(default_factory=list)
    state_shocks: list[StateShock] = Field(default_factory=list)
    provenance: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_duplicate_effects(self) -> "AgentScenario":
        candidate_keys = [(item.uf, item.candidate_id) for item in self.candidate_shocks]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("candidate_shocks contém efeitos duplicados para UF/candidato")
        state_keys = [item.uf for item in self.state_shocks]
        if len(state_keys) != len(set(state_keys)):
            raise ValueError("state_shocks contém efeitos duplicados por UF")
        return self

    def confidence_weighted(self, strength: float = 1.0) -> "AgentScenario":
        """Return a conservatively shrunk copy of the scenario.

        Means are shrunk by confidence and an external strength parameter. Standard
        deviations are retained so uncertainty is not artificially reduced.
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength deve estar entre 0 e 1")
        candidate = [
            item.model_copy(
                update={"vote_shift_mean": item.vote_shift_mean * item.confidence * strength}
            )
            for item in self.candidate_shocks
        ]
        states = [
            item.model_copy(
                update={
                    "turnout_shift_mean": item.turnout_shift_mean * item.confidence * strength,
                    "undecided_shift_mean": item.undecided_shift_mean * item.confidence * strength,
                }
            )
            for item in self.state_shocks
        ]
        return self.model_copy(update={"candidate_shocks": candidate, "state_shocks": states})
