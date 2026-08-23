from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class PollObservation(BaseModel):
    poll_id: str
    field_date: date
    institute: str
    collection_mode: str = "unknown"
    target_population: str = "registered_voters"
    candidate_id: str
    candidate_name: str
    share: float = Field(ge=0, le=100)
    undecided_share: float = Field(default=0, ge=0, le=100)
    sample_size: int = Field(gt=0)
    margin_error: float = Field(gt=0, le=20)
    scope: Literal["national", "state"] = "national"
    uf: str | None = None
    institute_quality: float | None = Field(default=None, description="Deprecated and ignored in v0.2")


class CandidateFundamentals(BaseModel):
    candidate_id: str
    candidate_name: str
    rejection: float = Field(ge=0, le=100)
    incumbency: int = Field(ge=0, le=1)
    government_approval: float = Field(ge=0, le=100)
    inflation_12m: float
    unemployment: float = Field(ge=0, le=100)
    gdp_yoy: float
    search_share: float = Field(ge=0, le=100)
    media_sentiment: float = Field(ge=-1, le=1)
    search_reliability: float = Field(default=0.5, ge=0, le=1)
    sentiment_reliability: float = Field(default=0.5, ge=0, le=1)
    search_anomaly_score: float = Field(default=0, ge=0)
    sentiment_anomaly_score: float = Field(default=0, ge=0)
    ideology_score: float = Field(default=0, ge=-1, le=1)
    bloc: str = "independent"
    late_decider_score: float = Field(default=0, ge=-2, le=2)


class StatePriorObservation(BaseModel):
    uf: str
    region: str = "unknown"
    registered_voters: int = Field(gt=0)
    candidate_id: str
    candidate_name: str
    prior_share: float = Field(ge=0, le=100)
    prior_strength: float = Field(default=3.0, gt=0, le=50)


class StateTurnoutObservation(BaseModel):
    uf: str
    registered_voters: int = Field(gt=0)
    historical_turnout: float = Field(ge=0.3, le=0.99)
    abstention_trend: float = Field(default=0, ge=-0.2, le=0.2)
    registration_growth: float = Field(default=0, ge=-0.2, le=0.2)
    mobility_index: float = Field(default=0, ge=-3, le=3)
    weather_severity: float = Field(default=0, ge=0, le=3)
    competitiveness: float = Field(default=0.5, ge=0, le=1)


class PredictionRequest(BaseModel):
    as_of_date: date
    election_id: str = "SYNTHETIC-LAB"
    election_year: int = Field(default=2026, ge=1900, le=2200)
    dataset_type: Literal["synthetic", "historical", "operational"] = "synthetic"
    validation_status: Literal["unvalidated", "backtested", "independently_validated"] = "unvalidated"
    n_simulations: int = Field(default=50_000, ge=1_000, le=1_000_000)
    polls: list[PollObservation]
    fundamentals: list[CandidateFundamentals]
    state_priors: list[StatePriorObservation] = []
    turnout: list[StateTurnoutObservation] = []

    @model_validator(mode="after")
    def validate_candidates(self):
        poll_ids = {p.candidate_id for p in self.polls}
        fundamental_ids = {f.candidate_id for f in self.fundamentals}
        missing = poll_ids - fundamental_ids
        if missing:
            raise ValueError(f"Fundamentos ausentes para: {sorted(missing)}")
        if self.state_priors:
            prior_ids = {p.candidate_id for p in self.state_priors}
            if not poll_ids.issubset(prior_ids):
                raise ValueError("Os priors estaduais devem conter todos os candidatos pesquisados.")
        return self


class CandidatePrediction(BaseModel):
    candidate_id: str
    candidate_name: str
    poll_average: float
    poll_lower: float
    poll_upper: float
    poll_uncertainty: float
    ml_probability: float
    first_round_lead_probability: float
    win_probability: float
    expected_first_round_share: float
    expected_first_round_share_low: float
    expected_first_round_share_high: float


class StatePrediction(BaseModel):
    uf: str
    expected_turnout: float
    turnout_low: float
    turnout_high: float
    leading_candidate: str
    leader_probability: float
    expected_shares: dict[str, float]


class PredictionResponse(BaseModel):
    as_of_date: date
    election_id: str
    model_version: str
    simulations: int
    likely_winner: str | None
    publication_status: str
    watermark: str | None
    run_id: str
    data_versions: dict[str, str]
    candidates: list[CandidatePrediction]
    states: list[StatePrediction]
    institute_reliability: list[dict]
    diagnostics: dict
    warnings: list[str]
