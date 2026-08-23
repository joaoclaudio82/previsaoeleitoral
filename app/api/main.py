from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.core.config import settings
from app.data.versioning import DataVersionStore
from app.domain.schemas import CandidatePrediction, PredictionRequest, PredictionResponse, StatePrediction
from app.governance.publication_guard import assess_publication
from app.services.predictor import predict

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "API probabilística com agregação Bayesiana hierárquica, efeitos estaduais, "
        "nowcasting de comparecimento, transferência aprendida e linhagem de dados."
    ),
)


def _store() -> DataVersionStore:
    return DataVersionStore(settings.data_registry_path, settings.snapshots_path)


@app.get("/health")
def health() -> dict:
    models = {
        "winner": settings.model_path.exists(),
        "pollster_calibration": settings.pollster_calibration_path.exists(),
        "turnout": settings.turnout_model_path.exists(),
        "transfer": settings.transfer_model_path.exists(),
    }
    return {"status": "ok", "version": "0.2.0", "models": models, "all_models_ready": all(models.values())}


@app.get("/lineage/datasets")
def list_dataset_versions(dataset_name: str | None = None) -> list[dict]:
    return _store().list_versions(dataset_name)


@app.get("/lineage/runs")
def list_prediction_runs(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    return _store().list_prediction_runs(limit)


@app.get("/lineage/runs/{run_id}")
def get_prediction_run(run_id: str) -> dict:
    record = _store().get_prediction_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Execução não encontrada.")
    return record


@app.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        polls = pd.DataFrame([item.model_dump() for item in request.polls])
        fundamentals = pd.DataFrame([item.model_dump() for item in request.fundamentals])
        state_priors = pd.DataFrame([item.model_dump() for item in request.state_priors])
        turnout = pd.DataFrame([item.model_dump() for item in request.turnout])
        store = _store()
        is_synthetic = request.dataset_type == "synthetic"
        versions = {}
        for name, frame in {
            "polls": polls,
            "fundamentals": fundamentals,
            "state_priors": state_priors,
            "turnout": turnout,
        }.items():
            if frame.empty:
                continue
            registered = store.register_dataframe(
                name,
                frame,
                as_of_date=request.as_of_date,
                source_uri="api://inline-payload",
                is_synthetic=is_synthetic,
                metadata={"election_id": request.election_id, "election_year": request.election_year},
            )
            versions[name] = registered.version

        bundle = predict(
            polls=polls,
            fundamentals=fundamentals,
            state_priors=state_priors,
            turnout=turnout,
            as_of_date=request.as_of_date,
            model_path=settings.model_path,
            pollster_calibration_path=settings.pollster_calibration_path,
            turnout_model_path=settings.turnout_model_path,
            transfer_model_path=settings.transfer_model_path,
            n_simulations=request.n_simulations,
            posterior_draws=settings.posterior_draws,
            seed=settings.random_seed,
        )
        publication = assess_publication(
            request.dataset_type,
            request.election_year,
            request.validation_status,
        )
    except (ValueError, FileNotFoundError, np.linalg.LinAlgError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    candidates = [
        CandidatePrediction(
            candidate_id=row.candidate_id,
            candidate_name=row.candidate_name,
            poll_average=round(float(row.poll_mean), 3),
            poll_lower=round(float(row.poll_lower), 3),
            poll_upper=round(float(row.poll_upper), 3),
            poll_uncertainty=round(float(row.poll_uncertainty), 3),
            ml_probability=round(float(row.ml_probability), 6),
            first_round_lead_probability=round(float(row.first_round_lead_probability), 6),
            win_probability=round(float(row.win_probability), 6),
            expected_first_round_share=round(float(row.expected_first_round_share), 3),
            expected_first_round_share_low=round(float(row.expected_first_round_share_low), 3),
            expected_first_round_share_high=round(float(row.expected_first_round_share_high), 3),
        )
        for row in bundle.candidates.itertuples(index=False)
    ]
    states = [
        StatePrediction(
            uf=row.uf,
            expected_turnout=round(float(row.expected_turnout), 6),
            turnout_low=round(float(row.turnout_low), 6),
            turnout_high=round(float(row.turnout_high), 6),
            leading_candidate=row.leading_candidate,
            leader_probability=round(float(row.leader_probability), 6),
            expected_shares={key: round(float(value), 3) for key, value in row.expected_shares.items()},
        )
        for row in bundle.states.itertuples(index=False)
    ]
    output_payload = {
        "candidates": [candidate.model_dump() for candidate in candidates],
        "states": [state.model_dump() for state in states],
        "diagnostics": bundle.diagnostics,
    }
    run_id = store.record_prediction(
        election_id=request.election_id,
        election_year=request.election_year,
        as_of_date=request.as_of_date,
        model_version=bundle.model_version,
        input_versions=versions,
        output_payload=output_payload,
        dataset_type=request.dataset_type,
        publication_status=publication.status,
    )
    warnings = [
        *bundle.warnings,
        publication.reason,
        "Probabilidades são distribuições de incerteza, não certezas ou recomendações políticas.",
        "Pesquisas registradas não equivalem automaticamente a microdados públicos de intenção de voto.",
    ]
    return PredictionResponse(
        as_of_date=request.as_of_date,
        election_id=request.election_id,
        model_version=bundle.model_version,
        simulations=request.n_simulations,
        likely_winner=candidates[0].candidate_name if publication.allowed else None,
        publication_status=publication.status,
        watermark=publication.watermark,
        run_id=run_id,
        data_versions=versions,
        candidates=candidates,
        states=states,
        institute_reliability=bundle.institute_reliability.round(6).to_dict(orient="records"),
        diagnostics=bundle.diagnostics,
        warnings=warnings,
    )
