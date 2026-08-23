from datetime import date
from pathlib import Path
import pandas as pd

from app.services.predictor import predict

ROOT = Path(__file__).resolve().parents[1]


def test_full_v02_prediction_pipeline():
    bundle = predict(
        polls=pd.read_csv(ROOT / "data/raw/current_polls.csv"),
        fundamentals=pd.read_csv(ROOT / "data/raw/current_fundamentals.csv"),
        state_priors=pd.read_csv(ROOT / "data/raw/state_priors.csv"),
        turnout=pd.read_csv(ROOT / "data/raw/current_turnout.csv"),
        as_of_date=date(2026, 8, 1),
        model_path=ROOT / "models/winner_model.joblib",
        pollster_calibration_path=ROOT / "models/pollster_calibration.joblib",
        turnout_model_path=ROOT / "models/turnout_model.joblib",
        transfer_model_path=ROOT / "models/transfer_model.joblib",
        n_simulations=1200,
        posterior_draws=1000,
        seed=42,
    )
    assert abs(bundle.candidates["win_probability"].sum() - 1.0) < 1e-9
    assert len(bundle.states) == 27
    assert bundle.diagnostics["transfer_matrix_supplied"] is False
    assert bundle.diagnostics["turnout_mode"] == "bayesian_state_nowcast"
    assert bundle.diagnostics["correlated_institute_error"] is True
