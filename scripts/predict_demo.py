from __future__ import annotations

from datetime import date
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.governance.publication_guard import assess_publication
from app.services.predictor import predict

bundle = predict(
    polls=pd.read_csv(ROOT / "data/raw/current_polls.csv"),
    fundamentals=pd.read_csv(ROOT / "data/raw/current_fundamentals.csv"),
    state_priors=pd.read_csv(ROOT / "data/raw/state_priors.csv"),
    turnout=pd.read_csv(ROOT / "data/raw/current_turnout.csv"),
    as_of_date=date(2026, 8, 1),
    model_path=ROOT / settings.model_path,
    pollster_calibration_path=ROOT / settings.pollster_calibration_path,
    turnout_model_path=ROOT / settings.turnout_model_path,
    transfer_model_path=ROOT / settings.transfer_model_path,
    n_simulations=20_000,
    posterior_draws=4_000,
    seed=42,
)
publication = assess_publication("synthetic", 2026, "unvalidated")
print(publication.watermark)
print(f"Modelo {bundle.model_version} | status: {publication.status}")
print(bundle.candidates.to_string(index=False))
print("\nDiagnósticos:")
print(bundle.diagnostics)
