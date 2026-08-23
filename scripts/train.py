from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.model import WinnerModel
from app.ml.pollster import PollsterCalibration
from app.ml.transfer import TransferModel
from app.ml.turnout import TurnoutNowcaster

models = ROOT / "models"
models.mkdir(parents=True, exist_ok=True)

winner = WinnerModel.build().fit(pd.read_csv(ROOT / "data/processed/historical_training.csv"))
winner.save(models / "winner_model.joblib")

pollster = PollsterCalibration.fit(pd.read_csv(ROOT / "data/processed/historical_poll_calibration.csv"))
pollster.save(models / "pollster_calibration.joblib")

turnout = TurnoutNowcaster.fit(pd.read_csv(ROOT / "data/processed/historical_turnout.csv"))
turnout.save(models / "turnout_model.joblib")

transfer = TransferModel.fit(pd.read_csv(ROOT / "data/processed/runoff_transfer_training.csv"))
transfer.save(models / "transfer_model.joblib")

print("Modelos v0.2 salvos em models/.")
