from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.model import NUMERIC_FEATURES, WinnerModel

frame = pd.read_csv(ROOT / "data/processed/historical_training.csv")
groups = frame["election_id"]
oof = np.zeros(len(frame), dtype=float)

for train_idx, test_idx in GroupKFold(n_splits=5).split(frame, frame["won"], groups):
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx].copy()
    model = WinnerModel.build().fit(train)
    raw = model.pipeline.predict_proba(test[NUMERIC_FEATURES])[:, 1]
    test["raw"] = raw
    normalized = test.groupby("election_id")["raw"].transform(lambda x: x / x.sum())
    oof[test_idx] = normalized.to_numpy()

scored = frame[["election_id", "won", "poll_mean"]].copy()
scored["probability"] = oof
scored["poll_baseline"] = scored.groupby("election_id")["poll_mean"].transform(lambda x: x / x.sum())
model_brier = brier_score_loss(scored["won"], scored["probability"])
baseline_brier = brier_score_loss(scored["won"], scored["poll_baseline"])
model_logloss = log_loss(scored["won"], np.clip(scored["probability"], 1e-6, 1 - 1e-6))
predicted = scored.loc[scored.groupby("election_id")["probability"].idxmax()]
accuracy = predicted["won"].mean()

print("Avaliação v0.2 em dados SINTÉTICOS com GroupKFold por eleição")
print(f"Brier do modelo:      {model_brier:.4f}")
print(f"Brier baseline polls: {baseline_brier:.4f}")
print(f"Log loss binária:     {model_logloss:.4f}")
print(f"Acurácia do vencedor: {accuracy:.3%}")
print("Estes números não validam uso em uma eleição real.")
