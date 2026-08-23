from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "poll_mean", "poll_trend_14d", "rejection", "incumbency",
    "government_approval", "inflation_12m", "unemployment", "gdp_yoy",
    "search_share_guarded", "media_sentiment_guarded", "digital_signal_reliability",
]
TARGET = "won"


@dataclass
class WinnerModel:
    pipeline: Pipeline
    version: str = "0.2.0"

    @classmethod
    def build(cls) -> "WinnerModel":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3_000, class_weight="balanced", C=0.65)),
        ])
        return cls(pipeline)

    def fit(self, frame: pd.DataFrame) -> "WinnerModel":
        required = set(NUMERIC_FEATURES + [TARGET])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no treino: {sorted(missing)}")
        self.pipeline.fit(frame[NUMERIC_FEATURES], frame[TARGET])
        return self

    def predict_normalized(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(frame[NUMERIC_FEATURES])[:, 1]
        total = raw.sum()
        if total <= 0:
            return np.full(len(raw), 1.0 / len(raw))
        return raw / total

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "WinnerModel":
        return joblib.load(path)
