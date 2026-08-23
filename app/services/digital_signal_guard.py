from __future__ import annotations

import numpy as np
import pandas as pd


def guard_digital_signals(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Shrink manipulable digital signals and report coverage/anomaly risks."""
    guarded = frame.copy()
    n = max(len(guarded), 1)
    neutral_search = 100.0 / n
    search_reliability = pd.to_numeric(guarded.get("search_reliability", 0.5), errors="coerce").fillna(0.0).clip(0, 1)
    sentiment_reliability = pd.to_numeric(guarded.get("sentiment_reliability", 0.5), errors="coerce").fillna(0.0).clip(0, 1)
    search_anomaly = pd.to_numeric(guarded.get("search_anomaly_score", 0.0), errors="coerce").fillna(3.0).clip(lower=0)
    sentiment_anomaly = pd.to_numeric(guarded.get("sentiment_anomaly_score", 0.0), errors="coerce").fillna(3.0).clip(lower=0)
    search_weight = search_reliability * np.exp(-search_anomaly)
    sentiment_weight = sentiment_reliability * np.exp(-sentiment_anomaly)
    guarded["search_share_guarded"] = neutral_search + search_weight * (
        pd.to_numeric(guarded["search_share"], errors="coerce").fillna(neutral_search) - neutral_search
    )
    guarded["media_sentiment_guarded"] = sentiment_weight * pd.to_numeric(
        guarded["media_sentiment"], errors="coerce"
    ).fillna(0.0).clip(-1, 1)
    guarded["digital_signal_reliability"] = (search_weight + sentiment_weight) / 2

    warnings: list[str] = []
    if float(search_weight.mean()) < 0.35:
        warnings.append("O sinal de buscas recebeu forte regularização por baixa confiabilidade ou anomalia.")
    if float(sentiment_weight.mean()) < 0.35:
        warnings.append("O sentimento recebeu forte regularização por cobertura insuficiente ou anomalia.")
    flagged = guarded.loc[(search_anomaly > 1.5) | (sentiment_anomaly > 1.5), "candidate_id"].astype(str).tolist()
    if flagged:
        warnings.append(f"Sinais digitais potencialmente anômalos para: {', '.join(flagged)}.")
    diagnostics = {
        "mean_search_weight": float(search_weight.mean()),
        "mean_sentiment_weight": float(sentiment_weight.mean()),
        "flagged_candidates": flagged,
        "raw_signals_used_directly": False,
    }
    return guarded, warnings, diagnostics
