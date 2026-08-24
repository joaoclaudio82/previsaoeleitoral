from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ReleaseDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_release(
    *,
    dataset_type: str,
    validation_status: str,
    backtest_elections: int,
    brier_score: float | None,
    calibration_error: float | None,
    max_brier: float = 0.25,
    max_calibration_error: float = 0.10,
    min_backtest_elections: int = 2,
) -> ReleaseDecision:
    reasons: list[str] = []
    if dataset_type.lower() != "real":
        reasons.append("publication blocked: dataset is not real")
    if validation_status.lower() not in {"validated", "approved"}:
        reasons.append("publication blocked: dataset/model validation is incomplete")
    if backtest_elections < min_backtest_elections:
        reasons.append("publication blocked: insufficient historical elections in backtest")
    if brier_score is None or brier_score > max_brier:
        reasons.append("publication blocked: Brier score does not meet release threshold")
    if calibration_error is None or calibration_error > max_calibration_error:
        reasons.append("publication blocked: calibration error does not meet release threshold")
    return ReleaseDecision(allowed=not reasons, reasons=reasons)
