from app.governance.release_gate import evaluate_release


def test_synthetic_dataset_is_blocked() -> None:
    decision = evaluate_release(
        dataset_type="synthetic",
        validation_status="validated",
        backtest_elections=4,
        brier_score=0.18,
        calibration_error=0.05,
    )
    assert not decision.allowed
    assert any("not real" in reason for reason in decision.reasons)


def test_validated_real_forecast_can_pass_thresholds() -> None:
    decision = evaluate_release(
        dataset_type="real",
        validation_status="validated",
        backtest_elections=4,
        brier_score=0.18,
        calibration_error=0.05,
    )
    assert decision.allowed
    assert decision.reasons == []


def test_bad_calibration_blocks_release() -> None:
    decision = evaluate_release(
        dataset_type="real",
        validation_status="validated",
        backtest_elections=4,
        brier_score=0.18,
        calibration_error=0.20,
    )
    assert not decision.allowed
    assert any("calibration" in reason.lower() for reason in decision.reasons)
