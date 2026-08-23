from app.governance.publication_guard import assess_publication


def test_synthetic_2026_prediction_is_blocked():
    decision = assess_publication("synthetic", 2026, "unvalidated")
    assert decision.allowed is False
    assert decision.status == "BLOCKED_SYNTHETIC_DEMONSTRATION"
    assert "NÃO É PREVISÃO" in decision.watermark
