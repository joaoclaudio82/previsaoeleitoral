import numpy as np
import pandas as pd

from app.ml.turnout import TurnoutNowcaster


def test_turnout_nowcast_draws_have_valid_range():
    rows = []
    for i in range(30):
        rows.append({
            "uf": "SP" if i % 2 == 0 else "CE", "historical_turnout": 0.76 + (i % 3) * 0.01,
            "abstention_trend": 0.002, "registration_growth": 0.01,
            "mobility_index": 0.1, "weather_severity": 0.2,
            "competitiveness": 0.7, "turnout": 0.77 + (i % 4) * 0.005,
        })
    model = TurnoutNowcaster.fit(pd.DataFrame(rows))
    current = pd.DataFrame([
        {"uf": "SP", "historical_turnout": 0.78, "abstention_trend": 0, "registration_growth": 0.01, "mobility_index": 0, "weather_severity": 0.1, "competitiveness": 0.8},
        {"uf": "CE", "historical_turnout": 0.77, "abstention_trend": 0, "registration_growth": 0.01, "mobility_index": 0, "weather_severity": 0.1, "competitiveness": 0.8},
    ])
    draws = model.predict_draws(current, 500, 42)
    assert draws.shape == (500, 2)
    assert np.all((draws >= 0.35) & (draws <= 0.95))
