import numpy as np
import pandas as pd

from app.ml.transfer import TransferModel


def test_transfer_model_prefers_closer_finalist():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(250):
        advantage = rng.uniform(-1.5, 1.5)
        p = 1 / (1 + np.exp(-2.2 * advantage))
        total = 500
        to_a = rng.binomial(total - 50, p)
        rows.append({
            "distance_advantage_a": advantage, "rejection_advantage_a": 0,
            "same_bloc_advantage_a": 0, "incumbency_advantage_a": 0,
            "source_rejection": 40, "finalist_distance_sum": 1,
            "to_a": to_a, "to_b": total - 50 - to_a, "abstain": 50,
        })
    model = TransferModel.fit(pd.DataFrame(rows))
    fundamentals = pd.DataFrame([
        {"candidate_id": "S", "ideology_score": -0.8, "bloc": "x", "rejection": 40, "incumbency": 0},
        {"candidate_id": "A", "ideology_score": -0.7, "bloc": "x", "rejection": 35, "incumbency": 0},
        {"candidate_id": "B", "ideology_score": 0.8, "bloc": "y", "rejection": 35, "incumbency": 0},
    ])
    draws = model.precompute_pair_draws(fundamentals, 1000, 4)
    p_a, _ = draws[("S", "A", "B")]
    assert p_a.mean() > 0.7
