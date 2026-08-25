import pandas as pd

from app.ml.robustness import (
    cluster_bootstrap_difference,
    paired_difference_by_cluster,
    select_strength_leave_one_election_out,
)


def test_cluster_bootstrap_preserves_direction_of_paired_improvement():
    frame = pd.DataFrame(
        {
            "election_year": [2014, 2014, 2018, 2018, 2022, 2022],
            "model": ["historical", "neutral"] * 3,
            "vote_share_mae": [0.06, 0.09, 0.05, 0.08, 0.07, 0.10],
        }
    )
    paired = paired_difference_by_cluster(
        frame,
        cluster="election_year",
        treatment="historical",
        control="neutral",
        value="vote_share_mae",
    )
    result = cluster_bootstrap_difference(paired, draws=2000, seed=7)
    assert result.mean_difference < 0
    assert result.upper < 0
    assert result.probability_improvement == 1.0


def test_leave_one_election_out_strength_uses_training_elections_only():
    sensitivity = pd.DataFrame(
        {
            "election_year": [2014, 2014, 2018, 2018, 2022, 2022],
            "prior_strength": [1.0, 2.5] * 3,
            "vote_share_mae": [0.08, 0.06, 0.09, 0.07, 0.03, 0.20],
        }
    )
    strength, summary = select_strength_leave_one_election_out(
        sensitivity, holdout_year=2022
    )
    assert strength == 2.5
    assert set(summary["prior_strength"]) == {1.0, 2.5}
