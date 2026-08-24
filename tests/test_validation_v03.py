import pandas as pd

from app.data.fingerprints import dataframe_sha256
from app.governance.data_quality_gate import evaluate_dataset


def valid_polls() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "poll_id": "p1",
                "institute": "Institute A",
                "fieldwork_end": "2026-08-01",
                "sample_size": 2000,
                "candidate_id": "a",
                "share": 45.0,
            },
            {
                "poll_id": "p1",
                "institute": "Institute A",
                "fieldwork_end": "2026-08-01",
                "sample_size": 2000,
                "candidate_id": "b",
                "share": 40.0,
            },
        ]
    )


def test_valid_poll_dataset_passes_gate() -> None:
    decision = evaluate_dataset("polls", valid_polls())
    assert decision.allowed
    assert decision.errors == ()


def test_duplicate_candidate_in_poll_is_rejected() -> None:
    frame = pd.concat([valid_polls(), valid_polls().iloc[[0]]], ignore_index=True)
    decision = evaluate_dataset("polls", frame)
    assert not decision.allowed
    assert any("duplicate" in error for error in decision.errors)


def test_dataframe_hash_is_row_order_invariant() -> None:
    frame = valid_polls()
    assert dataframe_sha256(frame) == dataframe_sha256(frame.iloc[::-1].reset_index(drop=True))
