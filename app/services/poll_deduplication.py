from __future__ import annotations

import pandas as pd


def deduplicate_polls(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact/semantic duplicates and return rejected rows for audit."""
    if frame.empty:
        return frame.copy(), frame.copy()
    output = frame.copy()
    if "poll_id" not in output.columns or "candidate_id" not in output.columns:
        raise ValueError("poll_id and candidate_id are required for deduplication")

    semantic_columns = [
        column
        for column in (
            "institute",
            "fieldwork_start",
            "fieldwork_end",
            "sample_size",
            "uf",
            "mode",
            "target_population",
            "candidate_id",
            "share",
        )
        if column in output.columns
    ]
    exact_duplicate = output.duplicated(keep="first")
    key_duplicate = output.duplicated(["poll_id", "candidate_id"], keep="first")
    semantic_duplicate = output.duplicated(semantic_columns, keep="first") if semantic_columns else False
    rejected_mask = exact_duplicate | key_duplicate | semantic_duplicate
    rejected = output.loc[rejected_mask].copy()
    accepted = output.loc[~rejected_mask].copy()
    return accepted.reset_index(drop=True), rejected.reset_index(drop=True)
