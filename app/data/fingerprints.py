from __future__ import annotations

import hashlib

import pandas as pd


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Return a stable digest independent of row and column ordering."""
    if frame.empty:
        return hashlib.sha256(b"empty-dataframe").hexdigest()
    ordered_columns = sorted(map(str, frame.columns))
    normalized = frame.copy()
    normalized.columns = normalized.columns.map(str)
    normalized = normalized[ordered_columns]
    normalized = normalized.sort_values(ordered_columns, kind="mergesort", na_position="first")
    payload = normalized.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
