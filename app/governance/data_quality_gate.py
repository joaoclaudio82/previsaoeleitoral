from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.data.schema_registry import validate_registered_schema
from app.data.validation import validate_poll_frame


@dataclass(frozen=True, slots=True)
class DataQualityDecision:
    allowed: bool
    errors: tuple[str, ...]


def evaluate_dataset(dataset_name: str, frame: pd.DataFrame) -> DataQualityDecision:
    errors = list(validate_registered_schema(dataset_name, frame))
    if dataset_name == "polls" and not errors:
        errors.extend(validate_poll_frame(frame).errors)
    if frame.empty:
        errors.append(f"{dataset_name}: dataset is empty")
    return DataQualityDecision(allowed=not errors, errors=tuple(errors))


def require_dataset(dataset_name: str, frame: pd.DataFrame) -> None:
    decision = evaluate_dataset(dataset_name, frame)
    if not decision.allowed:
        raise ValueError("; ".join(decision.errors))
