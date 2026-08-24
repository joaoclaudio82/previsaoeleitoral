from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    name: str
    required_columns: frozenset[str]

    def validate(self, frame: pd.DataFrame) -> list[str]:
        missing = sorted(self.required_columns.difference(frame.columns))
        return [f"{self.name}: missing column {column}" for column in missing]


SCHEMAS: dict[str, DatasetSchema] = {
    "polls": DatasetSchema(
        "polls",
        frozenset({"poll_id", "institute", "fieldwork_end", "sample_size", "candidate_id", "share"}),
    ),
    "turnout": DatasetSchema(
        "turnout",
        frozenset({"year", "uf", "eligible_voters", "votes_cast"}),
    ),
    "fundamentals": DatasetSchema(
        "fundamentals",
        frozenset({"reference_date", "release_date"}),
    ),
}


def validate_registered_schema(dataset_name: str, frame: pd.DataFrame) -> list[str]:
    try:
        schema = SCHEMAS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset schema: {dataset_name}") from exc
    return schema.validate(frame)
