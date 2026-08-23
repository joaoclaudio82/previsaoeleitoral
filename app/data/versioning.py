from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import pandas as pd


@dataclass(frozen=True)
class DatasetVersion:
    dataset_name: str
    version: str
    sha256: str
    row_count: int
    snapshot_path: str
    is_synthetic: bool


class DataVersionStore:
    """Immutable dataset snapshots and prediction lineage backed by SQLite."""

    def __init__(self, db_path: str | Path, snapshots_path: str | Path):
        self.db_path = Path(db_path)
        self.snapshots_path = Path(snapshots_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    schema_json TEXT NOT NULL,
                    source_uri TEXT,
                    as_of_date TEXT,
                    is_synthetic INTEGER NOT NULL,
                    parent_version TEXT,
                    snapshot_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(dataset_name, version),
                    UNIQUE(dataset_name, sha256)
                );
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    run_id TEXT PRIMARY KEY,
                    election_id TEXT NOT NULL,
                    election_year INTEGER NOT NULL,
                    as_of_date TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_versions_json TEXT NOT NULL,
                    output_sha256 TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    publication_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _canonical_csv(frame: pd.DataFrame) -> bytes:
        canonical = frame.copy()
        canonical = canonical.reindex(sorted(canonical.columns), axis=1)
        if len(canonical.columns):
            sort_columns = list(canonical.columns)
            canonical = canonical.sort_values(sort_columns, kind="mergesort", na_position="last")
        return canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")

    def register_dataframe(
        self,
        dataset_name: str,
        frame: pd.DataFrame,
        *,
        as_of_date: date | str | None,
        source_uri: str | None = None,
        is_synthetic: bool = False,
        parent_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        content = self._canonical_csv(frame)
        digest = sha256(content).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_name = ? AND sha256 = ?",
                (dataset_name, digest),
            ).fetchone()
            if existing:
                return DatasetVersion(
                    dataset_name=existing["dataset_name"],
                    version=existing["version"],
                    sha256=existing["sha256"],
                    row_count=existing["row_count"],
                    snapshot_path=existing["snapshot_path"],
                    is_synthetic=bool(existing["is_synthetic"]),
                )

            count = conn.execute(
                "SELECT COUNT(*) AS n FROM dataset_versions WHERE dataset_name = ?",
                (dataset_name,),
            ).fetchone()["n"]
            version = f"v{count + 1:06d}"
            folder = self.snapshots_path / dataset_name
            folder.mkdir(parents=True, exist_ok=True)
            snapshot = folder / f"{version}-{digest[:12]}.csv"
            snapshot.write_bytes(content)
            schema = {column: str(dtype) for column, dtype in frame.dtypes.items()}
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO dataset_versions (
                    id, dataset_name, version, sha256, row_count, schema_json,
                    source_uri, as_of_date, is_synthetic, parent_version,
                    snapshot_path, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()), dataset_name, version, digest, len(frame),
                    json.dumps(schema, sort_keys=True), source_uri,
                    str(as_of_date) if as_of_date else None, int(is_synthetic),
                    parent_version, str(snapshot), json.dumps(metadata or {}, sort_keys=True), now,
                ),
            )
        return DatasetVersion(dataset_name, version, digest, len(frame), str(snapshot), is_synthetic)

    def record_prediction(
        self,
        *,
        election_id: str,
        election_year: int,
        as_of_date: date,
        model_version: str,
        input_versions: dict[str, str],
        output_payload: dict[str, Any],
        dataset_type: str,
        publication_status: str,
    ) -> str:
        run_id = str(uuid4())
        output_blob = json.dumps(output_payload, sort_keys=True, default=str).encode("utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_runs (
                    run_id, election_id, election_year, as_of_date, model_version,
                    input_versions_json, output_sha256, dataset_type,
                    publication_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, election_id, election_year, str(as_of_date), model_version,
                    json.dumps(input_versions, sort_keys=True), sha256(output_blob).hexdigest(),
                    dataset_type, publication_status, datetime.now(timezone.utc).isoformat(),
                ),
            )
        return run_id


    def list_prediction_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prediction_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_prediction_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prediction_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_versions(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM dataset_versions"
        params: tuple[Any, ...] = ()
        if dataset_name:
            query += " WHERE dataset_name = ?"
            params = (dataset_name,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
