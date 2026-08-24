from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_version: str
    created_at: str
    python_version: str
    artifacts: dict[str, str]
    metadata: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    model_version: str,
    artifacts: dict[str, str | Path],
    metadata: dict[str, str] | None = None,
) -> ModelManifest:
    hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = file_sha256(path)
    return ModelManifest(
        model_version=model_version,
        created_at=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        artifacts=hashes,
        metadata=metadata or {},
    )
