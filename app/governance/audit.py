from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Hash-chained JSONL log for lightweight tamper evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return "0" * 64
        return json.loads(last)["record_hash"]

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "previous_hash": self._last_hash(),
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        record["record_hash"] = hashlib.sha256(canonical).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return record
