from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable
import httpx


CKAN_BASE = "https://dadosabertos.tse.jus.br/api/3/action"


def package_metadata(package_id: str) -> dict:
    response = httpx.get(f"{CKAN_BASE}/package_show", params={"id": package_id}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError("O CKAN do TSE retornou success=false")
    return payload["result"]


def list_resources(package_id: str, formats: Iterable[str] = ("CSV", "ZIP")) -> list[dict]:
    allowed = {fmt.upper() for fmt in formats}
    resources = package_metadata(package_id).get("resources", [])
    return [r for r in resources if str(r.get("format", "")).upper() in allowed]


def download_resource(url: str, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=180, follow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return destination


def extract_zip_bytes(content: bytes, destination: str | Path) -> list[Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(destination)
        return [destination / name for name in archive.namelist()]
