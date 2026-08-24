from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.adapters.tse_ckan import list_resources
from app.data.historical_manifest import get_election


@dataclass(frozen=True, slots=True)
class DownloadedResource:
    path: Path
    source_url: str
    sha256: str
    resource_name: str


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def choose_resource(resources: list[dict], *terms: str) -> dict:
    folded_terms = [_fold(term) for term in terms]
    candidates: list[dict] = []
    for resource in resources:
        haystack = _fold(f"{resource.get('name', '')} {resource.get('description', '')}")
        if all(term in haystack for term in folded_terms):
            candidates.append(resource)
    if not candidates:
        raise LookupError(f"No TSE resource matched terms={terms!r}")
    candidates.sort(key=lambda item: (str(item.get("format", "")).upper() not in {"CSV", "ZIP"}, item.get("name", "")))
    return candidates[0]


def discover_presidential_results(year: int) -> dict:
    election = get_election(year)
    resources = list_resources(election.tse_results_package, formats=("CSV", "ZIP", "TXT"))
    for terms in (
        ("presidente", "votação por seção"),
        ("votação nominal", "município", "zona"),
        ("votacao nominal", "municipio", "zona"),
    ):
        try:
            return choose_resource(resources, *terms)
        except LookupError:
            continue
    raise LookupError(f"Could not locate presidential result resource for {year}")


def discover_turnout(year: int) -> dict:
    election = get_election(year)
    if election.tse_turnout_package is None:
        raise LookupError(f"No turnout package configured for {year}")
    resources = list_resources(election.tse_turnout_package, formats=("CSV", "ZIP"))
    return choose_resource(resources, "comparecimento", "abstenção")


def download_verified(resource: dict, destination_dir: str | Path, timeout: float = 300.0) -> DownloadedResource:
    url = str(resource["url"])
    name = str(resource.get("name") or "resource")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "resource"
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?", 1)[0]).suffix or ".bin"
    target = destination / f"{safe_name}{suffix}"
    digest = hashlib.sha256()
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                handle.write(chunk)
    return DownloadedResource(target, url, digest.hexdigest(), name)


def extract_if_archive(resource: DownloadedResource, destination_dir: str | Path) -> list[Path]:
    raw = resource.path.read_bytes()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        return [resource.path]
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        archive.extractall(destination)
        return [destination / member for member in archive.namelist() if not member.endswith("/")]
