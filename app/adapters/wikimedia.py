from __future__ import annotations

import io
from urllib.parse import urlparse

import httpx
import pandas as pd


USER_AGENT = "ElectionAI-research/0.3 (academic historical election forecasting; contact via GitHub)"


def _rest_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc or "/wiki/" not in parsed.path:
        return None
    title = parsed.path.split("/wiki/", 1)[1]
    return f"https://{parsed.netloc}/api/rest_v1/page/html/{title}"


def fetch_html(url: str, timeout: float = 60.0) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    candidates = [url]
    rest = _rest_url(url)
    if rest:
        candidates.append(rest)
    errors: list[str] = []
    for candidate in candidates:
        try:
            response = httpx.get(candidate, follow_redirects=True, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            errors.append(f"{candidate}: {type(exc).__name__}")
    raise RuntimeError("Unable to fetch Wikimedia page: " + "; ".join(errors))


def fetch_tables(url: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(fetch_html(url)), flavor="lxml")
