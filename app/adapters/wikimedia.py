from __future__ import annotations

import io
from urllib.parse import unquote, urlparse

import httpx
import pandas as pd


USER_AGENT = "ElectionAI-research/0.3 (academic historical election forecasting; contact via GitHub)"


def _wiki_parts(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc or "/wiki/" not in parsed.path:
        return None
    return parsed.netloc, unquote(parsed.path.split("/wiki/", 1)[1]).replace("_", " ")


def fetch_html(url: str, timeout: float = 60.0) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/json"}
    errors: list[str] = []
    try:
        response = httpx.get(url, follow_redirects=True, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        errors.append(f"page:{type(exc).__name__}")

    parts = _wiki_parts(url)
    if parts:
        host, title = parts
        rest_url = f"https://{host}/api/rest_v1/page/html/{title.replace(' ', '_')}"
        try:
            response = httpx.get(rest_url, follow_redirects=True, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            errors.append(f"rest:{type(exc).__name__}")

        api_url = f"https://{host}/w/api.php"
        try:
            response = httpx.get(api_url, params={"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2, "origin": "*"}, follow_redirects=True, timeout=timeout, headers=headers)
            response.raise_for_status()
            payload = response.json()
            html = payload.get("parse", {}).get("text")
            if html:
                return str(html)
            errors.append("api:missing_parse_text")
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"api:{type(exc).__name__}")

    raise RuntimeError("Unable to fetch Wikimedia page: " + "; ".join(errors))


def fetch_tables(url: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(fetch_html(url)), flavor="lxml")
