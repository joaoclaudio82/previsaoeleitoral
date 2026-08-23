from __future__ import annotations

from datetime import date
import pandas as pd
import httpx


BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"


def fetch_series(series_code: int, start: date, end: date) -> pd.DataFrame:
    url = BASE_URL.format(series_code=series_code)
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    response = httpx.get(url, params=params, timeout=60)
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["data"], format="%d/%m/%Y")
    frame["value"] = pd.to_numeric(frame["valor"].str.replace(",", "."), errors="coerce")
    return frame[["date", "value"]]
