from __future__ import annotations

import pandas as pd

UF_REGION = {
    "AC": "N", "AP": "N", "AM": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}


def add_macroregion(frame: pd.DataFrame, uf_column: str = "uf") -> pd.DataFrame:
    if uf_column not in frame.columns:
        raise ValueError(f"Missing UF column: {uf_column}")
    output = frame.copy()
    output["macroregion"] = output[uf_column].map(UF_REGION)
    if output["macroregion"].isna().any():
        invalid = sorted(output.loc[output["macroregion"].isna(), uf_column].astype(str).unique())
        raise ValueError(f"Unknown UF values: {invalid}")
    return output


def shrink_state_estimates(
    frame: pd.DataFrame,
    value_column: str,
    weight_column: str,
    strength: float = 5.0,
) -> pd.Series:
    if strength < 0:
        raise ValueError("strength must be non-negative")
    enriched = add_macroregion(frame)
    values = pd.to_numeric(enriched[value_column], errors="raise")
    weights = pd.to_numeric(enriched[weight_column], errors="raise").clip(lower=0)
    temp = enriched.assign(_value=values, _weight=weights)
    region_mean = temp.groupby("macroregion").apply(
        lambda group: (group["_value"] * group["_weight"]).sum() / max(group["_weight"].sum(), 1e-12),
        include_groups=False,
    )
    prior = temp["macroregion"].map(region_mean)
    return (weights * values + strength * prior) / (weights + strength)
