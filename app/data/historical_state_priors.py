from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.candidate_identity import canonical_candidate
from app.data.geography import BRAZIL_UF_SET


def build_state_priors(
    previous_results: pd.DataFrame,
    current_candidates: pd.DataFrame,
    *,
    prior_strength: float = 2.5,
    fallback_strength: float = 0.75,
) -> pd.DataFrame:
    """Build state priors using only elections completed before the forecast year.

    Priority: same canonical candidate -> same party -> neutral national fallback.
    The output is normalized within each UF and can be consumed by the hierarchical poll model.
    """
    required_results = {"uf", "candidate_name", "party", "vote_share"}
    required_candidates = {"candidate_id", "candidate_name"}
    missing = required_results.difference(previous_results.columns)
    if missing:
        raise ValueError(f"previous_results missing columns: {sorted(missing)}")
    missing = required_candidates.difference(current_candidates.columns)
    if missing:
        raise ValueError(f"current_candidates missing columns: {sorted(missing)}")

    prior = previous_results.copy()
    prior["uf"] = prior["uf"].astype(str).str.upper().str.strip()
    prior = prior[prior["uf"].isin(BRAZIL_UF_SET)].copy()
    prior["canonical"] = prior["candidate_name"].map(canonical_candidate)
    prior["party"] = prior["party"].astype(str).str.upper().str.strip()
    candidates = current_candidates.copy()
    candidates["canonical"] = candidates["candidate_name"].map(canonical_candidate)
    if "party" not in candidates.columns:
        candidates["party"] = "UNKNOWN"
    candidates["party"] = candidates["party"].astype(str).str.upper().str.strip()

    national = prior.groupby("canonical", as_index=True)["vote_share"].mean().to_dict()
    party_national = prior.groupby("party", as_index=True)["vote_share"].mean().to_dict()
    neutral = 1.0 / max(len(candidates), 1)
    rows: list[dict[str, object]] = []
    ufs = sorted(uf for uf in prior["uf"].dropna().astype(str).unique() if uf in BRAZIL_UF_SET)

    for uf in ufs:
        state = prior[prior["uf"] == uf]
        state_by_candidate = state.groupby("canonical")["vote_share"].sum().to_dict()
        state_by_party = state.groupby("party")["vote_share"].sum().to_dict()
        values: list[tuple[pd.Series, float, float, str]] = []
        for _, candidate in candidates.iterrows():
            canonical = str(candidate["canonical"])
            party = str(candidate["party"])
            if canonical in state_by_candidate:
                value, strength, source = state_by_candidate[canonical], prior_strength, "same_candidate"
            elif party != "UNKNOWN" and party in state_by_party:
                value, strength, source = state_by_party[party], prior_strength * 0.8, "same_party"
            elif canonical in national:
                value, strength, source = national[canonical], fallback_strength, "candidate_national"
            elif party != "UNKNOWN" and party in party_national:
                value, strength, source = party_national[party], fallback_strength, "party_national"
            else:
                value, strength, source = neutral, fallback_strength * 0.5, "neutral"
            values.append((candidate, float(value), float(strength), source))

        total = sum(value for _, value, _, _ in values)
        if not np.isfinite(total) or total <= 0:
            total = 1.0
        for candidate, value, strength, source in values:
            rows.append(
                {
                    "uf": uf,
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_name": str(candidate["candidate_name"]),
                    "prior_share": value / total * 100.0,
                    "prior_strength": strength,
                    "prior_source": source,
                }
            )
    return pd.DataFrame(rows)
