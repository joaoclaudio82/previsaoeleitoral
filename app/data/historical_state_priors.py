from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.candidate_identity import canonical_candidate
from app.data.geography import BRAZIL_UF_SET


def _normalized_map(frame: pd.DataFrame, key: str) -> dict[str, float]:
    """Return national shares using votes when available, otherwise state-share means."""
    if "votes" in frame.columns:
        votes = pd.to_numeric(frame["votes"], errors="coerce").fillna(0.0)
        if float(votes.sum()) > 0:
            grouped = frame.assign(_votes=votes).groupby(key)["_votes"].sum()
        else:
            grouped = frame.groupby(key)["vote_share"].mean()
    else:
        grouped = frame.groupby(key)["vote_share"].mean()
    grouped = pd.to_numeric(grouped, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(grouped.sum())
    if total <= 0:
        return {}
    return {str(index): float(value / total) for index, value in grouped.items()}


def build_state_priors(
    previous_results: pd.DataFrame,
    current_candidates: pd.DataFrame,
    *,
    prior_strength: float = 2.5,
    fallback_strength: float = 0.75,
) -> pd.DataFrame:
    """Build leakage-safe state leans from elections completed before the forecast year.

    The prior stores both the previous-election state composition and the corresponding
    national composition for the *current* candidate set. The hierarchical model uses
    their log-ratio difference as a geographic lean and applies that lean to the current
    national forecast. This avoids freezing a state at its previous absolute vote share.

    Matching priority is: same canonical candidate -> same party -> national candidate
    fallback -> national party fallback -> neutral. ``prior_strength`` controls both the
    amount of shrinkage when state polls exist and the concentration of prior draws.
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
    prior["vote_share"] = pd.to_numeric(prior["vote_share"], errors="coerce").fillna(0.0)

    candidates = current_candidates.copy()
    candidates["canonical"] = candidates["candidate_name"].map(canonical_candidate)
    if "party" not in candidates.columns:
        candidates["party"] = "UNKNOWN"
    candidates["party"] = candidates["party"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

    national = _normalized_map(prior, "canonical")
    party_national = _normalized_map(prior, "party")
    neutral = 1.0 / max(len(candidates), 1)
    rows: list[dict[str, object]] = []
    ufs = sorted(uf for uf in prior["uf"].dropna().astype(str).unique() if uf in BRAZIL_UF_SET)

    for uf in ufs:
        state = prior[prior["uf"] == uf]
        state_by_candidate = state.groupby("canonical")["vote_share"].sum().to_dict()
        state_by_party = state.groupby("party")["vote_share"].sum().to_dict()
        values: list[tuple[pd.Series, float, float, float, str]] = []
        for _, candidate in candidates.iterrows():
            canonical = str(candidate["canonical"])
            party = str(candidate["party"])
            if canonical in state_by_candidate:
                state_value = float(state_by_candidate[canonical])
                national_value = float(national.get(canonical, neutral))
                strength, source = prior_strength, "same_candidate"
            elif party != "UNKNOWN" and party in state_by_party:
                state_value = float(state_by_party[party])
                national_value = float(party_national.get(party, neutral))
                strength, source = prior_strength * 0.8, "same_party"
            elif canonical in national:
                state_value = national_value = float(national[canonical])
                strength, source = fallback_strength, "candidate_national"
            elif party != "UNKNOWN" and party in party_national:
                state_value = national_value = float(party_national[party])
                strength, source = fallback_strength, "party_national"
            else:
                state_value = national_value = neutral
                strength, source = fallback_strength * 0.5, "neutral"
            values.append((candidate, state_value, national_value, float(strength), source))

        state_total = sum(value for _, value, _, _, _ in values)
        national_total = sum(value for _, _, value, _, _ in values)
        if not np.isfinite(state_total) or state_total <= 0:
            state_total = 1.0
        if not np.isfinite(national_total) or national_total <= 0:
            national_total = 1.0

        for candidate, state_value, national_value, strength, source in values:
            rows.append(
                {
                    "uf": uf,
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_name": str(candidate["candidate_name"]),
                    "prior_share": state_value / state_total * 100.0,
                    "national_prior_share": national_value / national_total * 100.0,
                    "prior_strength": strength,
                    "prior_concentration": max(6.0, strength * 20.0),
                    "prior_source": source,
                }
            )
    return pd.DataFrame(rows)
