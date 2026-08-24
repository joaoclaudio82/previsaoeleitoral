import pandas as pd

from app.data.historical_state_priors import build_state_priors


def test_state_prior_prefers_same_candidate_then_neutral_fallback() -> None:
    previous = pd.DataFrame(
        {
            "uf": ["CE", "CE", "SP", "SP"],
            "candidate_name": ["JAIR MESSIAS BOLSONARO", "FERNANDO HADDAD", "JAIR MESSIAS BOLSONARO", "FERNANDO HADDAD"],
            "party": ["PSL", "PT", "PSL", "PT"],
            "vote_share": [0.30, 0.70, 0.60, 0.40],
        }
    )
    current = pd.DataFrame(
        {
            "candidate_id": ["bolsonaro", "lula"],
            "candidate_name": ["Bolsonaro", "Lula"],
            "party": ["PL", "PT"],
        }
    )
    priors = build_state_priors(previous, current)
    assert set(priors["uf"]) == {"CE", "SP"}
    assert priors.groupby("uf")["prior_share"].sum().round(8).eq(100.0).all()
    bolsonaro_ce = priors[(priors["uf"] == "CE") & (priors["candidate_id"] == "bolsonaro")].iloc[0]
    assert bolsonaro_ce["prior_source"] == "same_candidate"
