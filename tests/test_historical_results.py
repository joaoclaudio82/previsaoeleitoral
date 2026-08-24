import pandas as pd

from app.data.historical_results import normalize_presidential_results


def test_normalize_presidential_results_by_uf() -> None:
    raw = pd.DataFrame(
        {
            "ANO_ELEICAO": [2022, 2022, 2022, 2022],
            "NR_TURNO": [1, 1, 1, 1],
            "SG_UF": ["CE", "CE", "SP", "SP"],
            "DS_CARGO": ["PRESIDENTE"] * 4,
            "NM_VOTAVEL": ["LULA", "JAIR BOLSONARO", "LULA", "JAIR BOLSONARO"],
            "SG_PARTIDO": ["PT", "PL", "PT", "PL"],
            "QT_VOTOS": [60, 40, 45, 55],
        }
    )
    result = normalize_presidential_results(raw, 2022)
    totals = result.groupby("uf")["vote_share"].sum()
    assert totals.round(10).eq(1.0).all()
    ce_lula = result[(result["uf"] == "CE") & (result["candidate_name"] == "LULA")].iloc[0]
    assert ce_lula["party"] == "PT"
    assert ce_lula["vote_share"] == 0.60
