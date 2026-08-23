from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

UF_INFO = [
    ("AC", "Norte", 600_000), ("AL", "Nordeste", 2_300_000), ("AP", "Norte", 550_000),
    ("AM", "Norte", 2_900_000), ("BA", "Nordeste", 11_200_000), ("CE", "Nordeste", 6_900_000),
    ("DF", "Centro-Oeste", 2_200_000), ("ES", "Sudeste", 3_000_000), ("GO", "Centro-Oeste", 5_100_000),
    ("MA", "Nordeste", 5_100_000), ("MT", "Centro-Oeste", 2_500_000), ("MS", "Centro-Oeste", 2_100_000),
    ("MG", "Sudeste", 16_000_000), ("PA", "Norte", 6_100_000), ("PB", "Nordeste", 3_100_000),
    ("PR", "Sul", 8_600_000), ("PE", "Nordeste", 7_000_000), ("PI", "Nordeste", 2_500_000),
    ("RJ", "Sudeste", 12_800_000), ("RN", "Nordeste", 2_600_000), ("RS", "Sul", 8_500_000),
    ("RO", "Norte", 1_200_000), ("RR", "Norte", 420_000), ("SC", "Sul", 5_600_000),
    ("SP", "Sudeste", 34_500_000), ("SE", "Nordeste", 1_700_000), ("TO", "Norte", 1_100_000),
]


def _normalize(values: np.ndarray, total: float = 100.0) -> np.ndarray:
    values = np.clip(values, 0.05, None)
    return values / values.sum() * total


def generate(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    legacy_transfer_matrix = RAW / "transfer_matrix.csv"
    if legacy_transfer_matrix.exists():
        legacy_transfer_matrix.unlink()

    candidate_ids = ["A", "B", "C", "D"]
    candidate_names = ["Candidata Aurora", "Candidato Boreal", "Candidata Ceres", "Candidato Dourado"]
    institutes = ["Instituto Alfa", "Instituto Beta", "Instituto Gama", "Instituto Delta"]
    modes = {"Instituto Alfa": "telephone", "Instituto Beta": "online", "Instituto Gama": "face_to_face", "Instituto Delta": "mixed"}
    populations = {"Instituto Alfa": "registered_voters", "Instituto Beta": "adults", "Instituto Gama": "likely_voters", "Instituto Delta": "registered_voters"}
    institute_bias = {
        "Instituto Alfa": np.array([0.8, -0.4, -0.2, -0.2]),
        "Instituto Beta": np.array([-0.6, 0.7, 0.2, -0.3]),
        "Instituto Gama": np.array([0.1, -0.2, 0.4, -0.3]),
        "Instituto Delta": np.array([0.0, 0.0, 0.0, 0.0]),
    }
    mode_bias = {
        "telephone": np.array([0.3, -0.2, -0.1, 0.0]),
        "online": np.array([-0.4, 0.2, 0.4, -0.2]),
        "face_to_face": np.array([0.2, 0.1, -0.2, -0.1]),
        "mixed": np.zeros(4),
    }
    region_shift = {
        "Norte": np.array([-1.5, 0.2, 0.8, 0.5]),
        "Nordeste": np.array([4.0, -2.0, 0.5, -2.5]),
        "Centro-Oeste": np.array([-2.5, 2.8, -0.5, 0.2]),
        "Sudeste": np.array([0.5, 0.5, -0.4, -0.6]),
        "Sul": np.array([-3.0, 3.5, -0.8, 0.3]),
    }
    end = date(2026, 8, 1)
    national_base = np.array([37.0, 31.0, 19.0, 13.0])

    state_prior_rows: list[dict] = []
    state_truth: dict[str, np.ndarray] = {}
    for uf, region, electorate in UF_INFO:
        local = _normalize(national_base + region_shift[region] + rng.normal(0, 1.4, 4))
        state_truth[uf] = local
        for cid, cname, share in zip(candidate_ids, candidate_names, local):
            state_prior_rows.append({
                "uf": uf, "region": region, "registered_voters": electorate,
                "candidate_id": cid, "candidate_name": cname,
                "prior_share": round(float(share), 4), "prior_strength": 3.5,
            })
    state_priors = pd.DataFrame(state_prior_rows)
    state_priors.to_csv(RAW / "state_priors.csv", index=False)

    poll_rows: list[dict] = []
    for i in range(16):
        poll_date = end - timedelta(days=(15 - i) * 4)
        institute = institutes[i % len(institutes)]
        drift = np.array([0.10 * i, -0.055 * i, 0.005 * i, -0.05 * i])
        decided = _normalize(
            national_base + drift + institute_bias[institute] + mode_bias[modes[institute]] + rng.multivariate_normal(np.zeros(4), np.full((4, 4), 0.25) + np.eye(4) * 0.8)
        )
        undecided = float(np.clip(16.0 - 0.45 * i + rng.normal(0, 0.8), 5, 20))
        reported = decided * (1.0 - undecided / 100.0)
        sample_size = int(rng.integers(1200, 3200))
        margin = round(float(rng.uniform(1.7, 2.8)), 1)
        poll_id = f"DEMO-NAT-{i + 1:02d}"
        for cid, cname, share in zip(candidate_ids, candidate_names, reported):
            poll_rows.append({
                "poll_id": poll_id, "field_date": poll_date.isoformat(), "institute": institute,
                "collection_mode": modes[institute], "target_population": populations[institute],
                "candidate_id": cid, "candidate_name": cname, "share": round(float(share), 3),
                "undecided_share": round(undecided, 3), "sample_size": sample_size,
                "margin_error": margin, "scope": "national", "uf": "BR",
            })

    for uf, region, electorate in UF_INFO:
        poll_count = 3 if electorate > 7_000_000 else (2 if electorate > 2_000_000 else 1)
        for j in range(poll_count):
            poll_date = end - timedelta(days=int(rng.integers(2, 48)))
            institute = institutes[(j + sum(ord(c) for c in uf)) % len(institutes)]
            decided = _normalize(
                state_truth[uf] + institute_bias[institute] + mode_bias[modes[institute]] + rng.normal(0, 1.6, 4)
            )
            undecided = float(np.clip(13 + (end - poll_date).days * 0.06 + rng.normal(0, 1), 5, 22))
            reported = decided * (1.0 - undecided / 100.0)
            sample_size = int(rng.integers(700, 1800))
            margin = round(float(rng.uniform(2.2, 3.8)), 1)
            poll_id = f"DEMO-{uf}-{j + 1:02d}"
            for cid, cname, share in zip(candidate_ids, candidate_names, reported):
                poll_rows.append({
                    "poll_id": poll_id, "field_date": poll_date.isoformat(), "institute": institute,
                    "collection_mode": modes[institute], "target_population": populations[institute],
                    "candidate_id": cid, "candidate_name": cname, "share": round(float(share), 3),
                    "undecided_share": round(undecided, 3), "sample_size": sample_size,
                    "margin_error": margin, "scope": "state", "uf": uf,
                })
    polls = pd.DataFrame(poll_rows)
    polls.to_csv(RAW / "current_polls.csv", index=False)

    fundamentals = pd.DataFrame([
        ["A", candidate_names[0], 36, 1, 48, 4.2, 7.1, 2.1, 41, 0.18, 0.72, 0.65, 0.15, 0.20, -0.45, "progressive", 0.12],
        ["B", candidate_names[1], 42, 0, 48, 4.2, 7.1, 2.1, 33, 0.08, 0.68, 0.61, 0.20, 0.25, 0.55, "conservative", 0.05],
        ["C", candidate_names[2], 28, 0, 48, 4.2, 7.1, 2.1, 17, 0.22, 0.45, 0.58, 0.80, 0.35, -0.10, "center", 0.22],
        ["D", candidate_names[3], 49, 0, 48, 4.2, 7.1, 2.1, 9, -0.08, 0.30, 0.42, 1.90, 1.30, 0.20, "independent", -0.10],
    ], columns=[
        "candidate_id", "candidate_name", "rejection", "incumbency", "government_approval",
        "inflation_12m", "unemployment", "gdp_yoy", "search_share", "media_sentiment",
        "search_reliability", "sentiment_reliability", "search_anomaly_score", "sentiment_anomaly_score",
        "ideology_score", "bloc", "late_decider_score",
    ])
    fundamentals.to_csv(RAW / "current_fundamentals.csv", index=False)

    turnout_rows = []
    for uf, region, electorate in UF_INFO:
        baseline = 0.77 + {"Norte": -0.015, "Nordeste": 0.015, "Centro-Oeste": -0.005, "Sudeste": 0.005, "Sul": 0.012}[region]
        local = state_truth[uf]
        competitiveness = 1.0 - abs(np.sort(local)[-1] - np.sort(local)[-2]) / 30.0
        turnout_rows.append({
            "uf": uf, "registered_voters": electorate,
            "historical_turnout": round(float(np.clip(baseline + rng.normal(0, 0.018), 0.68, 0.86)), 5),
            "abstention_trend": round(float(rng.normal(0.003, 0.008)), 5),
            "registration_growth": round(float(rng.normal(0.018, 0.009)), 5),
            "mobility_index": round(float(rng.normal(0, 0.45)), 5),
            "weather_severity": round(float(np.clip(rng.gamma(1.2, 0.22), 0, 2.5)), 5),
            "competitiveness": round(float(np.clip(competitiveness, 0, 1)), 5),
        })
    current_turnout = pd.DataFrame(turnout_rows)
    current_turnout.to_csv(RAW / "current_turnout.csv", index=False)

    calibration_rows = []
    for election in range(35):
        result = _normalize(rng.lognormal(0, 0.55, 4))
        for j in range(int(rng.integers(5, 12))):
            institute = institutes[j % len(institutes)]
            days_before = int(rng.integers(1, 75))
            error_scale = {"Instituto Alfa": 1.2, "Instituto Beta": 1.8, "Instituto Gama": 1.45, "Instituto Delta": 1.0}[institute]
            common = rng.normal(0, 0.9)
            poll = _normalize(result + institute_bias[institute] + common + rng.normal(0, error_scale, 4))
            poll_id = f"CAL-{election:03d}-{j:02d}"
            for cid, poll_share, result_share in zip(candidate_ids, poll, result):
                calibration_rows.append({
                    "election_id": f"CAL-{election:03d}", "poll_id": poll_id,
                    "institute": institute, "collection_mode": modes[institute],
                    "target_population": populations[institute], "days_before": days_before,
                    "candidate_id": cid, "poll_share": poll_share, "result_share": result_share,
                })
    pd.DataFrame(calibration_rows).to_csv(PROCESSED / "historical_poll_calibration.csv", index=False)

    turnout_history = []
    for election in range(24):
        national_shock = rng.normal(0, 0.012)
        for uf, region, electorate in UF_INFO:
            historical = float(np.clip(rng.normal(0.78, 0.025), 0.68, 0.88))
            abstention_trend = float(rng.normal(0, 0.012))
            registration_growth = float(rng.normal(0.015, 0.012))
            mobility = float(rng.normal(0, 0.65))
            weather = float(np.clip(rng.gamma(1.3, 0.28), 0, 3))
            competitiveness = float(rng.uniform(0.15, 1.0))
            latent = (
                np.log(historical / (1 - historical)) - 0.65 * abstention_trend
                + 0.18 * registration_growth - 0.025 * mobility - 0.055 * weather
                + 0.10 * competitiveness + national_shock + rng.normal(0, 0.025)
            )
            turnout_value = 1 / (1 + np.exp(-latent))
            turnout_history.append({
                "election_id": f"TURN-{election:03d}", "uf": uf,
                "historical_turnout": historical, "abstention_trend": abstention_trend,
                "registration_growth": registration_growth, "mobility_index": mobility,
                "weather_severity": weather, "competitiveness": competitiveness,
                "turnout": float(np.clip(turnout_value, 0.55, 0.92)),
            })
    pd.DataFrame(turnout_history).to_csv(PROCESSED / "historical_turnout.csv", index=False)

    transfer_rows = []
    for _ in range(1_100):
        distance_advantage = float(rng.uniform(-1.8, 1.8))
        rejection_advantage = float(rng.normal(0, 18))
        bloc_advantage = float(rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]))
        incumbency_advantage = float(rng.choice([-1, 0, 1], p=[0.12, 0.76, 0.12]))
        source_rejection = float(rng.uniform(18, 72))
        distance_sum = float(rng.uniform(0.2, 2.8))
        p_a = 1 / (1 + np.exp(-(1.75 * distance_advantage + 0.026 * rejection_advantage + 0.85 * bloc_advantage + 0.22 * incumbency_advantage)))
        p_abs = 1 / (1 + np.exp(-(-3.0 + 0.018 * source_rejection + 0.38 * distance_sum)))
        total = int(rng.integers(250, 1800))
        abstain = int(rng.binomial(total, p_abs))
        valid = total - abstain
        to_a = int(rng.binomial(valid, p_a))
        transfer_rows.append({
            "distance_advantage_a": distance_advantage, "rejection_advantage_a": rejection_advantage,
            "same_bloc_advantage_a": bloc_advantage, "incumbency_advantage_a": incumbency_advantage,
            "source_rejection": source_rejection, "finalist_distance_sum": distance_sum,
            "to_a": to_a, "to_b": valid - to_a, "abstain": abstain,
        })
    pd.DataFrame(transfer_rows).to_csv(PROCESSED / "runoff_transfer_training.csv", index=False)

    history_rows = []
    for election in range(420):
        latent = rng.normal(0, 1, 4)
        poll = _normalize(np.exp(latent))
        rejection = np.clip(55 - poll * 0.55 + rng.normal(0, 8, 4), 10, 80)
        incumbent_idx = int(rng.integers(0, 4))
        approval = float(np.clip(rng.normal(48, 12), 15, 80))
        inflation = float(np.clip(rng.normal(5.2, 2.5), 0, 18))
        unemployment = float(np.clip(rng.normal(9, 3), 2, 22))
        gdp = float(rng.normal(1.8, 2.7))
        search_raw = _normalize(np.clip(poll + rng.normal(0, 7, 4), 1, None))
        sentiment_raw = np.clip(rng.normal((poll - poll.mean()) / 40, 0.30), -1, 1)
        digital_reliability = np.clip(rng.beta(4, 2, 4), 0.05, 1)
        search_guarded = 25 + digital_reliability * (search_raw - 25)
        sentiment_guarded = digital_reliability * sentiment_raw
        trend = rng.normal(0, 2.0, 4)
        score = (
            poll * 0.12 - rejection * 0.035 + search_guarded * 0.020 + sentiment_guarded * 0.75 + trend * 0.05
            + np.array([1 if i == incumbent_idx else 0 for i in range(4)]) * ((approval - 50) / 14)
            - inflation * 0.03 - unemployment * 0.02 + gdp * 0.04 + rng.normal(0, 0.45, 4)
        )
        winner = int(np.argmax(score))
        for i in range(4):
            history_rows.append({
                "election_id": f"SYN-{election:04d}", "candidate_id": candidate_ids[i],
                "poll_mean": poll[i], "poll_trend_14d": trend[i], "rejection": rejection[i],
                "incumbency": int(i == incumbent_idx), "government_approval": approval,
                "inflation_12m": inflation, "unemployment": unemployment, "gdp_yoy": gdp,
                "search_share_guarded": search_guarded[i], "media_sentiment_guarded": sentiment_guarded[i],
                "digital_signal_reliability": digital_reliability[i], "won": int(i == winner),
            })
    pd.DataFrame(history_rows).to_csv(PROCESSED / "historical_training.csv", index=False)

    sample = {
        "as_of_date": end.isoformat(), "election_id": "SYNTHETIC-LAB",
        "election_year": 2026, "dataset_type": "synthetic", "validation_status": "unvalidated",
        "n_simulations": 20_000,
        "polls": polls.to_dict(orient="records"),
        "fundamentals": fundamentals.to_dict(orient="records"),
        "state_priors": state_priors.to_dict(orient="records"),
        "turnout": current_turnout.to_dict(orient="records"),
    }
    (ROOT / "data" / "sample_request.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate()
    print("Dados sintéticos v0.2 gerados. Eles são exclusivamente demonstrativos.")
