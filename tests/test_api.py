from fastapi.testclient import TestClient
from app.api.main import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "0.2.0"


def test_synthetic_api_response_never_exposes_likely_winner():
    polls = []
    for candidate_id, name, share in [("A", "Aurora", 52), ("B", "Boreal", 38)]:
        polls.append({
            "poll_id": "P1", "field_date": "2026-08-01", "institute": "Alpha",
            "collection_mode": "telephone", "target_population": "registered_voters",
            "candidate_id": candidate_id, "candidate_name": name, "share": share,
            "undecided_share": 10, "sample_size": 1200, "margin_error": 2.5,
            "scope": "national", "uf": "BR",
        })
    fundamentals = [
        {"candidate_id": "A", "candidate_name": "Aurora", "rejection": 35, "incumbency": 1, "government_approval": 50, "inflation_12m": 4, "unemployment": 7, "gdp_yoy": 2, "search_share": 55, "media_sentiment": 0.1, "ideology_score": -0.4, "bloc": "x"},
        {"candidate_id": "B", "candidate_name": "Boreal", "rejection": 45, "incumbency": 0, "government_approval": 50, "inflation_12m": 4, "unemployment": 7, "gdp_yoy": 2, "search_share": 45, "media_sentiment": 0.0, "ideology_score": 0.4, "bloc": "y"},
    ]
    response = TestClient(app).post("/predict", json={
        "as_of_date": "2026-08-01", "election_id": "SYNTHETIC-LAB", "election_year": 2026,
        "dataset_type": "synthetic", "validation_status": "unvalidated", "n_simulations": 1000,
        "polls": polls, "fundamentals": fundamentals,
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["likely_winner"] is None
    assert payload["publication_status"] == "BLOCKED_SYNTHETIC_DEMONSTRATION"
    assert payload["run_id"]
    lineage = TestClient(app).get(f"/lineage/runs/{payload['run_id']}")
    assert lineage.status_code == 200
    assert lineage.json()["publication_status"] == "BLOCKED_SYNTHETIC_DEMONSTRATION"
