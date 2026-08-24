from datetime import date

import pandas as pd

from app.data.historical_snapshots import build_snapshots


def test_snapshot_never_uses_future_poll_and_locks_latest_slate() -> None:
    frame = pd.DataFrame(
        [
            {"poll_id": "old", "field_date": "2018-08-01", "candidate_id": "lula", "candidate_name": "Lula", "share": 40},
            {"poll_id": "old", "field_date": "2018-08-01", "candidate_id": "bolsonaro", "candidate_name": "Bolsonaro", "share": 30},
            {"poll_id": "new", "field_date": "2018-09-20", "candidate_id": "haddad", "candidate_name": "Haddad", "share": 22},
            {"poll_id": "new", "field_date": "2018-09-20", "candidate_id": "bolsonaro", "candidate_name": "Bolsonaro", "share": 35},
            {"poll_id": "future", "field_date": "2018-10-06", "candidate_id": "haddad", "candidate_name": "Haddad", "share": 40},
            {"poll_id": "future", "field_date": "2018-10-06", "candidate_id": "bolsonaro", "candidate_name": "Bolsonaro", "share": 41},
        ]
    )
    snapshots = build_snapshots(frame, date(2018, 10, 7), offsets=(15,))
    snapshot = snapshots[15]
    assert set(snapshot["poll_id"]) == {"new"}
    assert set(snapshot["candidate_id"]) == {"haddad", "bolsonaro"}
    assert pd.to_datetime(snapshot["field_date"]).max().date() <= date(2018, 9, 22)
