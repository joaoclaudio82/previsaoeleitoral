from datetime import date
import pandas as pd

from app.data.versioning import DataVersionStore


def test_dataset_versions_are_immutable_and_deduplicated(tmp_path):
    store = DataVersionStore(tmp_path / "registry.sqlite3", tmp_path / "snapshots")
    frame = pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    first = store.register_dataframe("polls", frame, as_of_date=date(2026, 8, 1), is_synthetic=True)
    repeated = store.register_dataframe("polls", frame.iloc[::-1], as_of_date=date(2026, 8, 1), is_synthetic=True)
    changed = store.register_dataframe("polls", pd.concat([frame, pd.DataFrame([{"a": 3, "b": "z"}])]), as_of_date=date(2026, 8, 1), is_synthetic=True)
    assert first.version == repeated.version == "v000001"
    assert changed.version == "v000002"
    assert len(store.list_versions("polls")) == 2
