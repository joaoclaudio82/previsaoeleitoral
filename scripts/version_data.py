from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.data.versioning import DataVersionStore

store = DataVersionStore(ROOT / settings.data_registry_path, ROOT / settings.snapshots_path)
files = {
    "polls": ROOT / settings.polls_path,
    "fundamentals": ROOT / settings.fundamentals_path,
    "state_priors": ROOT / settings.state_priors_path,
    "turnout": ROOT / settings.turnout_path,
}
for name, path in files.items():
    version = store.register_dataframe(
        name, pd.read_csv(path), as_of_date=date(2026, 8, 1),
        source_uri=str(path), is_synthetic=True,
        metadata={"election_id": "SYNTHETIC-LAB", "version": "0.2"},
    )
    print(name, version.version, version.sha256[:12], version.snapshot_path)
