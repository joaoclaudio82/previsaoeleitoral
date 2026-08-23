from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ElectionAI"
    environment: str = "development"
    model_path: Path = Path("models/winner_model.joblib")
    pollster_calibration_path: Path = Path("models/pollster_calibration.joblib")
    turnout_model_path: Path = Path("models/turnout_model.joblib")
    transfer_model_path: Path = Path("models/transfer_model.joblib")
    polls_path: Path = Path("data/raw/current_polls.csv")
    fundamentals_path: Path = Path("data/raw/current_fundamentals.csv")
    state_priors_path: Path = Path("data/raw/state_priors.csv")
    turnout_path: Path = Path("data/raw/current_turnout.csv")
    data_registry_path: Path = Path("data/registry/election_ai.sqlite3")
    snapshots_path: Path = Path("data/snapshots")
    n_simulations: int = 50_000
    posterior_draws: int = 8_000
    random_seed: int = 42

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
