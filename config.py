from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Data
    default_ticker: str = "AAPL"
    default_start_date: str = "2022-01-01"
    data_raw_dir: Path = Path("data/raw")
    data_prices_dir: Path = data_raw_dir / "prices"
    data_news_dir: Path = data_raw_dir / "news"
    data_processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")
    
    # ML
    n_cv_folds: int = 12
    cv_gap_days: int = 1
    n_regimes: int = 3
    conformal_alpha: float = 0.1
    signal_buy_threshold: float = 0.65
    signal_sell_threshold: float = 0.35
    random_state: int = 42
    
    # Backtest
    initial_capital: float = 100_000
    transaction_cost_pct: float = 0.001
    bootstrap_n: int = 10_000
    
    # Drift
    psi_threshold: float = 0.2
    accuracy_degradation_threshold: float = 0.10
    drift_window_days: int = 20
    
    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "stock-signal-platform"
    
    # APIs
    news_api_key: str = ""  # from .env
    gemini_api_key: str = ""  # from .env
    
    class Config:
        env_file = ".env.LOCAL"

settings = Settings()
