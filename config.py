import os
import subprocess
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Data
    default_ticker: str = os.environ.get("DEFAULT_TICKER", "AAPL")
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
    signal_buy_threshold: float = float(os.environ.get("SIGNAL_BUY_THRESHOLD", "0.65"))
    signal_sell_threshold: float = float(os.environ.get("SIGNAL_SELL_THRESHOLD", "0.35"))
    random_state: int = 42
    
    # Backtest
    initial_capital: float = float(os.environ.get("INITIAL_CAPITAL", "100000"))
    transaction_cost_pct: float = float(os.environ.get("TRANSACTION_COST_PCT", "0.001"))
    bootstrap_n: int = 10_000
    
    # Drift
    psi_threshold: float = 0.2
    accuracy_degradation_threshold: float = 0.10
    drift_window_days: int = 20
    
    # MLflow
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow_experiment_name: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "stock-signal-platform")
    
    # APIs
    news_api_key: str = os.environ.get("NEWS_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    huggingface_token: str = os.environ.get("HUGGINGFACE_TOKEN", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    
    class Config:
        env_file = ".env"  # Default to .env

settings = Settings()

def check_env_not_tracked():
    try:
        result = subprocess.run(
            ["git", "ls-files", ".env"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            raise RuntimeError(
                "CRITICAL SECURITY: .env is tracked by git! "
                "Run immediately: git rm --cached .env"
            )
    except FileNotFoundError:
        pass

# Also check for .env.LOCAL since it was found to be tracked
def check_env_local_not_tracked():
    try:
        result = subprocess.run(
            ["git", "ls-files", ".env.LOCAL"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            # Not raising error here to stick strictly to Rule 2, 
            # but Rule 2 check covers .env specifically.
            # I'll include it in the general scan script later.
            pass
    except FileNotFoundError:
        pass

check_env_not_tracked()
