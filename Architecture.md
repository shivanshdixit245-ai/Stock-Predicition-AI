# Architecture — System Design

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                          │
│  yfinance (OHLCV)  │  NewsAPI (headlines)  │  Yahoo RSS         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering Layer                     │
│  Technical indicators  │  Sentiment NLP  │  Lag features        │
│  Market regime (HMM)   │  SHAP selection │  Calendar features   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Training Layer                             │
│  Walk-forward CV   │  XGBoost + LightGBM + LR                   │
│  Optuna hyperopt   │  Calibration (Platt)                       │
│  Ensemble          │  Conformal prediction (MAPIE)              │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────┐     ┌─────────────────────────────────────┐
│  MLflow Tracking    │     │  Backtesting Engine                 │
│  Parameters         │     │  Vectorised simulation              │
│  Metrics            │     │  Transaction cost model             │
│  Artefacts          │     │  Bootstrap significance test        │
│  Model Registry     │     │  Benchmark comparisons              │
└─────────────────────┘     └─────────────────────────────────────┘
               │                          │
               └──────────────┬───────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Drift Monitoring Layer                        │
│  PSI per feature   │  Rolling accuracy   │  Retrain trigger     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                            │
│  Streamlit Dashboard   │   FastAPI (optional REST layer)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `src/data/loader.py`
- Single responsibility: fetch and cache raw data
- `load_stock_data(ticker, start, end) → pd.DataFrame`
- `load_news(ticker, start, end) → pd.DataFrame`
- Caches to `data/raw/{ticker}_{start}_{end}.parquet`
- Raises `DataUnavailableError` on API failure

### `src/data/preprocessor.py`
- Single responsibility: clean and normalise data
- `preprocess(df) → pd.DataFrame`
- Fill forward missing values (max 2 days)
- Remove weekends, holidays
- Compute log returns, validate OHLC ordering
- Normalise features with `RobustScaler` (outlier-resistant)
- CRITICAL: scaler must be fit only on training data, applied to val/test

### `src/features/technical.py`
- Compute all technical indicators using `pandas-ta`
- `compute_indicators(df) → pd.DataFrame`
- Compute lag features for selected indicators
- Returns feature DataFrame aligned to price index

### `src/features/sentiment.py`
- `compute_sentiment(news_df, price_df) → pd.DataFrame`
- VADER baseline + FinBERT
- Aggregate to daily granularity
- Align to next trading day (prevent lookahead)

### `src/features/selector.py`
- `select_features(X_train, y_train) → list[str]`
- Fit XGBoost on training data
- Compute SHAP values
- Return feature list with cumulative importance ≥ 95%

### `src/models/regime.py`
- `fit_regime_model(df) → HMMRegimeModel`
- `predict_regime(df, model) → pd.Series`
- Uses (log_return, realised_vol_20) as HMM observations
- Returns regime labels {0, 1, 2}

### `src/models/trainer.py`
- Walk-forward training loop
- `train_walk_forward(X, y, config) → WalkForwardResult`
- Per-fold: train, calibrate, evaluate, log to MLflow
- Returns all fold models + metrics

### `src/models/ensemble.py`
- `build_ensemble(models, weights) → EnsembleModel`
- Soft-voting with validation-performance weights
- `predict_proba(X) → np.ndarray`

### `src/models/uncertainty.py`
- Wrap ensemble in MAPIE conformal predictor
- `fit_conformal(ensemble, X_cal, y_cal) → MapieClassifier`
- `predict_with_uncertainty(X) → (predictions, prediction_sets)`

### `src/backtest/engine.py`
- `run_backtest(prices, signals, config) → BacktestResult`
- Vectorised (no row-by-row loops)
- Returns equity curve, trades log, metrics dict

### `src/backtest/stats.py`
- `compute_metrics(returns) → dict`
- `bootstrap_significance(returns, n=10000) → dict`
- `plot_equity_curve(results) → plotly.Figure`

### `src/drift/monitor.py`
- `DriftMonitor` class (see ModelDrift.md)
- Called on every new batch of predictions in Streamlit dashboard

### `api/routes.py`
- FastAPI routes (see API.md)
- Thin wrapper around `src/` modules

### `app.py`
- Streamlit entry point
- Calls `src/` modules, never contains business logic
- See UIUX.md for full dashboard spec

---

## Data Flow Diagram

```
Raw OHLCV  ──► Preprocessor ──► Feature Engineer ──► Feature Selector
                                                              │
News Data  ──► Sentiment NLP ─────────────────────────────► │
                                                              ▼
                                                     Feature Matrix (X)
                                                              │
                              ┌───────────────────────────── │
                              ▼                               │
                      HMM Regime Model                        │
                              │                               │
                              ▼                               ▼
                      Regime Labels ──────────────► Walk-Forward Trainer
                                                              │
                                              ┌───────────────┤
                                              │               │
                                              ▼               ▼
                                         MLflow Log    Trained Models
                                                               │
                                                               ▼
                                                     Calibrated Ensemble
                                                               │
                                                               ▼
                                                     MAPIE Conformal
                                                               │
                                                    ┌──────────┤
                                                    │          │
                                                    ▼          ▼
                                              Signals     Backtest Engine
                                                               │
                                                         Bootstrap Test
                                                               │
                                                         Drift Monitor
                                                               │
                                                    ┌──────────┤
                                                    │          │
                                                    ▼          ▼
                                              Streamlit     FastAPI
```

---

## Configuration

All constants in `config.py` using `pydantic-settings`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Data
    default_ticker: str = "AAPL"
    default_start_date: str = "2019-01-01"
    data_cache_dir: str = "data/raw"
    model_dir: str = "data/models"
    
    # ML
    n_cv_folds: int = 12
    cv_gap_days: int = 1
    n_regimes: int = 3
    conformal_alpha: float = 0.1
    signal_buy_threshold: float = 0.65
    signal_sell_threshold: float = 0.35
    
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
    
    class Config:
        env_file = ".env"

settings = Settings()
```
