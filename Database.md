# Database & Data Storage — Specification

## Philosophy

This project uses file-based storage (Parquet + JSON) rather than a database server. Rationale: no infrastructure to spin up, fast columnar reads with Parquet, simple enough for a portfolio project while demonstrating real-world data engineering practices.

---

## Storage Layout

```
data/
├── raw/
│   ├── prices/
│   │   └── {TICKER}_{start}_{end}.parquet       ← cached OHLCV
│   └── news/
│       └── {TICKER}_{start}_{end}.parquet       ← cached headlines
│
├── processed/
│   ├── features/
│   │   └── {TICKER}_{start}_{end}_v{version}.parquet  ← engineered features
│   ├── sentiment/
│   │   └── {TICKER}_{start}_{end}_sentiment.parquet   ← daily sentiment scores
│   └── regimes/
│       └── {TICKER}_{start}_{end}_regimes.parquet     ← HMM regime labels
│
├── models/
│   └── {TICKER}/
│       ├── xgb_model.pkl
│       ├── lgb_model.pkl
│       ├── calibrated_ensemble.pkl
│       ├── mapie_conformal.pkl
│       ├── hmm_regime.pkl
│       ├── scaler.pkl
│       ├── feature_list.json
│       └── training_metadata.json
│
└── results/
    └── {TICKER}/
        ├── backtest_results.json
        ├── walk_forward_metrics.json
        └── drift_baseline.parquet      ← reference distribution for drift monitoring
```

---

## Parquet Schema: Prices

```
Column          Type        Description
─────────────────────────────────────────────────────
date            datetime64  Trading date (index)
open            float64     Open price
high            float64     Daily high
low             float64     Daily low
close           float64     Adjusted close price
volume          int64       Daily volume
ticker          string      Ticker symbol
```

---

## Parquet Schema: Features

```
Column              Type        Description
─────────────────────────────────────────────────────
date                datetime64  Trading date (index)
close               float64     Adjusted close
log_return          float64     log(close_t / close_t-1)
target              int8        1=up tomorrow, 0=down/flat
sma_5 ... sma_200   float64     Simple moving averages
ema_9 ... ema_55    float64     Exponential moving averages
rsi_7               float64     RSI (7-day)
rsi_14              float64     RSI (14-day)
macd                float64     MACD line
macd_signal         float64     MACD signal line
macd_hist           float64     MACD histogram
bb_upper            float64     Bollinger upper band
bb_lower            float64     Bollinger lower band
bb_width            float64     Band width
bb_pct              float64     %B position
atr_14              float64     Average True Range
realised_vol_5      float64     5-day realised vol
realised_vol_20     float64     20-day realised vol
obv                 float64     On-Balance Volume
volume_zscore_20    float64     Volume z-score
vwap                float64     VWAP
close_lag_1..10     float64     Lagged close prices
rsi_lag_1..5        float64     Lagged RSI values
return_lag_1..5     float64     Lagged returns
sentiment_finbert   float64     Daily FinBERT sentiment [-1, 1]
sentiment_vader     float64     Daily VADER compound score
sentiment_ma3       float64     3-day sentiment moving average
sentiment_change    float64     Day-over-day sentiment delta
regime              int8        HMM regime label {0, 1, 2}
regime_bull         int8        One-hot encoded
regime_bear         int8        One-hot encoded
regime_sideways     int8        One-hot encoded
day_of_week         int8        0=Mon to 4=Fri
month               int8        Month number
is_month_end        int8        Binary flag
```

---

## JSON Schema: training_metadata.json

```json
{
  "ticker": "AAPL",
  "version": "v3",
  "trained_at": "2024-01-15T14:32:00Z",
  "train_start": "2019-01-01",
  "train_end": "2023-01-01",
  "feature_version": "v3",
  "n_features": 24,
  "model_type": "calibrated_ensemble",
  "walk_forward_folds": 12,
  "metrics": {
    "mean_f1": 0.61,
    "std_f1": 0.04,
    "mean_auc": 0.64,
    "sharpe_ratio": 1.42,
    "bootstrap_pvalue": 0.023,
    "conformal_coverage": 0.913
  },
  "mlflow_run_id": "abc123...",
  "feature_list_hash": "sha256:def456..."
}
```

---

## Caching Strategy

```python
from pathlib import Path
import pandas as pd
import hashlib

def get_cache_path(data_dir: str, ticker: str, start: str, end: str, suffix: str) -> Path:
    key = f"{ticker}_{start}_{end}"
    return Path(data_dir) / f"{key}_{suffix}.parquet"

def load_with_cache(fetch_fn, cache_path: Path, **kwargs) -> pd.DataFrame:
    """Load from cache if exists, otherwise fetch and cache."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    df = fetch_fn(**kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=True, compression="snappy")
    return df
```

---

## Why Parquet Over CSV

| Aspect | Parquet | CSV |
|--------|---------|-----|
| Read speed | ~10x faster | Baseline |
| File size | ~5-10x smaller (snappy compression) | Baseline |
| Data types | Preserved (datetime, int8, float32) | Everything becomes string |
| Columnar reads | Only load needed columns | Load everything |
| Schema | Enforced | No enforcement |

For 5 years of daily AAPL data + features (~1,250 rows × 60 cols), Parquet reads in ~5ms vs. CSV ~50ms. Negligible here, but demonstrates professional practice.
