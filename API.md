# API — FastAPI Specification

## Overview

A lightweight REST API layer wrapping the ML pipeline. Priority P2 — build after the core pipeline and dashboard are working. Demonstrates production engineering mindset.

---

## Base URL
`http://localhost:8000`

Docs auto-generated at: `http://localhost:8000/docs` (FastAPI OpenAPI UI)

---

## Endpoints

### GET /health
Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-01-15T14:00:00Z"
}
```

---

### GET /tickers/{ticker}/data
Fetch and return processed feature data for a ticker.

**Path params:**
- `ticker` (str): Stock ticker (e.g., AAPL)

**Query params:**
- `start_date` (str): ISO date, default 2 years ago
- `end_date` (str): ISO date, default today

**Response:**
```json
{
  "ticker": "AAPL",
  "start_date": "2022-01-01",
  "end_date": "2024-01-15",
  "n_rows": 504,
  "columns": ["close", "rsi_14", "macd", "..."],
  "preview": [{"date": "2024-01-15", "close": 182.5, "rsi_14": 58.2}]
}
```

---

### POST /tickers/{ticker}/train
Trigger model training for a ticker.

**Request body:**
```json
{
  "ticker": "AAPL",
  "start_date": "2019-01-01",
  "end_date": "2023-12-31",
  "use_sentiment": true,
  "use_regime": true,
  "n_cv_folds": 12,
  "model_types": ["xgboost", "lightgbm", "logistic"]
}
```

**Response:**
```json
{
  "run_id": "abc123def456",
  "status": "training_complete",
  "metrics": {
    "mean_f1": 0.61,
    "std_f1": 0.04,
    "sharpe_ratio": 1.42,
    "bootstrap_pvalue": 0.023
  },
  "mlflow_run_url": "http://localhost:5000/#/experiments/1/runs/abc123def456"
}
```

---

### GET /tickers/{ticker}/predict
Generate signal for today (or a specified date).

**Query params:**
- `date` (str, optional): Default today

**Response:**
```json
{
  "ticker": "AAPL",
  "date": "2024-01-15",
  "signal": "BUY",
  "buy_probability": 0.73,
  "sell_probability": 0.27,
  "prediction_set": ["BUY"],
  "confidence": "high",
  "top_features": [
    {"feature": "rsi_14", "shap_value": 0.12, "feature_value": 58.2},
    {"feature": "macd_hist", "shap_value": 0.09, "feature_value": 0.42},
    {"feature": "sentiment_finbert", "shap_value": 0.07, "feature_value": 0.34}
  ],
  "regime": "bull",
  "model_version": "v3",
  "trained_at": "2024-01-10T09:00:00Z"
}
```

---

### GET /tickers/{ticker}/backtest
Run backtest and return results.

**Query params:**
- `start_date` (str)
- `end_date` (str)
- `transaction_cost_pct` (float, default 0.001)

**Response:**
```json
{
  "ticker": "AAPL",
  "period": "2022-01-01 to 2024-01-15",
  "metrics": {
    "total_return": 0.712,
    "cagr": 0.143,
    "sharpe_ratio": 1.42,
    "calmar_ratio": 0.78,
    "max_drawdown": -0.182,
    "win_rate": 0.574,
    "n_trades": 137
  },
  "benchmarks": {
    "buy_and_hold": {"total_return": 0.341, "sharpe": 0.81},
    "momentum": {"total_return": 0.412, "sharpe": 0.97}
  },
  "significance": {
    "p_value": 0.023,
    "is_significant": true,
    "null_sharpe_95th_pct": 1.12
  }
}
```

---

### GET /tickers/{ticker}/drift
Get current drift status.

**Response:**
```json
{
  "ticker": "AAPL",
  "last_checked": "2024-01-15T06:00:00Z",
  "overall_status": "drift_detected",
  "should_retrain": true,
  "reason": "Feature drift detected in: macd, realised_vol_20",
  "features": {
    "rsi_14": {"psi": 0.08, "status": "stable"},
    "macd": {"psi": 0.24, "status": "drift"},
    "realised_vol_20": {"psi": 0.31, "status": "drift"},
    "sentiment_finbert": {"psi": 0.06, "status": "stable"}
  },
  "rolling_accuracy": {
    "current_20d": 0.52,
    "baseline": 0.61,
    "degraded": false
  }
}
```

---

## FastAPI Implementation Skeleton

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Market Signal API",
    description="AI-powered stock signal generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

```python
# api/routes.py
from fastapi import APIRouter
from src.data.loader import load_stock_data
from src.models.trainer import train_walk_forward
from src.backtest.engine import run_backtest
from src.drift.monitor import DriftMonitor

router = APIRouter(prefix="/tickers/{ticker}")

@router.get("/predict")
async def predict(ticker: str, date: str = None):
    try:
        model = load_trained_model(ticker)
        features = compute_latest_features(ticker, date)
        signal = generate_signal_with_explanation(model, features)
        return signal
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"No trained model found for {ticker}")
```

---

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Both Streamlit and FastAPI can run simultaneously:
- Streamlit: `python -m streamlit run app.py` (port 8501)
- FastAPI: `uvicorn api.main:app --port 8000`
