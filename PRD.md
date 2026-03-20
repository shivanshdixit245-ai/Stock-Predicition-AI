# PRD — Product Requirements Document

## Project Name
**Adaptive Market Intelligence & Risk Signal Platform**

## Version
1.0.0

## Author
Data Science Portfolio Project — FAANG/MAANG Candidate

---

## 1. Problem Statement

Retail and institutional investors face the challenge of identifying statistically significant trading signals in noisy financial time-series data. Existing open-source projects either lack rigorous ML validation (data leakage via random splits), ignore market regime shifts, or fail to quantify prediction uncertainty. This project addresses all three gaps in a single, production-ready system.

---

## 2. Goals

| Goal | Success Metric |
|------|---------------|
| Predict next-day stock direction | Out-of-sample F1 ≥ 0.58 across 12 walk-forward windows |
| Generate statistically significant alpha | Bootstrap p-value < 0.05 vs. random-entry baseline |
| Quantify prediction uncertainty | Conformal coverage ≥ 90% on holdout set |
| Detect and respond to model drift | Drift detection latency < 1 trading day |
| Provide interactive exploration | Streamlit dashboard with < 3s response time |

---

## 3. Non-Goals

- Real-money trading execution (no brokerage API integration)
- High-frequency trading (signals are daily granularity)
- Multi-asset portfolio optimisation (single ticker at a time, extendable)
- Regulatory compliance (not a financial product)

---

## 4. Users

**Primary:** Data science hiring managers and interviewers reviewing this as a portfolio piece.

**Secondary:** Self (learning data science, ML, and production engineering).

---

## 5. Core Requirements

### 5.1 Data Layer
- Fetch OHLCV (Open, High, Low, Close, Volume) data via `yfinance` for any ticker
- Fetch financial news headlines via NewsAPI (free tier) and Yahoo Finance RSS
- Store raw data in Parquet format for fast I/O
- Handle missing values, splits, dividends, and delistings gracefully

### 5.2 Feature Engineering
- Compute ≥ 20 technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Create lag features (t-1 through t-10)
- Compute rolling statistics (mean, std, z-score over 5/10/20/50 day windows)
- Compute sentiment score per trading day (FinBERT + VADER)
- Perform SHAP-based feature selection — retain only features with non-zero SHAP importance

### 5.3 Machine Learning
- Detect market regimes (bull / bear / sideways) using Hidden Markov Model
- Train per-regime models OR add regime label as feature — compare both
- Use walk-forward cross-validation (TimeSeriesSplit, 12 folds minimum)
- Train XGBoost, LightGBM, and logistic regression baseline
- Stack via soft-voting ensemble
- Calibrate output probabilities using Platt scaling (CalibratedClassifierCV)
- Generate conformal prediction intervals (MAPIE library) at 90% coverage level

### 5.4 Backtesting
- Vectorised backtest with realistic transaction cost model (0.1% slippage per trade)
- Report: total return, annualised return, Sharpe ratio, Calmar ratio, max drawdown, win rate
- Compare vs. three benchmarks: buy-and-hold, simple momentum (20-day MA crossover), random entry
- Bootstrap permutation test (n=10,000) on Sharpe ratio to confirm statistical significance

### 5.5 Model Monitoring
- Track rolling prediction accuracy on 20-day window
- Compute Population Stability Index (PSI) on feature distributions
- Trigger alert and optional auto-retrain when PSI > 0.2 or rolling accuracy drops > 10% from baseline

### 5.6 Experiment Tracking
- Log every training run to MLflow: hyperparameters, features used, walk-forward metrics, artefacts
- Store best model artefact per ticker
- Compare runs via MLflow UI

### 5.7 Dashboard
- Streamlit app with sidebar: ticker selection, date range, model configuration
- Tabs: Overview, Signals, Backtest Results, Model Explainability, Regime Analysis, Drift Monitor
- Candlestick chart with indicator overlays (Plotly)
- Buy/Sell signal annotations on price chart
- SHAP waterfall plots per prediction
- Backtest equity curve vs. benchmarks

---

## 6. Technical Constraints

- Zero monetary cost — all libraries and data sources are free
- Runs locally (no cloud required, though deployable to Streamlit Cloud)
- Python 3.11+
- Max dashboard load time: 5 seconds for 2 years of data

---

## 7. Milestones

| Phase | Deliverable | Priority |
|-------|------------|----------|
| 1 | Data pipeline + feature engineering | P0 |
| 2 | Walk-forward model training + ensemble | P0 |
| 3 | Backtesting engine + significance testing | P0 |
| 4 | Streamlit dashboard | P0 |
| 5 | Regime detection | P1 |
| 6 | Conformal prediction | P1 |
| 7 | Drift monitoring | P1 |
| 8 | MLflow integration | P1 |
| 9 | FastAPI layer | P2 |
| 10 | Unit tests | P2 |
