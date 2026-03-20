# Features — Full Specification

## Feature Categories

---

## 1. Data Ingestion Features

### F-001: Multi-ticker stock data loader
- Fetch OHLCV data for any ticker using `yfinance`
- Configurable date range
- Auto-adjust for splits and dividends
- Cache to local Parquet on first fetch, reload from cache on subsequent runs
- Validate data completeness: flag gaps > 3 trading days

### F-002: Financial news ingestion
- Sources: NewsAPI (free tier, 100 req/day), Yahoo Finance RSS feed
- Query by ticker symbol and company name
- Store headline, source, publication timestamp, URL
- Deduplicate by URL hash
- Align news timestamps to next trading day open (prevent lookahead bias)

### F-003: Data quality monitor
- Report: % missing values per column, date coverage, OHLC sanity checks (High ≥ Low, etc.)
- Log all anomalies to `data_quality.log`

---

## 2. Technical Indicator Features

All indicators computed using `pandas-ta`. Every indicator also generates lag versions (t-1 to t-5).

### Trend indicators
| Feature | Description | Window(s) |
|---------|-------------|-----------|
| `sma_N` | Simple moving average | 5, 10, 20, 50, 200 |
| `ema_N` | Exponential moving average | 9, 21, 55 |
| `macd` | MACD line | 12/26/9 |
| `macd_signal` | MACD signal line | 12/26/9 |
| `macd_hist` | MACD histogram | 12/26/9 |
| `price_vs_sma20` | (Close - SMA20) / SMA20 | 20 |

### Momentum indicators
| Feature | Description | Window(s) |
|---------|-------------|-----------|
| `rsi_14` | Relative Strength Index | 14 |
| `rsi_7` | Short RSI | 7 |
| `stoch_k` | Stochastic %K | 14 |
| `stoch_d` | Stochastic %D | 3 |
| `williams_r` | Williams %R | 14 |
| `roc_10` | Rate of Change | 10 |

### Volatility indicators
| Feature | Description | Window(s) |
|---------|-------------|-----------|
| `bb_upper` | Bollinger upper band | 20, 2σ |
| `bb_lower` | Bollinger lower band | 20, 2σ |
| `bb_width` | Band width (volatility proxy) | 20 |
| `bb_pct` | %B position within bands | 20 |
| `atr_14` | Average True Range | 14 |
| `realised_vol_20` | Rolling std of log returns | 20 |
| `realised_vol_5` | Short-window volatility | 5 |

### Volume indicators
| Feature | Description |
|---------|-------------|
| `obv` | On-Balance Volume |
| `volume_zscore_20` | Volume z-score vs. 20-day mean |
| `vwap` | Volume-weighted average price |
| `volume_change_pct` | % change in volume vs. prior day |

### Calendar features
| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Monday to 4=Friday (encoded) |
| `month` | Month number (seasonality) |
| `is_month_end` | Binary flag |

### Lag features (auto-generated for top SHAP features)
- `close_lag_1` through `close_lag_10`
- `rsi_lag_1` through `rsi_lag_5`
- `return_lag_1` through `return_lag_5`

---

## 3. Sentiment Features

### F-003: VADER baseline sentiment
- Apply VADER to each headline
- Output: compound score [-1, 1]
- Aggregate per trading day: mean, max, min, count of headlines

### F-004: FinBERT financial sentiment
- Model: `ProsusAI/finbert` from HuggingFace
- Output: positive / negative / neutral probability per headline
- Aggregate per trading day: weighted mean of positive - negative probabilities
- **Compare FinBERT vs VADER correlation with next-day returns (report in README)**

### F-005: Sentiment momentum
- `sentiment_ma3`: 3-day rolling average sentiment
- `sentiment_change`: day-over-day sentiment delta
- `sentiment_volume_interaction`: sentiment × volume_zscore (amplify signal on high-volume news days)

---

## 4. Market Regime Features

### F-006: HMM regime label
- Fit 3-state Gaussian HMM on (log returns, realised volatility) using `hmmlearn`
- Output: `regime` ∈ {0=bull, 1=bear, 2=sideways} per trading day
- Train HMM only on training data; apply fitted model to validation/test (no leakage)
- One-hot encode regime for use as model feature
- Also use as stratification variable: report per-regime model performance

---

## 5. Target Variable

### F-007: Binary direction target
```python
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
# 1 = price goes up tomorrow, 0 = price goes down or flat
```
- Drop last row (NaN target)
- Class imbalance check: if imbalance > 55/45, apply `scale_pos_weight` in XGBoost

### F-008 (optional extension): Multi-class target
```python
# UP if return > +0.5%, DOWN if < -0.5%, SIDEWAYS otherwise
```

---

## 6. Feature Selection

### F-009: SHAP-based selection
- Train XGBoost on full feature set (walk-forward fold 1)
- Compute SHAP values on validation set
- Rank features by mean absolute SHAP value
- Retain top N features where cumulative importance ≥ 95%
- This set is fixed for all subsequent folds (prevents selection bias per fold)

### F-010: Permutation importance validation
- Cross-validate permutation importance vs. SHAP ranking
- Log correlation between the two rankings to MLflow

---

## 7. Dashboard Features

| Feature | Description |
|---------|-------------|
| Ticker input | Text field, validates against yfinance |
| Date range picker | Min 1 year of history |
| Candlestick chart | OHLC with volume bars, Plotly |
| Indicator overlay | Toggle RSI, MACD, Bollinger bands on chart |
| Signal markers | Green triangle (BUY), red triangle (SELL) on price chart |
| Prediction confidence | Probability bar with uncertainty interval |
| Regime overlay | Colour-code chart background by regime |
| Backtest equity curve | Strategy vs. all 3 benchmarks |
| Metrics table | Sharpe, drawdown, CAGR, Calmar, win rate |
| SHAP waterfall | Explain most recent prediction |
| Drift status | Green/amber/red indicator per feature group |
| MLflow link | Button to open experiment comparison view |
