# Tech Stack — Full Specification

## Summary

Every technology choice below is justified. In interviews, be ready to explain WHY you chose each tool.

---

## Core Language

| Tool | Version | Why |
|------|---------|-----|
| Python | 3.11+ | Industry standard for data science; 3.11 is significantly faster than 3.9/3.10 |

---

## Data Ingestion

| Tool | Purpose | Why this, not alternatives |
|------|---------|---------------------------|
| `yfinance` | Stock OHLCV data | Free, covers most tickers, adjusts for splits/dividends automatically |
| `NewsAPI` | Financial headlines | Free tier (100 req/day), clean JSON, date-queryable |
| `requests` + BeautifulSoup | Yahoo Finance RSS fallback | Backup when NewsAPI quota exhausted |
| `pyarrow` + `fastparquet` | Parquet storage | 10-100x faster than CSV for columnar financial data; preserves dtypes |

---

## Feature Engineering

| Tool | Purpose | Why |
|------|---------|-----|
| `pandas` 2.0+ | DataFrame operations | Industry standard; 2.0 has significant performance improvements with Copy-on-Write |
| `numpy` | Numerical operations | Vectorised operations, no for-loops |
| `pandas-ta` | Technical indicators | 130+ indicators, clean API, pandas-native |
| `scipy` | Statistical functions | Needed for KL divergence, statistical tests |

---

## NLP / Sentiment

| Tool | Purpose | Why |
|------|---------|-----|
| `vaderSentiment` | Baseline sentiment | Lightweight, no GPU needed, good baseline |
| `transformers` (HuggingFace) | FinBERT inference | `ProsusAI/finbert` is fine-tuned specifically on financial text — outperforms general BERT |
| `torch` | PyTorch backend for FinBERT | Required by HuggingFace transformers |

Performance note: FinBERT inference is slow on CPU. For a portfolio project, run once and cache results to Parquet. Use `--no-cache` flag to force re-computation.

---

## Machine Learning

| Tool | Purpose | Why |
|------|---------|-----|
| `scikit-learn` | Preprocessing, baseline models, calibration | Industry standard; TimeSeriesSplit for walk-forward CV |
| `xgboost` | Primary gradient boosting model | Best single model for tabular financial data; `scale_pos_weight` for imbalance |
| `lightgbm` | Ensemble member | Faster than XGBoost on large datasets; often competitive |
| `optuna` | Hyperparameter optimisation | Better than GridSearch (Bayesian optimisation); integrates with MLflow |
| `shap` | Feature importance + explainability | Model-agnostic, shows both direction and magnitude of feature contributions |
| `mapie` | Conformal prediction | Production-grade uncertainty quantification; works with any sklearn-compatible model |
| `hmmlearn` | Hidden Markov Model (regime detection) | Standard HMM implementation; well-documented |
| `ruptures` | Changepoint detection (alternative to HMM) | Fast, pure-Python, good for detecting structural breaks |

---

## Backtesting & Finance

| Tool | Purpose | Why |
|------|---------|-----|
| `vectorbt` | Vectorised backtesting | 100x faster than event-driven engines; handles large date ranges efficiently |
| `quantstats` | Financial metrics reporting | Generates full tearsheet (Sharpe, drawdown, etc.) in one call |
| `pyfolio` | Portfolio analysis | Tear sheet visualisations; used at Quantopian |

---

## Experiment Tracking

| Tool | Purpose | Why |
|------|---------|-----|
| `mlflow` | Experiment tracking + model registry | Free, local, production-quality; industry standard for ML ops |

---

## Monitoring

| Tool | Purpose | Why |
|------|---------|-----|
| `evidently` | Data + model drift reports | Production-grade drift detection; generates HTML reports |
| `apscheduler` | Scheduled drift checks | Lightweight Python job scheduler |

---

## Dashboard

| Tool | Purpose | Why |
|------|---------|-----|
| `streamlit` 1.30+ | Dashboard framework | Fastest way to build interactive data apps; perfect for demos |
| `plotly` | Interactive charts | Best interactive charting library for Python; native Streamlit support |
| `plotly.express` | Quick chart generation | Reduces boilerplate for standard charts |

---

## API Layer (optional P2)

| Tool | Purpose | Why |
|------|---------|-----|
| `fastapi` | REST API framework | Fastest Python API framework; auto-generates OpenAPI docs |
| `uvicorn` | ASGI server | Production-grade async server for FastAPI |
| `pydantic` v2 | Data validation | FastAPI uses it natively; validates all request/response schemas |

---

## Testing

| Tool | Purpose | Why |
|------|---------|-----|
| `pytest` | Test runner | Industry standard |
| `pytest-cov` | Coverage reporting | Enforces minimum coverage |
| `hypothesis` | Property-based testing | Tests edge cases you wouldn't think to write manually |

---

## Code Quality

| Tool | Purpose | Why |
|------|---------|-----|
| `ruff` | Linting + formatting | 10-100x faster than flake8; replaces black + isort |
| `mypy` | Static type checking | Catches type errors before runtime |
| `loguru` | Logging | Better API than stdlib logging; structured logs |
| `pydantic-settings` | Config management | Type-safe settings from environment variables |

---

## `requirements.txt`

```
# Data
yfinance>=0.2.36
newsapi-python>=0.2.7
pyarrow>=14.0.0

# Feature engineering
pandas>=2.0.0
numpy>=1.26.0
pandas-ta>=0.3.14b
scipy>=1.11.0

# NLP
vaderSentiment>=3.3.2
transformers>=4.37.0
torch>=2.1.0

# ML
scikit-learn>=1.4.0
xgboost>=2.0.3
lightgbm>=4.2.0
optuna>=3.5.0
shap>=0.44.1
mapie>=0.8.2
hmmlearn>=0.3.2
ruptures>=1.1.9

# Backtesting
vectorbt>=0.26.2
quantstats>=0.0.62

# Experiment tracking
mlflow>=2.10.0

# Monitoring
evidently>=0.4.13
apscheduler>=3.10.4

# Dashboard
streamlit>=1.30.0
plotly>=5.18.0

# API
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Code quality
loguru>=0.7.2
ruff>=0.2.0
mypy>=1.8.0
pytest>=7.4.0
pytest-cov>=4.1.0
```
