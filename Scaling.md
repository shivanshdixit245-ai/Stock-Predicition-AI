# Scaling Strategy

## Current Scope (Portfolio Project)

Single ticker, daily granularity, local execution. This document describes how the system would scale to production — important for FAANG interviews where they ask "how would you scale this?"

---

## Scaling Dimensions

### 1. Data volume scaling
Current: 1 ticker, 5 years daily = ~1,250 rows

Production target: 500 tickers, 10 years = 1.25M rows

Solution: Move from Parquet files to Apache Parquet + DuckDB for analytical queries without a database server. DuckDB queries Parquet files directly with SQL semantics and columnar performance.

```python
import duckdb

conn = duckdb.connect()
result = conn.execute("""
    SELECT ticker, date, close, rsi_14
    FROM read_parquet('data/processed/features/*.parquet')
    WHERE ticker = 'AAPL' AND date >= '2022-01-01'
""").df()
```

For 100+ tickers, switch to a managed database: TimescaleDB (PostgreSQL extension optimised for time-series).

### 2. Inference latency scaling
Current: Compute features + predict on each dashboard load (~2-3 seconds)

Production: Pre-compute all signals nightly, serve from cache, sub-100ms latency

```python
# Scheduled nightly job (APScheduler or cron)
def nightly_signal_refresh():
    for ticker in watch_list:
        signals = generate_signals(ticker)
        save_to_cache(signals, f"data/signals/{ticker}_latest.parquet")
```

### 3. Model training scaling
Current: Single process, ~5 minutes per ticker

Production: Parallelise across tickers using `joblib.Parallel` or Celery task queue

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(train_ticker)(ticker, config)
    for ticker in ticker_list
)
```

### 4. MLflow scaling
Current: Local `./mlruns` directory

Production: MLflow Tracking Server with PostgreSQL backend + S3 artefact store

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@host/mlflow \
  --default-artifact-root s3://your-bucket/mlflow-artefacts
```

---

## Batch vs. Real-Time Inference

| Mode | When to use | Latency | Cost |
|------|------------|---------|------|
| Batch (nightly) | Daily signals, no urgency | Hours | Low |
| Near-real-time | Intraday alerting | Minutes | Medium |
| Real-time | HFT (out of scope) | Microseconds | Very high |

This project uses batch inference. Signals computed after market close, cached, served next morning.

---

## Production ML Additions (future work)

If this were a real production system, next steps would be:
1. Feature store (Feast or Tecton) — centralise feature computation, avoid recomputation
2. Model serving (BentoML or Seldon) — containerised model server with versioning
3. A/B testing framework — compare model versions on live data with statistical rigour
4. Monitoring alerts (PagerDuty / Slack) — automated alerts on drift triggers
5. Retraining pipeline (Airflow DAG) — scheduled, automated, audited retraining
