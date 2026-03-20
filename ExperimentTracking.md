# Experiment Tracking — MLflow Specification

## Overview

Every model training run must be logged to MLflow. This creates a reproducible record of what was tried, what worked, and why the final model was chosen.

**Interview talking point:** "I tracked 60+ experiments with MLflow — every hyperparameter, feature set, and evaluation metric is logged, so I can reproduce any result and justify model selection decisions."

---

## 1. MLflow Setup

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Local setup — runs in mlruns/ directory, free, no cloud needed
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("stock-signal-platform")
```

Launch UI:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## 2. What to Log Per Run

### Parameters (hyperparameters + config)
```python
mlflow.log_params({
    # Model config
    "model_type": "xgboost",
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    
    # Data config
    "ticker": "AAPL",
    "train_start": "2019-01-01",
    "train_end": "2023-01-01",
    "feature_set": "v3_with_sentiment",
    "n_features": 24,
    "target": "binary_direction",
    
    # Validation config
    "cv_strategy": "walk_forward",
    "n_folds": 12,
    "gap_days": 1,
    
    # Calibration
    "calibration_method": "sigmoid",
    "conformal_alpha": 0.1,
    
    # Regime
    "regime_strategy": "feature_augmentation",
    "n_regimes": 3,
})
```

### Metrics (performance)
```python
mlflow.log_metrics({
    # Walk-forward metrics
    "wf_f1_mean": 0.61,
    "wf_f1_std": 0.04,
    "wf_f1_worst": 0.53,
    "wf_auc_mean": 0.64,
    "wf_brier_mean": 0.23,
    "wf_ece": 0.03,           # Expected Calibration Error
    
    # Backtest metrics
    "sharpe_ratio": 1.42,
    "calmar_ratio": 0.78,
    "max_drawdown": 0.182,
    "cagr": 0.143,
    "win_rate": 0.574,
    "total_return": 0.712,
    "bootstrap_pvalue": 0.023,
    
    # Conformal coverage
    "conformal_coverage": 0.913,
    "hold_rate": 0.28,         # fraction of days with HOLD signal
})
```

### Artefacts (files)
```python
mlflow.log_artifact("data/models/AAPL/xgb_model.pkl")
mlflow.log_artifact("data/models/AAPL/feature_list.json")
mlflow.log_artifact("data/models/AAPL/training_metadata.json")
mlflow.log_artifact("reports/shap_summary.png")
mlflow.log_artifact("reports/calibration_curve.png")
mlflow.log_artifact("reports/backtest_equity_curve.png")
mlflow.log_artifact("reports/walk_forward_metrics.png")
mlflow.log_artifact("reports/bootstrap_significance.png")
```

---

## 3. Full Logging Context Manager

Wrap every training run:

```python
def train_and_log(ticker: str, config: dict) -> str:
    """Train model and log everything to MLflow. Returns run_id."""
    
    with mlflow.start_run(run_name=f"{ticker}_{config['model_type']}_{config['feature_set']}") as run:
        mlflow.set_tags({
            "ticker": ticker,
            "model_family": "gradient_boosting",
            "has_sentiment": str(config.get("use_sentiment", False)),
            "has_regime": str(config.get("use_regime", False)),
            "stage": "development",
        })
        
        mlflow.log_params(config)
        
        # Train
        results = run_walk_forward_pipeline(ticker, config)
        
        # Log metrics
        mlflow.log_metrics(results["metrics"])
        
        # Save and log artefacts
        save_model_artefacts(ticker, results["models"])
        mlflow.log_artifacts(f"data/models/{ticker}/")
        mlflow.log_artifacts("reports/")
        
        return run.info.run_id
```

---

## 4. Experiment Naming Convention

```
Run name format: {TICKER}_{MODEL_TYPE}_{FEATURE_SET}_{TIMESTAMP}

Examples:
  AAPL_xgboost_v1_baseline_20240115
  AAPL_xgboost_v2_with_sentiment_20240116
  AAPL_ensemble_v3_regime_aware_20240117
  MSFT_xgboost_v3_regime_aware_20240117
```

---

## 5. Model Registry

Promote the best-performing model to "Production" stage in the MLflow Model Registry:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/xgb_model"
mv = mlflow.register_model(model_uri, f"stock-signal-{ticker}")

# Promote to Production
client.transition_model_version_stage(
    name=f"stock-signal-{ticker}",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)
```

---

## 6. Comparing Experiments

Key comparisons to always run and document in README:

| Experiment | Mean F1 | Sharpe | Notes |
|-----------|---------|--------|-------|
| Baseline (LR, no sentiment) | 0.53 | 0.72 | Sanity check |
| XGBoost, no sentiment | 0.58 | 1.01 | ML adds value |
| XGBoost + VADER sentiment | 0.59 | 1.08 | Small sentiment lift |
| XGBoost + FinBERT | 0.61 | 1.24 | FinBERT > VADER |
| Ensemble + regime | 0.63 | 1.42 | Regime detection helps |
| Ensemble + regime + conformal | 0.61* | 1.55 | *Lower trades, better Sharpe |

*HOLD signals from conformal prediction reduce trade count, improving risk-adjusted returns.
