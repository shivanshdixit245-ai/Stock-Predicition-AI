# AI Instructions — Antigravity Agent Guide

## Your Role

You are a senior data scientist and ML engineer helping build a production-grade stock signal platform for a FAANG-level portfolio project. Your code must be clean, well-commented, and demonstrably correct — this will be reviewed by technical interviewers.

---

## Non-Negotiable Rules

### 1. Never use random train/test split on time-series data
```python
# NEVER DO THIS
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# ALWAYS USE THIS
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=12, gap=1)
```

### 2. Never fit scalers or feature selectors on the full dataset
```python
# WRONG — data leakage
scaler.fit(X_all)

# CORRECT — fit only on training fold
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

### 3. Never use `.shift()` incorrectly
```python
# Target must shift backward (tomorrow's direction):
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
# Drop last row — it has NaN target

# Indicators must NOT look forward:
df['sma_20'] = df['close'].rolling(20).mean()  # ✓ correct
```

### 4. Always add type hints and docstrings
```python
def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """
    Compute annualised Sharpe ratio.
    
    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate (default: 4%)
    
    Returns:
        Annualised Sharpe ratio
    
    Interview note: Sharpe = (mean_return - rfr) / std_return * sqrt(252)
    """
    excess = returns - risk_free_rate / 252
    return (excess.mean() / excess.std()) * np.sqrt(252)
```

### 5. Always use loguru for logging, never print()
```python
from loguru import logger
logger.info(f"Training fold {fold_idx}: F1={f1:.3f}, AUC={auc:.3f}")
logger.warning(f"Class imbalance detected: {ratio:.2f}")
logger.error(f"Data fetch failed for {ticker}: {e}")
```

### 6. Always log to MLflow inside training functions
Every training run must call:
- `mlflow.log_params(...)` — all hyperparameters
- `mlflow.log_metrics(...)` — all evaluation metrics
- `mlflow.log_artifact(...)` — model files and plots

### 7. Use config.py for all constants
Never hardcode thresholds or file paths in business logic:
```python
# WRONG
if prob > 0.65:  # hardcoded
    return "BUY"

# CORRECT
from config import settings
if prob > settings.signal_buy_threshold:
    return "BUY"
```

---

## Code Style

- Line length: 100 characters max
- Imports: standard lib → third party → local (separated by blank lines)
- Function length: max ~40 lines; extract helpers if longer
- Variable names: descriptive (`walk_forward_results`, not `wfr`)
- No magic numbers — use named constants
- Error messages must say what went wrong AND what to do about it

---

## Module Build Protocol

When asked to build a module:

1. Write the module docstring first (what it does, what it imports, what it returns)
2. Define all function signatures with type hints
3. Implement functions one by one
4. Add a `if __name__ == "__main__":` block that demonstrates usage with example data
5. Write a corresponding test file `tests/test_{module_name}.py`

---

## Explanation Blocks

At the top of every module, add a comment block for learning:

```python
"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Computes technical indicators from OHLCV price data.

Why we use pandas-ta instead of computing manually:
  pandas-ta implements 130+ indicators correctly and is pandas-native.
  Computing RSI manually is error-prone; use battle-tested libraries.

What a FAANG interviewer might ask:
  Q: "How does RSI work mathematically?"
  A: RSI = 100 - 100/(1 + RS) where RS = avg_gain/avg_loss over 14 days.
     It measures momentum: >70 = overbought, <30 = oversold.
  
  Q: "What's the difference between SMA and EMA?"
  A: EMA gives more weight to recent prices via exponential decay.
     EMA reacts faster to price changes; SMA is smoother/slower.

Data leakage risk in this module:
  None — all indicators are computed using only past data (no shift(-1)).
"""
```

---

## Testing Requirements

Every module needs a test file. Minimum tests:

```python
# tests/test_features.py
import pandas as pd
import numpy as np
from src.features.technical import compute_indicators

def test_no_future_lookahead():
    """Ensure no NaN at start means no backward fill from future."""
    df = make_sample_data(100)
    features = compute_indicators(df)
    # First 20 rows should have NaN for 20-day indicators (not filled from future)
    assert features["sma_20"].iloc[:19].isna().all()

def test_feature_count():
    df = make_sample_data(200)
    features = compute_indicators(df)
    assert len(features.columns) >= 20

def test_no_inf_values():
    df = make_sample_data(200)
    features = compute_indicators(df)
    assert not features.isin([np.inf, -np.inf]).any().any()
```

---

## Priority Order

If unclear what to build next, follow this priority:
1. Get data pipeline working end-to-end first (even with simple features)
2. Get a basic walk-forward training loop running with logistic regression
3. Get backtesting showing results
4. Get Streamlit dashboard showing something
5. Then layer in XGBoost, ensemble, regime, conformal, sentiment, drift

A working simple system beats a broken complex one. Always.
