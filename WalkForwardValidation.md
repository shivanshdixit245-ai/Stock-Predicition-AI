# Walk-Forward Validation — Deep Specification

## The Core Problem: Data Leakage in Time Series

This is the most common and most damaging mistake in financial ML projects. Understanding and fixing it is what separates a genuine data science project from a toy.

**The mistake:**
```python
# WRONG — DO NOT DO THIS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
```

When `shuffle=True` (which is the default), row 1000 (year 2023) ends up in training, row 1001 (year 2022) ends up in test. The model learns patterns that implicitly encode future information. Validation scores are meaningless.

**Even `shuffle=False` with a single split is not enough.** It only tests on one time period — you don't know if performance holds across different market conditions.

---

## Walk-Forward Validation Explained

Walk-forward validation (also called anchored rolling validation) respects time ordering and tests across multiple market periods.

### Expanding Window (what we use)
```
Fold 1:  Train [2019-01 → 2020-06]  |  Validate [2020-07 → 2020-09]
Fold 2:  Train [2019-01 → 2020-09]  |  Validate [2020-10 → 2020-12]
Fold 3:  Train [2019-01 → 2020-12]  |  Validate [2021-01 → 2021-03]
...
Fold 12: Train [2019-01 → 2022-06]  |  Validate [2022-07 → 2022-09]
```

The training window expands — the model always sees all historical data up to that point, mirroring real deployment.

---

## Implementation

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, model_factory, n_splits: int = 12) -> dict:
    """
    Walk-forward cross-validation for time-series classification.
    
    Args:
        X: Feature matrix, must be sorted by date ascending
        y: Target vector
        model_factory: Callable that returns a fresh untrained model
        n_splits: Number of validation folds
    
    Returns:
        Dictionary of per-fold and aggregated metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fresh model each fold — no state leakage between folds
        model = model_factory()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        metrics = compute_classification_metrics(y_val, preds, probs)
        metrics["fold"] = fold_idx + 1
        metrics["train_size"] = len(train_idx)
        metrics["val_size"] = len(val_idx)
        metrics["val_start"] = X_val.index[0]
        metrics["val_end"] = X_val.index[-1]
        
        fold_metrics.append(metrics)
    
    results = {
        "per_fold": fold_metrics,
        "mean_f1": np.mean([m["f1"] for m in fold_metrics]),
        "std_f1": np.std([m["f1"] for m in fold_metrics]),
        "worst_f1": np.min([m["f1"] for m in fold_metrics]),
        "best_f1": np.max([m["f1"] for m in fold_metrics]),
        "mean_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
    }
    
    return results
```

---

## Leakage Checklist

Before training, verify the following are all FALSE:

| Leakage Type | Check |
|-------------|-------|
| Target calculated with future data | `y_t` uses only data up to time t |
| Feature uses future prices | All indicators lag-adjusted |
| News timestamps aligned correctly | Headlines aligned to NEXT day open |
| Scaler fit on full dataset | Scaler fitted ONLY on training fold |
| Regime HMM fit on full dataset | HMM fitted ONLY on training fold |
| Feature selection on full dataset | SHAP selection done on fold 1 training data only |
| Hyperparameter tuning leaks test set | Optuna runs on validation, final evaluation on separate holdout |

---

## Gap Parameter

The `gap=1` parameter in `TimeSeriesSplit` inserts a 1-day gap between training end and validation start. This prevents the last training day's close from implicitly informing the first validation day's features. Always use it.

---

## Reporting Results

Never report a single number. Report distribution:

```
Walk-Forward Cross-Validation Results (12 folds)
─────────────────────────────────────────────────
F1 Score:    0.61 ± 0.04  (worst: 0.53, best: 0.68)
ROC-AUC:     0.64 ± 0.03  (worst: 0.58, best: 0.70)
Precision:   0.62 ± 0.05
Recall:      0.60 ± 0.06
Brier Score: 0.23 ± 0.02  (lower is better)
```

This is the kind of output a FAANG interviewer wants to see. It shows you understand variance in model performance, not just point estimates.
