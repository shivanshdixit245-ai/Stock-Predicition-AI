# ML Pipeline — Full Specification

## Overview

The ML pipeline is the core of this project. It is designed to avoid the most common mistakes in financial ML: data leakage, overfitting to a single test period, and ignoring non-stationarity.

**Interview talking point:** "I designed the pipeline around three constraints that most tutorials ignore: temporal ordering in cross-validation, probability calibration for reliable signal thresholds, and market regime conditioning."

---

## 1. Walk-Forward Cross-Validation

### Why this matters (critical)
Standard `train_test_split(shuffle=True)` on time-series data is a critical bug. It allows the model to learn from "future" data during training, producing inflated validation scores that collapse on real forward data. Walk-forward validation mimics real deployment.

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=12, gap=1)
# gap=1 prevents the model seeing the day it's predicting
```

Each fold:
- Training window expands (expanding window, not rolling)
- Gap of 1 day between train end and validation start
- Validate on next ~20 trading days
- Log per-fold metrics to MLflow
- Report: mean F1, std F1, worst-fold F1 (important for robustness claim)

### Reporting standard
Never report a single accuracy number. Always report:
```
F1: 0.61 ± 0.04 (mean ± std across 12 folds)
Worst fold: 0.54 | Best fold: 0.68
```

---

## 2. Regime-Conditional Training

### Two strategies (implement both, compare)

**Strategy A — Feature augmentation**
Add one-hot encoded regime columns as features. Single model learns regime-specific patterns implicitly.

**Strategy B — Separate models**
Train one XGBoost model per regime. At inference, detect current regime, route to corresponding model.

Log both strategies to MLflow. Report which achieves better walk-forward F1.

---

## 3. Model Specifications

### 3.1 XGBoost (primary model)
```python
params = {
    "n_estimators": 500,
    "max_depth": 4,           # shallow to prevent overfitting
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": auto, # handle class imbalance
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "random_state": 42
}
```
Tune with Optuna (50 trials), log all trials to MLflow.

### 3.2 LightGBM (secondary model)
```python
params = {
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "random_state": 42
}
```

### 3.3 Logistic Regression (baseline — CRITICAL)
Always include a simple baseline. If your complex model barely beats logistic regression, that's an important finding to report honestly.
```python
from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression(max_iter=1000, C=0.1)
```

### 3.4 Ensemble
Soft-voting ensemble: weighted average of probability outputs.
```python
weights = [xgb_val_f1, lgb_val_f1, lr_val_f1]  # weight by validation performance
ensemble_prob = np.average([xgb_probs, lgb_probs, lr_probs], weights=weights, axis=0)
```

---

## 4. Probability Calibration

### Why this matters
Raw XGBoost probability outputs are not well-calibrated — "70% confidence" from the model often doesn't mean 70% empirical accuracy. Calibration fixes this, which is essential for reliable signal thresholds.

### Implementation
```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Platt scaling
    cv='prefit'        # model already fitted
)
calibrated_model.fit(X_val, y_val)
```

### Validation
Plot reliability diagram (calibration curve):
- x-axis: mean predicted probability (binned)
- y-axis: fraction of positives in that bin
- Perfect calibration = diagonal line
- Log calibration error (ECE) to MLflow

---

## 5. Conformal Prediction (Uncertainty Quantification)

### Why this matters
Point predictions are not enough for financial decisions. Conformal prediction gives statistically valid prediction sets — guaranteed coverage without distributional assumptions.

### Implementation
```python
from mapie.classification import MapieClassifier

mapie = MapieClassifier(estimator=calibrated_ensemble, method="score")
mapie.fit(X_train, y_train)
y_pred, y_set = mapie.predict(X_test, alpha=0.1)  # 90% coverage
```

### Output
Each prediction includes:
- Point prediction (BUY / SELL)
- Prediction set: may include both classes on uncertain days (= "no signal" condition)
- Empirical coverage on holdout: should be ≥ 90%

### Signal generation rule
```python
# Only trade when prediction set has exactly one class (high confidence)
if len(prediction_set) == 1:
    signal = "BUY" if prediction_set[0] == 1 else "SELL"
else:
    signal = "HOLD"  # uncertain — stay flat
```
This HOLD condition dramatically reduces false signals.

---

## 6. Signal Generation

```python
def generate_signal(prob: float, pred_set: list) -> str:
    if len(pred_set) != 1:
        return "HOLD"
    if prob > 0.65:
        return "BUY"
    elif prob < 0.35:
        return "SELL"
    else:
        return "HOLD"
```

Configurable thresholds in `config.py`. Dashboard exposes slider to explore threshold sensitivity.

---

## 7. Model Artefact Management

```
data/models/
├── {ticker}/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── calibrated_ensemble.pkl
│   ├── mapie_model.pkl
│   ├── hmm_regime.pkl
│   ├── feature_list.json        ← exact features the model was trained on
│   ├── training_metadata.json   ← date range, fold performance, version
│   └── scaler.pkl
```

Always save `feature_list.json` alongside the model. This prevents silent mismatch between training features and inference features.

---

## 8. Evaluation Metrics

| Metric | Why |
|--------|-----|
| F1 score | Balanced metric for imbalanced classes |
| Precision | When we say BUY, how often are we right? |
| Recall | What fraction of actual UP days did we catch? |
| ROC-AUC | Overall discrimination ability |
| Brier score | Probability calibration quality |
| ECE | Expected Calibration Error |
| Log-loss | Per-prediction probability quality |

Report all metrics per fold. Report mean ± std across folds.

---

## 9. Hyperparameter Optimisation

Use Optuna for Bayesian hyperparameter search. Log every trial to MLflow.

```python
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_est", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("mcw", 1, 20),
    }
    # cross-validate with walk-forward
    return mean_walk_forward_f1(params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```
