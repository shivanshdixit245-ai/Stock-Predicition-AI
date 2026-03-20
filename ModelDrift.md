# Model Drift Detection & Monitoring

## Overview

Financial models degrade over time because markets evolve. A model trained on 2021 data may fail in 2023 due to regime shifts, new correlations, and changing market microstructure. This module monitors for drift and triggers retraining when needed.

**Interview talking point:** "I built a drift monitor because in production, model accuracy doesn't stay constant — markets change and models need to adapt. This shows I think beyond the notebook."

---

## 1. Types of Drift

| Type | What changes | Detection method |
|------|-------------|-----------------|
| Feature drift | Input distribution shifts | PSI, KL divergence |
| Concept drift | Relationship between features and target changes | Rolling accuracy monitor |
| Label drift | Target class distribution changes | Rolling class ratio monitor |

---

## 2. Population Stability Index (PSI)

PSI measures how much a feature's distribution has changed between a reference period (training data) and current data.

### Formula
```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
```

### Interpretation
| PSI Value | Action |
|-----------|--------|
| < 0.1 | Stable — no action needed |
| 0.1 – 0.2 | Minor shift — monitor closely |
| > 0.2 | Significant drift — trigger retraining |

### Implementation
```python
import numpy as np
import pandas as pd

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between reference and current distributions.
    
    Args:
        reference: Feature values from training period
        current: Feature values from recent window
        n_bins: Number of bins for discretisation
    
    Returns:
        PSI value
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    
    def get_bin_freqs(data, breakpoints):
        counts = np.histogram(data, bins=breakpoints)[0]
        freqs = counts / len(data)
        freqs = np.where(freqs == 0, 0.0001, freqs)  # avoid log(0)
        return freqs
    
    ref_freqs = get_bin_freqs(reference, breakpoints)
    cur_freqs = get_bin_freqs(current, breakpoints)
    
    psi = np.sum((cur_freqs - ref_freqs) * np.log(cur_freqs / ref_freqs))
    return psi


def monitor_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.2
) -> dict:
    """Compute PSI for all monitored features."""
    results = {}
    for feature in features:
        psi = compute_psi(reference_df[feature].values, current_df[feature].values)
        results[feature] = {
            "psi": round(psi, 4),
            "status": "drift" if psi > threshold else "stable",
        }
    return results
```

---

## 3. Rolling Accuracy Monitor

Track model accuracy on a rolling 20-day window. Alert when accuracy degrades significantly.

```python
def rolling_accuracy_monitor(
    predictions: pd.Series,
    actuals: pd.Series,
    window: int = 20,
    baseline_accuracy: float = None,
    degradation_threshold: float = 0.10
) -> pd.DataFrame:
    """
    Compute rolling accuracy and flag degradation.
    
    Args:
        predictions: Model predictions (0 or 1)
        actuals: True labels
        window: Rolling window in trading days
        baseline_accuracy: Accuracy on validation set (from training)
        degradation_threshold: Alert if rolling accuracy drops this much below baseline
    
    Returns:
        DataFrame with rolling accuracy, rolling F1, and alert flags
    """
    correct = (predictions == actuals).astype(int)
    rolling_acc = correct.rolling(window).mean()
    
    if baseline_accuracy:
        alert = rolling_acc < (baseline_accuracy - degradation_threshold)
    else:
        alert = rolling_acc < rolling_acc.rolling(60).mean() - degradation_threshold
    
    return pd.DataFrame({
        "rolling_accuracy": rolling_acc,
        "alert": alert,
        "baseline": baseline_accuracy,
    })
```

---

## 4. Automated Retraining Trigger

```python
class DriftMonitor:
    def __init__(self, reference_df, baseline_accuracy, psi_threshold=0.2, acc_drop_threshold=0.10):
        self.reference_df = reference_df
        self.baseline_accuracy = baseline_accuracy
        self.psi_threshold = psi_threshold
        self.acc_drop_threshold = acc_drop_threshold
        self.monitored_features = ["rsi_14", "macd", "realised_vol_20", "bb_pct", "sentiment_score"]
    
    def check(self, current_df, recent_predictions, recent_actuals) -> dict:
        psi_results = monitor_feature_drift(
            self.reference_df, current_df, self.monitored_features, self.psi_threshold
        )
        
        drifted_features = [f for f, r in psi_results.items() if r["status"] == "drift"]
        
        rolling_acc = rolling_accuracy_monitor(
            recent_predictions, recent_actuals,
            baseline_accuracy=self.baseline_accuracy,
            degradation_threshold=self.acc_drop_threshold
        )
        accuracy_degraded = rolling_acc["alert"].iloc[-1]
        
        should_retrain = len(drifted_features) > 2 or accuracy_degraded
        
        return {
            "psi_results": psi_results,
            "drifted_features": drifted_features,
            "accuracy_degraded": accuracy_degraded,
            "should_retrain": should_retrain,
            "retrain_reason": self._format_reason(drifted_features, accuracy_degraded),
        }
    
    def _format_reason(self, drifted_features, accuracy_degraded):
        reasons = []
        if drifted_features:
            reasons.append(f"Feature drift detected in: {', '.join(drifted_features)}")
        if accuracy_degraded:
            reasons.append("Rolling accuracy degraded > 10% from baseline")
        return " | ".join(reasons) if reasons else "No drift detected"
```

---

## 5. Drift Dashboard Tab

In Streamlit, add a "Drift Monitor" tab with:
- Table of all monitored features, their PSI, and status (green/amber/red)
- Line chart of rolling accuracy over time with baseline reference line
- Alert banner if retraining is recommended
- "Retrain Now" button that triggers the training pipeline
- Last retrain date and model version

---

## 6. Evidently AI (Optional Alternative)

For a more production-ready monitoring setup, use the Evidently library:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("drift_report.html")
```

This generates a full HTML report with statistical tests, visualisations, and pass/fail status per feature. Can be embedded in Streamlit via `st.components.v1.html`.
