import pytest
import numpy as np
import pandas as pd
from src.drift.monitor import compute_psi, monitor_feature_drift, rolling_accuracy_monitor, DriftMonitor

def test_psi_zero_for_identical_distributions():
    """Verify PSI=0 (or very close) when reference equals current."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    current = reference.copy()
    
    psi = compute_psi(reference, current)
    
    # Due to float precision and epsilon handling, it might be slightly > 0
    # but for identical it should be tiny.
    assert psi < 0.001

def test_psi_high_for_shifted_distribution():
    """Verify PSI > 0.2 when distributions are very different."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(2, 1, 1000) # Significant shift
    
    psi = compute_psi(reference, current)
    
    assert psi > 0.2

def test_should_retrain_when_many_features_drifted():
    """Verify should_retrain=True when 3+ features drifted (red status)."""
    # Create reference and current DFs where 3 features drift
    features = ["rsi_14", "macd", "realised_vol_20", "bb_pct", "sentiment_finbert"]
    
    ref_data = {f: np.random.normal(0, 1, 100) for f in features}
    cur_data = {f: np.random.normal(0, 1, 100) for f in features}
    
    # Apply high drift to 3 features
    cur_data["rsi_14"] = np.random.normal(5, 1, 100)
    cur_data["macd"] = np.random.normal(5, 1, 100)
    cur_data["realised_vol_20"] = np.random.normal(5, 1, 100)
    
    reference_df = pd.DataFrame(ref_data)
    current_df = pd.DataFrame(cur_data)
    
    # Mock predictions and actuals (perfect accuracy to isolate feature drift)
    preds = pd.Series([1] * 20)
    actuals = pd.Series([1] * 20)
    
    monitor = DriftMonitor(reference_df, baseline_accuracy=0.9)
    result = monitor.check(current_df, preds, actuals)
    
    assert len(result["drifted_features"]) == 3
    assert result["should_retrain"] is True
    assert "Drift detected in 3 features" in result["retrain_reason"]

def test_rolling_accuracy_alert_triggers_correctly():
    """Verify alert fires when accuracy drops below threshold."""
    # 80% baseline
    baseline_acc = 0.80
    degradation_threshold = 0.10
    
    # Window of 10 for testing
    window = 10
    
    # Case 1: High accuracy (90%)
    preds_good = pd.Series([1] * 10)
    actuals_good = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]) # 90%
    
    df_good = rolling_accuracy_monitor(preds_good, actuals_good, window=window, 
                                       baseline_accuracy=baseline_acc, 
                                       degradation_threshold=degradation_threshold)
    
    assert df_good["alert"].iloc[-1] == False
    
    # Case 2: Low accuracy (60%)
    # Baseline 0.8 - threshold 0.1 = 0.7. 
    # 60% < 70% -> Alert!
    preds_bad = pd.Series([1] * 10)
    actuals_bad = pd.Series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) # 60%
    
    df_bad = rolling_accuracy_monitor(preds_bad, actuals_bad, window=window, 
                                      baseline_accuracy=baseline_acc, 
                                      degradation_threshold=degradation_threshold)
    
    assert df_bad["alert"].iloc[-1] == True
