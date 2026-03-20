import pytest
import numpy as np
import pandas as pd
from src.models.uncertainty import generate_signal, compute_empirical_coverage, fit_conformal

class MockMapie:
    def __init__(self, coverage=0.95):
        self.coverage = coverage
    def predict(self, X, alpha=0.1):
        n = len(X)
        y_pred = np.ones(n)
        # return sets that cover if coverage condition met
        y_set = np.zeros((n, 2, 1), dtype=bool)
        for i in range(n):
            if i < n * self.coverage:
                y_set[i, 1, 0] = True # Covers True label (1)
            else:
                y_set[i, 0, 0] = True # Misses
        return y_pred, y_set

def test_empirical_coverage_meets_nominal():
    """Verify coverage calculation and check against threshold."""
    y_true = np.ones(100)
    y_set = np.zeros((100, 2, 1), dtype=bool)
    y_set[:95, 1, 0] = True # 95% coverage
    
    coverage = compute_empirical_coverage(y_true, y_set)
    assert coverage >= 0.90

def test_hold_signal_when_prediction_set_ambiguous():
    """verify HOLD returned when both classes in set."""
    # prediction_set: [True, True]
    signal = generate_signal(0.8, [True, True])
    assert signal == "HOLD"
    
    # prediction_set: [False, False] (empty set)
    signal = generate_signal(0.8, [False, False])
    assert signal == "HOLD"

def test_no_trade_signal_below_threshold():
    """verify no BUY when prob < 0.65."""
    # Prob 0.6 is not enough for BUY even if set is {1}
    signal = generate_signal(0.6, [False, True])
    assert signal == "HOLD"
    
    # Prob 0.7 IS enough
    signal = generate_signal(0.7, [False, True])
    assert signal == "BUY"

def test_conformal_fitted_on_cal_data_only():
    """
    Conceptual test. In practice, we verify this by ensuring 
    MapieClassifier(cv='prefit') is used.
    """
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))
    
    ensemble = {
        "models": {"m1": MockModel()},
        "weights": {"m1": 1.0}
    }
    X_cal = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y_cal = pd.Series([0, 1])
    
    mapie = fit_conformal(ensemble, X_cal, y_cal)
    assert mapie.cv == "prefit"
    assert mapie.method == "score"
