import pytest
import numpy as np
import pandas as pd
from src.models.ensemble import build_ensemble, compute_calibration_error, ensemble_predict_proba

def test_weights_sum_to_one():
    """Verify ensemble weights normalise to 1.0."""
    models = {"m1": None, "m2": None}
    weights = {"m1": 0.8, "m2": 0.4}
    ensemble = build_ensemble(models, weights)
    
    assert sum(ensemble["weights"].values()) == pytest.approx(1.0)
    assert ensemble["weights"]["m1"] == pytest.approx(0.8 / 1.2)

def test_calibration_improves_ece():
    """
    Verify ECE calculation logic.
    We mock y_true and y_prob for a 'good' and 'bad' case.
    """
    y_true = np.array([0, 0, 1, 1])
    
    # Bad calibration: high confidence in wrong class
    y_prob_bad = np.array([0.9, 0.8, 0.1, 0.2])
    ece_bad = compute_calibration_error(y_true, y_prob_bad)
    
    # Good calibration: probabilities close to actual labels
    y_prob_good = np.array([0.1, 0.1, 0.9, 0.9])
    ece_good = compute_calibration_error(y_true, y_prob_good)
    
    assert ece_good < ece_bad

def test_ensemble_proba_between_zero_and_one():
    """Verify all output probabilities are in [0, 1]."""
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.7, 0.3]])
            
    models = {"m1": MockModel(), "m2": MockModel()}
    weights = {"m1": 0.5, "m2": 0.5}
    ensemble = build_ensemble(models, weights)
    
    X = pd.DataFrame([[1], [2]])
    probs = ensemble_predict_proba(ensemble, X)
    
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
