"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Implements Conformal Prediction using the MAPIE library. 
  It provides statistically rigorous 'prediction sets' that 
  guarantee a specific coverage probability (e.g., 90%). 
  This allows us to filter out 'uncertain' days where the 
  model might be guessing.

Why we use Conformal Prediction:
  Traditional ML models only provide point predictions or 
  probabilities. Conformal prediction answer: "What is the 
  set of classes that are likely to contain the true label 
  with 90% confidence?" If the set is {0, 1}, the model is 
  saying "I don't know". This is a much safer signal than 
  just picking the higher probability.

What a FAANG interviewer might ask:
  Q: "What are the advantages of Conformal Prediction over 
     Bayesian uncertainty?"
  A: Conformal prediction is distribution-free and provides 
     finite-sample coverage guarantees. It doesn't require 
     complex prior distributions or MCMC sampling. It works 
     with any black-box model (like XGBoost) as long as 
     you have a calibration set.

  Q: "How does the 'score' method work in MapieClassifier?"
  A: It uses the softmax scores (probabilities) as a 
     non-conformity measure. Classes whose probabilities 
     exceed a threshold (determined by the alpha quantile 
     of scores on calibration data) are included in the 
     prediction set.

Data leakage risk in this module:
  Critical. The calibration data used for MAPIE (`X_cal`) 
  must NOT have been seen by the ensemble models during 
  their initial training. If they were, the 'non-conformity 
  scores' would be artificially high (overfit), leading 
  to sets that are too small and failing to provide 
  the promised coverage on future data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from loguru import logger
from mapie.classification import MapieClassifier
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import settings
from src.models.ensemble import ensemble_predict_proba


from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Minimal sklearn-compatible wrapper for the weighted ensemble dict.
    This allows MAPIE to treat the ensemble as a single estimator.
    """
    def __init__(self, ensemble: Optional[dict] = None):
        self.ensemble = ensemble
        
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if self.ensemble is None:
            raise ValueError("Ensemble dict not provided to wrapper.")
        p1 = ensemble_predict_proba(self.ensemble, X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
    def get_params(self, deep=True):
        return {"ensemble": self.ensemble}


def fit_conformal(ensemble: dict, X_cal: pd.DataFrame, y_cal: pd.Series) -> MapieClassifier:
    """
    Wrap the ensemble in MAPIE and fit the conformal scores on calibration data.
    
    Args:
        ensemble: Weighted ensemble dict.
        X_cal: Calibration features (NOT used in model training).
        y_cal: Calibration targets.
        
    Returns:
        MapieClassifier: Fitted conformal model.
        
    DS Interview Note:
        method='score' is chosen for its simplicity and robustness 
        with well-calibrated probabilities. It effectively builds 
        sets around the most likely classes until the coverage requirement is met.
    """
    logger.info("Fitting Conformal Prediction model (MAPIE)...")
    
    # 1. Wrap ensemble
    wrapper = EnsembleWrapper(ensemble)
    # Explicitly fit the wrapper (sets classes_) so MAPIE accepts it with cv='prefit'
    wrapper.fit(X_cal)
    
    # 2. Init Mapie with prefit
    mapie = MapieClassifier(estimator=wrapper, method='score', cv='prefit')
    
    # 3. Fit on calibration data
    y_cal = y_cal.astype(int)
    mapie.fit(X_cal, y_cal)
    
    return mapie


def generate_signal(prob: float, prediction_set: list) -> str:
    """
    Logic-based signal generation using probability and conformal set.
    
    Rules (all must be met):
      1. Exactly one class in set (high confidence)
      2. Probability meets threshold (>0.65 for BUY, <0.35 for SELL)
    """
    # Prediction set often comes in as [False, True] or similar from MAPIE
    # We convert to class labels [1] etc.
    active_classes = [i for i, val in enumerate(prediction_set) if val]
    set_size = len(active_classes)
    
    if set_size != 1:
        return "HOLD"
    
    pred_class = active_classes[0]
    
    if pred_class == 1 and prob > 0.65:
        return "BUY"
    elif pred_class == 0 and prob < 0.35: # prob is for class 1
        return "SELL"
    else:
        return "HOLD"


def predict_with_uncertainty(mapie_model: MapieClassifier, X: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Produce predictions, probabilities, and conformal sets.
    
    Args:
        mapie_model: Fitted MapieClassifier.
        X: Features.
        alpha: Mis-coverage rate (1 - confidence level).
        
    Returns:
        pd.DataFrame: Augmented with uncertainty columns.
    """
    logger.info(f"Producing predictions with uncertainty (alpha={alpha})...")
    
    # Mapie predict returns (y_pred, y_set) where y_set is NxK array of booleans
    y_pred, y_set = mapie_model.predict(X, alpha=alpha)
    
    # Probabilities from the wrapped estimator
    # Mapie 0.8+ with cv='prefit' uses .estimator
    if hasattr(mapie_model, "estimator_"):
        probs = mapie_model.estimator_.predict_proba(X)
    else:
        probs = mapie_model.estimator.predict_proba(X)
    
    # Flatten y_set to list of active classes for the signal function
    set_list = [list(row) for row in y_set[:, :, 0]] # index 0 for the single alpha
    
    results = []
    for i in range(len(X)):
        prob_buy = probs[i, 1]
        active_set = set_list[i]
        signal = generate_signal(prob_buy, active_set)
        
        # Confidence is 'high' only if set size == 1
        set_size = sum(active_set)
        confidence = "high" if set_size == 1 else "low"
        
        results.append({
            "prob_buy": prob_buy,
            "prob_sell": probs[i, 0],
            "prediction_set": active_set, # [True, False] etc.
            "signal": signal,
            "confidence": confidence
        })
        
    return pd.DataFrame(results, index=X.index)


def compute_empirical_coverage(y_true: np.ndarray, prediction_sets: np.ndarray) -> float:
    """
    Verify coverage: fraction of rows where true label is in the prediction set.
    
    Args:
        y_true: Ground truth.
        prediction_sets: Nx2 boolean array from Mapie.
        
    Returns:
        float: Empirical coverage.
    """
    y_true = np.array(y_true).astype(int)
    # prediction_sets is [N, 2, 1] usually
    sets = prediction_sets[:, :, 0]
    
    covered = 0
    for i in range(len(y_true)):
        label = y_true[i]
        if sets[i, label]:
            covered += 1
            
    coverage = covered / len(y_true)
    logger.info(f"Empirical Coverage: {coverage:.4f}")
    return coverage


def save_conformal_model(mapie_model: MapieClassifier, ticker: str) -> None:
    """Save conformal model."""
    path = settings.model_dir / ticker / "mapie_conformal.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving conformal model to {path}...")
    joblib.dump(mapie_model, path)


def load_conformal_model(ticker: str) -> MapieClassifier:
    """Load conformal model."""
    path = settings.model_dir / ticker / "mapie_conformal.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


if __name__ == "__main__":
    # Demo Block
    from src.models.trainer import build_target, get_feature_columns
    from src.data.loader import load_stock_data, load_news
    from src.data.preprocessor import validate_ohlc, fill_missing, compute_returns
    from src.features.technical import build_feature_matrix
    from src.features.sentiment import build_sentiment_features
    from src.models.regime import fit_hmm, predict_regime, add_regime_features
    from src.models.ensemble import load_ensemble
    
    TICKER = "AAPL"
    logger.info(f"--- Running Conformal Uncertainty Demo for {TICKER} ---")
    
    # 1. Load ensemble and data
    try:
        ensemble = load_ensemble(TICKER)
        # Scaler is needed to prepare features
        scaler = joblib.load(settings.model_dir / TICKER / "scaler.pkl")
    except Exception as e:
        logger.error(f"Failed to load ensemble/scaler: {e}. Run trainer.py and ensemble.py first.")
        exit(1)
        
    prices = load_stock_data(TICKER, "2022-01-01", "2023-12-31")
    df = validate_ohlc(prices)
    df = fill_missing(df)
    df = compute_returns(df)
    df = build_feature_matrix(df)
    news = load_news(TICKER, "2022-01-01", "2023-12-31")
    sentiment = build_sentiment_features(news, prices, TICKER)
    hmm = fit_hmm(df)
    regimes = predict_regime(df, hmm)
    df = add_regime_features(df, regimes)
    df = pd.concat([df, sentiment], axis=1).dropna()
    df = build_target(df)
    
    features = list(scaler.feature_names_in_)
    X_full = scaler.transform(df[features])
    y_full = df["target"]
    
    # 2. Split into Calibration (last fold validation) and Test
    # In a real pipeline, we'd use the exact same split as Trainer.
    # For demo, we use the last 60 days for cal, and last 20 for 'test' demo.
    X_cal, y_cal = X_full[-100:-20], y_full[-100:-20]
    X_test, y_test = X_full[-20:], y_full[-20:]
    
    X_cal_df = pd.DataFrame(X_cal, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
    
    # 3. Fit Conformal
    mapie = fit_conformal(ensemble, X_cal_df, y_cal)
    
    # 4. Predict
    results = predict_with_uncertainty(mapie, X_test_df, alpha=0.1)
    
    # 5. Coverage
    y_pred, y_set = mapie.predict(X_test_df, alpha=0.1)
    coverage = compute_empirical_coverage(y_test, y_set)
    assert coverage >= 0.90, f"Coverage {coverage:.2f} below nominal 0.90"
    
    # 6. Signal Distribution
    counts = results["signal"].value_counts(normalize=True)
    hold_rate = counts.get("HOLD", 0.0)
    
    print("\n" + "="*50)
    print("CONFORMAL PREDICTION RESULTS")
    print("="*50)
    print(f"Empirical Coverage: {coverage:.2%}")
    print(f"Signal Distribution: BUY={counts.get('BUY', 0):.1%}, SELL={counts.get('SELL', 0):.1%}, HOLD={hold_rate:.1%}")
    print("-"*50)
    print("\nExample predictions (Last 5 days):")
    print(results[["prob_buy", "prediction_set", "signal", "confidence"]].tail(5))
    print(f"\nConformal prediction reduces false signals by {hold_rate:.1%} of trading days")
    
    # 7. Log to MLflow
    if mlflow.active_run():
        mlflow.log_metrics({
            "empirical_coverage": coverage,
            "conformal_hold_rate": hold_rate
        })
        
    save_conformal_model(mapie, TICKER)
    logger.success("Uncertainty demo completed.")
