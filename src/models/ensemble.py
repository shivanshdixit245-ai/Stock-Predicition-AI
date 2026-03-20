"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Combines multiple models into a weighted ensemble and calibrates 
  their output probabilities. It also implements Expected Calibration 
  Error (ECE) as a metric.

Why we use Weighted Ensembling:
  Not all models are equal. By weighting models by their validation 
  F1 score, we ensure that the strongest learner has the most 
  influence on the final signal, while still benefiting from the 
  diversity of the ensemble.

Why we Calibrate (Platt Scaling):
  Tree-based models (XGBoost, LightGBM) are notorious for producing 
  poorly calibrated probabilities. They tend to push predictions 
  towards 0 and 1, or stay away from the extremes. Calibration 
  (using Sigmoid/Platt scaling) maps the model's 'scores' to 
  actual empirical frequencies, making the probabilities 
  mathematically meaningful.

What a FAANG interviewer might ask:
  Q: "How do you measure if your probabilities are 'reliable'?"
  A: I use the Expected Calibration Error (ECE) and Reliability Diagrams. 
     ECE measures the average gap between predicted confidence and actual 
     accuracy across several bins. A perfectly calibrated model has ECE = 0.

  Q: "Why use 'prefit' in CalibratedClassifierCV?"
  A: Because we already trained our models using walk-forward validation. 
     If we didn't use 'prefit', sklearn would try to cross-validate 
     or re-train the base models internally, which would break our 
     strict time-series temporal ordering.

Data leakage risk in this module:
  Medium. Calibration must ONLY happen on validation data that the 
  model has NOT seen during training. Calibrating on training data 
  would result in 'over-calibration', where the model looks perfectly 
  reliable on its own training set but fails on new data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from pathlib import Path
from typing import Dict, Any, Tuple

from config import settings


def build_ensemble(models: dict, weights: dict) -> dict:
    """
    Stitch models together with normalized weights based on F1 scores.
    
    Args:
        models: Dict of model objects {name: model}.
        weights: Dict of F1 scores or weights {name: weight}.
        
    Returns:
        dict: Ensemble info containing 'models' and 'normalized_weights'.
        
    DS Interview Note:
        Weighting by F1 is better than equal weighting because it 
        penalizes models that have high variance or poor precision/recall 
        balance in historical backtests.
    """
    logger.info("Building weighted ensemble...")
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        logger.warning("All weights are zero. Falling back to equal weights.")
        normalized_weights = {k: 1.0/len(weights) for k in weights.keys()}
    else:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    logger.info(f"Ensemble weights: {normalized_weights}")
    
    return {
        "models": models,
        "weights": normalized_weights
    }


def ensemble_predict_proba(ensemble: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Compute weighted average of probabilities from all ensemble models.
    """
    models = ensemble["models"]
    weights = ensemble["weights"]
    
    final_probs = None
    
    for name, model in models.items():
        # Handle if model is calibrated or raw
        probs = model.predict_proba(X)[:, 1]
        weighted_probs = probs * weights[name]
        
        if final_probs is None:
            final_probs = weighted_probs
        else:
            final_probs += weighted_probs
            
    return final_probs


def calibrate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
    """
    Apply Platt Scaling (Sigmoid) to a pre-trained model using validation data.
    
    DS Interview Note:
        cv='prefit' is essential here. It tells sklearn the model is 
        already trained and to only fit the sigmoid mapping. 
        This prevents data leakage and preserves the work-forward setup.
    """
    logger.info(f"Calibrating model {type(model).__name__}...")
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    
    # Ensure y_val is int
    y_val = y_val.astype(int)
    calibrated.fit(X_val, y_val)
    
    return calibrated


def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE = sum( (bin_count / total) * abs(bin_accuracy - bin_confidence) )
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_count = len(y_true)
    
    for lower, upper in zip(bin_lowers, bin_uppers):
        # Indices of predictions in this bin
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, ticker: str) -> None:
    """
    Visualize calibration and save as artifact.
    """
    logger.info(f"Plotting calibration curve for {ticker}...")
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = compute_calibration_error(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"Model (ECE={ece:.4f})")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Reliability Diagram: {ticker}")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    save_path = Path(f"reports/calibration_curve_{ticker}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.success(f"Calibration curve saved to {save_path}")


def save_ensemble(ensemble: dict, ticker: str) -> None:
    """Save calibrated ensemble models and weights."""
    model_path = settings.model_dir / ticker / "calibrated_ensemble.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving ensemble to {model_path}...")
    joblib.dump(ensemble, model_path)


def load_ensemble(ticker: str) -> dict:
    """Load calibrated ensemble."""
    model_path = settings.model_dir / ticker / "calibrated_ensemble.pkl"
    if not model_path.exists():
        logger.error(f"Ensemble file not found: {model_path}")
        raise FileNotFoundError(f"No ensemble found for {ticker}")
        
    return joblib.load(model_path)


if __name__ == "__main__":
    # Demo Block
    from src.models.trainer import build_target, get_feature_columns
    from src.data.loader import load_stock_data
    from src.data.preprocessor import validate_ohlc, fill_missing, compute_returns
    from src.features.technical import build_feature_matrix
    
    TICKER = "AAPL"
    logger.info(f"--- Running Ensemble & Calibration Demo for {TICKER} ---")
    
    # 1. Load best models and data
    model_dir = settings.model_dir / TICKER
    try:
        xgb = joblib.load(model_dir / "xgb_model.pkl")
        lgb = joblib.load(model_dir / "lgb_model.pkl")
        lr = joblib.load(model_dir / "lr_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
    except FileNotFoundError:
        logger.error("Best models not found. Run trainer.py first.")
        exit(1)

    # 2. Prepare validation data
    from src.features.sentiment import build_sentiment_features
    from src.models.regime import fit_hmm, predict_regime, add_regime_features
    from src.data.loader import load_news

    prices = load_stock_data(TICKER, "2022-01-01", "2023-12-31")
    df = validate_ohlc(prices)
    df = fill_missing(df)
    df = compute_returns(df)
    
    # Technical features
    df = build_feature_matrix(df)
    
    # Sentiment features
    news = load_news(TICKER, "2022-01-01", "2023-12-31")
    sentiment = build_sentiment_features(news, prices, TICKER)
    
    # Regime features
    hmm = fit_hmm(df)
    regime_series = predict_regime(df, hmm)
    df = add_regime_features(df, regime_series)
    
    # Merge
    df = pd.concat([df, sentiment], axis=1).dropna()
    df = build_target(df)
    
    # Ensure feature order matches what the model saw
    features = scaler.feature_names_in_.tolist()
    X = scaler.transform(df[features])
    y = df["target"]
    
    # Split validation (calibration) set
    X_val, y_val = X[-60:], y[-60:] # Small set for demo
    
    # 3. Initial Ensemble (Raw)
    # Weights from a dummy f1 dict for demo (normally from CV results)
    weights = {"xgb": 0.62, "lgb": 0.58, "lr": 0.55}
    raw_ensemble = build_ensemble({"xgb": xgb, "lgb": lgb, "lr": lr}, weights)
    raw_probs = ensemble_predict_proba(raw_ensemble, X_val)
    ece_before = compute_calibration_error(y_val, raw_probs)
    
    # 4. Calibrate each model
    calibrated_models = {
        "xgb": calibrate_model(xgb, X_val, y_val),
        "lgb": calibrate_model(lgb, X_val, y_val),
        "lr": calibrate_model(lr, X_val, y_val)
    }
    
    # 5. Build Calibrated Ensemble
    cal_ensemble = build_ensemble(calibrated_models, weights)
    cal_probs = ensemble_predict_proba(cal_ensemble, X_val)
    ece_after = compute_calibration_error(y_val, cal_probs)
    
    # 6. Report & Plot
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"Initial ECE: {ece_before:.4f}")
    print(f"Calibrated ECE: {ece_after:.4f}")
    print(f"Improvement: {ece_before - ece_after:.4f}")
    print("-"*50)
    
    plot_calibration_curve(y_val, cal_probs, TICKER)
    save_ensemble(cal_ensemble, TICKER)
    
    # Log to MLflow if current run exists
    if mlflow.active_run():
        mlflow.log_metrics({
            "ece_before_calibration": ece_before,
            "ece_after_calibration": ece_after
        })
    
    logger.success("Ensemble & Calibration demo completed.")
