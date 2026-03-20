"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Detects underlying 'market regimes' (Bull, Bear, Sideways) using a 
  Hidden Markov Model (HMM). It uses log returns and realized volatility 
  to infer the hidden state of the market.

Why we use HMM:
  Markets are non-stationary; the same strategy that works in a Bull market 
  often fails in a Bear market. HMMs are unsupervised models that assume 
  the market switches between discrete 'hidden' states. By identifying the 
  current state, we can conditionally adjust our strategy or add the regime 
  as a critical feature to the downstream XGBoost model.

What a FAANG interviewer might ask:
  Q: "Why use GaussianHMM for regime detection instead of K-Means?"
  A: K-Means treats each day as independent. HMM incorporates temporal 
     dependency through its transition matrix (the probability of staying 
     in the same state vs. switching). In financial markets, regimes are 
     persistent (market states 'cluster' in time), making HMM far more 
     appropriate than simple clustering.

  Q: "How do you handle 'state switching' in live inference?"
  A: HMM labels are arbitrary (e.g., State 0, 1, 2). To make them useful, 
     we must map them to economic meaning. We label the regime with the 
     highest mean return as 'Bull' and the lowest as 'Bear'.

Data leakage risk in this module:
  High. If you fit the HMM on the entire dataset, the state transitions 
  of the training period will be influenced by the future volatility in 
  the test set. We must fit ONLY on training data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
from typing import Tuple, Dict

from config import settings
from src.data.preprocessor import compute_returns, get_train_val_split


def fit_hmm(df: pd.DataFrame, n_regimes: int = 3) -> GaussianHMM:
    """
    Fit a Gaussian HMM on log_return and realised_vol_20.
    
    Args:
        df: Input DataFrame containing features.
        n_regimes: Number of hidden states to detect.
        
    Returns:
        GaussianHMM: Fitted HMM model.
        
    DS Interview Note:
        The choice of 3 regimes is standard: high-mean/low-vol (Bull), 
        low-mean/high-vol (Bear), and low-mean/low-vol (Sideways).
    """
    logger.info(f"Fitting GaussianHMM with {n_regimes} regimes...")
    
    # Requirement: use ONLY log_return and realised_vol_20
    # If realised_vol_20 isn't there, compute it (though it should be from Technical module)
    if "realised_vol_20" not in df.columns:
        logger.warning("realised_vol_20 not found. Computing simple rolling std.")
        df["realised_vol_20"] = df["log_return"].rolling(20).std()
        
    # Drop rows with NaNs in features
    features = ["log_return", "realised_vol_20"]
    X = df[features].dropna()
    
    # Fit model
    model = GaussianHMM(
        n_components=n_regimes, 
        covariance_type="diag", 
        n_iter=100, 
        random_state=settings.random_state
    )
    model.fit(X)
    
    if model.monitor_.iter < 100:
        logger.success(f"HMM converged in {model.monitor_.iter} iterations.")
    else:
        logger.warning("HMM reached max iterations without full convergence.")
        
    return model


def predict_regime(df: pd.DataFrame, model: GaussianHMM) -> pd.Series:
    """
    Predict regimes for a dataset using a fitted HMM.
    
    Args:
        df: Input DataFrame.
        model: Fitted GaussianHMM.
        
    Returns:
        pd.Series: Predicted regime labels (preserved index).
    """
    logger.info("Predicting regimes...")
    
    features = ["log_return", "realised_vol_20"]
    # We must handle NaNs (e.g. at the start of vol window)
    mask = ~df[features].isna().any(axis=1)
    X = df.loc[mask, features]
    
    preds = model.predict(X)
    
    # Map back to original index
    regime_series = pd.Series(index=df.index, dtype="Int64")
    regime_series.loc[mask] = preds
    
    # Fill leading NaNs with the first available prediction
    regime_series = regime_series.bfill().ffill()
    
    return regime_series


def label_regimes(regime_series: pd.Series, df: pd.DataFrame) -> Dict[int, str]:
    """
    Map HMM state integers to economic labels: Bull, Bear, Sideways.
    Highest mean return = Bull, Lowest = Bear, Middle = Sideways.
    
    Args:
        regime_series: Series of state integers.
        df: DataFrame with log_return column.
        
    Returns:
        Dict[int, str]: Mapping from int to label.
    """
    logger.info("Automatic labeling of regimes based on mean returns...")
    
    # Calculate mean return per state
    temp_df = pd.DataFrame({
        "regime": regime_series,
        "returns": df["log_return"]
    }).dropna()
    
    means = temp_df.groupby("regime")["returns"].mean().sort_values()
    
    # Map based on rank of means
    mapping = {}
    if len(means) >= 3:
        mapping[means.index[0]] = "bear"
        mapping[means.index[-1]] = "bull"
        # Everything in between is sideways
        for i in range(1, len(means) - 1):
            mapping[means.index[i]] = "sideways"
    elif len(means) == 2:
        mapping[means.index[0]] = "bear"
        mapping[means.index[1]] = "bull"
    elif len(means) == 1:
        mapping[means.index[0]] = "sideways"
    
    # Log the means for verification in demo
    for state, label in mapping.items():
        logger.info(f"Regime {state} labelled as {label.upper()} (Mean Return: {means.loc[state]:.6f})")
        
    return mapping


def save_regime_model(model: GaussianHMM, ticker: str) -> None:
    """Save HMM model to disk."""
    save_path = settings.model_dir / ticker / "hmm_regime.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"HMM model saved to {save_path}")


def load_regime_model(ticker: str) -> GaussianHMM:
    """Load HMM model from disk."""
    load_path = settings.model_dir / ticker / "hmm_regime.pkl"
    if not load_path.exists():
        raise FileNotFoundError(f"No HMM model found for {ticker} at {load_path}")
    model = joblib.load(load_path)
    logger.info(f"HMM model loaded from {load_path}")
    return model


def add_regime_features(df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    """
    Add regime features to the DataFrame: label and one-hot encoded.
    
    Args:
        df: Input DataFrame.
        regime_series: Series of regime labels (strings or ints).
        
    Returns:
        pd.DataFrame: DataFrame with regime columns.
    """
    logger.info("Adding regime features (label + one-hot)...")
    
    # If those are ints, we need to map them to labels first for consistent one-hot column names
    # (Assuming label_regimes mapping is available or passed)
    # Actually, the user says add: regime (int), regime_bull, regime_bear, regime_sideways
    
    # For the purpose of this module's requirement, we assume regime_series is already converted 
    # to bull/bear/sideways strings or we do it here if we have the means.
    # We will compute the mapping once and apply.
    
    mapping = label_regimes(regime_series, df)
    df["regime_label"] = regime_series.map(mapping)
    df["regime"] = regime_series.astype(int)
    
    # One-hot encoding
    df["regime_bull"] = (df["regime_label"] == "bull").astype(int)
    df["regime_bear"] = (df["regime_label"] == "bear").astype(int)
    df["regime_sideways"] = (df["regime_label"] == "sideways").astype(int)
    
    # Drop temp label
    df = df.drop(columns=["regime_label"])
    
    return df


if __name__ == "__main__":
    # Demo block
    TICKER = "AAPL"
    DATA_PATH = settings.data_prices_dir / f"{TICKER}_2023-01-01_2023-12-31.parquet"
    
    if not DATA_PATH.exists():
        logger.error(f"Data not found: {DATA_PATH}. Run loader.py first.")
    else:
        # 1. Load data
        df = pd.read_parquet(DATA_PATH)
        
        # 2. Compute returns and volatility
        df = compute_returns(df)
        df["realised_vol_20"] = df["log_return"].rolling(20).std()
        df = df.dropna(subset=["realised_vol_20"])
        
        # 3. Split into train (80%) and val (20%)
        train_df, val_df = get_train_val_split(df, val_size=0.2)
        
        # 4. Fit HMM on training data ONLY
        model = fit_hmm(train_df)
        save_regime_model(model, TICKER)
        
        # 5. Predict on full dataset
        full_regimes = predict_regime(df, model)
        
        # 6. Label and add features
        df_with_regimes = add_regime_features(df, full_regimes)
        
        # 7. Print distribution
        counts = df_with_regimes["regime"].value_counts(normalize=True).sort_index()
        print("\nRegime Distribution:")
        print(counts)
        
        # 8. ASCII Chart (Simple representation)
        print("\nRegime Timeline (last 50 days):")
        # 2=Bull(^), 1=Sideways(-), 0=Bear(v) - after mapping it might differ
        mapping = label_regimes(full_regimes, df)
        sym_map = {"bull": "^", "sideways": "-", "bear": "v"}
        
        timeline = [sym_map[mapping[r]] for r in full_regimes.tail(50)]
        print("".join(timeline))
        print("Legend: ^ Bull, - Sideways, v Bear")
        
        logger.success("Regime detection demo completed successfully.")
