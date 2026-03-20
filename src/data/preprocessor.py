"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Cleans raw OHLCV data, computes log returns, and handles feature scaling.

Why we use RobustScaler:
  Financial data is often non-normal and contains significant outliers (fat tails).
  StandardScaler uses mean and variance, which are sensitive to outliers. 
  RobustScaler uses the median and the interquartile range (IQR), making it 
  much more stable for market data.

What a FAANG interviewer might ask:
  Q: "What is the most critical rule when scaling financial data for ML?"
  A: Data leakage prevention. You must fit the scaler ONLY on the training data 
     and then apply (transform) it to the validation and test data. Fitting on 
     the entire dataset allows information from the future to leak into the training process.
  
  Q: "Why perform OHLC validation?"
  A: Raw data from APIs can have errors where High is lower than Low or Open/Close. 
     Garbage in, garbage with. Sanity checks ensure the fundamental integrity of the time-series.

Data leakage risk in this module:
  High. The `get_train_val_split` must never shuffle, as time-series data has temporal 
  dependency. Splitting randomly would cause lookahead bias.
"""

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler
from pathlib import Path

from config import settings


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLC integrity: High >= Low, High >= Open, High >= Close, etc.
    
    Args:
        df: Input price DataFrame.
        
    Returns:
        pd.DataFrame: Validated DataFrame.
        
    DS Interview Note:
        In production, we log dropped rows to a data quality dashboard (like Evidently).
        Consistent failures indicate upstream API issues.
    """
    initial_len = len(df)
    
    # Define validation rules
    valid_mask = (
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    )
    
    df = df[valid_mask].copy()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to OHLC validation failure.")
    else:
        logger.info("OHLC validation passed for all rows.")
        
    return df


def fill_missing(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    Forward-fill missing values with a limit. Drop rows where gaps exceed the limit.
    
    Args:
        df: Input DataFrame.
        max_gap: Maximum consecutive NaNs to fill.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
        
    DS Interview Note:
        Forward-filling (ffill) is safe because it only uses past data. 
        However, large gaps indicate data ingestion failure. Dropping them 
        prevents the model from learning from "stale" filled values.
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values detected.")
        return df

    # Identify gaps
    mask = df.isnull().all(axis=1)
    # Group consecutive NaNs
    groups = (mask != mask.shift()).cumsum()
    gap_sizes = mask.groupby(groups).transform("sum")
    
    # Identify rows to drop (where gap size > max_gap and row was NaN)
    to_drop = mask & (gap_sizes > max_gap)
    
    # Fill only the rows that are NOT in the to_drop set
    df_filled = df.ffill(limit=max_gap)
    
    # Drop the rows that were part of a too-large gap
    initial_len = len(df)
    df_cleaned = df_filled[~to_drop].copy()
    
    # Also drop any remaining NaNs (e.g. at the very beginning)
    df_cleaned = df_cleaned.dropna()
    
    dropped = initial_len - len(df_cleaned)
    logger.info(f"Filled missing values (limit={max_gap}). Dropped {dropped} rows.")
    
    return df_cleaned


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns and simple returns.
    
    Args:
        df: Input price DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with 'return' and 'log_return' columns.
        
    DS Interview Note:
        Log returns are preferred in ML because they are additive over time and 
        usually more normally distributed than simple price changes.
    """
    # Simple percentage return
    df["return"] = df["close"].pct_change()
    
    # Log return: log(p_t / p_t-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Drop first row (NaN)
    df = df.dropna(subset=["log_return"])
    
    logger.info("Computed log returns and simple returns.")
    return df


def fit_scaler(X_train: pd.DataFrame, ticker: str) -> RobustScaler:
    """
    Fit a RobustScaler on training data and save it.
    
    Args:
        X_train: Training feature matrix.
        ticker: Symbol for identifying the model directory.
        
    Returns:
        RobustScaler: Fitted scaler object.
        
    DS Interview Note:
        We use RobustScaler because it uses the IQR, making it resistant to 
        the extreme volatility spikes common in stock markets.
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    
    # Save the scaler
    save_path = settings.model_dir / ticker / "scaler.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    
    logger.info(f"Fitted RobustScaler on training data and saved to {save_path}")
    return scaler


def transform(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """
    Transform data using a fitted scaler.
    
    Args:
        X: Feature matrix to scale.
        scaler: Fitted RobustScaler.
        
    Returns:
        pd.DataFrame: Scaled DataFrame with original index/columns.
        
    DS Interview Note:
        Always ensure the transform results are cast back to the original 
        DataFrame format to maintain feature names for model explainability (SHAP).
    """
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    logger.info("Applied feature scaling.")
    return X_scaled_df


def get_train_val_split(df: pd.DataFrame, val_size: float = 0.2) -> tuple:
    """
    Split data into training and validation sets while preserving time order.
    
    Args:
        df: Input DataFrame.
        val_size: Fraction of data to use for validation.
        
    Returns:
        tuple: (train_df, val_df)
        
    DS Interview Note:
        NEVER shuffle a time-series split. It destroys the autocorrelation 
        the model is trying to learn and leads to massive lookahead bias.
    """
    split_idx = int(len(df) * (1 - val_size))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    logger.info(f"Split data into train ({len(train_df)} rows) and val ({len(val_df)} rows).")
    return train_df, val_df


if __name__ == "__main__":
    # Demo block
    TICKER = "AAPL"
    RAW_DATA_PATH = settings.data_prices_dir / f"{TICKER}_2023-01-01_2023-12-31.parquet"
    
    if not RAW_DATA_PATH.exists():
        logger.error(f"Demo failed: {RAW_DATA_PATH} not found. Run loader.py first.")
    else:
        logger.info(f"--- Running Preprocessor Demo for {TICKER} ---")
        
        # 1. Load data
        df = pd.read_parquet(RAW_DATA_PATH)
        logger.info(f"Initial shape: {df.shape}")
        
        # 2. Validate OHLC
        df = validate_ohlc(df)
        logger.info(f"After validation shape: {df.shape}")
        
        # 3. Fill missing
        df = fill_missing(df)
        logger.info(f"After filling shape: {df.shape}")
        
        # 4. Compute returns
        df = compute_returns(df)
        logger.info(f"After returns shape: {df.shape}")
        
        # 5. Split data
        train_df, val_df = get_train_val_split(df)
        
        # 6. Fit and transform scaler (using only Price columns for demo)
        # In actual pipeline, this would be done on feature matrix
        price_cols = ["open", "high", "low", "close", "volume"]
        scaler = fit_scaler(train_df[price_cols], TICKER)
        
        train_scaled = transform(train_df[price_cols], scaler)
        val_scaled = transform(val_df[price_cols], scaler)
        
        logger.success("Preprocessor demo completed successfully.")
        print("\nTail of scaled validation data:")
        print(val_scaled.tail(3))
