"""
=== DATA SCIENTIST LEARNING NOTES ===

What this module does:
  Computes technical indicators, lag features, and calendar features for the ML model.

Why we use pandas-ta:
  It is a well-vetted, pandas-native library that implements over 130 technical indicators
  accurately. Implementing these manually (like RSI or Bollinger Bands) is error-prone 
  due to complex rolling updates and smoothing techniques.

What a FAANG interviewer might ask:
  Q: "How do you ensure there is no data leakage in your feature engineering?"
  A: I ensure no feature uses `.shift(-1)` or any forward-looking operation. 
     Furthermore, indicators like SMA(200) result in 199 NaNs at the start; I do 
     not backfill these because that would leak future mean-reverting information 
     into the training set.

  Q: "What is the importance of lag features in financial time-series?"
  A: Lag features create a 'sliding window' of historical state, allowing 
     stationary models like XGBoost to capture temporal dependencies (autocorrelation) 
     without needing many sequential layers like an RNN.

Data leakage risk in this module:
  Low, as long as we avoid any negative shifts. We must preserve NaNs at the beginning 
  of the series to reflect the 'warm-up' period of the indicators.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger
from typing import List
from pathlib import Path

from config import settings
from src.data.preprocessor import compute_returns


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Trend, Momentum, Volatility, and Volume indicators using pandas-ta.
    
    Args:
        df: Input DataFrame with OHLCV columns.
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added.
        
    DS Interview Note:
        VWAP (Volume Weighted Average Price) is calculated as sum(Price * Volume) / sum(Volume).
        It is a key indicator for institutional order flow because it reflects 
        the true 'average' price paid, weighted by liquid activity.
    """
    logger.info("Computing technical indicators...")
    
    # Trend: SMA and EMA
    for n in [5, 10, 20, 50, 200]:
        df[f"sma_{n}"] = ta.sma(df["close"], length=n)
    
    for n in [9, 21, 55]:
        df[f"ema_{n}"] = ta.ema(df["close"], length=n)
        
    # MACD (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    
    # Price vs SMA20
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    
    # Momentum: RSI, Stoch, Williams%R, ROC
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)
    
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["roc_10"] = ta.roc(df["close"], length=10)
    
    # Volatility: Bollinger Bands, ATR, Realised Vol
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bbands["BBU_20_2.0_2.0"]
    df["bb_lower"] = bbands["BBL_20_2.0_2.0"]
    df["bb_width"] = bbands["BBB_20_2.0_2.0"]
    df["bb_pct"] = bbands["BBP_20_2.0_2.0"]
    
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    # Realised Vol: rolling std of log returns
    if "log_return" not in df.columns:
        logger.warning("log_return not found. Computing on the fly for volatility.")
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
    df["realised_vol_20"] = df["log_return"].rolling(20).std()
    df["realised_vol_5"] = df["log_return"].rolling(5).std()
    
    # Volume: OBV, Z-Score, VWAP
    df["obv"] = ta.obv(df["close"], df["volume"])
    
    vol_mean_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20
    
    # VWAP requires an anchor (defaulting to the start of the series)
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["volume_change_pct"] = df["volume"].pct_change()
    
    return df


def compute_lag_features(df: pd.DataFrame, columns: list[str], lags: int = 5) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to lag.
        lags: Number of lags to create.
        
    Returns:
        pd.DataFrame: DataFrame with lagged columns.
        
    DS Interview Note:
        Autocorrelation is the correlation of a signal with a delayed copy 
        of itself. Lag features allow the model to detect persistence 
        (momentum) or mean-reversion patterns directly.
    """
    logger.info(f"Computing {lags} lags for: {columns}")
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping lags.")
            continue
            
        for i in range(1, lags + 1):
            df[f"{col}_lag_{i}"] = df[col].shift(i)
            
    return df


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the date index.
    
    Args:
        df: Input DataFrame with a DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with calendar features.
        
    DS Interview Note:
        Calendar effects like 'The Monday Effect' or 'Turn-of-the-month' 
        can be statistically significant. Encoding these as categorical 
        features helps models capture seasonality.
    """
    logger.info("Computing calendar features...")
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Index is not DatetimeIndex. Cannot compute calendar features.")
        return df
        
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full feature engineering pipeline.
    
    Args:
        df: Input price DataFrame.
        
    Returns:
        pd.DataFrame: Final feature matrix.
        
    DS Interview Note:
        This function represents a reproducible 'Feature Store' operation. 
        Consistent naming and order are critical for model inference.
    """
    logger.info("Building full feature matrix...")
    
    # 1. Indicators
    df = compute_indicators(df)
    
    # 2. Lags for specific columns
    lag_cols = ["close", "rsi_14", "log_return"]
    df = compute_lag_features(df, lag_cols, lags=5)
    
    # 3. Calendar
    df = compute_calendar_features(df)
    
    # Drop rows where all columns are empty (not needed here but good practice)
    # Actually, user says: do NOT fill NaNs, leave as NaN.
    
    logger.success(f"Built feature matrix with {len(df.columns)} columns.")
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
        
        # 2. Compute returns (needed for realised vol and lag features)
        df = compute_returns(df)
        
        # 3. Build matrix
        feature_df = build_feature_matrix(df)
        
        # 4. Results
        print(f"\nFeature Matrix Shape: {feature_df.shape}")
        print("\nColumns:")
        print(feature_df.columns.tolist())
        
        # Find first row with all non-NaN values (usually after SMA200)
        first_valid_idx = feature_df["sma_200"].first_valid_index()
        if first_valid_idx:
            print("\nFirst 3 valid rows (around SMA200 warm-up):")
            print(feature_df.loc[first_valid_idx:].head(3))
        
        # 5. Proof of no lookahead (SMA200 assertion)
        first_199 = feature_df["sma_200"].iloc[:199]
        assert first_199.isna().all(), "Lookahead Detected: SMA_200 should be NaN for first 199 rows."
        logger.success("Lookahead check passed: SMA_200 preserves correct warm-up period.")
