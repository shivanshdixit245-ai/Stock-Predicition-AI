import pytest
import pandas as pd
import numpy as np
from src.features.technical import build_feature_matrix
from src.data.preprocessor import compute_returns

def make_sample_data(n: int = 300) -> pd.DataFrame:
    """Helper to create dummy OHLCV data."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": np.random.randn(n) + 100,
        "high": np.random.randn(n) + 102,
        "low": np.random.randn(n) + 98,
        "close": np.random.randn(n) + 100,
        "volume": np.random.randint(1000, 5000, n)
    }, index=dates)
    return df

def test_no_lookahead_in_indicators():
    """Ensure SMA(200) has 199 lead NaNs."""
    df = make_sample_data(300)
    df = compute_returns(df)
    features = build_feature_matrix(df)
    
    # SMA200 needs 200 data points. First 199 must be NaN.
    assert features["sma_200"].iloc[:199].isna().all()
    assert not np.isnan(features["sma_200"].iloc[205])

def test_lag_column_naming_convention():
    """Verify {col}_lag_N naming."""
    df = make_sample_data(50)
    df = compute_returns(df)
    features = build_feature_matrix(df)
    
    assert "close_lag_1" in features.columns
    assert "close_lag_5" in features.columns
    assert "rsi_14_lag_3" in features.columns
    
    # Check that lag 1 matches previous close
    # Row 10's lag_1 should be Row 9's close
    val_9 = features["close"].iloc[9]
    lag_10 = features["close_lag_1"].iloc[10]
    assert val_9 == lag_10

def test_build_feature_matrix_minimum_column_count():
    """Ensure all requested groups are present."""
    df = make_sample_data(250)
    df = compute_returns(df)
    features = build_feature_matrix(df)
    
    # We expect roughly:
    # 5 SMAs, 3 EMAs, 3 MACD, 1 SMA_VS, 2 RSIs, 2 Stoch, 1 WillR, 1 ROC (18)
    # 4 BBands, 1 ATR, 2 RealisedVol (7)
    # 4 Volume items (4)
    # 3 Calendar (3)
    # 5 lags for 3 cols (15)
    # Plus original OHLCV (5) + returns (2)
    # Total should be > 50
    assert len(features.columns) >= 50
    
    # Specific checks
    expected_cols = ["sma_200", "macd_hist", "bb_pct", "volume_zscore_20", "day_of_week"]
    for col in expected_cols:
        assert col in features.columns
