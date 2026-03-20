import pytest
import pandas as pd
import numpy as np
from src.models.regime import fit_hmm, predict_regime, add_regime_features, label_regimes
from src.data.preprocessor import get_train_val_split

def make_sample_data(n: int = 500) -> pd.DataFrame:
    """Create synthetic data with two distinct regimes."""
    dates = pd.date_range("2020-01-01", periods=n)
    
    # State 0: High return, low vol
    # State 1: Low return, high vol
    # State 2: Zero return, low vol
    
    # First half: Bullish
    r1 = np.random.normal(0.01, 0.01, n // 2)
    v1 = np.random.uniform(0.005, 0.01, n // 2)
    
    # Second half: Bearish
    r2 = np.random.normal(-0.01, 0.03, n // 2)
    v2 = np.random.uniform(0.02, 0.05, n // 2)
    
    df = pd.DataFrame({
        "log_return": np.concatenate([r1, r2]),
        "realised_vol_20": np.concatenate([v1, v2])
    }, index=dates)
    
    return df

def test_hmm_fit_on_train_only():
    """Verify model can be fitted and used for out-of-sample prediction."""
    df = make_sample_data()
    train_df, val_df = get_train_val_split(df, val_size=0.2)
    
    model = fit_hmm(train_df)
    
    # Predict on validation data
    val_regimes = predict_regime(val_df, model)
    
    assert len(val_regimes) == len(val_df)
    assert not val_regimes.isna().any()

def test_regime_labels_correct():
    """Verify bull regime has highest mean return."""
    df = make_sample_data()
    model = fit_hmm(df)
    regimes = predict_regime(df, model)
    
    mapping = label_regimes(regimes, df)
    
    # Check returns per label
    temp_df = pd.DataFrame({"r": df["log_return"], "l": regimes.map(mapping)})
    means = temp_df.groupby("l")["r"].mean()
    
    if "bull" in means and "bear" in means:
        assert means["bull"] > means["bear"]
    if "bull" in means and "sideways" in means:
        assert means["bull"] >= means["sideways"]

def test_one_hot_encoding_sum_to_one():
    """Verify one-hot regime encoding logic."""
    df = make_sample_data(120)
    # Create 3 distinct regimes for the test
    regimes = pd.Series([0]*40 + [1]*40 + [2]*40, index=df.index)
    
    df_feat = add_regime_features(df.copy(), regimes)
    
    # Sum of bull, bear, sideways should be 1 for all rows
    sum_one_hot = df_feat["regime_bull"] + df_feat["regime_bear"] + df_feat["regime_sideways"]
    assert (sum_one_hot == 1).all()
