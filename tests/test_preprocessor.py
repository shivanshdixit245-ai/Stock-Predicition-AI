import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocessor import fit_scaler, fill_missing, compute_returns
from config import settings
import joblib

def test_no_future_data_in_scaler(tmp_path):
    """Ensure scaler fitted on train doesn't change when transformed on test."""
    # Mock settings.model_dir for test
    original_model_dir = settings.model_dir
    settings.model_dir = tmp_path / "models"
    
    train_data = pd.DataFrame({"feat": [10, 20, 30, 40, 50]})
    test_data = pd.DataFrame({"feat": [100, 200]})
    
    scaler = fit_scaler(train_data, "TEST")
    
    # Scale test data
    test_scaled = scaler.transform(test_data)
    
    # Check that the scaler still reflects train data stats
    assert pytest.approx(scaler.center_[0]) == 30.0  # Median
    
    # Restore settings
    settings.model_dir = original_model_dir

def test_forward_fill_max_2_days():
    """Verify max_gap=2 limit in fill_missing."""
    df = pd.DataFrame({
        "price": [100, np.nan, np.nan, np.nan, 105]  # Gap of 3
    })
    
    cleaned = fill_missing(df, max_gap=2)
    
    # Should drop the 3 NaNs. The 2 valid prices remain.
    assert len(cleaned) == 2
    assert list(cleaned["price"]) == [100.0, 105.0]

def test_log_return_no_lookahead():
    """Ensure log_return at time t uses only price at t and t-1."""
    df = pd.DataFrame({
        "close": [100, 102, 101, 105]
    })
    
    returns_df = compute_returns(df)
    
    expected_log_return_1 = np.log(102/100)
    assert pytest.approx(returns_df["log_return"].iloc[0]) == expected_log_return_1
    
    # Ensure shape is input_len - 1
    assert len(returns_df) == 3
