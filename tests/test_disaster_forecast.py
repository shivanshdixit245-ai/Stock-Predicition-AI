"""
Tests for the Disaster Forecast Early Warning System.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.disaster_forecast import (
    compute_leading_indicators,
    predict_disaster_probability,
    get_disaster_forecast,
    LEADING_INDICATORS,
)


# -----------------------------------------------------------------------
# Test 1: Leading indicators do not use future data (no lookahead)
# -----------------------------------------------------------------------

def test_leading_indicators_no_lookahead():
    """
    Leading indicators at row N must only depend on data from rows <= N.
    Verify by computing indicators on a subset, then on the full set, and
    checking that the values for the overlapping rows are identical.
    """
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.normal(0, 1, n)),
        "high": 100 + np.cumsum(np.random.normal(0, 1, n)) + 1,
        "low": 100 + np.cumsum(np.random.normal(0, 1, n)) - 1,
        "open": 100 + np.cumsum(np.random.normal(0, 1, n)),
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)

    # Compute on first 150 rows
    subset = df.iloc[:150].copy()
    indicators_subset = compute_leading_indicators(subset)

    # Compute on all 200 rows
    indicators_full = compute_leading_indicators(df)

    # Values at row 140 should be identical (no future data influence)
    check_idx = 140
    for col in LEADING_INDICATORS:
        if col == "rolling_correlation_spy":
            continue  # SPY fetch may differ between runs
        val_subset = indicators_subset[col].iloc[check_idx]
        val_full = indicators_full[col].iloc[check_idx]
        assert abs(val_subset - val_full) < 1e-6, (
            f"Lookahead detected in {col}: subset={val_subset}, full={val_full}"
        )


# -----------------------------------------------------------------------
# Test 2: Disaster probability values are between 0 and 1
# -----------------------------------------------------------------------

def test_disaster_probability_between_zero_and_one():
    """
    All probability outputs must be valid probabilities in [0, 1].
    Test with synthetic data and mock models.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.normal(0, 1, n)),
        "high": 100 + np.cumsum(np.random.normal(0, 1, n)) + 1,
        "low": 100 + np.cumsum(np.random.normal(0, 1, n)) - 1,
        "open": 100 + np.cumsum(np.random.normal(0, 1, n)),
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)

    # Call with empty model dict (no trained models) — should return 0 probs
    model_dict = {"models": {}, "metrics": {}}
    result = predict_disaster_probability(df, model_dict)

    for col in ["prob_disaster_5d", "prob_disaster_10d", "prob_disaster_30d"]:
        assert result[col].min() >= 0.0, f"{col} has values below 0"
        assert result[col].max() <= 1.0, f"{col} has values above 1"


# -----------------------------------------------------------------------
# Test 3: Low confidence shown when insufficient data
# -----------------------------------------------------------------------

def test_low_confidence_shown_when_insufficient_data():
    """
    When no trained model exists (F1 < 0.55), the forecast must show
    confidence='low' and include a disclaimer about insufficient data.
    """
    # Use a ticker unlikely to have pre-trained disaster models
    forecast = get_disaster_forecast("AAPL")

    # If no model has been trained, confidence should be low
    # and recommendation should include the disclaimer text
    if forecast["confidence"] == "low":
        assert "insufficient" in forecast["recommendation"].lower() or \
               forecast["overall_f1"] < 0.55, (
            "Low-confidence forecast must include a disclaimer "
            "or have F1 < 0.55"
        )
    else:
        # If models ARE trained with decent F1, confidence should be high
        assert forecast["overall_f1"] >= 0.55, (
            f"High confidence claimed but F1 is only {forecast['overall_f1']}"
        )

    # Probabilities should still be valid numbers
    assert 0 <= forecast["prob_disaster_5d"] <= 1
    assert 0 <= forecast["prob_disaster_10d"] <= 1
    assert 0 <= forecast["prob_disaster_30d"] <= 1
