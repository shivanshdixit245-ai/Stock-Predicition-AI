"""
Tests for the Market Stress Detector & Circuit Breaker module.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.stress_detector import (
    apply_circuit_breaker,
    compute_market_stress_score,
    classify_stress_level,
    detect_black_swan,
)


# -----------------------------------------------------------------------
# Test 1: Circuit breaker overrides signals during extreme stress
# -----------------------------------------------------------------------

def test_circuit_breaker_overrides_signals_during_extreme_stress():
    """
    When stress >= 75  → every signal should become HOLD.
    When stress 50-74  → signal should get _CAUTION suffix.
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    signals = pd.Series(
        ["BUY", "SELL", "HOLD", "BUY", "SELL", "BUY"],
        index=dates,
    )
    stress = pd.Series(
        [80, 90, 75, 60, 55, 10],  # 3 extreme, 2 high, 1 normal
        index=dates,
        dtype=float,
    )

    result = apply_circuit_breaker(signals, stress)

    # Extreme (>=75): indices 0, 1, 2 → all HOLD
    assert result.iloc[0] == "HOLD"
    assert result.iloc[1] == "HOLD"
    assert result.iloc[2] == "HOLD"

    # High (50-74): indices 3, 4 → _CAUTION suffix
    assert result.iloc[3] == "BUY_CAUTION"
    assert result.iloc[4] == "SELL_CAUTION"

    # Normal (<50): index 5 → unchanged
    assert result.iloc[5] == "BUY"


# -----------------------------------------------------------------------
# Test 2: Normal conditions leave signals unchanged
# -----------------------------------------------------------------------

def test_normal_conditions_signals_unchanged():
    """All signals should pass through untouched when stress < 50."""
    dates = pd.date_range("2024-06-01", periods=5, freq="B")
    signals = pd.Series(
        ["BUY", "SELL", "HOLD", "BUY", "SELL"],
        index=dates,
    )
    stress = pd.Series(
        [5, 15, 30, 45, 10],
        index=dates,
        dtype=float,
    )

    result = apply_circuit_breaker(signals, stress)

    assert list(result) == ["BUY", "SELL", "HOLD", "BUY", "SELL"]


# -----------------------------------------------------------------------
# Test 3: Black swan detects COVID crash in real data
# -----------------------------------------------------------------------

def test_black_swan_detects_covid_crash():
    """
    Use real AAPL data from Feb–Apr 2020.
    The COVID crash should flag at least 5 days as black swan events.
    """
    try:
        import yfinance as yf

        df = yf.download("AAPL", start="2020-02-01", end="2020-04-30", progress=False)
        if df.empty:
            pytest.skip("Could not download AAPL data for COVID period")

        # Normalise columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        result = detect_black_swan(df)

        n_flagged = result["is_black_swan"].sum()
        assert n_flagged >= 5, (
            f"Expected at least 5 black-swan days during COVID crash, got {n_flagged}"
        )

        # Verify at least one flash crash was detected in March 2020
        march_events = result.loc["2020-03-01":"2020-03-31"]
        assert march_events["is_black_swan"].any(), (
            "No black-swan events detected in March 2020 — COVID crash was missed"
        )
    except ImportError:
        pytest.skip("yfinance not installed")
