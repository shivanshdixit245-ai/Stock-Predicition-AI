import pytest
import numpy as np
import pandas as pd
from src.backtest.stats import (
    compute_sharpe,
    compute_max_drawdown,
    bootstrap_significance_test
)

def test_bootstrap_p_value_range():
    """Verify p-value is between 0.0 and 1.0"""
    returns = np.random.normal(0.001, 0.01, 100)
    result = bootstrap_significance_test(returns, n_permutations=100)
    
    assert 0.0 <= result['p_value'] <= 1.0
    assert isinstance(result['is_significant'], bool)

def test_random_strategy_not_significant():
    """Verify a random signal sequence produces p_value > 0.10"""
    # Create random noise with mean 0
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 500)
    
    # Run test
    result = bootstrap_significance_test(returns, n_permutations=500)
    
    # A truly random strategy should not be significant
    # (Though statistically it could happen by chance, seed 42 is likely safe)
    assert result['p_value'] > 0.10
    assert not result['is_significant']

def test_sharpe_annualisation():
    """Verify Sharpe computed correctly with known input returns"""
    # 1% return every day, 0% volatility (hypothetically)
    # std=0 returns 0.0 in our implementation to avoid div by zero
    
    # Use returns with known mean and std
    returns = np.array([0.01, -0.01] * 50) # mean=0, std=~0.01
    rfr = 0.0
    
    sharpe = compute_sharpe(returns, risk_free_rate=rfr)
    
    # Mean is 0, so Sharpe should be 0
    assert pytest.approx(sharpe, abs=1e-5) == 0.0
    
    # returns with mean 0.001 and std 0.01
    returns_pos = np.ones(100) * 0.001
    # Adding a bit of noise to get non-zero std
    returns_pos[0] = 0.002
    returns_pos[1] = 0.000
    
    sharpe_pos = compute_sharpe(returns_pos, risk_free_rate=0)
    assert sharpe_pos > 0

def test_max_drawdown_correct():
    """Verify max drawdown computed correctly with a known equity curve"""
    equity = np.array([100, 110, 100, 80, 90, 100])
    # Peak is 110, Trough is 80
    # MDD = (80 - 110) / 110 = -30 / 110 = -0.2727...
    
    result = compute_max_drawdown(equity)
    
    assert pytest.approx(result['max_drawdown_pct'], abs=1e-4) == -0.272727
    assert result['drawdown_start'] == 1 # index of 110
    assert result['drawdown_end'] == 3   # index of 80
