"""Shared test fixtures for the cross-asset lead-lag test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_var_data(rng):
    """Synthetic VAR(1) data where asset A leads asset B.

    Asset A: random noise
    Asset B: 0.6 * A[t-1] + noise (A leads B with lag=1)

    Returns:
        Tuple (A, B) of 1-D numpy arrays of length 500.
    """
    T = 500
    A = rng.standard_normal(T)
    B = np.zeros(T)
    B[0] = rng.standard_normal()
    for t in range(1, T):
        B[t] = 0.6 * A[t - 1] + 0.3 * rng.standard_normal()
    return A, B


@pytest.fixture(scope="session")
def synthetic_regime_data(rng):
    """Synthetic 2-regime data with different VAR coefficients.

    Regime 0: low volatility, weak autocorrelation
    Regime 1: high volatility, strong autocorrelation

    Returns:
        Tuple (data, regime_labels) where data.shape=(500, 2).
    """
    T = 500
    n_vars = 2
    data = np.zeros((T, n_vars))
    labels = np.zeros(T, dtype=int)

    regime = 0
    for t in range(1, T):
        # Regime switching with transition probability
        if regime == 0 and rng.random() < 0.05:
            regime = 1
        elif regime == 1 and rng.random() < 0.1:
            regime = 0
        labels[t] = regime

        if regime == 0:
            # Low vol regime
            data[t, 0] = 0.1 * data[t - 1, 0] + 0.2 * rng.standard_normal()
            data[t, 1] = 0.2 * data[t - 1, 0] + 0.1 * data[t - 1, 1] + 0.2 * rng.standard_normal()
        else:
            # High vol regime
            data[t, 0] = 0.4 * data[t - 1, 0] + 0.8 * rng.standard_normal()
            data[t, 1] = 0.5 * data[t - 1, 0] + 0.3 * data[t - 1, 1] + 0.8 * rng.standard_normal()

    return data, labels


@pytest.fixture(scope="session")
def sample_returns_dict(rng):
    """Dict of return arrays for testing TE and MI functions.

    Contains 3 assets: A leads B, C is independent.

    Returns:
        Dict mapping asset name to 1-D numpy array.
    """
    T = 300
    A = rng.standard_normal(T)
    B = np.zeros(T)
    for t in range(1, T):
        B[t] = 0.5 * A[t - 1] + 0.5 * rng.standard_normal()
    C = rng.standard_normal(T)  # Independent
    return {"A": A, "B": B, "C": C}


@pytest.fixture(scope="session")
def sample_returns_panel(rng):
    """DataFrame of returns for backtesting and portfolio tests."""
    T = 504  # 2 years
    assets = ["SPX", "HY_OAS", "VIX", "COPPER", "DXY"]
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns = pd.DataFrame(
        rng.standard_normal((T, len(assets))) * 0.01,
        index=dates,
        columns=assets,
    )
    return returns
