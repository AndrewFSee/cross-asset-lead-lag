"""Tests for Markov-Switching VAR model."""

from __future__ import annotations

import numpy as np

from models.ms_var import MarkovSwitchingVAR


class TestMarkovSwitchingVAR:
    """Tests for MarkovSwitchingVAR."""

    def test_fit_returns_self(self, synthetic_regime_data):
        """fit() should return self and set _fitted=True."""
        data, _ = synthetic_regime_data
        model = MarkovSwitchingVAR(n_vars=2, n_lags=1, n_regimes=2)
        result = model.fit(data[:100], max_iter=5)
        assert result is model
        assert model._fitted

    def test_smoothed_probs_shape(self, synthetic_regime_data):
        """Smoothed probabilities should have shape (T-p, n_regimes)."""
        data, _ = synthetic_regime_data
        T = 150
        model = MarkovSwitchingVAR(n_vars=2, n_lags=1, n_regimes=2)
        model.fit(data[:T], max_iter=5)
        probs = model.smoothed_probs
        assert probs is not None
        assert probs.shape == (T - 1, 2), f"Expected ({T-1}, 2), got {probs.shape}"

    def test_smoothed_probs_sum_to_one(self, synthetic_regime_data):
        """Smoothed probabilities should sum to 1 at each time step."""
        data, _ = synthetic_regime_data
        model = MarkovSwitchingVAR(n_vars=2, n_lags=1, n_regimes=2)
        model.fit(data[:100], max_iter=5)
        probs = model.smoothed_probs
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_forecast_shape(self, synthetic_regime_data):
        """forecast() should return array of shape (horizon, n_vars)."""
        data, _ = synthetic_regime_data
        model = MarkovSwitchingVAR(n_vars=2, n_lags=1, n_regimes=2)
        model.fit(data[:100], max_iter=5)
        fc = model.forecast(data[:100], horizon=3)
        assert fc.shape == (3, 2), f"Expected (3, 2), got {fc.shape}"

    def test_get_regime_coefficients(self, synthetic_regime_data):
        """get_regime_coefficients should return dict with n_regimes entries."""
        data, _ = synthetic_regime_data
        model = MarkovSwitchingVAR(n_vars=2, n_lags=1, n_regimes=2)
        model.fit(data[:100], max_iter=5)
        coefs = model.get_regime_coefficients()
        assert len(coefs) == 2
        for r, B in coefs.items():
            assert B.shape == (2, 3), f"Expected (2,3) for regime {r}, got {B.shape}"
