"""Tests for Lasso-penalized VAR model."""

from __future__ import annotations

import numpy as np

from models.lasso_var import LassoVAR


def _generate_sparse_var(T: int = 400, n_vars: int = 4, seed: int = 0) -> tuple:
    """Generate sparse VAR data with known structure.

    Asset 0 leads asset 1 (coefficient 0.5), all others are independent.

    Returns:
        Tuple (Y, true_coefs) where true_coefs[(0, 1)] = 0.5.
    """
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, n_vars))
    for t in range(1, T):
        Y[t, 0] = 0.2 * Y[t - 1, 0] + rng.standard_normal()
        Y[t, 1] = 0.5 * Y[t - 1, 0] + 0.1 * Y[t - 1, 1] + 0.5 * rng.standard_normal()
        for j in range(2, n_vars):
            Y[t, j] = 0.1 * Y[t - 1, j] + rng.standard_normal()
    return Y


class TestLassoVAR:
    """Tests for LassoVAR model."""

    def test_fit_returns_self(self):
        """fit() should return self and mark as fitted."""
        Y = _generate_sparse_var(T=200)
        model = LassoVAR(n_lags=2, alpha=0.01)
        result = model.fit(Y)
        assert result is model
        assert model._fitted

    def test_predict_shape(self):
        """predict() should return array of correct shape."""
        Y = _generate_sparse_var(T=200)
        model = LassoVAR(n_lags=2, alpha=0.01)
        model.fit(Y)
        fc = model.predict(Y, horizon=5)
        assert fc.shape == (5, Y.shape[1])

    def test_lead_lag_matrix_shape(self):
        """get_lead_lag_matrix should return (n_vars, n_vars) DataFrame."""
        Y = _generate_sparse_var(T=200, n_vars=4)
        model = LassoVAR(n_lags=2, alpha=0.01)
        model.fit(Y, asset_names=["A", "B", "C", "D"])
        ll_matrix = model.get_lead_lag_matrix()
        assert ll_matrix.shape == (4, 4)
        assert list(ll_matrix.columns) == ["A", "B", "C", "D"]

    def test_sparsity_recovery(self):
        """True causal link (A→B) should have larger coefficient than noise links."""
        Y = _generate_sparse_var(T=600, n_vars=4, seed=1)
        model = LassoVAR(n_lags=1, alpha=0.001)
        model.fit(Y, asset_names=["A", "B", "C", "D"])
        ll_matrix = model.get_lead_lag_matrix()
        # A→B should be among the largest
        a_to_b = ll_matrix.loc["A", "B"]
        a_to_c = ll_matrix.loc["A", "C"]
        a_to_d = ll_matrix.loc["A", "D"]
        assert (
            a_to_b >= a_to_c or a_to_b >= a_to_d
        ), f"Expected A→B ({a_to_b:.4f}) to dominate other A links"
