"""Tests for the KSG-based transfer entropy estimator."""

from __future__ import annotations

from discovery.transfer_entropy import compute_te_matrix, transfer_entropy_knn


class TestTransferEntropyKnn:
    """Tests for transfer_entropy_knn."""

    def test_te_causal_direction(self, synthetic_var_data):
        """TE(A→B) should be larger than TE(B→A) when A causally leads B."""
        A, B = synthetic_var_data
        te_ab = transfer_entropy_knn(A, B, lag=1, k=5)
        te_ba = transfer_entropy_knn(B, A, lag=1, k=5)
        assert te_ab > te_ba, f"Expected TE(A→B)={te_ab:.4f} > TE(B→A)={te_ba:.4f}"

    def test_te_independent_series_near_zero(self, rng):
        """TE between independent series should be close to zero."""
        T = 400
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        te = transfer_entropy_knn(X, Y, lag=1, k=5)
        assert te < 0.05, f"Expected TE ≈ 0 for independent series, got {te:.4f}"

    def test_te_nonnegative(self, synthetic_var_data):
        """Transfer entropy should always be non-negative (clipped at 0)."""
        A, B = synthetic_var_data
        for lag in [1, 2, 5]:
            te = transfer_entropy_knn(A, B, lag=lag, k=5)
            assert te >= 0.0, f"TE should be >= 0, got {te} for lag={lag}"

    def test_te_larger_lag_decreases(self, synthetic_var_data):
        """For a lag-1 causal system, TE(lag=1) should be largest."""
        A, B = synthetic_var_data
        te_lag1 = transfer_entropy_knn(A, B, lag=1, k=5)
        te_lag5 = transfer_entropy_knn(A, B, lag=5, k=5)
        # Lag-1 should generally be stronger for a lag-1 DGP
        assert te_lag1 >= te_lag5, "Expected TE(lag=1) >= TE(lag=5)"

    def test_compute_te_matrix_shape(self, sample_returns_dict):
        """compute_te_matrix should return correctly shaped DataFrames."""
        result = compute_te_matrix(sample_returns_dict, lags=[1, 2], k=5)
        assert set(result.keys()) == {1, 2}
        for lag, df in result.items():
            assert df.shape == (3, 3), f"Expected (3,3) matrix for lag={lag}"
            # Diagonal should be zero
            for i in range(3):
                assert df.iloc[i, i] == 0.0
