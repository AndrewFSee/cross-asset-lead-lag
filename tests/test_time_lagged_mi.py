"""Tests for the KSG-based time-lagged mutual information estimator."""

from __future__ import annotations

import numpy as np

from discovery.time_lagged_mi import (
    compute_tlmi_matrix,
    mutual_information_knn,
    time_lagged_mi,
)


class TestMutualInformationKnn:
    """Tests for mutual_information_knn."""

    def test_mi_nonnegative(self, rng):
        """MI should always be >= 0."""
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        mi = mutual_information_knn(x, y, k=5)
        assert mi >= 0.0

    def test_mi_dependent_higher_than_independent(self, rng):
        """MI should be higher for correlated variables."""
        x = rng.standard_normal(300)
        y_dep = x + 0.3 * rng.standard_normal(300)
        y_ind = rng.standard_normal(300)
        mi_dep = mutual_information_knn(x, y_dep, k=5)
        mi_ind = mutual_information_knn(x, y_ind, k=5)
        assert mi_dep > mi_ind

    def test_mi_short_series_returns_zero(self):
        """Very short series should return 0.0 gracefully."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        mi = mutual_information_knn(x, y, k=5)
        assert mi == 0.0


class TestTimeLaggedMI:
    """Tests for time_lagged_mi."""

    def test_lagged_mi_causal_direction(self, synthetic_var_data):
        """MI(A_t, B_{t+1}) should be higher than MI(B_t, A_{t+1}) for A→B."""
        A, B = synthetic_var_data
        mi_ab = time_lagged_mi(A, B, lag=1, k=5)
        mi_ba = time_lagged_mi(B, A, lag=1, k=5)
        assert mi_ab > mi_ba, f"Expected MI(A→B)={mi_ab:.4f} > MI(B→A)={mi_ba:.4f}"

    def test_negative_lag(self, rng):
        """Negative lag should work without error."""
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        mi = time_lagged_mi(x, y, lag=-1, k=5)
        assert mi >= 0.0

    def test_lag_exceeding_length_returns_zero(self):
        """Lag >= series length should return 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        mi = time_lagged_mi(x, y, lag=10, k=5)
        assert mi == 0.0


class TestComputeTlmiMatrix:
    """Tests for compute_tlmi_matrix."""

    def test_matrix_shape(self, sample_returns_dict):
        """TLMI matrix should be (n_assets, n_assets) for each lag."""
        result = compute_tlmi_matrix(sample_returns_dict, lags=[1, 3], k=5)
        assert set(result.keys()) == {1, 3}
        for lag, df in result.items():
            assert df.shape == (3, 3)

    def test_diagonal_zero(self, sample_returns_dict):
        """Self-MI should be zero (skipped by convention)."""
        result = compute_tlmi_matrix(sample_returns_dict, lags=[1], k=5)
        for i in range(3):
            assert result[1].iloc[i, i] == 0.0
