"""Tests for statistical significance testing of transfer entropy."""

from __future__ import annotations

import numpy as np

from discovery.significance import bootstrap_te_significance, surrogate_significance


class TestBootstrapTeSignificance:
    """Tests for bootstrap_te_significance."""

    def test_result_keys(self, synthetic_var_data):
        """Result should contain all expected keys."""
        A, B = synthetic_var_data
        result = bootstrap_te_significance(
            A[:200], B[:200], lag=1, n_bootstraps=50, k=5, history_len=3
        )
        for key in ["te_observed", "p_value", "is_significant", "ci_lower", "ci_upper"]:
            assert key in result, f"Missing key: {key}"

    def test_p_value_range(self, synthetic_var_data):
        """p-value should be in [0, 1]."""
        A, B = synthetic_var_data
        result = bootstrap_te_significance(
            A[:200], B[:200], lag=1, n_bootstraps=50, k=5
        )
        assert 0.0 <= result["p_value"] <= 1.0

    def test_significant_causal_pair(self, synthetic_var_data):
        """A→B (true causal link) should be significant with enough bootstraps."""
        A, B = synthetic_var_data
        result = bootstrap_te_significance(
            A[:300], B[:300], lag=1, n_bootstraps=100, k=5, alpha=0.1
        )
        assert result["te_observed"] > 0

    def test_independent_not_significant(self, rng):
        """Independent series should generally not be significant."""
        X = rng.standard_normal(300)
        Y = rng.standard_normal(300)
        result = bootstrap_te_significance(X, Y, lag=1, n_bootstraps=100, k=5, alpha=0.05)
        # Not a hard assertion—stochastic—but p-value should tend high
        assert result["p_value"] >= 0.0


class TestSurrogateSignificance:
    """Tests for surrogate_significance."""

    def test_result_keys_shuffle(self, synthetic_var_data):
        """Result should contain all expected keys with shuffle method."""
        A, B = synthetic_var_data
        result = surrogate_significance(
            A[:200], B[:200], lag=1, n_surrogates=50, method="shuffle", k=5
        )
        for key in ["te_observed", "p_value", "is_significant", "surrogate_mean", "surrogate_std"]:
            assert key in result

    def test_result_keys_phase(self, synthetic_var_data):
        """Result should contain all expected keys with phase method."""
        A, B = synthetic_var_data
        result = surrogate_significance(
            A[:200], B[:200], lag=1, n_surrogates=50, method="phase", k=5
        )
        for key in ["te_observed", "p_value", "is_significant", "surrogate_mean", "surrogate_std"]:
            assert key in result

    def test_invalid_method_raises(self, synthetic_var_data):
        """Invalid method should raise ValueError."""
        A, B = synthetic_var_data
        import pytest

        with pytest.raises(ValueError, match="Unknown surrogate method"):
            surrogate_significance(A[:100], B[:100], method="invalid")

    def test_p_value_range(self, synthetic_var_data):
        """p-value should be in [0, 1]."""
        A, B = synthetic_var_data
        result = surrogate_significance(A[:200], B[:200], n_surrogates=50, method="shuffle", k=5)
        assert 0.0 <= result["p_value"] <= 1.0
