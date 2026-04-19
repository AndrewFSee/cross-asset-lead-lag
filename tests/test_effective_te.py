"""Tests for effective (bias-corrected) transfer entropy."""

from __future__ import annotations

import numpy as np

from discovery.transfer_entropy import effective_transfer_entropy


class TestEffectiveTE:
    def test_independent_series_effective_te_near_zero(self, rng):
        """On independent Gaussian noise, effective TE should be ~0 because
        the raw TE bias is absorbed by the surrogate mean."""
        T = 400
        source = rng.standard_normal(T)
        target = rng.standard_normal(T)
        result = effective_transfer_entropy(
            source, target, lag=1, k=5, history_len=3, n_surrogates=30,
        )
        assert result["te_effective"] < 0.02, (
            f"Effective TE should be ~0 for independent series, got "
            f"{result['te_effective']:.4f}"
        )
        # p-value should not be tiny on a genuinely independent pair. With
        # only 30 surrogates the smallest observable non-zero value is
        # 1/30 ≈ 0.033, so we require > 0.05 — a threshold that still fails
        # clean causal pairs (which hit p=0 routinely) but tolerates the
        # occasional sampling fluctuation under the null.
        assert result["p_value"] > 0.05, (
            f"p-value under null should not be tiny: got {result['p_value']:.3f}"
        )

    def test_causal_series_effective_te_positive(self, rng):
        """On a pair where A causes B, effective TE (A→B) should be clearly
        positive and the p-value clearly significant."""
        T = 400
        source = rng.standard_normal(T)
        target = np.zeros(T)
        for t in range(1, T):
            target[t] = 0.7 * source[t - 1] + 0.3 * rng.standard_normal()

        result = effective_transfer_entropy(
            source, target, lag=1, k=5, history_len=3, n_surrogates=30,
        )
        assert result["te_effective"] > 0.05, (
            f"Effective TE too small for known causal pair: "
            f"{result['te_effective']:.4f}"
        )
        assert result["p_value"] < 0.1, (
            f"p-value should reject null for causal pair: got {result['p_value']:.3f}"
        )

    def test_result_keys(self, sample_returns_dict):
        result = effective_transfer_entropy(
            sample_returns_dict["A"], sample_returns_dict["B"],
            lag=1, k=5, history_len=3, n_surrogates=10,
        )
        for key in ("te_raw", "te_surrogate_mean", "te_effective", "p_value"):
            assert key in result

    def test_effective_te_nonnegative(self, sample_returns_dict):
        result = effective_transfer_entropy(
            sample_returns_dict["A"], sample_returns_dict["C"],
            lag=1, k=5, history_len=3, n_surrogates=10,
        )
        assert result["te_effective"] >= 0.0
