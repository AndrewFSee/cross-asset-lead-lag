"""Tests for variable-lag transfer entropy."""

from __future__ import annotations

import numpy as np

from discovery.variable_lag import best_lag_effective_te, compute_variable_lag_matrix


class TestBestLagEffectiveTE:
    def test_recovers_known_lag(self, rng):
        """When B[t] = A[t-5] + noise, the best lag should be 5. We use
        history_len=1 to prevent the KSG embedding from confounding adjacent
        lags (a longer history covers many lags at once)."""
        T = 600
        true_lag = 5
        A = rng.standard_normal(T)
        B = np.zeros(T)
        for t in range(true_lag, T):
            B[t] = 0.9 * A[t - true_lag] + 0.15 * rng.standard_normal()

        result = best_lag_effective_te(
            A, B, candidate_lags=[1, 3, 5, 8, 12],
            k=5, history_len=1, n_surrogates=15,
        )
        assert result["best_lag"] == true_lag, (
            f"Expected best lag {true_lag}, got {result['best_lag']} "
            f"(all_lags: {result['all_lags']})"
        )
        assert result["te_effective"] > 0.0
        assert result["p_value"] < 0.2

    def test_returns_stability_cv(self, sample_returns_dict):
        r = best_lag_effective_te(
            sample_returns_dict["A"], sample_returns_dict["B"],
            candidate_lags=[1, 2, 3],
            k=5, history_len=3, n_surrogates=10,
        )
        # stability_cv should be a float (nan if too short) >= 0 when defined
        cv = r["stability_cv"]
        assert isinstance(cv, float)
        if not np.isnan(cv):
            assert cv >= 0.0


class TestComputeVariableLagMatrix:
    def test_returns_tidy_frame(self, sample_returns_dict):
        df = compute_variable_lag_matrix(
            sample_returns_dict,
            candidate_lags=[1, 2],
            k=5, history_len=3, n_surrogates=5,
        )
        expected_cols = {"source", "target", "best_lag", "te_effective",
                         "p_value", "stability_cv"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0
        # No self-pairs
        assert (df["source"] != df["target"]).all()

    def test_respects_target_subset(self, sample_returns_dict):
        df = compute_variable_lag_matrix(
            sample_returns_dict,
            candidate_lags=[1],
            k=5, history_len=3, n_surrogates=5,
            target_subset=["B"],
        )
        assert set(df["target"].unique()) == {"B"}
