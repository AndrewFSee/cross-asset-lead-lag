"""Tests for deflated Sharpe, PBO, and bootstrap SR confidence intervals."""

from __future__ import annotations

import numpy as np

from signals.metrics import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probability_of_backtest_overfitting,
)


class TestExpectedMaxSharpe:
    def test_grows_with_n_trials(self):
        a = expected_maximum_sharpe(10)
        b = expected_maximum_sharpe(1000)
        assert b > a > 0.0

    def test_trivial_at_one(self):
        assert expected_maximum_sharpe(1) == 0.0


class TestDeflatedSharpe:
    def test_deflates_with_more_trials(self):
        low = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=5)
        high = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=500)
        assert high["deflated_sharpe"] < low["deflated_sharpe"]

    def test_strong_sharpe_passes_threshold(self):
        r = deflated_sharpe_ratio(3.0, n_obs=2000, n_trials=5)
        assert r["passes_95"]
        assert r["deflated_sharpe"] >= 0.95

    def test_marginal_sharpe_fails(self):
        r = deflated_sharpe_ratio(0.3, n_obs=252, n_trials=100)
        assert not r["passes_95"]


class TestPBO:
    def test_pbo_separates_noise_from_signal(self):
        """Averaged across seeds, pure noise should produce a higher PBO
        than a panel containing one dominant strategy. We don't pin a
        hard threshold on a single Monte Carlo draw (it's too noisy on
        600 bars × 10-20 trials) — instead we check the expected
        *ordering* across repeated draws, which is the robust
        theoretical prediction."""
        noise_pbos = []
        edge_pbos = []
        T, M = 600, 12
        for seed in range(8):
            rng_local = np.random.default_rng(seed)
            R_noise = rng_local.standard_normal((T, M)) * 0.01
            noise_pbos.append(probability_of_backtest_overfitting(R_noise)["pbo"])

            R_edge = rng_local.standard_normal((T, M)) * 0.01
            R_edge[:, 0] += 0.002
            edge_pbos.append(probability_of_backtest_overfitting(R_edge)["pbo"])
        # Expect noise to overfit *more* than a panel with a real edge
        assert np.mean(noise_pbos) > np.mean(edge_pbos), (
            f"noise mean PBO {np.mean(noise_pbos):.2f} should exceed "
            f"dominant-trial mean PBO {np.mean(edge_pbos):.2f}"
        )
        # And edge case should be a low PBO on its own merits
        assert np.mean(edge_pbos) < 0.3

    def test_too_few_observations_returns_nan(self, rng):
        R = rng.standard_normal((40, 3))
        r = probability_of_backtest_overfitting(R)
        assert np.isnan(r["pbo"])


class TestBootstrapSharpeCI:
    def test_ci_covers_point_estimate(self, rng):
        r = rng.standard_normal(500) * 0.01 + 0.0005
        out = bootstrap_sharpe_ci(r, n_boot=200, ci=0.95)
        assert out["lower"] <= out["sharpe"] <= out["upper"]

    def test_wider_ci_for_shorter_series(self, rng):
        short = rng.standard_normal(100) * 0.01
        long = rng.standard_normal(2000) * 0.01
        ci_short = bootstrap_sharpe_ci(short, n_boot=200, random_state=0)
        ci_long = bootstrap_sharpe_ci(long, n_boot=200, random_state=0)
        width_s = ci_short["upper"] - ci_short["lower"]
        width_l = ci_long["upper"] - ci_long["lower"]
        assert width_s > width_l
