"""Tests for walk-forward backtesting."""

from __future__ import annotations

import pandas as pd

from signals.backtest import WalkForwardBacktest


def equal_weight_signal(returns: pd.DataFrame) -> dict:
    """Simple equal-weight signal function for testing."""
    n = len(returns.columns)
    w = 1.0 / n
    return {col: w for col in returns.columns}


def zero_signal(returns: pd.DataFrame) -> dict:
    """Zero-weight signal for testing neutral portfolio."""
    return {col: 0.0 for col in returns.columns}


class TestWalkForwardBacktest:
    """Tests for WalkForwardBacktest."""

    def test_run_produces_portfolio_returns(self, sample_returns_panel):
        """After run(), portfolio returns should have correct length."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=equal_weight_signal,
            initial_window=100,
            step_size=20,
        )
        bt.run()
        assert bt._portfolio_returns is not None
        assert len(bt._portfolio_returns) > 0

    def test_no_lookahead_bias(self, sample_returns_panel):
        """Signal function should only see past data at each step."""
        seen_slices = []

        def recording_signal(returns: pd.DataFrame) -> dict:
            seen_slices.append(returns.index[-1])
            return equal_weight_signal(returns)

        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=recording_signal,
            initial_window=100,
            step_size=50,
        )
        bt.run()

        # Each seen slice should only be before the current step
        for i, date in enumerate(seen_slices):
            step_t = 100 + i * 50
            if step_t < len(sample_returns_panel):
                assert (
                    date <= sample_returns_panel.index[step_t]
                ), f"Look-ahead bias: signal saw data beyond step {step_t}"

    def test_compute_metrics_keys(self, sample_returns_panel):
        """compute_metrics should return all required metric keys."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=equal_weight_signal,
            initial_window=100,
            step_size=20,
        )
        bt.run()
        metrics = bt.compute_metrics()
        required_keys = [
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "hit_rate",
            "avg_turnover",
            "total_return",
            "annual_return",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_equity_curve_starts_near_one(self, sample_returns_panel):
        """Equity curve initial value should be close to 1."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=equal_weight_signal,
            initial_window=100,
            step_size=20,
        )
        bt.run()
        eq = bt.equity_curve()
        assert abs(eq.iloc[0] - 1.0) < 0.1

    def test_zero_weights_produce_zero_returns(self, sample_returns_panel):
        """Zero-weight portfolio should have near-zero returns."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=zero_signal,
            initial_window=100,
            step_size=20,
        )
        bt.run()
        assert bt._portfolio_returns.abs().max() < 1e-10
