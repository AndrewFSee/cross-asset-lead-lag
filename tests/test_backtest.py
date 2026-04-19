"""Tests for walk-forward backtesting."""

from __future__ import annotations

import numpy as np
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
            tc_bps=0.0,
        )
        bt.run()
        assert bt._portfolio_returns is not None
        assert len(bt._portfolio_returns) > 0

    def test_no_lookahead_bias(self, sample_returns_panel):
        """Signal function must strictly see dates BEFORE the rebalance bar."""
        seen_slices = []

        def recording_signal(returns: pd.DataFrame) -> dict:
            seen_slices.append(returns.index[-1])
            return equal_weight_signal(returns)

        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=recording_signal,
            initial_window=100,
            step_size=50,
            tc_bps=0.0,
            execution_lag=0,
        )
        bt.run()

        # Signal at step t must see data STRICTLY BEFORE index[t]. The old
        # assertion used `<=`, which silently allowed a one-bar leak.
        for i, date in enumerate(seen_slices):
            step_t = 100 + i * 50
            if step_t < len(sample_returns_panel):
                assert (
                    date < sample_returns_panel.index[step_t]
                ), f"Look-ahead bias: signal saw index[{step_t}]={sample_returns_panel.index[step_t]}"

    def test_no_current_bar_leakage(self, sample_returns_panel):
        """Poisoning the current bar must not affect weights — proves the
        signal function cannot peek at data it should not see.

        We build a backtest where the signal simply remembers the latest
        value it saw on a specific asset. We then flip values at the
        rebalance bars to sentinel garbage and re-run — if the signal
        function ever read the current bar, the sentinel would show up.
        """
        sentinel = -9999.0
        seen_values = []

        def peek_signal(returns: pd.DataFrame) -> dict:
            # What is the "most recent" value the signal sees on asset 0?
            seen_values.append(float(returns.iloc[-1, 0]))
            n = len(returns.columns)
            return {col: 1.0 / n for col in returns.columns}

        panel_a = sample_returns_panel.copy()
        panel_b = sample_returns_panel.copy()
        # Poison bars at t=100, 150, 200, ... in panel_b (the rebalance bars
        # with initial_window=100, step_size=50)
        for t in range(100, len(panel_b), 50):
            panel_b.iloc[t, 0] = sentinel

        seen_values.clear()
        WalkForwardBacktest(
            returns=panel_a, signal_func=peek_signal,
            initial_window=100, step_size=50, tc_bps=0.0, execution_lag=0,
        ).run()
        seen_a = list(seen_values)

        seen_values.clear()
        WalkForwardBacktest(
            returns=panel_b, signal_func=peek_signal,
            initial_window=100, step_size=50, tc_bps=0.0, execution_lag=0,
        ).run()
        seen_b = list(seen_values)

        # If the harness excludes the current bar from training, the two
        # runs must see identical values at every rebalance.
        assert seen_a == seen_b, (
            "Signal function saw poisoned current bar — training slice leaks data at t."
        )

    def test_transaction_costs_reduce_returns(self, sample_returns_panel):
        """Higher TC must strictly reduce realised portfolio return on a
        strategy with non-zero turnover."""
        # Alternate between two weight profiles each rebalance to guarantee
        # turnover, otherwise equal-weight has zero Δw after the first step.
        flipper = {"i": 0}
        assets = sample_returns_panel.columns.tolist()

        def flipping_signal(returns: pd.DataFrame) -> dict:
            flipper["i"] += 1
            if flipper["i"] % 2 == 0:
                return {a: 1.0 / len(assets) for a in assets}
            return {a: (-1.0 / len(assets) if k % 2 else 1.0 / len(assets))
                    for k, a in enumerate(assets)}

        bt_no_tc = WalkForwardBacktest(
            returns=sample_returns_panel, signal_func=flipping_signal,
            initial_window=100, step_size=20, tc_bps=0.0, execution_lag=0,
        )
        bt_no_tc.run()
        flipper["i"] = 0
        bt_hi_tc = WalkForwardBacktest(
            returns=sample_returns_panel, signal_func=flipping_signal,
            initial_window=100, step_size=20, tc_bps=10.0, execution_lag=0,
        )
        bt_hi_tc.run()

        total_no = float((1 + bt_no_tc._portfolio_returns).prod() - 1)
        total_hi = float((1 + bt_hi_tc._portfolio_returns).prod() - 1)
        assert total_hi < total_no, (
            f"TC did not reduce returns: no-TC={total_no:.6f}, hi-TC={total_hi:.6f}"
        )

    def test_execution_lag_shifts_realised_returns(self, sample_returns_panel):
        """execution_lag=1 must produce different realised returns vs 0 on
        any non-constant return series."""
        bt0 = WalkForwardBacktest(
            returns=sample_returns_panel, signal_func=equal_weight_signal,
            initial_window=100, step_size=20, tc_bps=0.0, execution_lag=0,
        )
        bt0.run()
        bt1 = WalkForwardBacktest(
            returns=sample_returns_panel, signal_func=equal_weight_signal,
            initial_window=100, step_size=20, tc_bps=0.0, execution_lag=1,
        )
        bt1.run()
        # Same weights on every bar, but shifted realisation: the equity
        # curves must be visibly different and the same length rule should
        # hold (bt1 simply trades one bar later).
        assert not np.allclose(
            bt0._portfolio_returns.values[:10],
            bt1._portfolio_returns.values[:10],
        ), "execution_lag had no effect on realised returns"

    def test_compute_metrics_keys(self, sample_returns_panel):
        """compute_metrics should return all required metric keys."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=equal_weight_signal,
            initial_window=100,
            step_size=20,
            tc_bps=0.0,
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
            tc_bps=0.0,
        )
        bt.run()
        eq = bt.equity_curve()
        assert abs(eq.iloc[0] - 1.0) < 0.1

    def test_zero_weights_produce_zero_returns(self, sample_returns_panel):
        """Zero-weight portfolio should have near-zero returns (no TC)."""
        bt = WalkForwardBacktest(
            returns=sample_returns_panel,
            signal_func=zero_signal,
            initial_window=100,
            step_size=20,
            tc_bps=0.0,
        )
        bt.run()
        assert bt._portfolio_returns.abs().max() < 1e-10
