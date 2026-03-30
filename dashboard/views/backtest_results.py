"""Backtest results visualization."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_backtest_results(
    backtest_or_equity,
    metrics_dict=None,
    benchmark_equity=None,
    benchmark_metrics=None,
) -> None:
    """Render backtest equity curve, drawdown, rolling Sharpe and metrics.

    Args:
        backtest_or_equity: Fitted WalkForwardBacktest instance, or a
            pd.Series equity curve when called with pre-computed data.
        metrics_dict: Optional dict of pre-computed metrics.
        benchmark_equity: Optional pd.Series benchmark equity curve.
        benchmark_metrics: Optional dict of benchmark metrics.
    """
    st.subheader("Backtest Results")

    if isinstance(backtest_or_equity, pd.Series):
        eq_curve = backtest_or_equity
        metrics = metrics_dict or {}
        dd_rolling_max = eq_curve.cummax()
        dd_series = (eq_curve - dd_rolling_max) / dd_rolling_max
        port_returns = eq_curve.pct_change().dropna()
    else:
        backtest = backtest_or_equity
        try:
            metrics = backtest.compute_metrics()
            eq_curve = backtest.equity_curve()
            dd_series = backtest.drawdown_series()
            port_returns = backtest._portfolio_returns
        except Exception as exc:
            st.error(f"Could not compute backtest results: {exc}")
            return

    has_bench = benchmark_equity is not None and benchmark_metrics is not None

    # ── Strategy comparison metrics ──────────────────────────────────────────
    if has_bench:
        st.write("### Lead-Lag Signals vs. Benchmark (Inverse-Vol)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Lead-Lag Signal Strategy**")
        with col_b:
            st.markdown("**Benchmark (Inverse-Vol Risk Parity)**")

        metric_defs = [
            ("Sharpe Ratio", "sharpe", "{:.2f}"),
            ("Sortino Ratio", "sortino", "{:.2f}"),
            ("Max Drawdown", "max_drawdown", "{:.1%}"),
            ("Total Return", "total_return", "{:.1%}"),
            ("Annual Return", "annual_return", "{:.1%}"),
            ("Daily Hit Rate", "hit_rate", "{:.1%}"),
            ("Avg Turnover", "avg_turnover", "{:.4f}"),
            ("Calmar Ratio", "calmar", "{:.2f}"),
        ]
        for label, key, fmt in metric_defs:
            ca, cb = st.columns(2)
            val_s = metrics.get(key, 0.0)
            val_b = benchmark_metrics.get(key, 0.0)
            ca.metric(label, fmt.format(val_s))
            cb.metric(label, fmt.format(val_b))

        st.divider()
    else:
        st.write("### Performance Metrics")
        metric_cols = st.columns(4)
        metric_labels = [
            ("Sharpe Ratio", "sharpe", "{:.2f}"),
            ("Sortino Ratio", "sortino", "{:.2f}"),
            ("Max Drawdown", "max_drawdown", "{:.1%}"),
            ("Total Return", "total_return", "{:.1%}"),
        ]
        for col, (label, key, fmt) in zip(metric_cols, metric_labels):
            val = metrics.get(key, 0.0)
            col.metric(label, fmt.format(val))

    # ── Equity curve + drawdown ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Equity Curve", "Drawdown"],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Scatter(
            x=eq_curve.index, y=eq_curve.values,
            name="Lead-Lag Signals", line=dict(color="#26a69a", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_series.index, y=dd_series.values,
            name="Signal DD", fill="tozeroy",
            line=dict(color="red", width=1),
        ),
        row=2, col=1,
    )

    if has_bench:
        bench_dd_max = benchmark_equity.cummax()
        bench_dd = (benchmark_equity - bench_dd_max) / bench_dd_max
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index, y=benchmark_equity.values,
                name="Benchmark (Inv-Vol)", line=dict(color="#7e57c2", width=2, dash="dot"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bench_dd.index, y=bench_dd.values,
                name="Bench DD", fill="tozeroy",
                line=dict(color="#ff9800", width=1, dash="dot"),
            ),
            row=2, col=1,
        )

    fig.update_layout(height=650, title_text="Walk-Forward Backtest")
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Sharpe ───────────────────────────────────────────────────────
    st.write("### Rolling 63-day Sharpe Ratio")
    rolling_sharpe = port_returns.rolling(63).mean() / port_returns.rolling(63).std() * (252**0.5)

    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(
        go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name="Lead-Lag Signals", line=dict(color="#26a69a"),
        )
    )

    if has_bench:
        bench_rets = benchmark_equity.pct_change().dropna()
        bench_sharpe = bench_rets.rolling(63).mean() / bench_rets.rolling(63).std() * (252**0.5)
        fig_sharpe.add_trace(
            go.Scatter(
                x=bench_sharpe.index, y=bench_sharpe.values,
                name="Benchmark", line=dict(color="#7e57c2", dash="dot"),
            )
        )

    fig_sharpe.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_sharpe.update_layout(height=300)
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Full metrics table ───────────────────────────────────────────────────
    with st.expander("Full metrics"):
        if has_bench:
            compare = pd.DataFrame({
                "Lead-Lag Signals": pd.Series(metrics),
                "Benchmark (Inv-Vol)": pd.Series(benchmark_metrics),
            }).round(4)
            st.dataframe(compare)
        else:
            st.dataframe(
                pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).round(4)
            )
