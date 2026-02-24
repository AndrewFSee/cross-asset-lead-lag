"""Backtest results visualization."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_backtest_results(backtest) -> None:
    """Render backtest equity curve, drawdown, rolling Sharpe and metrics.

    Args:
        backtest: Fitted WalkForwardBacktest instance.
    """
    st.subheader("Backtest Results")

    try:
        metrics = backtest.compute_metrics()
        eq_curve = backtest.equity_curve()
        dd_series = backtest.drawdown_series()
    except Exception as exc:
        st.error(f"Could not compute backtest results: {exc}")
        return

    # ── Metrics table ────────────────────────────────────────────────────────
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
        rows=2, cols=1,
        subplot_titles=["Equity Curve", "Drawdown"],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Scatter(x=eq_curve.index, y=eq_curve.values, name="Portfolio", line=dict(color="blue")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=2, col=1,
    )

    fig.update_layout(height=600, title_text="Walk-Forward Backtest")
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Sharpe ───────────────────────────────────────────────────────
    st.write("### Rolling 63-day Sharpe Ratio")
    port_returns = backtest._portfolio_returns
    rolling_sharpe = (
        port_returns.rolling(63).mean() / port_returns.rolling(63).std() * (252 ** 0.5)
    )
    fig_sharpe = go.Figure(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Rolling Sharpe")
    )
    fig_sharpe.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_sharpe.update_layout(height=300)
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Full metrics table ───────────────────────────────────────────────────
    st.write("### Full Metrics")
    st.dataframe(
        pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).round(4)
    )
