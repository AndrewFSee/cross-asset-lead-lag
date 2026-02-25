"""Live signal monitoring dashboard."""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_signal_monitor(current_signals: List[dict]) -> None:
    """Render the current active signals dashboard.

    Args:
        current_signals: List of signal dicts with keys:
            leader, follower, te_score, expected_return, confidence.
    """
    st.subheader("Active Lead-Lag Signals")

    if not current_signals:
        st.info("No active signals above threshold. Adjust parameters or fetch fresh data.")
        return

    df = pd.DataFrame(current_signals)

    # Confidence bars using progress indicators
    st.write(f"**{len(df)} active signals**")

    for _, row in df.iterrows():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            direction = "▲" if row.get("expected_return", 0) > 0 else "▼"
            st.write(f"{direction} **{row.get('leader', '')}** → {row.get('follower', '')}")
        with col2:
            conf = float(row.get("confidence", 0.0))
            st.progress(conf, text=f"Confidence: {conf:.1%}")
        with col3:
            ret = row.get("expected_return", 0.0)
            color = "green" if ret > 0 else "red"
            st.markdown(f":{color}[{ret:+.4f}]")

    # P&L attribution bar chart
    if "expected_return" in df.columns and "follower" in df.columns:
        st.subheader("Expected Return Attribution")
        fig = go.Figure(
            go.Bar(
                x=df["follower"].tolist(),
                y=df["expected_return"].tolist(),
                marker_color=["green" if v > 0 else "red" for v in df["expected_return"].tolist()],
            )
        )
        fig.update_layout(title="Expected Return by Follower Asset", height=350)
        st.plotly_chart(fig, use_container_width=True)
