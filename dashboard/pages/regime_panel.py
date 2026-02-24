"""Regime probability panel visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.components.charts import heatmap


def render_regime_panel(
    smoothed_probs: np.ndarray,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    regime_coefs: dict = None,
) -> None:
    """Render regime probability time series and coefficient heatmaps.

    Args:
        smoothed_probs: Smoothed regime probabilities (T, n_regimes).
        returns: Asset returns DataFrame.
        dates: Date index for the probability series.
        regime_coefs: Optional dict mapping regime index to coefficient matrix.
    """
    st.subheader("Regime Panel")

    n_regimes = smoothed_probs.shape[1] if smoothed_probs.ndim > 1 else 2

    # Regime probability time series
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Regime Probabilities", "Asset Returns"],
        vertical_spacing=0.1,
    )

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for r in range(n_regimes):
        prob_series = smoothed_probs[:, r] if smoothed_probs.ndim > 1 else smoothed_probs
        fig.add_trace(
            go.Scatter(
                x=dates[-len(prob_series):],
                y=prob_series,
                name=f"Regime {r}",
                fill="tozeroy",
                line=dict(color=colors[r % len(colors)]),
            ),
            row=1, col=1,
        )

    if not returns.empty:
        for col in returns.columns[:3]:  # Show first 3 assets
            fig.add_trace(
                go.Scatter(x=returns.index, y=returns[col].cumsum(), name=col, mode="lines"),
                row=2, col=1,
            )

    fig.update_layout(height=600, title_text="Regime Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Coefficient heatmaps per regime
    if regime_coefs:
        st.subheader("Regime-Conditional VAR Coefficients")
        cols = st.columns(min(n_regimes, 3))
        for r, coef_matrix in regime_coefs.items():
            with cols[r % len(cols)]:
                st.write(f"**Regime {r}**")
                df_coef = pd.DataFrame(coef_matrix)
                fig_heat = heatmap(df_coef, title=f"Regime {r} Coefficients", zmid=0.0)
                st.plotly_chart(fig_heat, use_container_width=True)
