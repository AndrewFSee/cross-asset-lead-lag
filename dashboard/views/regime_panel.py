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

    # Let user pick assets for the returns subplot
    preferred = ["SPX", "VIX", "HY_OAS", "DXY", "GOLD", "UST_10Y"]
    available = [c for c in preferred if c in returns.columns]
    if not available:
        available = returns.columns[:4].tolist()
    selected_assets = st.multiselect(
        "Assets to overlay on regime chart",
        returns.columns.tolist(),
        default=available[:4],
    )

    # ── Build subplot figure ──────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "Regime Probabilities  (0 = Calm · 1 = Stress)",
            "Cumulative Asset Returns",
        ],
        vertical_spacing=0.12,
        row_heights=[0.4, 0.6],
    )

    regime_colors = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726"]
    regime_names = {0: "Calm", 1: "Stress", 2: "Regime 2", 3: "Regime 3"}

    # Top: regime probabilities as stacked area
    for r in range(n_regimes):
        prob_series = smoothed_probs[:, r] if smoothed_probs.ndim > 1 else smoothed_probs
        fig.add_trace(
            go.Scatter(
                x=dates[-len(prob_series):],
                y=prob_series,
                name=f"Regime {r} ({regime_names.get(r, '')})",
                fill="tozeroy" if r == 0 else "tonexty",
                line=dict(color=regime_colors[r % len(regime_colors)], width=0.5),
                stackgroup="regimes",
            ),
            row=1,
            col=1,
        )

    # Bottom: cumulative returns with regime background shading
    if selected_assets and not returns.empty:
        asset_colors = [
            "#42A5F5", "#EF5350", "#66BB6A", "#FFA726",
            "#AB47BC", "#FF7043", "#26C6DA", "#D4E157",
        ]
        for i, col in enumerate(selected_assets):
            if col in returns.columns:
                cum = returns[col].cumsum()
                fig.add_trace(
                    go.Scatter(
                        x=cum.index,
                        y=cum.values,
                        name=col,
                        mode="lines",
                        line=dict(width=1.5, color=asset_colors[i % len(asset_colors)]),
                    ),
                    row=2,
                    col=1,
                )

        # Add stress-regime shading as vertical rectangles on returns chart
        if n_regimes >= 2:
            stress_prob = smoothed_probs[:, 1] if smoothed_probs.ndim > 1 else np.zeros(len(dates))
            in_stress = stress_prob > 0.5
            # Find contiguous stress blocks
            blocks = []
            start = None
            for i in range(len(in_stress)):
                if in_stress[i] and start is None:
                    start = i
                elif not in_stress[i] and start is not None:
                    blocks.append((start, i - 1))
                    start = None
            if start is not None:
                blocks.append((start, len(in_stress) - 1))

            for s, e in blocks:
                fig.add_vrect(
                    x0=dates[s], x1=dates[e],
                    fillcolor="rgba(239, 83, 80, 0.12)",
                    line_width=0,
                    row=2, col=1,
                )

    fig.update_layout(
        height=800,
        title_text="Regime Analysis",
        legend=dict(font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Probability", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Regime summary metrics ────────────────────────────────────────
    st.subheader("Regime Summary")
    labels = smoothed_probs.argmax(axis=1) if smoothed_probs.ndim > 1 else smoothed_probs
    total_days = len(labels)
    cols = st.columns(n_regimes)
    for r in range(n_regimes):
        mask = labels == r
        n_days = int(mask.sum())
        with cols[r]:
            color = regime_colors[r % len(regime_colors)]
            name = regime_names.get(r, f"Regime {r}")
            st.metric(
                f"Regime {r}: {name}",
                f"{n_days} days ({n_days/total_days*100:.1f}%)",
            )

    current_regime = int(labels[-1])
    current_name = regime_names.get(current_regime, f"Regime {current_regime}")
    current_prob = smoothed_probs[-1, current_regime] if smoothed_probs.ndim > 1 else 1.0
    st.info(f"**Current regime: {current_regime} ({current_name})** — "
            f"probability {current_prob:.1%} as of {dates[-1].date()}")

    st.caption(
        "Red-shaded regions on the returns chart indicate stress regime "
        "(probability > 50%). Regime features: SPX realized volatility, "
        "credit spreads, yield curve slope, and VIX."
    )

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
