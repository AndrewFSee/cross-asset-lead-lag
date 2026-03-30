"""Reusable Plotly chart components for the dashboard."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go


def line_chart(
    series_dict: Dict[str, pd.Series],
    title: str = "",
    x_label: str = "Date",
    y_label: str = "Value",
    height: int = 400,
) -> go.Figure:
    """Create a multi-line time series chart.

    Args:
        series_dict: Dict mapping series name to pd.Series.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    for name, series in series_dict.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines"))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=height)
    return fig


def heatmap(
    matrix: pd.DataFrame,
    title: str = "",
    colorscale: str = "RdBu",
    height: int = 500,
    zmid: float = 0.0,
) -> go.Figure:
    """Create a heatmap from a DataFrame.

    Args:
        matrix: DataFrame to visualize.
        title: Chart title.
        colorscale: Plotly colorscale name.
        height: Chart height in pixels.
        zmid: Center value for diverging colorscale.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            zmid=zmid,
        )
    )
    fig.update_layout(title=title, height=height)
    return fig


def bar_chart(
    data: Dict[str, float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color_positive: str = "green",
    color_negative: str = "red",
    height: int = 400,
) -> go.Figure:
    """Create a bar chart with positive/negative coloring.

    Args:
        data: Dict mapping label to value.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        color_positive: Color for positive bars.
        color_negative: Color for negative bars.
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    labels = list(data.keys())
    values = list(data.values())
    colors = [color_positive if v >= 0 else color_negative for v in values]

    fig = go.Figure(data=go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=height)
    return fig


def network_chart(
    nodes: List[Dict],
    edges: List[Dict],
    title: str = "",
    height: int = 800,
) -> go.Figure:
    """Create a network graph using Plotly scatter traces.

    Args:
        nodes: List of dicts with keys: id, x, y, color, size, label.
            Optional: asset_class (str) for legend grouping.
        edges: List of dicts with keys: source_x, source_y, target_x,
            target_y, weight.  Optional: hover (str).
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # ── Edges: single consolidated trace + arrow annotations ─────────
    if edges:
        weights = [e.get("weight", 0.01) for e in edges]
        w_min, w_max = min(weights), max(weights)
        w_range = w_max - w_min if w_max > w_min else 1.0

        edge_x: List[float | None] = []
        edge_y: List[float | None] = []
        edge_colors: list[str] = []
        for edge in edges:
            edge_x += [edge["source_x"], edge["target_x"], None]
            edge_y += [edge["source_y"], edge["target_y"], None]

        # One trace for all edge lines (much faster than per-edge traces)
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.0, color="rgba(150,150,150,0.4)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Arrow annotations for direction + color by weight
        for edge in edges:
            w = edge.get("weight", 0.01)
            norm = (w - w_min) / w_range  # 0..1
            # Square root scale so mid-range edges are still visible
            norm_vis = norm ** 0.4
            # Color: muted blue (weak) → bright orange/red (strong)
            r = int(120 + 135 * norm_vis)
            g = int(160 - 60 * norm_vis)
            b = int(200 - 180 * norm_vis)
            opacity = 0.5 + 0.5 * norm_vis
            color = f"rgba({r},{g},{b},{opacity:.2f})"
            line_w = 1.0 + 2.5 * norm_vis

            # Shorten arrow so it doesn't overlap the target node
            sx, sy = edge["source_x"], edge["source_y"]
            tx, ty = edge["target_x"], edge["target_y"]
            dx, dy = tx - sx, ty - sy
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                shrink = 0.06  # stop 6% before center of target node
                ax = sx + dx * (1 - shrink)
                ay = sy + dy * (1 - shrink)
            else:
                ax, ay = sx, sy

            fig.add_annotation(
                x=tx, y=ty, ax=sx, ay=sy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0 + 0.6 * norm,
                arrowwidth=line_w,
                arrowcolor=color,
                opacity=opacity,
            )

    # ── Nodes: one trace per asset class for legend grouping ─────────
    class_groups: Dict[str, List[Dict]] = {}
    for n in nodes:
        cls = n.get("asset_class", "other")
        class_groups.setdefault(cls, []).append(n)

    for cls, group in class_groups.items():
        nx_ = [n["x"] for n in group]
        ny_ = [n["y"] for n in group]
        labels = [n["label"] for n in group]
        colors = [n.get("color", "#9E9E9E") for n in group]
        sizes = [n.get("size", 12) for n in group]
        hover = [
            f"<b>{n['label']}</b><br>Connections: {n.get('degree', 0)}"
            for n in group
        ]

        fig.add_trace(
            go.Scatter(
                x=nx_,
                y=ny_,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=1, color="rgba(255,255,255,0.6)"),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hovertext=hover,
                hoverinfo="text",
                name=cls.title(),
                legendgroup=cls,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        legend=dict(
            title="Asset Class",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
