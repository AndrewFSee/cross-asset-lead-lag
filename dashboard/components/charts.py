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
    height: int = 600,
) -> go.Figure:
    """Create a network graph using Plotly scatter traces.

    Args:
        nodes: List of dicts with keys: id, x, y, color, size, label.
        edges: List of dicts with keys: source_x, source_y, target_x, target_y, weight.
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Draw edges
    for edge in edges:
        fig.add_trace(
            go.Scatter(
                x=[edge["source_x"], edge["target_x"], None],
                y=[edge["source_y"], edge["target_y"], None],
                mode="lines",
                line=dict(width=max(0.5, edge.get("weight", 1.0) * 3), color="grey"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Draw nodes
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_text = [n["label"] for n in nodes]
    node_color = [n.get("color", "blue") for n in nodes]
    node_size = [n.get("size", 10) for n in nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=node_size, color=node_color),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
