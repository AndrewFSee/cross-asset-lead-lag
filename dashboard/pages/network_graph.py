"""Network graph visualization of lead-lag relationships."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import streamlit as st

from dashboard.components.charts import network_chart

logger = logging.getLogger(__name__)

ASSET_CLASS_COLORS: Dict[str, str] = {
    "equity": "#2196F3",
    "rates": "#4CAF50",
    "credit": "#FF9800",
    "commodities": "#9C27B0",
    "fx": "#F44336",
    "volatility": "#795548",
    "crypto": "#FF5722",
    "macro": "#607D8B",
    "default": "#9E9E9E",
}


def render_network_graph(
    te_matrix: pd.DataFrame,
    threshold: float = 0.01,
    asset_class_map: Optional[Dict[str, str]] = None,
) -> None:
    """Render an interactive network graph of lead-lag relationships.

    Args:
        te_matrix: Transfer entropy matrix (rows=source, cols=target).
        threshold: Minimum TE value to include an edge.
        asset_class_map: Optional dict mapping asset name to asset class string.
    """
    st.subheader("Lead-Lag Network Graph")

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("TE Threshold", 0.0, 0.5, threshold, 0.005)
    with col2:
        layout = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"])

    # Build networkx graph
    G = nx.DiGraph()
    assets = te_matrix.index.tolist()
    G.add_nodes_from(assets)

    for src in assets:
        for tgt in assets:
            if src != tgt:
                te_val = float(te_matrix.loc[src, tgt])
                if te_val >= threshold:
                    G.add_edge(src, tgt, weight=te_val)

    if G.number_of_edges() == 0:
        st.warning(f"No edges above TE threshold {threshold:.4f}. Try lowering the threshold.")
        return

    # Compute layout
    try:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Build node/edge lists for chart
    degree = dict(G.degree())
    nodes = []
    for node in assets:
        color = ASSET_CLASS_COLORS.get((asset_class_map or {}).get(node, "default"), "#9E9E9E")
        nodes.append(
            {
                "id": node,
                "x": float(pos.get(node, [0, 0])[0]),
                "y": float(pos.get(node, [0, 0])[1]),
                "color": color,
                "size": 10 + degree.get(node, 0) * 3,
                "label": node,
            }
        )

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append(
            {
                "source_x": float(pos[src][0]),
                "source_y": float(pos[src][1]),
                "target_x": float(pos[tgt][0]),
                "target_y": float(pos[tgt][1]),
                "weight": float(data.get("weight", 0.01)),
            }
        )

    fig = network_chart(nodes, edges, title="Lead-Lag Transfer Entropy Network", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**Nodes:** {G.number_of_nodes()} | **Edges:** {G.number_of_edges()}")
