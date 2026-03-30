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
    "equity": "#42A5F5",
    "rates": "#66BB6A",
    "credit": "#FFA726",
    "commodities": "#AB47BC",
    "fx": "#EF5350",
    "volatility": "#8D6E63",
    "crypto": "#FF7043",
    "macro": "#78909C",
    "default": "#9E9E9E",
}

# Map every asset to its class
ASSET_TO_CLASS: Dict[str, str] = {
    # Equities
    "SPX": "equity", "NDX": "equity", "RTY": "equity",
    "XLB": "equity", "XLC": "equity", "XLE": "equity",
    "XLF": "equity", "XLI": "equity", "XLK": "equity",
    "XLP": "equity", "XLRE": "equity", "XLU": "equity",
    "XLV": "equity", "XLY": "equity",
    # Rates
    "UST_2Y": "rates", "UST_5Y": "rates", "UST_10Y": "rates",
    "UST_30Y": "rates", "REAL_YIELD_10Y": "rates",
    "BREAKEVEN_5Y": "rates", "BREAKEVEN_10Y": "rates",
    "SPREAD_2s10s": "rates",
    # Credit
    "IG_OAS": "credit", "HY_OAS": "credit", "BBB_OAS": "credit",
    "CCC_OAS": "credit", "NFCI": "credit",
    # Commodities
    "WTI": "commodities", "GOLD": "commodities", "SILVER": "commodities",
    "COPPER": "commodities", "NAT_GAS": "commodities",
    # FX
    "DXY": "fx", "EURUSD": "fx", "USDJPY": "fx", "AUDUSD": "fx",
    # Volatility
    "VIX": "volatility", "MOVE": "volatility", "OVX": "volatility",
    # Crypto
    "BTC": "crypto", "ETH": "crypto",
    # Macro
    "ISM_PMI": "macro", "UMICH_SENTIMENT": "macro",
    "INITIAL_CLAIMS": "macro", "FED_FUNDS": "macro",
}


def render_network_graph(
    te_matrix: pd.DataFrame,
    threshold: float = 0.05,
    asset_class_map: Optional[Dict[str, str]] = None,
) -> None:
    """Render an interactive network graph of lead-lag relationships.

    Args:
        te_matrix: Transfer entropy matrix (rows=source, cols=target).
        threshold: Minimum TE value to include an edge.
        asset_class_map: Optional dict mapping asset name to asset class string.
    """
    st.subheader("Lead-Lag Network Graph")

    class_map = asset_class_map or ASSET_TO_CLASS

    # ── Controls ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        threshold = st.slider(
            "TE Threshold", 0.0, 0.5, threshold, 0.005,
            help="Higher = fewer, stronger edges. Try 0.04-0.06 for clarity.",
        )
    with col2:
        layout = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"])
    with col3:
        top_n = st.selectbox("Max edges", [20, 40, 60, 100, 200, 0], index=1,
                             format_func=lambda x: "All" if x == 0 else str(x))

    # Build networkx graph
    G = nx.DiGraph()
    assets = te_matrix.index.tolist()
    G.add_nodes_from(assets)

    # Collect all candidate edges above threshold
    candidate_edges = []
    for src in assets:
        for tgt in assets:
            if src != tgt:
                te_val = float(te_matrix.loc[src, tgt])
                if te_val >= threshold:
                    candidate_edges.append((src, tgt, te_val))

    # Sort by weight and keep top N if requested
    candidate_edges.sort(key=lambda x: x[2], reverse=True)
    if top_n > 0:
        candidate_edges = candidate_edges[:top_n]

    for src, tgt, w in candidate_edges:
        G.add_edge(src, tgt, weight=w)

    # Remove isolated nodes (no edges) for cleaner visual
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_edges() == 0:
        st.warning(f"No edges above TE threshold {threshold:.4f}. Try lowering the threshold.")
        return

    # Compute layout
    try:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=2.0 / (G.number_of_nodes() ** 0.3),
                                   iterations=80)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Build node/edge lists for chart
    degree = dict(G.degree())
    nodes = []
    for node in G.nodes():
        cls = class_map.get(node, "default")
        color = ASSET_CLASS_COLORS.get(cls, ASSET_CLASS_COLORS["default"])
        d = degree.get(node, 0)
        nodes.append(
            {
                "id": node,
                "x": float(pos[node][0]),
                "y": float(pos[node][1]),
                "color": color,
                "size": 12 + d * 2,
                "label": node,
                "asset_class": cls,
                "degree": d,
            }
        )

    edges = []
    for src, tgt, data in G.edges(data=True):
        w = float(data.get("weight", 0.01))
        edges.append(
            {
                "source_x": float(pos[src][0]),
                "source_y": float(pos[src][1]),
                "target_x": float(pos[tgt][0]),
                "target_y": float(pos[tgt][1]),
                "weight": w,
                "hover": f"{src} → {tgt}: {w:.4f}",
            }
        )

    fig = network_chart(
        nodes, edges,
        title="Lead-Lag Transfer Entropy Network",
        height=850,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Nodes", G.number_of_nodes())
    col_b.metric("Edges", G.number_of_edges())

    # Top sources (most outgoing edges)
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    if out_deg:
        col_c.metric("Top Source", out_deg[0][0], f"{out_deg[0][1]} outgoing")
    # Top sinks (most incoming edges)
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    if in_deg:
        col_d.metric("Top Sink", in_deg[0][0], f"{in_deg[0][1]} incoming")

    if isolates:
        st.caption(f"Hidden {len(isolates)} isolated nodes: {', '.join(sorted(isolates))}")
