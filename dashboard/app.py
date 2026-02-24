"""Main Streamlit dashboard application."""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross-Asset Lead-Lag Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_sample_te_matrix():
    """Load or generate a sample TE matrix for demo purposes."""
    import numpy as np
    import pandas as pd

    assets = ["SPX", "NDX", "HY_OAS", "VIX", "COPPER", "DXY", "BTC", "UST_10Y"]
    n = len(assets)
    rng = np.random.default_rng(42)
    data = np.abs(rng.random((n, n))) * 0.1
    np.fill_diagonal(data, 0.0)
    return pd.DataFrame(data, index=assets, columns=assets)


def main() -> None:
    """Main dashboard application entry point."""
    st.sidebar.title("Cross-Asset Lead-Lag")
    st.sidebar.markdown("_Stochastic Lead-Lag Discovery Engine_")

    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Overview", "🕸️ Network Graph", "📊 Regime Panel", "📡 Signal Monitor", "📈 Backtest"],
    )

    if page == "🏠 Overview":
        _render_overview()
    elif page == "🕸️ Network Graph":
        _render_network_page()
    elif page == "📊 Regime Panel":
        _render_regime_page()
    elif page == "📡 Signal Monitor":
        _render_signals_page()
    elif page == "📈 Backtest":
        _render_backtest_page()


def _render_overview() -> None:
    st.title("Cross-Asset Lead-Lag Discovery Engine")
    st.markdown("""
    This dashboard provides an interactive view of cross-asset lead-lag relationships
    discovered using information-theoretic measures and neural Granger causality.

    ### Quick Start
    1. Configure your FRED API key in `.env`
    2. Run `make data` to fetch market data
    3. Use the sidebar to explore different views

    ### Pages
    - **Network Graph**: Interactive visualization of TE-based lead-lag network
    - **Regime Panel**: Markov-Switching VAR regime probabilities
    - **Signal Monitor**: Active trading signals with confidence scores
    - **Backtest**: Walk-forward backtest results and performance metrics
    """)
    st.info("⚡ Using demo data. Configure API keys and run `make data` for live data.")


def _render_network_page() -> None:
    from dashboard.pages.network_graph import render_network_graph

    te_matrix = load_sample_te_matrix()
    render_network_graph(te_matrix, threshold=0.02)


def _render_regime_page() -> None:
    import numpy as np

    from dashboard.pages.regime_panel import render_regime_panel

    T = 500
    rng = np.random.default_rng(42)
    smoothed_probs = rng.dirichlet([1, 1], size=T)
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns = pd.DataFrame(rng.standard_normal((T, 3)), index=dates, columns=["SPX", "HY_OAS", "VIX"])
    render_regime_panel(smoothed_probs, returns, dates)


def _render_signals_page() -> None:
    from dashboard.pages.signal_monitor import render_signal_monitor

    demo_signals = [
        {"leader": "VIX", "follower": "HY_OAS", "te_score": 0.15, "expected_return": 0.002, "confidence": 0.7},
        {"leader": "COPPER", "follower": "XLI", "te_score": 0.12, "expected_return": 0.003, "confidence": 0.65},
        {"leader": "HY_OAS", "follower": "SPX", "te_score": 0.10, "expected_return": -0.001, "confidence": 0.55},
    ]
    render_signal_monitor(demo_signals)


def _render_backtest_page() -> None:
    st.subheader("Backtest Results")
    st.info(
        "Run a backtest first by executing:\n```python\nfrom signals.backtest import WalkForwardBacktest\n```"
    )


if __name__ == "__main__":
    main()
