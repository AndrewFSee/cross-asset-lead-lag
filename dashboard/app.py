"""Main Streamlit dashboard application."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross-Asset Lead-Lag Dashboard",
    page_icon="📊",
    layout="wide",
)

OUTPUT_DIR = Path("data/outputs")


@st.cache_data(ttl=3600)
def load_te_matrix(lag: int = 1) -> pd.DataFrame:
    """Load TE matrix from pipeline outputs, or generate demo data."""
    path = OUTPUT_DIR / f"te_matrix_lag{lag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Fallback demo
    assets = ["SPX", "NDX", "HY_OAS", "VIX", "COPPER", "DXY", "BTC", "UST_10Y"]
    n = len(assets)
    rng = np.random.default_rng(42)
    data = np.abs(rng.random((n, n))) * 0.1
    np.fill_diagonal(data, 0.0)
    return pd.DataFrame(data, index=assets, columns=assets)


@st.cache_data(ttl=3600)
def load_lasso_matrix() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "lasso_var_matrix.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_returns_panel() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "returns_panel.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_regime_labels() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "regime_labels.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_backtest_equity() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "backtest_equity.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_backtest_metrics() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "backtest_metrics.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_benchmark_equity() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "backtest_benchmark_equity.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_benchmark_metrics() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "backtest_benchmark_metrics.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _has_real_data() -> bool:
    return (OUTPUT_DIR / "te_matrix_lag1.parquet").exists()


@st.cache_data(ttl=3600)
def load_te_decay() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "te_decay.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_regime_probs() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "regime_probs.parquet"
    return pd.read_parquet(path) if path.exists() else None


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

    if _has_real_data():
        st.success("Pipeline outputs detected — showing real data.")
        panel = load_returns_panel()
        if panel is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Assets", len(panel.columns))
            col2.metric("Observations", len(panel))
            col3.metric("Date Range", f"{panel.index.min().date()} → {panel.index.max().date()}")

        te = load_te_matrix(1)
        decay = load_te_decay()

        # ── Tradable Signals (primary view) ──────────────────────────
        if decay is not None and not decay.empty:
            _render_tradable_signals(decay)

        # ── Full TE Decay Analysis (expandable) ─────────────────────
        if decay is not None and not decay.empty:
            with st.expander("Full TE Decay Analysis (all 40 pairs)", expanded=False):
                _render_decay_section(decay)

        if te is not None:
            with st.expander("Raw Top Lead-Lag Pairs (TE lag=1)", expanded=False):
                pairs = []
                for src in te.index:
                    for tgt in te.columns:
                        if src != tgt:
                            pairs.append({"Source": src, "Target": tgt, "TE": te.loc[src, tgt]})
                top = pd.DataFrame(pairs).nlargest(10, "TE")
                st.dataframe(top, use_container_width=True, hide_index=True)

    else:
        st.info("No pipeline outputs found. Run `python run_pipeline.py` first.")

    st.markdown("""
    ### Pages
    - **Network Graph**: Interactive TE-based lead-lag network
    - **Regime Panel**: Markov-Switching VAR regime probabilities
    - **Signal Monitor**: Active trading signals with confidence scores
    - **Backtest**: Walk-forward backtest results
    """)


def _render_tradable_signals(decay: pd.DataFrame) -> None:
    """Show only tradable pairs prominently at the top of the overview."""
    st.subheader("Tradable Lead-Lag Signals")
    st.caption(
        "Pairs with information transfer that persists long enough for "
        "daily execution — sorted by TE strength.  "
        "**Next-day**: signal holds to day 2.  "
        "**Swing**: signal persists 5+ days."
    )

    tradable_cats = {"Next-day tradable", "Swing (tradable)", "Slow decay (tradable)"}
    lag1 = decay[decay["lag"] == decay.groupby(["source", "target"])["lag"].transform("min")]
    tradable = lag1[lag1["category"].isin(tradable_cats)].copy()

    if tradable.empty:
        st.warning("No tradable pairs found in current data.")
        return

    tradable = tradable.sort_values("te", ascending=False)

    cat_icons = {
        "Next-day tradable": "\U0001f7e1",
        "Swing (tradable)": "\U0001f7e2",
        "Slow decay (tradable)": "\U0001f7e2",
    }

    rows = []
    for _, r in tradable.iterrows():
        arrow = "\u2191\u2191" if r.get("direction") == "same" else "\u2191\u2193"
        rows.append({
            "Pair": f"{r['source']} \u2192 {r['target']}",
            "TE": round(r["te"], 4),
            "Direction": f"{arrow} {r.get('direction', '')}",
            "Corr (lag-1)": f"{r.get('lagged_corr', 0):+.3f}",
            "Half-Life": f"{int(r['half_life'])}d",
            "Category": f"{cat_icons.get(r['category'], '')} {r['category']}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick metric row for top 3
    top3 = tradable.head(3)
    cols = st.columns(min(3, len(top3)))
    for col, (_, r) in zip(cols, top3.iterrows()):
        arrow = "\u2191\u2191" if r.get("direction") == "same" else "\u2191\u2193"
        col.metric(
            f"{r['source']} \u2192 {r['target']} {arrow}",
            f"TE {r['te']:.4f}",
            f"half-life {int(r['half_life'])}d",
        )


def _render_decay_section(decay: pd.DataFrame) -> None:
    """Render the TE decay analysis: chart + tradability table."""
    import plotly.graph_objects as go

    st.subheader("TE Decay Profiles")
    st.caption(
        "How quickly does information transfer decay across lags?  "
        "Pairs where TE persists over multiple days are potentially "
        "tradable; pairs that collapse immediately require HFT infrastructure."
    )

    # Build summary table: one row per pair
    pair_rows = []
    has_direction = "direction" in decay.columns
    for (src, tgt), grp in decay.groupby(["source", "target"], sort=False):
        lag1_te = grp.loc[grp["lag"] == grp["lag"].min(), "te"].iloc[0]
        hl = grp["half_life"].iloc[0]
        cat = grp["category"].iloc[0]
        row = {
            "Pair": f"{src} \u2192 {tgt}",
            "TE (lag=1)": round(lag1_te, 4),
            "Half-Life (days)": hl,
            "Category": cat,
        }
        if has_direction:
            corr = grp["lagged_corr"].iloc[0]
            direction = grp["direction"].iloc[0]
            arrow = "\u2191\u2191" if direction == "same" else "\u2191\u2193"
            row["Direction"] = f"{arrow} {direction} (r={corr:+.3f})"
        pair_rows.append(row)
    summary_df = pd.DataFrame(pair_rows)

    # Color-code categories
    cat_colors = {
        "HFT only": "\U0001f534",
        "Short-term": "\U0001f7e0",
        "Next-day tradable": "\U0001f7e1",
        "Swing (tradable)": "\U0001f7e2",
        "Slow decay (tradable)": "\U0001f7e2",
    }
    summary_df["Category"] = summary_df["Category"].map(
        lambda c: f"{cat_colors.get(c, '')} {c}"
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # \u2500\u2500 Decay chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Filter for chart
    categories = sorted(decay["category"].unique())
    selected_cats = st.multiselect(
        "Filter decay chart by category",
        categories,
        default=[c for c in categories if "tradable" in c.lower()],
    )
    chart_pairs = decay[decay["category"].isin(selected_cats)] if selected_cats else decay

    fig = go.Figure()
    for (src, tgt), grp in chart_pairs.groupby(["source", "target"], sort=False):
        cat = grp["category"].iloc[0]
        dash = "solid" if "tradable" in cat.lower() else "dot"
        if has_direction:
            arrow = "\u2191\u2191" if grp["direction"].iloc[0] == "same" else "\u2191\u2193"
            label = f"{src} \u2192 {tgt} {arrow}"
        else:
            label = f"{src} \u2192 {tgt}"
        fig.add_trace(
            go.Scatter(
                x=grp["lag"],
                y=grp["te_norm"],
                mode="lines+markers",
                name=label,
                line=dict(dash=dash),
            )
        )

    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="50% (half-life threshold)")
    fig.update_layout(
        title="Normalized TE Decay (1.0 = lag-1 value)",
        xaxis_title="Lag (business days)",
        yaxis_title="TE / TE(lag=1)",
        height=450,
        yaxis=dict(range=[0, 1.1]),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Solid lines** = tradable pairs (next-day / swing / slow decay).  "
        "**Dotted lines** = fast-decaying pairs (HFT / short-term).  "
        "Grey dashed line = 50% half-life threshold.  "
        "**\u2191\u2191** = same direction, **\u2191\u2193** = inverse."
    )


def _render_network_page() -> None:
    from dashboard.views.network_graph import render_network_graph

    available_lags = sorted(
        int(p.stem.replace("te_matrix_lag", ""))
        for p in OUTPUT_DIR.glob("te_matrix_lag*.parquet")
    )
    if available_lags:
        lag = st.sidebar.selectbox("TE Lag", available_lags, index=0)
    else:
        lag = 1
    te_matrix = load_te_matrix(lag)
    render_network_graph(te_matrix, threshold=0.05)


def _render_regime_page() -> None:
    from dashboard.views.regime_panel import render_regime_panel

    regime_labels = load_regime_labels()
    regime_probs = load_regime_probs()
    panel = load_returns_panel()

    if regime_labels is not None and panel is not None:
        labels = regime_labels["regime"].values
        n_regimes = int(labels.max()) + 1
        dates = regime_labels.index

        # Use smooth posterior probs if available, else one-hot from labels
        if regime_probs is not None:
            smoothed_probs = regime_probs.values
            # Align lengths
            min_len = min(len(smoothed_probs), len(dates))
            smoothed_probs = smoothed_probs[:min_len]
            dates = dates[:min_len]
        else:
            T = len(labels)
            smoothed_probs = np.zeros((T, n_regimes))
            for i, lab in enumerate(labels):
                smoothed_probs[i, int(lab)] = 1.0

        # Use panel columns that overlap with regime period
        overlap = panel.loc[panel.index.isin(dates)]
        if len(overlap) > len(dates):
            overlap = overlap.iloc[:len(dates)]
        elif len(overlap) < len(smoothed_probs):
            smoothed_probs = smoothed_probs[:len(overlap)]
            dates = dates[:len(overlap)]
        render_regime_panel(smoothed_probs, overlap, dates)
    else:
        st.info("No regime data found. Run the full pipeline first.")


def _render_signals_page() -> None:
    from dashboard.views.signal_monitor import render_signal_monitor

    te = load_te_matrix(1)
    panel = load_returns_panel()
    decay = load_te_decay()

    if not (_has_real_data() and te is not None and panel is not None):
        st.info("No pipeline data found. Run `python run_pipeline.py` first.")
        return

    # Market-type classification for timing analysis
    # FX/crypto trade outside equity hours; FRED data is weekly
    MARKET_TYPE = {
        "SPX": "equity", "NDX": "equity", "RTY": "equity",
        "XLB": "equity", "XLC": "equity", "XLE": "equity",
        "XLF": "equity", "XLI": "equity", "XLK": "equity",
        "XLP": "equity", "XLRE": "equity", "XLU": "equity",
        "XLV": "equity", "XLY": "equity", "VIX": "equity",
        "EURUSD": "fx", "USDJPY": "fx", "GBPUSD": "fx",
        "AUDUSD": "fx", "USDCAD": "fx", "USDCNH": "fx", "DXY": "fx",
        "BTC": "crypto", "ETH": "crypto",
        "GOLD": "commodity", "COPPER": "commodity", "BRENT": "commodity",
        "UST_2Y": "bond", "UST_10Y": "bond", "UST_30Y": "bond",
        "REAL_YIELD_10Y": "bond", "BREAKEVEN_5Y": "bond",
        "BREAKEVEN_10Y": "bond", "MOVE": "bond",
        "IG_OAS": "credit", "HY_OAS": "credit", "BBB_OAS": "credit",
        "CCC_OAS": "credit",
        "NFCI": "weekly_macro", "CFNAI": "weekly_macro",
    }

    def _assess_timing(src: str, tgt: str) -> str:
        """Is the signal actionable given market hours?

        Returns a timing tag:
          'actionable'  – follower hasn't traded yet when leader closes
          'likely priced' – follower trades concurrently / reacts before
                           you can act on the leader's close
          'delayed data' – leader is weekly/monthly macro data
        """
        src_type = MARKET_TYPE.get(src, "other")
        tgt_type = MARKET_TYPE.get(tgt, "other")

        # Weekly macro data (NFCI, CFNAI) — signals are valid but act on
        # release day, not daily close
        if src_type == "weekly_macro":
            return "delayed data"

        # FX/crypto → equity: FX moves overnight, equity hasn't opened
        if src_type in ("fx", "crypto") and tgt_type == "equity":
            return "actionable"
        # Equity → equity (same close time → react next open)
        if src_type == "equity" and tgt_type == "equity":
            return "actionable"
        # Bond/credit → equity
        if src_type in ("bond", "credit") and tgt_type == "equity":
            return "actionable"
        # Equity → FX: FX trades ~24h, likely reprices before next equity open
        if src_type == "equity" and tgt_type in ("fx", "crypto"):
            return "likely priced"
        # FX → FX (same session, concurrent)
        if src_type == "fx" and tgt_type == "fx":
            return "likely priced"
        # Anything → crypto (24/7, reacts immediately)
        if tgt_type == "crypto":
            return "likely priced"
        # Bond → FX, credit → FX — FX reacts in real time
        if src_type in ("bond", "credit") and tgt_type == "fx":
            return "likely priced"
        return "actionable"

    # Find the last *trading* day (skip weekends/holidays where most assets are 0)
    for offset in range(min(5, len(panel))):
        row = panel.iloc[-(1 + offset)]
        if (row.abs() > 1e-10).sum() > len(panel.columns) * 0.4:
            break
    latest_returns = row
    last_date = panel.index[-(1 + offset)].date()

    # Build lagged correlation lookup from decay data
    corr_lookup = {}
    if decay is not None and "lagged_corr" in decay.columns:
        lag1 = decay[decay["lag"] == decay.groupby(["source", "target"])["lag"].transform("min")]
        for _, r in lag1.iterrows():
            corr_lookup[(r["source"], r["target"])] = {
                "lagged_corr": r["lagged_corr"],
                "direction": r.get("direction", ""),
                "category": r.get("category", ""),
                "half_life": r.get("half_life", 0),
            }

    # For pairs without decay data, compute lagged corr on the fly
    def _get_lagged_corr(src: str, tgt: str) -> float:
        if (src, tgt) in corr_lookup:
            return corr_lookup[(src, tgt)]["lagged_corr"]
        if src in panel.columns and tgt in panel.columns:
            import numpy as _np
            s = panel[src].dropna().values
            t = panel[tgt].dropna().values
            n = min(len(s), len(t)) - 1
            if n > 50:
                return float(_np.corrcoef(s[-n-1:-1], t[-n:])[0, 1])
        return 0.0

    # Compute regression beta: beta = corr * std(follower) / std(leader)
    stds = panel.iloc[-252:].std()  # trailing 1yr vol

    # Pre-compute directional hit rate for top pairs over trailing 1yr
    import numpy as _np
    _lookback = min(252, len(panel) - 1)
    _leader_rets = panel.iloc[-_lookback - 1:-1]   # day t
    _follower_rets = panel.iloc[-_lookback:]        # day t+1
    # Align indices
    _leader_rets = _leader_rets.iloc[:_lookback]
    _follower_rets = _follower_rets.iloc[:_lookback]

    def _hit_rate(src: str, tgt: str, corr: float) -> tuple:
        """Compute directional hit rate: fraction of days where
        follower moved in the beta-predicted direction.
        Returns (hit_rate, n_obs)."""
        if src not in _leader_rets.columns or tgt not in _follower_rets.columns:
            return (float('nan'), 0)
        s = _leader_rets[src].values
        t = _follower_rets[tgt].values
        # Only count days where leader actually moved (|ret| > 1e-8)
        mask = _np.abs(s) > 1e-8
        if mask.sum() < 30:
            return (float('nan'), int(mask.sum()))
        predicted_dir = _np.sign(s[mask] * corr)  # +corr: same dir, -corr: opposite
        actual_dir = _np.sign(t[mask])
        # Exclude days where follower didn't move
        moved = actual_dir != 0
        if moved.sum() < 20:
            return (float('nan'), int(moved.sum()))
        hits = (predicted_dir[moved] == actual_dir[moved]).sum()
        return (float(hits / moved.sum()), int(moved.sum()))

    # Build top pairs sorted by TE
    pairs = []
    for src in te.index:
        for tgt in te.columns:
            if src != tgt:
                pairs.append((src, tgt, float(te.loc[src, tgt])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    signals = []
    for src, tgt, te_val in pairs[:20]:
        corr = _get_lagged_corr(src, tgt)
        leader_ret = float(latest_returns.get(src, 0.0))

        # Simple linear forecast: E[follower_t+1] = beta * leader_t
        std_src = stds.get(src, 1e-10)
        std_tgt = stds.get(tgt, 1e-10)
        beta = corr * (std_tgt / std_src) if std_src > 1e-10 else 0.0
        expected_ret = beta * leader_ret

        meta = corr_lookup.get((src, tgt), {})
        timing = _assess_timing(src, tgt)
        hr, hr_n = _hit_rate(src, tgt, corr)
        signals.append({
            "leader": src,
            "follower": tgt,
            "te_score": te_val,
            "expected_return": expected_ret,
            "te_strength": min(te_val / pairs[0][2], 1.0),
            "hit_rate": hr,
            "hit_rate_n": hr_n,
            "leader_return": leader_ret,
            "lagged_corr": corr,
            "beta": beta,
            "direction": meta.get("direction", "same" if corr >= 0 else "inverse"),
            "category": meta.get("category", ""),
            "half_life": meta.get("half_life", 0),
            "timing": timing,
        })

    render_signal_monitor(signals, as_of_date=last_date)


def _render_backtest_page() -> None:
    from dashboard.views.backtest_results import render_backtest_results
    from dashboard.views.robustness_panel import render_robustness_panel

    equity = load_backtest_equity()
    metrics = load_backtest_metrics()
    bench_equity = load_benchmark_equity()
    bench_metrics = load_benchmark_metrics()

    if equity is not None and metrics is not None:
        metrics_dict = metrics["value"].to_dict()
        bench_eq = bench_equity["equity"] if bench_equity is not None else None
        bench_m = bench_metrics["value"].to_dict() if bench_metrics is not None else None
        render_backtest_results(
            equity["equity"], metrics_dict,
            benchmark_equity=bench_eq, benchmark_metrics=bench_m,
        )
        st.divider()
        port_returns = equity["equity"].pct_change().dropna()
        render_robustness_panel(
            port_returns, output_dir=OUTPUT_DIR,
            n_trials_tested=int(metrics_dict.get("n_trials_tested", 30)),
        )
    else:
        st.subheader("Backtest Results")
        st.info(
            "No backtest results found. Run:\n"
            "```\npython run_pipeline.py --skip-fetch --backtest\n```"
        )


if __name__ == "__main__":
    main()
