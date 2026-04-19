"""Backtest-robustness dashboard panels: TC sensitivity, deflated Sharpe, PBO.

These panels answer "how much of the headline Sharpe survives after
selection bias, transaction costs, and cross-sectional trial inflation?".
They read optional parquets produced by ``run_pipeline.py --backtest``
and gracefully degrade if the pipeline hasn't written them yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signals.metrics import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def render_robustness_panel(
    port_returns: pd.Series,
    output_dir: Path,
    n_trials_tested: int = 30,
) -> None:
    """Render deflated Sharpe, bootstrap CI, PBO, TC sensitivity, pair churn.

    Args:
        port_returns: Daily portfolio returns from the live (leakage-fixed)
            backtest. Must be indexed by date.
        output_dir: Pipeline output directory — read optional artefacts
            (tc_sensitivity, per_regime_pnl, pair_churn, trial_returns).
        n_trials_tested: How many strategy configurations were evaluated.
            Used as the selection-bias multiplier for deflated Sharpe.
    """
    st.subheader("Backtest Robustness")
    st.caption(
        "Complements the headline Sharpe with selection-bias and cost "
        "adjustments. If these panels aren't loading, rerun "
        "`python run_pipeline.py --backtest` with the current codebase."
    )

    rets = port_returns.dropna().values
    if len(rets) < 60:
        st.warning("Need at least 60 return observations for robustness stats.")
        return

    # ── Deflated Sharpe + bootstrap CI ───────────────────────────────────────
    sr = float(rets.mean() / (rets.std(ddof=1) + 1e-12) * np.sqrt(252))
    skew = float(((rets - rets.mean()) ** 3).mean() / (rets.std() ** 3 + 1e-12))
    kurt = float(((rets - rets.mean()) ** 4).mean() / (rets.std() ** 4 + 1e-12))

    dsr = deflated_sharpe_ratio(
        sharpe=sr / np.sqrt(252),
        n_obs=len(rets),
        n_trials=max(n_trials_tested, 2),
        skew=skew,
        kurtosis=kurt,
    )
    ci = bootstrap_sharpe_ci(rets, n_boot=500, random_state=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe (annualised)", f"{sr:.2f}")
    c2.metric("Bootstrap 95% CI", f"[{ci['lower']:.2f}, {ci['upper']:.2f}]")
    c3.metric("Deflated SR", f"{dsr['deflated_sharpe']:.2f}",
              "passes 95%" if dsr["passes_95"] else "below 95%")
    c4.metric("DSR p-value", f"{dsr['p_value']:.3f}")

    st.caption(
        f"Deflated SR assumes {n_trials_tested} configurations were tried "
        "during research (adjust via the pipeline's `--trials-tested` flag). "
        "Higher n_trials → more aggressive deflation."
    )

    # ── PBO (if multi-trial return matrix saved) ─────────────────────────────
    trial_path = output_dir / "backtest_trial_returns.parquet"
    if trial_path.exists():
        trials = pd.read_parquet(trial_path)
        pbo = probability_of_backtest_overfitting(trials.values)
        ca, cb = st.columns(2)
        ca.metric("Probability of Backtest Overfitting",
                  f"{pbo['pbo']:.2f}" if not np.isnan(pbo["pbo"]) else "n/a")
        cb.metric("CPCV splits", pbo["n_splits"])
        if not np.isnan(pbo["pbo"]) and pbo["pbo"] >= 0.5:
            st.warning(
                "PBO ≥ 0.5 — the top in-sample strategy is below median "
                "out-of-sample more often than not. Treat headline results "
                "as overfit."
            )
    else:
        st.info(
            "Pair-level trial returns not found (`backtest_trial_returns.parquet`). "
            "PBO requires multiple strategy configurations evaluated on the same bars."
        )

    # ── Transaction-cost sensitivity ─────────────────────────────────────────
    tc_path = output_dir / "backtest_tc_sensitivity.parquet"
    if tc_path.exists():
        tc = pd.read_parquet(tc_path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tc["tc_bps"], y=tc["sharpe"], mode="lines+markers",
            name="Sharpe vs TC", line=dict(color="#26a69a", width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
                      annotation_text="Sharpe = 1")
        fig.update_layout(
            title="Transaction-Cost Sensitivity",
            xaxis_title="Round-trip TC (bps)",
            yaxis_title="Annualised Sharpe",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Strategy Sharpe as round-trip transaction cost increases. "
            "A robust signal should stay profitable at ~1 bp (liquid ETFs) "
            "and degrade gracefully above."
        )
    else:
        st.info(
            "No TC sensitivity artefact (`backtest_tc_sensitivity.parquet`). "
            "Add a sweep in `run_pipeline.py` over `--tc-bps` values."
        )

    # ── Per-regime attribution ───────────────────────────────────────────────
    regime_path = output_dir / "backtest_per_regime.parquet"
    if regime_path.exists():
        rg = pd.read_parquet(regime_path)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rg.index.astype(str), y=rg.get("sharpe", pd.Series(dtype=float)).values,
            name="Sharpe by regime", marker_color="#7e57c2",
        ))
        fig.update_layout(
            title="Per-Regime Sharpe Attribution",
            xaxis_title="Regime", yaxis_title="Annualised Sharpe",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pair churn over time ─────────────────────────────────────────────────
    churn_path = output_dir / "backtest_pair_churn.parquet"
    if churn_path.exists():
        churn = pd.read_parquet(churn_path)
        fig = go.Figure()
        if "active_pairs" in churn.columns:
            fig.add_trace(go.Scatter(
                x=churn.index, y=churn["active_pairs"], mode="lines",
                name="Active pairs", line=dict(color="#ff9800"),
            ))
        if "new_pairs" in churn.columns:
            fig.add_trace(go.Scatter(
                x=churn.index, y=churn["new_pairs"], mode="lines",
                name="New pairs this refresh", line=dict(color="#26a69a", dash="dot"),
            ))
        fig.update_layout(
            title="Pair Churn Over Time",
            xaxis_title="Date", yaxis_title="Pairs",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "High churn = the model re-selects a different leader set each "
            "refresh. Persistent edges should show low churn."
        )

    # ── Turnover (computed if not saved) ─────────────────────────────────────
    turnover_path = output_dir / "backtest_turnover.parquet"
    if turnover_path.exists():
        turn = pd.read_parquet(turnover_path)
        if "turnover" in turn.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=turn.index, y=turn["turnover"],
                mode="lines", line=dict(color="#ef5350"),
                name="Turnover (Σ|Δw|)",
            ))
            fig.update_layout(
                title="Daily Portfolio Turnover",
                xaxis_title="Date", yaxis_title="Σ|Δweights|",
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
            avg_to = float(turn["turnover"].mean())
            st.metric("Avg daily turnover", f"{avg_to:.3f}")
