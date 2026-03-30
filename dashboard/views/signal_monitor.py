"""Live signal monitoring dashboard."""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_signal_monitor(
    current_signals: List[dict],
    as_of_date=None,
) -> None:
    """Render the current active signals dashboard.

    Args:
        current_signals: List of signal dicts with keys:
            leader, follower, te_score, expected_return, confidence,
            leader_return, lagged_corr, beta, direction, category, half_life.
        as_of_date: Date of the latest observation.
    """
    st.subheader("Active Lead-Lag Signals")

    if as_of_date:
        st.caption(f"As of **{as_of_date}** (leader moved → follower expected next day)")

    if not current_signals:
        st.info("No active signals above threshold. Adjust parameters or fetch fresh data.")
        return

    df = pd.DataFrame(current_signals)
    nonzero = df[df["expected_return"].abs() > 1e-8] if "expected_return" in df.columns else df
    has_hit_rate = "hit_rate" in df.columns

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active signals", len(df))
    if len(nonzero):
        best = nonzero.iloc[nonzero["expected_return"].abs().argmax()]
        c2.metric(
            "Strongest signal",
            f"{best['leader']} → {best['follower']}",
            f"{best['expected_return']:+.2%}",
        )
        c3.metric("Mean |expected return|", f"{nonzero['expected_return'].abs().mean():.2%}")
    if has_hit_rate:
        valid_hr = df["hit_rate"].dropna()
        if len(valid_hr):
            best_hr = df.loc[df["hit_rate"].idxmax()]
            c4.metric(
                "Best hit rate",
                f"{best_hr['hit_rate']:.0%} ({best_hr['leader']}→{best_hr['follower']})",
                f"n={int(best_hr.get('hit_rate_n', 0))}",
            )

    st.divider()

    # Timing color map
    TIMING_STYLE = {
        "actionable": ("green", "✅ actionable"),
        "likely priced": ("orange", "⚠️ likely priced"),
        "delayed data": ("blue", "📅 delayed data"),
    }

    # Separate actionable from likely-priced
    has_timing = "timing" in df.columns
    if has_timing:
        actionable = df[df["timing"] == "actionable"]
        other = df[df["timing"] != "actionable"]
        if len(actionable):
            st.markdown("#### Actionable signals")
            st.caption("Follower market hasn't reacted yet — tradable at next open")
        sections = [("actionable", actionable)]
        if len(other):
            sections.append(("other", other))
    else:
        sections = [("all", df)]

    for section_key, section_df in sections:
        if has_timing and section_key == "other" and len(section_df):
            st.markdown("#### Likely already priced")
            st.caption("Follower market trades concurrently with leader — signal may be stale")

        for _, row in section_df.iterrows():
            exp_ret = row.get("expected_return", 0.0)
            leader_ret = row.get("leader_return", 0.0)
            corr = row.get("lagged_corr", 0.0)
            direction = row.get("direction", "")
            category = row.get("category", "")
            conf = float(row.get("confidence", 0.0))
            timing = row.get("timing", "")

            dir_label = "+corr" if direction == "same" else "−corr"
            ret_arrow = "▲" if exp_ret > 0 else ("▼" if exp_ret < 0 else "—")
            color = "green" if exp_ret > 0 else ("red" if exp_ret < 0 else "gray")

            hit_rate = row.get("hit_rate", float('nan'))
            hit_n = int(row.get("hit_rate_n", 0))
            te_str = float(row.get("te_strength", row.get("confidence", 0.0)))

            col1, col2, col3, col4, col5 = st.columns([2.2, 1.2, 1.3, 1.3, 1.0])
            with col1:
                label = f"**{row['leader']}** → **{row['follower']}**  `{dir_label}`"
                if category:
                    label += f"  `{category}`"
                st.write(label)
            with col2:
                st.write(f"Leader: **{leader_ret:+.3%}**")
            with col3:
                st.markdown(f":{color}[{ret_arrow} E\\[r\\] = **{exp_ret:+.4f}**]")
            with col4:
                import math
                if not math.isnan(hit_rate):
                    hr_color = "green" if hit_rate >= 0.55 else ("orange" if hit_rate >= 0.50 else "red")
                    st.markdown(f":{hr_color}[Hit rate: **{hit_rate:.0%}** (n={hit_n})]")
                else:
                    st.write("Hit rate: —")
            with col5:
                st.progress(te_str, text=f"TE {te_str:.0%}")

    # Expected return bar chart
    if "expected_return" in df.columns and "follower" in df.columns:
        st.subheader("Expected Return by Signal")
        labels = [f"{r['leader']}→{r['follower']}" for _, r in df.iterrows()]
        vals = df["expected_return"].tolist()
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=["#26a69a" if v > 0 else "#ef5350" for v in vals],
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=400,
            yaxis_title="Expected Return",
            xaxis_tickangle=-45,
            margin=dict(b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detail expander
    with st.expander("Signal details"):
        detail_cols = ["leader", "follower", "te_score", "leader_return",
                       "lagged_corr", "beta", "expected_return", "hit_rate",
                       "hit_rate_n", "direction", "category", "half_life",
                       "timing", "te_strength"]
        show = [c for c in detail_cols if c in df.columns]
        st.dataframe(df[show].style.format({
            "te_score": "{:.4f}",
            "leader_return": "{:+.4%}",
            "lagged_corr": "{:+.3f}",
            "beta": "{:+.4f}",
            "expected_return": "{:+.4f}",
            "hit_rate": "{:.1%}",
            "half_life": "{:.1f}",
            "te_strength": "{:.1%}",
        }), use_container_width=True)
