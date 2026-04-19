"""Tests for regime-conditional signal selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.generator import compute_per_regime_te, regime_conditional_te_weights


class TestRegimeConditionalWeights:
    def test_picks_current_regime_matrix(self):
        te0 = pd.DataFrame(
            [[0.0, 0.5, 0.0], [0.1, 0.0, 0.0], [0.05, 0.0, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        te1 = pd.DataFrame(
            [[0.0, 0.02, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        w0 = regime_conditional_te_weights(
            {0: te0, 1: te1}, current_regime=0,
            followers=["B"], te_threshold=0.01, top_k_per_follower=2,
        )
        w1 = regime_conditional_te_weights(
            {0: te0, 1: te1}, current_regime=1,
            followers=["B"], te_threshold=0.01, top_k_per_follower=2,
        )
        assert w0["B"][0]["leader"] == "A"
        assert w1["B"][0]["leader"] == "A"
        # Regime 0 gives A a stronger TE -> larger weight than regime 1 for B
        assert w0["B"][0]["te"] > w1["B"][0]["te"]

    def test_weights_sum_to_one_per_follower(self):
        te = pd.DataFrame(
            [[0.0, 0.4, 0.3], [0.2, 0.0, 0.1], [0.15, 0.05, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        out = regime_conditional_te_weights(
            {0: te}, current_regime=0, followers=["A", "B"],
            te_threshold=0.01, top_k_per_follower=3,
        )
        for fol, edges in out.items():
            s = sum(e["weight"] for e in edges)
            assert abs(s - 1.0) < 1e-9, f"weights for {fol} sum to {s}"

    def test_fallback_regime(self):
        te = pd.DataFrame(
            [[0.0, 0.3], [0.1, 0.0]], index=["A", "B"], columns=["A", "B"],
        )
        out = regime_conditional_te_weights(
            {0: te}, current_regime=99,
            followers=["B"], fallback_regime=0,
        )
        assert "B" in out
        out_none = regime_conditional_te_weights(
            {0: te}, current_regime=99, followers=["B"],
        )
        assert out_none == {}


class TestPerRegimeTE:
    def test_skips_sparse_regimes(self, rng):
        T = 400
        returns = pd.DataFrame(
            rng.standard_normal((T, 3)) * 0.01,
            columns=["A", "B", "C"],
            index=pd.date_range("2022-01-01", periods=T, freq="B"),
        )
        labels = np.zeros(T, dtype=int)
        labels[-30:] = 1  # only 30 bars in regime 1

        def fake_te(series_dict, **kwargs):
            names = list(series_dict)
            # Uniform tidy DF
            rows = [
                {"source": s, "target": t, "te": 0.1}
                for s in names for t in names if s != t
            ]
            return pd.DataFrame(rows)

        out = compute_per_regime_te(
            returns, labels, te_fn=fake_te, min_bars_per_regime=100,
        )
        assert 0 in out
        assert 1 not in out
