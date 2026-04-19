"""Tests for DeltaLag cross-attention lead-lag model."""

from __future__ import annotations

import numpy as np

from models.delta_lag import (
    DeltaLagModel,
    delta_lag_leader_graph,
    fit_delta_lag,
    predict_delta_lag,
)


class TestDeltaLagFit:
    def test_recovers_known_lag(self, rng):
        """With target[t] = 0.8 * leader0[t-3] + noise, the fitted model's
        argmax attention for leader 0 should be 3."""
        T = 600
        N = 4
        true_lag = 3
        leaders = rng.standard_normal((T, N)).astype(np.float32)
        target = np.zeros(T, dtype=np.float32)
        for t in range(true_lag, T):
            target[t] = 0.8 * leaders[t - true_lag, 0] + 0.2 * rng.standard_normal()

        model, info = fit_delta_lag(
            leaders, target,
            max_lag=6, n_epochs=400, learning_rate=5e-2,
            l1_penalty=1e-4, random_state=0,
        )
        lags = model.leader_lags()
        assert lags[0] == true_lag, f"Expected leader 0 lag={true_lag}, got {lags[0]}"
        assert info["train_ic"] > 0.2

    def test_sparsity_pushes_noise_leaders_down(self, rng):
        """With l1 penalty, a leader unrelated to the target should have
        small leader_scale vs the true driver."""
        T = 500
        N = 3
        leaders = rng.standard_normal((T, N)).astype(np.float32)
        target = np.zeros(T, dtype=np.float32)
        for t in range(2, T):
            target[t] = 0.9 * leaders[t - 2, 0] + 0.15 * rng.standard_normal()

        model, _ = fit_delta_lag(
            leaders, target,
            max_lag=5, n_epochs=400, learning_rate=5e-2,
            l1_penalty=5e-2, random_state=0,
        )
        import torch.nn.functional as F
        scales = F.softplus(model.leader_scale).detach().numpy()
        assert scales[0] > scales[1] and scales[0] > scales[2]


class TestDeltaLagPredict:
    def test_output_shape_and_finite(self, rng):
        T = 200
        N = 3
        leaders = rng.standard_normal((T, N)).astype(np.float32)
        target = rng.standard_normal(T).astype(np.float32)
        model, _ = fit_delta_lag(
            leaders, target, max_lag=4, n_epochs=50, random_state=0,
        )
        pred = predict_delta_lag(model, leaders)
        assert pred.shape == (T - model.max_lag,)
        assert np.all(np.isfinite(pred))


class TestLeaderGraph:
    def test_returns_tidy_edges(self, rng):
        T = 300
        leaders = rng.standard_normal((T, 2)).astype(np.float32)
        target = rng.standard_normal(T).astype(np.float32)
        model, _ = fit_delta_lag(
            leaders, target, max_lag=3, n_epochs=30, random_state=0,
        )
        edges = delta_lag_leader_graph(
            model, leader_names=["A", "B"], target_name="TGT", min_weight=0.0,
        )
        assert len(edges) > 0
        for e in edges:
            assert e["target"] == "TGT"
            assert e["source"] in {"A", "B"}
            assert 1 <= e["lag"] <= model.max_lag


class TestTopKMask:
    def test_top_k_produces_sparse_attention(self, rng):
        model = DeltaLagModel(n_leaders=3, max_lag=8, top_k=2)
        with __import__("torch").no_grad():
            model.lag_logits.data = __import__("torch").randn(3, 8)
        att = model.attention().detach().numpy()
        # Each row must have exactly `top_k` non-negligible entries
        for i in range(3):
            assert (att[i] > 1e-4).sum() == 2
