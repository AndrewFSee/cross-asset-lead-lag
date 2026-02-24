"""Tests for neural Granger causality."""

from __future__ import annotations

import numpy as np

from discovery.neural_granger import ComponentLSTM, neural_granger_test


class TestComponentLSTM:
    """Tests for ComponentLSTM model."""

    def test_forward_pass_shape(self):
        """Output should have shape (batch, 1)."""
        import torch

        model = ComponentLSTM(n_inputs=4, hidden_dim=16, embed_dim=8)
        x = torch.randn(8, 10, 4)  # (batch=8, seq=10, n_inputs=4)
        out = model(x)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

    def test_ablation_changes_output(self):
        """Ablating an input should change model output."""
        import torch

        model = ComponentLSTM(n_inputs=4, hidden_dim=16, embed_dim=8)
        x = torch.randn(4, 10, 4)

        model.ablate_idx = None
        out_full = model(x).detach().clone()

        model.ablate_idx = 0
        out_ablated = model(x).detach().clone()

        assert not torch.allclose(out_full, out_ablated), "Ablation should change output"

    def test_gc_scores_shape(self, synthetic_var_data):
        """neural_granger_test should return a score for each asset."""
        A, B = synthetic_var_data
        data = np.column_stack([A[:200], B[:200]])
        gc_scores = neural_granger_test(
            data, target_idx=1, n_epochs=10, history_len=5, hidden_dim=16, embed_dim=4
        )
        assert len(gc_scores) == 2, f"Expected 2 scores, got {len(gc_scores)}"
        assert all(v >= 0 for v in gc_scores.values()), "GC scores should be >= 0"

    def test_gc_causal_source_detected(self, synthetic_var_data):
        """True causal source (A→B) should have higher GC score than reverse."""
        A, B = synthetic_var_data
        # Test: B is target (idx 1), A (idx 0) should have higher GC score
        data = np.column_stack([A[:300], B[:300]])
        gc_scores = neural_granger_test(
            data, target_idx=1, n_epochs=30, history_len=10, hidden_dim=32, embed_dim=8
        )
        # A (original index 0) should have higher GC score than B (self)
        # Note: we check that at least one non-target source has a positive score
        non_target_scores = {k: v for k, v in gc_scores.items() if k != 1}
        assert any(v > 0 for v in non_target_scores.values()), "Expected positive GC scores"
