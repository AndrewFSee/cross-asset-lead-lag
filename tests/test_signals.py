"""Tests for signal generation and Bayesian model averaging."""

from __future__ import annotations

from signals.generator import LeadSignal, bayesian_model_average


class TestLeadSignal:
    """Tests for LeadSignal dataclass."""

    def test_raw_signal_computation(self):
        """raw_signal should return coefficient * leader_return."""
        sig = LeadSignal(
            leader="A",
            follower="B",
            lag=1,
            te_score=0.1,
            regime=0,
            coefficient=0.5,
            leader_return=0.02,
        )
        assert abs(sig.raw_signal - 0.01) < 1e-10

    def test_confidence_in_range(self):
        """Confidence should be in [0, 1]."""
        for te in [0.0, 0.05, 0.1, 0.5, 1.0]:
            sig = LeadSignal("A", "B", 1, te, 0, 0.5, 0.01)
            assert 0.0 <= sig.confidence <= 1.0

    def test_positive_te_higher_confidence(self):
        """Higher TE score should yield higher confidence."""
        sig_low = LeadSignal("A", "B", 1, 0.01, 0, 0.5, 0.01)
        sig_high = LeadSignal("A", "B", 1, 0.5, 0, 0.5, 0.01)
        assert sig_high.confidence > sig_low.confidence


class TestBayesianModelAverage:
    """Tests for bayesian_model_average."""

    def test_output_keys(self):
        """BMA output should contain expected keys."""
        signals = [
            LeadSignal("A", "C", 1, 0.1, 0, 0.5, 0.02),
            LeadSignal("B", "C", 1, 0.2, 0, 0.3, -0.01),
        ]
        result = bayesian_model_average(signals)
        assert "C" in result
        assert all(
            k in result["C"]
            for k in ["expected_return", "confidence", "n_leaders", "dominant_leader", "leaders"]
        )

    def test_dominant_leader_is_highest_te(self):
        """Dominant leader should be the one with highest TE score."""
        signals = [
            LeadSignal("A", "C", 1, 0.1, 0, 0.5, 0.01),
            LeadSignal("B", "C", 1, 0.3, 0, 0.5, 0.01),
        ]
        result = bayesian_model_average(signals)
        assert result["C"]["dominant_leader"] == "B"

    def test_weights_sum_to_one(self):
        """Leader weights in BMA should sum to 1."""
        signals = [
            LeadSignal("A", "C", 1, 0.1, 0, 0.5, 0.01),
            LeadSignal("B", "C", 1, 0.2, 0, 0.3, 0.02),
            LeadSignal("D", "C", 1, 0.05, 0, 0.1, -0.01),
        ]
        result = bayesian_model_average(signals)
        total_weight = sum(item["weight"] for item in result["C"]["leaders"])
        assert abs(total_weight - 1.0) < 1e-9
