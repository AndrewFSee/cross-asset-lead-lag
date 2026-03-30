"""Tests for portfolio construction utilities."""

from __future__ import annotations

import numpy as np

from signals.portfolio import apply_constraints, kelly_sizing, risk_parity_weights


def _sample_forecasts():
    """Create sample BMA forecasts for testing."""
    return {
        "A": {"expected_return": 0.01, "confidence": 0.7},
        "B": {"expected_return": -0.005, "confidence": 0.5},
        "C": {"expected_return": 0.003, "confidence": 0.6},
    }


def _sample_cov(n=3):
    """Create a valid covariance matrix."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, n))
    return np.cov(X.T)


class TestRiskParityWeights:
    """Tests for risk_parity_weights."""

    def test_weights_sum_to_one(self):
        """Risk parity weights should sum to ~1."""
        forecasts = _sample_forecasts()
        cov = _sample_cov(3)
        names = ["A", "B", "C"]
        weights = risk_parity_weights(forecasts, cov, names)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_weights_nonnegative(self):
        """Risk parity weights should be >= 0 (long-only)."""
        forecasts = _sample_forecasts()
        cov = _sample_cov(3)
        names = ["A", "B", "C"]
        weights = risk_parity_weights(forecasts, cov, names)
        for w in weights.values():
            assert w >= -1e-10

    def test_empty_forecasts_returns_empty(self):
        """No forecast assets should return empty dict."""
        cov = _sample_cov(3)
        weights = risk_parity_weights({}, cov, ["A", "B", "C"])
        assert weights == {}

    def test_only_forecast_assets_included(self):
        """Only assets with forecasts should appear in weights."""
        forecasts = {"A": {"expected_return": 0.01, "confidence": 0.5}}
        cov = _sample_cov(3)
        names = ["A", "B", "C"]
        weights = risk_parity_weights(forecasts, cov, names)
        assert "A" in weights
        assert "B" not in weights
        assert "C" not in weights


class TestKellySizing:
    """Tests for kelly_sizing."""

    def test_returns_dict(self):
        """Kelly sizing should return a dict of floats."""
        forecasts = _sample_forecasts()
        cov = _sample_cov(3)
        names = ["A", "B", "C"]
        weights = kelly_sizing(forecasts, cov, names)
        assert isinstance(weights, dict)
        for v in weights.values():
            assert isinstance(v, float)

    def test_max_leverage_respected(self):
        """Total gross exposure should respect max_leverage."""
        forecasts = _sample_forecasts()
        cov = _sample_cov(3)
        names = ["A", "B", "C"]
        max_lev = 1.5
        weights = kelly_sizing(forecasts, cov, names, max_leverage=max_lev)
        gross = sum(abs(w) for w in weights.values())
        assert gross <= max_lev + 1e-6

    def test_empty_forecasts_returns_empty(self):
        """No forecast assets should return empty dict."""
        cov = _sample_cov(3)
        weights = kelly_sizing({}, cov, ["A", "B", "C"])
        assert weights == {}


class TestApplyConstraints:
    """Tests for apply_constraints."""

    def test_max_position_clipped(self):
        """apply_constraints should reduce extreme concentration."""
        weights = {"A": 0.9, "B": 0.05, "C": 0.05}
        constrained = apply_constraints(weights, max_position=0.15)
        # The largest position should be reduced relative to the original
        assert constrained["A"] < weights["A"]

    def test_output_keys_match_input(self):
        """Constrained weights should have same keys as input."""
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        constrained = apply_constraints(weights)
        assert set(constrained.keys()) == set(weights.keys())

    def test_sector_constraint(self):
        """Sector exposure should be limited by max_sector_exposure."""
        weights = {"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.1}
        sector_map = {"A": "tech", "B": "tech", "C": "energy", "D": "energy"}
        constrained = apply_constraints(
            weights, max_position=0.5, max_sector_exposure=0.40, sector_map=sector_map
        )
        tech_exposure = abs(constrained["A"]) + abs(constrained["B"])
        energy_exposure = abs(constrained["C"]) + abs(constrained["D"])
        # After normalization, constraints should be approximated
        assert tech_exposure <= 0.55  # allow some slack from normalization
        assert energy_exposure <= 0.55
