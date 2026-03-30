"""Tests for the RegimeDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.regime_detector import RegimeDetector


class TestRegimeDetector:
    """Tests for RegimeDetector."""

    def test_fit_returns_self(self, synthetic_regime_data):
        """fit() should return self and set _fitted."""
        data, _ = synthetic_regime_data
        # Use KMeans fallback (no hmmlearn dependency required)
        det = RegimeDetector(n_regimes=2, method="hmm")
        result = det.fit(data)
        assert result is det
        assert det._fitted

    def test_regime_history_length(self, synthetic_regime_data):
        """regime_history should have same length as filtered data."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data)
        history = det.regime_history()
        assert len(history) == len(data)

    def test_regime_labels_in_range(self, synthetic_regime_data):
        """All regime labels should be in [0, n_regimes-1]."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data)
        labels = det.regime_history().values
        assert set(labels).issubset({0, 1})

    def test_predict_regime_returns_int(self, synthetic_regime_data):
        """predict_regime should return an integer."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data)
        regime = det.predict_regime(data[-5:])
        assert isinstance(regime, int)
        assert regime in {0, 1}

    def test_regime_summary_keys(self, synthetic_regime_data):
        """regime_summary should return expected keys per regime."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data)
        summary = det.regime_summary()
        assert len(summary) == 2
        for r in range(2):
            assert all(k in summary[r] for k in ["mean", "std", "n_obs", "pct"])

    def test_regime_percentages_sum_to_one(self, synthetic_regime_data):
        """Regime percentages should sum to approximately 1."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data)
        summary = det.regime_summary()
        total_pct = sum(s["pct"] for s in summary.values())
        assert abs(total_pct - 1.0) < 1e-9

    def test_with_dates(self, synthetic_regime_data):
        """fit with dates should produce a DatetimeIndex on regime_history."""
        data, _ = synthetic_regime_data
        dates = pd.date_range("2020-01-01", periods=len(data), freq="B")
        det = RegimeDetector(n_regimes=2, method="hmm")
        det.fit(data, dates=dates)
        history = det.regime_history()
        assert isinstance(history.index, pd.DatetimeIndex)

    def test_unfitted_raises(self):
        """Calling methods before fit should raise RuntimeError."""
        det = RegimeDetector()
        with pytest.raises(RuntimeError):
            det.regime_history()
        with pytest.raises(RuntimeError):
            det.predict_regime(np.array([[1.0, 2.0]]))
        with pytest.raises(RuntimeError):
            det.regime_summary()

    def test_invalid_method_raises(self, synthetic_regime_data):
        """Unknown method should raise ValueError."""
        data, _ = synthetic_regime_data
        det = RegimeDetector(method="invalid")
        with pytest.raises(ValueError):
            det.fit(data)
