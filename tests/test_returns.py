"""Tests for the returns panel builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.returns import _apply_transform, build_returns_panel


class TestApplyTransform:
    """Tests for _apply_transform."""

    def test_log_return_values(self):
        """Log returns should equal ln(P_t / P_{t-1})."""
        prices = pd.Series([100.0, 105.0, 110.0, 108.0])
        result = _apply_transform(prices, "log_return")
        expected = np.log(prices / prices.shift(1))
        pd.testing.assert_series_equal(result, expected)

    def test_diff_transform(self):
        """diff transform should produce first differences."""
        series = pd.Series([1.0, 1.5, 1.3, 1.8])
        result = _apply_transform(series, "diff")
        expected = series.diff()
        pd.testing.assert_series_equal(result, expected)

    def test_unknown_transform_defaults_to_log_return(self):
        """Unknown transforms should fall back to log_return."""
        prices = pd.Series([100.0, 110.0, 105.0])
        result = _apply_transform(prices, "unknown_xform")
        expected = np.log(prices / prices.shift(1))
        pd.testing.assert_series_equal(result, expected)


class TestBuildReturnsPanel:
    """Tests for build_returns_panel."""

    def test_returns_shape(self):
        """Panel should have all assets as columns."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        data_dict = {
            "equities": pd.DataFrame(
                np.exp(np.cumsum(np.random.randn(100, 2) * 0.01, axis=0)) * 100,
                index=dates,
                columns=["SPX", "NDX"],
            ),
            "commodities": pd.DataFrame(
                np.exp(np.cumsum(np.random.randn(100, 1) * 0.01, axis=0)) * 50,
                index=dates,
                columns=["GOLD"],
            ),
        }
        panel = build_returns_panel(data_dict)
        assert set(panel.columns) == {"SPX", "NDX", "GOLD"}
        assert len(panel) > 0

    def test_empty_dict_returns_empty(self):
        """Empty input should return empty DataFrame."""
        panel = build_returns_panel({})
        assert panel.empty

    def test_none_entries_skipped(self):
        """None or empty DataFrames in the dict should be skipped."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        data_dict = {
            "equities": pd.DataFrame(
                np.exp(np.cumsum(np.random.randn(50, 1) * 0.01, axis=0)) * 100,
                index=dates,
                columns=["SPX"],
            ),
            "rates": None,
            "credit": pd.DataFrame(),
        }
        panel = build_returns_panel(data_dict)
        assert "SPX" in panel.columns

    def test_no_all_nan_rows(self):
        """Panel should not contain rows that are entirely NaN."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        data_dict = {
            "equities": pd.DataFrame(
                np.exp(np.cumsum(np.random.randn(60, 2) * 0.01, axis=0)) * 100,
                index=dates,
                columns=["SPX", "NDX"],
            ),
        }
        panel = build_returns_panel(data_dict)
        assert not panel.isna().all(axis=1).any()
