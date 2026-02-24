"""Tests for data preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import (
    handle_missing,
    stationarity_check,
    winsorize_returns,
)


class TestWinsorizeReturns:
    """Tests for winsorize_returns."""

    def test_clips_extreme_values(self):
        """Values beyond n_sigma should be clipped."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"A": rng.standard_normal(200)})
        df.loc[0, "A"] = 100.0  # Extreme outlier
        df.loc[1, "A"] = -100.0

        result = winsorize_returns(df, n_sigma=4.0)
        mu = df["A"].mean()
        sigma = df["A"].std()
        assert result["A"].max() <= mu + 4.0 * sigma + 1e-10
        assert result["A"].min() >= mu - 4.0 * sigma - 1e-10

    def test_normal_values_unchanged(self):
        """Values within n_sigma should not be changed."""
        df = pd.DataFrame({"A": np.array([0.01, -0.01, 0.005, 0.0])})
        result = winsorize_returns(df, n_sigma=4.0)
        pd.testing.assert_frame_equal(result, df)

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        df = pd.DataFrame(np.random.randn(100, 5))
        result = winsorize_returns(df, n_sigma=3.0)
        assert result.shape == df.shape


class TestHandleMissing:
    """Tests for handle_missing."""

    def test_forward_fills_within_limit(self):
        """NaNs within ffill_limit should be filled."""
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]})
        filled, _ = handle_missing(df, ffill_limit=2)
        assert not filled["A"].isna().any()

    def test_does_not_fill_beyond_limit(self):
        """NaN runs longer than ffill_limit should remain."""
        df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]})
        filled, _ = handle_missing(df, ffill_limit=2)
        # At least one NaN should remain (the 3rd and 4th consecutive NaNs)
        assert filled["A"].isna().any()

    def test_gap_report_columns(self):
        """Gap report should have 'remaining_nans' column."""
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        _, report = handle_missing(df)
        assert "remaining_nans" in report.columns


class TestStationarityCheck:
    """Tests for stationarity_check."""

    def test_stationary_series_detected(self):
        """White noise should be detected as stationary."""
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(300))
        result = stationarity_check(series)
        # White noise is stationary: ADF should reject unit root
        assert result["adf_is_stationary"] == True  # noqa: E712

    def test_nonstationary_series_detected(self):
        """Random walk should be detected as non-stationary by ADF."""
        rng = np.random.default_rng(42)
        series = pd.Series(np.cumsum(rng.standard_normal(300)))
        result = stationarity_check(series)
        assert result["adf_is_stationary"] == False  # noqa: E712

    def test_result_keys(self):
        """Result should contain expected keys."""
        series = pd.Series(np.random.randn(100))
        result = stationarity_check(series)
        expected_keys = [
            "adf_stat",
            "adf_pvalue",
            "adf_is_stationary",
            "kpss_stat",
            "kpss_pvalue",
            "kpss_is_stationary",
            "is_stationary",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
