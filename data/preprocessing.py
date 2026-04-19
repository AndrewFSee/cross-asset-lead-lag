"""Data preprocessing utilities for the lead-lag discovery engine."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stationarity_check(
    series: pd.Series,
) -> Dict[str, object]:
    """Run ADF and KPSS stationarity tests on a time series.

    Args:
        series: Time series to test (NaNs are dropped before testing).

    Returns:
        Dict with keys: adf_stat, adf_pvalue, adf_is_stationary,
        kpss_stat, kpss_pvalue, kpss_is_stationary, is_stationary.
    """
    from statsmodels.tsa.stattools import adfuller, kpss  # noqa: PLC0415

    clean = series.dropna()
    if len(clean) < 20:
        logger.warning("Series too short for stationarity test (n=%d)", len(clean))
        return {
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "adf_is_stationary": None,
            "kpss_stat": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_is_stationary": None,
            "is_stationary": None,
        }

    # ADF: H0 = unit root (non-stationary); reject => stationary
    adf_result = adfuller(clean, autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_is_stationary = adf_pvalue < 0.05

    # KPSS: H0 = stationary; reject => non-stationary
    try:
        kpss_result = kpss(clean, regression="c", nlags="auto")
        kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
        kpss_is_stationary = kpss_pvalue >= 0.05
    except Exception as exc:
        logger.warning("KPSS test failed: %s", exc)
        kpss_stat, kpss_pvalue, kpss_is_stationary = np.nan, np.nan, None

    # Both tests agree => confident classification
    is_stationary = bool(adf_is_stationary) and bool(kpss_is_stationary)

    return {
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_is_stationary": adf_is_stationary,
        "kpss_stat": float(kpss_stat) if not np.isnan(kpss_stat) else np.nan,
        "kpss_pvalue": float(kpss_pvalue) if not np.isnan(kpss_pvalue) else np.nan,
        "kpss_is_stationary": kpss_is_stationary,
        "is_stationary": is_stationary,
    }


def winsorize_returns(df: pd.DataFrame, n_sigma: float = 4.0) -> pd.DataFrame:
    """Winsorize returns at ±n_sigma standard deviations.

    Args:
        df: DataFrame of returns (rows = dates, columns = assets).
        n_sigma: Number of standard deviations for clipping threshold.

    Returns:
        Winsorized DataFrame of the same shape.
    """
    result = df.copy()
    for col in result.columns:
        series = result[col].dropna()
        if len(series) == 0:
            continue
        mu = series.mean()
        sigma = series.std()
        if sigma == 0:
            continue
        lower = mu - n_sigma * sigma
        upper = mu + n_sigma * sigma
        result[col] = result[col].clip(lower=lower, upper=upper)
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            logger.debug("Winsorized %d values in column %s", n_clipped, col)
    return result


def handle_missing(
    df: pd.DataFrame,
    ffill_limit: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle missing values by forward-filling with a limit.

    Args:
        df: DataFrame with potential missing values.
        ffill_limit: Maximum number of consecutive NaNs to forward-fill.

    Returns:
        Tuple of (filled_df, gap_report) where gap_report shows remaining
        missing value counts per column.
    """
    filled = df.ffill(limit=ffill_limit)
    remaining_nans = filled.isnull().sum()
    gaps = remaining_nans[remaining_nans > 0]
    if len(gaps) > 0:
        logger.warning(
            "Remaining NaN counts after forward-fill:\n%s",
            gaps.to_string(),
        )
    return filled, pd.DataFrame({"remaining_nans": remaining_nans})


def align_calendars(
    data_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Align all series to a common business-day calendar.

    Crypto trades 7 days/week while equities trade ~5 days/week. This
    function reindexes all series to the union of business days, using
    forward-fill (limit=3) to bridge weekends for non-crypto series.

    Args:
        data_dict: Dict mapping asset class names to DataFrames.

    Returns:
        Dict with the same keys but all DataFrames on a shared index.
    """
    if not data_dict:
        return data_dict

    # Build union of all dates present across asset classes
    all_indices = [df.index for df in data_dict.values() if df is not None and not df.empty]
    if not all_indices:
        return data_dict

    union_index = all_indices[0]
    for idx in all_indices[1:]:
        union_index = union_index.union(idx)

    # Keep only dates that are either business days or present in crypto
    crypto_df = data_dict.get("crypto")
    if crypto_df is not None and not crypto_df.empty:
        business_days = pd.bdate_range(start=union_index.min(), end=union_index.max())
        combined_index = business_days.union(crypto_df.index)
    else:
        combined_index = pd.bdate_range(start=union_index.min(), end=union_index.max())

    aligned: Dict[str, pd.DataFrame] = {}
    for asset_class, df in data_dict.items():
        if df is None or df.empty:
            aligned[asset_class] = df
            continue
        # Reindex to the combined calendar. For tradable daily-priced classes
        # we forward-fill short gaps (e.g. a holiday) with limit=3. For
        # weekly/monthly release classes (macro) we do NOT ffill the level —
        # ffilling the level then taking a diff creates spurious zero-runs
        # that leak day-of-week / release-calendar structure into the TE
        # matrix. `build_returns_panel` now handles these series correctly
        # by computing returns at native frequency.
        reindexed = df.reindex(combined_index)
        if asset_class in ("rates", "credit", "volatility"):
            reindexed = reindexed.ffill(limit=5)
        elif asset_class != "macro":
            reindexed = reindexed.ffill(limit=3)
        # macro: leave NaN on non-release days so that downstream logic can
        # detect and exclude low-coverage series.
        aligned[asset_class] = reindexed
        logger.debug("Aligned %s: %d rows -> %d rows", asset_class, len(df), len(reindexed))

    return aligned
