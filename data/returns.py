"""Returns panel construction for the lead-lag discovery engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "universe.yaml"


def _load_transforms() -> Dict[str, str]:
    """Load transform rules from universe.yaml.

    Returns:
        Dict mapping asset name to transform type ('log_return', 'diff', etc.)
    """
    with open(_UNIVERSE_PATH) as f:
        universe = yaml.safe_load(f)

    transforms: Dict[str, str] = {}
    for asset_class, assets in universe.items():
        if asset_class == "equities":
            for _subgroup, items in assets.items():
                for name in items:
                    transforms[name] = "log_return"  # prices default
        else:
            for name, cfg in assets.items():
                transforms[name] = cfg.get("transform", "log_return")
    return transforms


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """Apply a named transform to a price/level series.

    Args:
        series: Raw price or level series.
        transform: One of 'log_return', 'diff'.

    Returns:
        Transformed series.
    """
    if transform == "log_return":
        result = np.log(series / series.shift(1))
    elif transform == "diff":
        result = series.diff()
    else:
        logger.warning("Unknown transform '%s'; defaulting to log_return", transform)
        result = np.log(series / series.shift(1))
    return result


def build_returns_panel(
    data_dict: Dict[str, pd.DataFrame],
    macro_ffill: bool = True,
    universe_path: Optional[str] = None,
) -> pd.DataFrame:
    """Merge all asset classes into a single aligned returns panel.

    Applies the correct transform per series (log_return for prices, diff for
    yields/spreads), forward-fills macro data to daily before differencing, and
    returns a single aligned DataFrame with all series as columns.

    Args:
        data_dict: Dict mapping asset class names to DataFrames of raw levels.
        macro_ffill: If True, forward-fill macro series to daily before diffing.
        universe_path: Optional override for universe YAML path.

    Returns:
        DataFrame with dates as index and all assets as columns (returns).
    """
    if universe_path:
        with open(universe_path) as f:
            universe = yaml.safe_load(f)
        transforms = {}
        for asset_class, assets in universe.items():
            if asset_class == "equities":
                for subgroup in assets.values():
                    for name in subgroup:
                        transforms[name] = "log_return"
            else:
                for name, cfg in assets.items():
                    transforms[name] = cfg.get("transform", "log_return")
    else:
        transforms = _load_transforms()

    returns_frames = []

    for asset_class, df in data_dict.items():
        if df is None or df.empty:
            logger.warning("Skipping empty asset class: %s", asset_class)
            continue

        class_returns = pd.DataFrame(index=df.index)

        for col in df.columns:
            transform = transforms.get(col, "log_return")
            raw = df[col]
            # Compute the return on *observed* values only. Pre-diff ffill of
            # the price/level creates long runs of zeros on non-release days
            # (e.g. weekly macro ffilled to daily then diffed) which then
            # correlate spuriously with day-of-week/release-calendar effects
            # in the TE matrix. Leaving NaN on non-release days lets the
            # downstream non-zero filter in transfer_entropy.compute_te_matrix
            # correctly identify and exclude low-coverage series.
            observed = raw.dropna()
            ret = _apply_transform(observed, transform)
            ret = ret.reindex(df.index)  # NaN on non-release days
            ret.name = col
            class_returns[col] = ret

        returns_frames.append(class_returns)
        logger.debug("Computed returns for %d %s series", len(df.columns), asset_class)

    if not returns_frames:
        logger.error("No data available to build returns panel")
        return pd.DataFrame()

    # Merge on outer index then drop rows that are all NaN
    panel = returns_frames[0]
    for frame in returns_frames[1:]:
        panel = panel.join(frame, how="outer")

    panel = panel.dropna(how="all")

    # Summary statistics
    n_assets = len(panel.columns)
    n_obs = len(panel)
    logger.info(
        "Returns panel: %d assets × %d observations (%s to %s)",
        n_assets,
        n_obs,
        panel.index.min().date() if not panel.empty else "N/A",
        panel.index.max().date() if not panel.empty else "N/A",
    )

    return panel
