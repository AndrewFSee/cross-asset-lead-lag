"""Portfolio construction and risk budgeting utilities."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def risk_parity_weights(
    forecasts: Dict[str, Dict],
    cov_matrix: np.ndarray,
    asset_names: list,
) -> Dict[str, float]:
    """Compute risk parity weights targeting equal risk contributions.

    Uses the inverse-volatility approximation as an initial estimate,
    then refines via numerical optimization.

    Args:
        forecasts: BMA forecast dict from bayesian_model_average().
        cov_matrix: Covariance matrix (n_assets, n_assets).
        asset_names: Ordered list of asset names matching cov_matrix.

    Returns:
        Dict mapping asset name to weight. Weights sum to 1.
    """
    n = len(asset_names)
    vols = np.sqrt(np.diag(cov_matrix))
    vols = np.maximum(vols, 1e-8)

    # Only include assets with a forecast
    forecast_assets = [a for a in asset_names if a in forecasts]
    if not forecast_assets:
        return {}

    indices = [asset_names.index(a) for a in forecast_assets]
    sub_cov = cov_matrix[np.ix_(indices, indices)]
    sub_vols = vols[indices]

    # Inverse volatility starting point
    inv_vol = 1.0 / sub_vols
    w0 = inv_vol / inv_vol.sum()

    def risk_contribution(w: np.ndarray) -> float:
        """Sum of squared deviations from equal risk contributions."""
        port_var = w @ sub_cov @ w
        if port_var < 1e-10:
            return 0.0
        marginal_rc = sub_cov @ w
        rc = w * marginal_rc / port_var
        target_rc = 1.0 / len(w)
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * len(forecast_assets)

    result = minimize(
        risk_contribution,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    weights_arr = result.x if result.success else w0
    weights_arr = np.maximum(weights_arr, 0.0)
    weights_arr /= weights_arr.sum()

    return {asset: float(w) for asset, w in zip(forecast_assets, weights_arr)}


def kelly_sizing(
    forecasts: Dict[str, Dict],
    cov_matrix: np.ndarray,
    asset_names: list,
    max_leverage: float = 2.0,
) -> Dict[str, float]:
    """Compute Kelly-optimal position sizes.

    Solves the unconstrained Kelly problem: w* = Sigma^{-1} mu,
    then scales to respect max_leverage.

    Args:
        forecasts: BMA forecast dict from bayesian_model_average().
        cov_matrix: Covariance matrix (n_assets, n_assets).
        asset_names: Ordered list of asset names.
        max_leverage: Maximum gross exposure.

    Returns:
        Dict mapping asset name to weight.
    """
    forecast_assets = [a for a in asset_names if a in forecasts]
    if not forecast_assets:
        return {}

    indices = [asset_names.index(a) for a in forecast_assets]
    sub_cov = cov_matrix[np.ix_(indices, indices)]

    mu = np.array([forecasts[a]["expected_return"] for a in forecast_assets])

    # Regularize covariance
    reg = 1e-6 * np.eye(len(forecast_assets))
    try:
        w_kelly = np.linalg.solve(sub_cov + reg, mu)
    except np.linalg.LinAlgError:
        w_kelly = np.zeros(len(forecast_assets))

    # Scale to max_leverage
    gross = np.sum(np.abs(w_kelly))
    if gross > max_leverage:
        w_kelly *= max_leverage / gross

    return {asset: float(w) for asset, w in zip(forecast_assets, w_kelly)}


def apply_constraints(
    weights: Dict[str, float],
    max_position: float = 0.15,
    max_sector_exposure: float = 0.40,
    sector_map: Dict[str, str] = None,
) -> Dict[str, float]:
    """Apply position and sector constraints to portfolio weights.

    Args:
        weights: Dict mapping asset to unconstrained weight.
        max_position: Maximum absolute weight per position.
        max_sector_exposure: Maximum absolute sector exposure.
        sector_map: Optional dict mapping asset to sector.

    Returns:
        Dict of constrained weights (re-normalized to sum to ~1).
    """
    constrained = {k: np.clip(v, -max_position, max_position) for k, v in weights.items()}

    # Apply sector exposure constraints
    if sector_map is not None:
        from collections import defaultdict  # noqa: PLC0415

        sector_exposure: Dict[str, float] = defaultdict(float)
        for asset, w in constrained.items():
            sector = sector_map.get(asset, "other")
            sector_exposure[sector] += abs(w)

        for sector, exposure in sector_exposure.items():
            if exposure > max_sector_exposure:
                scale = max_sector_exposure / exposure
                for asset in constrained:
                    if sector_map.get(asset, "other") == sector:
                        constrained[asset] *= scale

    # Normalize long and short legs separately (keep beta-neutral sign)
    total = sum(abs(v) for v in constrained.values())
    if total > 1e-10:
        constrained = {k: v / total for k, v in constrained.items()}

    return constrained
