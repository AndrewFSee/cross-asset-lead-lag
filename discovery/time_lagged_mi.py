"""Time-lagged mutual information using the KSG estimator."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)


def mutual_information_knn(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate mutual information I(X;Y) using the KSG estimator.

    Implements Algorithm 1 from Kraskov, Stögbauer & Grassberger (2004):
    I(X;Y) = ψ(k) - <ψ(n_x+1) + ψ(n_y+1)> + ψ(N)

    Args:
        x: First variable (1-D array).
        y: Second variable (1-D array).
        k: Number of nearest neighbors.

    Returns:
        Estimated mutual information (clipped to >= 0).
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    # Remove NaN rows
    mask = np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    x, y = x[mask], y[mask]
    n = len(x)

    if n < max(20, 2 * k):
        logger.warning("Not enough data for MI estimation (n=%d)", n)
        return 0.0

    joint = np.concatenate([x, y], axis=1)
    tree_joint = KDTree(joint, metric="chebyshev")
    dist, _ = tree_joint.query(joint, k=k + 1)
    eps = dist[:, k]

    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")

    n_x = np.array(
        [tree_x.query_radius([x[i]], r=eps[i] - 1e-15, count_only=True)[0] for i in range(n)]
    )
    n_y = np.array(
        [tree_y.query_radius([y[i]], r=eps[i] - 1e-15, count_only=True)[0] for i in range(n)]
    )

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(0.0, float(mi))


def time_lagged_mi(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    k: int = 5,
) -> float:
    """Compute time-lagged mutual information I(X_t ; Y_{t+lag}).

    Args:
        x: Source variable array.
        y: Target variable array.
        lag: Time lag (positive = x leads y).
        k: KNN neighbors for MI estimator.

    Returns:
        Mutual information value (>= 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lag >= len(x):
        logger.warning("Lag %d >= series length %d", lag, len(x))
        return 0.0

    if lag >= 0:
        x_lagged = x[: len(x) - lag]
        y_future = y[lag:]
    else:
        abs_lag = abs(lag)
        x_lagged = x[abs_lag:]
        y_future = y[: len(y) - abs_lag]

    return mutual_information_knn(x_lagged, y_future, k=k)


def compute_tlmi_matrix(
    returns_dict: Dict[str, np.ndarray],
    lags: Optional[List[int]] = None,
    k: int = 5,
) -> Dict[int, pd.DataFrame]:
    """Compute pairwise time-lagged mutual information for multiple lags.

    Args:
        returns_dict: Dict mapping asset name to 1-D return array.
        lags: List of lag values. Defaults to [1, 2, 3, 5, 10, 20].
        k: KNN neighbors for MI estimator.

    Returns:
        Dict mapping lag to DataFrame of TLMI values (rows=source, cols=target).
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]

    assets = list(returns_dict.keys())
    n = len(assets)
    result: Dict[int, pd.DataFrame] = {}

    for lag in lags:
        mi_values = np.zeros((n, n))
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i == j:
                    mi_values[i, j] = 0.0
                    continue
                mi = time_lagged_mi(returns_dict[src], returns_dict[tgt], lag=lag, k=k)
                mi_values[i, j] = mi
        result[lag] = pd.DataFrame(mi_values, index=assets, columns=assets)
        logger.info("Computed TLMI matrix for lag=%d", lag)

    return result
