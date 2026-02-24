"""KNN-based transfer entropy estimator using the KSG method."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)


def _build_joint_embedding(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build joint and marginal embeddings for transfer entropy.

    Constructs the vectors:
      - target_future: T[t+lag]
      - target_history: (T[t], T[t-1], ..., T[t-history_len+1])
      - source_history: (S[t], S[t-1], ..., S[t-history_len+1])
      - joint: concatenation of all three

    Args:
        source: Source time series array.
        target: Target time series array.
        lag: Number of steps ahead for target future.
        history_len: Length of embedding history.

    Returns:
        Tuple of (joint, target_future, target_history, source_history).
    """
    n = len(source)
    start = history_len
    end = n - lag

    if end <= start:
        raise ValueError(f"Not enough data: n={n}, lag={lag}, history_len={history_len}")

    t_idx = np.arange(start, end)
    n_samples = len(t_idx)

    target_future = target[t_idx + lag].reshape(-1, 1)

    t_hist = np.zeros((n_samples, history_len))
    s_hist = np.zeros((n_samples, history_len))
    for h in range(history_len):
        t_hist[:, h] = target[t_idx - h]
        s_hist[:, h] = source[t_idx - h]

    joint = np.concatenate([target_future, t_hist, s_hist], axis=1)
    return joint, target_future, t_hist, s_hist


def transfer_entropy_knn(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
    history_len: int = 3,
) -> float:
    """Compute KSG-based transfer entropy from source to target.

    Implements the KNN mutual information estimator (Kraskov, Stögbauer &
    Grassberger 2004) applied to transfer entropy (Schreiber 2000).

    TE(S→T) = I(T_future ; S_history | T_history)
             = H(T_future | T_history) - H(T_future | T_history, S_history)

    Args:
        source: Source time series (1-D array).
        target: Target time series (1-D array).
        lag: Prediction horizon (in time steps).
        k: Number of nearest neighbors for KNN estimator.
        history_len: Embedding dimension for history vectors.

    Returns:
        Transfer entropy value (clipped to >= 0).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    # Remove NaN rows
    mask = np.isfinite(source) & np.isfinite(target)
    source, target = source[mask], target[mask]

    if len(source) < max(50, 3 * k):
        logger.warning("Not enough data for TE estimation (n=%d)", len(source))
        return 0.0

    try:
        joint, tf, th, sh = _build_joint_embedding(source, target, lag, history_len)
    except ValueError as exc:
        logger.warning("TE embedding failed: %s", exc)
        return 0.0

    n = len(joint)

    # KSG estimator: for each point find k-th neighbor in joint space
    tree_joint = KDTree(joint, metric="chebyshev")
    dist_joint, _ = tree_joint.query(joint, k=k + 1)
    eps = dist_joint[:, k]  # k-th neighbor distance (Chebyshev)

    # Count neighbors in marginal spaces within eps
    joint_th_tf = np.concatenate([th, tf], axis=1)
    joint_th_sh = np.concatenate([th, sh], axis=1)

    tree_th_tf = KDTree(joint_th_tf, metric="chebyshev")
    tree_th_sh = KDTree(joint_th_sh, metric="chebyshev")
    tree_th = KDTree(th, metric="chebyshev")

    # Number of points strictly within eps ball
    n_th_tf = np.array([
        tree_th_tf.query_radius([joint_th_tf[i]], r=eps[i] - 1e-15, count_only=True)[0]
        for i in range(n)
    ])
    n_th_sh = np.array([
        tree_th_sh.query_radius([joint_th_sh[i]], r=eps[i] - 1e-15, count_only=True)[0]
        for i in range(n)
    ])
    n_th = np.array([
        tree_th.query_radius([th[i]], r=eps[i] - 1e-15, count_only=True)[0]
        for i in range(n)
    ])

    te = (
        digamma(k)
        - np.mean(digamma(np.maximum(n_th_tf, 1)))
        - np.mean(digamma(np.maximum(n_th_sh, 1)))
        + np.mean(digamma(np.maximum(n_th, 1)))
    )

    return max(0.0, float(te))


def compute_te_matrix(
    returns_dict: Dict[str, np.ndarray],
    lags: Optional[List[int]] = None,
    k: int = 5,
    history_len: int = 3,
) -> Dict[int, pd.DataFrame]:
    """Compute pairwise transfer entropy matrix for multiple lags.

    Args:
        returns_dict: Dict mapping asset name to 1-D return array.
        lags: List of lag values. Defaults to [1, 2, 3, 5, 10, 20].
        k: KNN neighbors for TE estimator.
        history_len: Embedding history length.

    Returns:
        Dict mapping lag value to DataFrame of TE values (source, target).
        Rows = source assets, columns = target assets.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]

    assets = list(returns_dict.keys())
    n = len(assets)
    result: Dict[int, pd.DataFrame] = {}

    for lag in lags:
        te_values = np.zeros((n, n))
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i == j:
                    te_values[i, j] = 0.0
                    continue
                te = transfer_entropy_knn(
                    returns_dict[src],
                    returns_dict[tgt],
                    lag=lag,
                    k=k,
                    history_len=history_len,
                )
                te_values[i, j] = te
        result[lag] = pd.DataFrame(te_values, index=assets, columns=assets)
        logger.info("Computed TE matrix for lag=%d", lag)

    return result
