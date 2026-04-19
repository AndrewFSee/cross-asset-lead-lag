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
    noise_level: float = 1e-8,
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
        noise_level: Amplitude of jitter added to break ties (KSG requires
            continuous data; tied values from e.g. zero-filled series inflate
            estimates).  Set to 0 to disable.

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

    # Add tiny jitter to break ties from zero-heavy series.  Without this,
    # many points collapse to the same location and the KSG digamma terms
    # become degenerate, producing wildly inflated TE values.
    if noise_level > 0:
        rng = np.random.default_rng(0)
        joint = joint + rng.normal(0, noise_level, joint.shape)
        tf = joint[:, :tf.shape[1]]
        th = joint[:, tf.shape[1]:tf.shape[1] + th.shape[1]]
        sh = joint[:, tf.shape[1] + th.shape[1]:]

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

    # Vectorized neighbor counting in marginal spaces
    eps_adj = eps - 1e-15
    n_th_tf = np.array(
        tree_th_tf.query_radius(joint_th_tf, r=eps_adj, count_only=True)
    )
    n_th_sh = np.array(
        tree_th_sh.query_radius(joint_th_sh, r=eps_adj, count_only=True)
    )
    n_th = np.array(
        tree_th.query_radius(th, r=eps_adj, count_only=True)
    )

    te = (
        digamma(k)
        - np.mean(digamma(np.maximum(n_th_tf, 1)))
        - np.mean(digamma(np.maximum(n_th_sh, 1)))
        + np.mean(digamma(np.maximum(n_th, 1)))
    )

    return max(0.0, float(te))


def effective_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
    history_len: int = 3,
    n_surrogates: int = 50,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """Bias-corrected ("effective") transfer entropy (Marschinski & Kantz 2002).

    Finite-sample KSG TE has a positive bias even for independent series.
    The effective TE subtracts the mean TE estimated from `n_surrogates`
    circular shifts of the source series — shifts preserve the marginal
    distribution but destroy any causal link to the target. The difference
    is an unbiased estimate of the true causal signal; the fraction of
    surrogate TEs exceeding the raw value gives a one-sided p-value.

    Args:
        source: Source time series (1-D array).
        target: Target time series (1-D array).
        lag: Prediction horizon.
        k: KSG k-nearest-neighbour parameter.
        history_len: Embedding dimension for target/source history.
        n_surrogates: Number of circular-shift surrogates. 50 gives a
            reasonable null distribution; 200+ for publication-grade
            p-values.
        random_state: Seed for the shift distribution.

    Returns:
        Dict with keys: te_raw, te_surrogate_mean, te_effective, p_value.
    """
    te_raw = transfer_entropy_knn(
        source, target, lag=lag, k=k, history_len=history_len,
    )
    rng = np.random.default_rng(random_state)
    n = len(source)
    # Avoid shifts smaller than 2·history_len so that the shifted source
    # retains no overlap with its original position in the embedding.
    min_shift = max(2 * history_len + lag + 1, 10)
    max_shift = max(min_shift + 1, n - min_shift)

    surrogate_tes = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        shift = int(rng.integers(min_shift, max_shift))
        shuffled_source = np.roll(source, shift)
        surrogate_tes[i] = transfer_entropy_knn(
            shuffled_source, target, lag=lag, k=k, history_len=history_len,
        )

    te_surrogate_mean = float(surrogate_tes.mean())
    te_effective = max(0.0, float(te_raw - te_surrogate_mean))
    # One-sided p-value: fraction of surrogates at-or-above the raw TE.
    p_value = float((surrogate_tes >= te_raw).mean())

    return {
        "te_raw": float(te_raw),
        "te_surrogate_mean": te_surrogate_mean,
        "te_effective": te_effective,
        "p_value": p_value,
    }


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

    # Filter out zero-dominated series (e.g. monthly macro data forward-filled
    # to daily then differenced) — these have < 10% non-zero values and
    # produce unreliable TE estimates even with jitter.
    min_nonzero_frac = 0.10
    valid_assets = []
    for name in assets:
        arr = returns_dict[name]
        frac = np.count_nonzero(arr) / max(len(arr), 1)
        if frac >= min_nonzero_frac:
            valid_assets.append(name)
        else:
            logger.info(
                "Excluding %s from TE (%.1f%% non-zero < %.0f%% threshold)",
                name, frac * 100, min_nonzero_frac * 100,
            )
    assets = valid_assets
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


def compute_te_decay(
    returns_dict: Dict[str, np.ndarray],
    pairs: List[tuple[str, str]],
    lags: Optional[List[int]] = None,
    k: int = 5,
    history_len: int = 3,
) -> pd.DataFrame:
    """Compute TE at multiple lags for specific pairs to build decay profiles.

    For each pair, estimates the half-life (lag at which TE drops to 50% of
    its lag-1 value) and classifies tradability.

    Args:
        returns_dict: Dict mapping asset name to 1-D return array.
        pairs: List of (source, target) tuples.
        lags: Lag values to probe.  Defaults to [1, 2, 3, 5, 10, 20].
        k: KNN neighbors for TE estimator.
        history_len: Embedding history length.

    Returns:
        DataFrame with columns: source, target, lag, te, te_norm, half_life,
        category, direction, lagged_corr.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]

    rows = []
    for src, tgt in pairs:
        if src not in returns_dict or tgt not in returns_dict:
            continue

        s_arr = returns_dict[src]
        t_arr = returns_dict[tgt]

        # Compute lagged Pearson correlation at lag=1 to determine direction
        s_lead = s_arr[:-1]
        t_follow = t_arr[1:]
        mask = np.isfinite(s_lead) & np.isfinite(t_follow)
        if mask.sum() > 20:
            lagged_corr = float(np.corrcoef(s_lead[mask], t_follow[mask])[0, 1])
        else:
            lagged_corr = 0.0
        direction = "same" if lagged_corr >= 0 else "inverse"

        te_lag1 = None
        lag_values = []
        for lag in lags:
            te = transfer_entropy_knn(
                returns_dict[src], returns_dict[tgt],
                lag=lag, k=k, history_len=history_len,
            )
            if te_lag1 is None:
                te_lag1 = max(te, 1e-10)
            lag_values.append((lag, te))

        # Estimate half-life: first lag where TE drops below 50% of lag-1
        half_life = lags[-1]  # default: beyond our measurement range
        for lag, te in lag_values[1:]:
            if te < te_lag1 * 0.5:
                half_life = lag
                break

        # Lag-2 retention ratio (how much TE survives to next day)
        lag2_retention = 0.0
        for lag, te in lag_values:
            if lag == 2:
                lag2_retention = te / te_lag1
                break

        # Classify tradability
        if half_life <= 1:
            category = "HFT only"
        elif half_life <= 3 and lag2_retention >= 0.30:
            category = "Next-day tradable"
        elif half_life <= 3:
            category = "Short-term"
        elif half_life <= 10:
            category = "Swing (tradable)"
        else:
            category = "Slow decay (tradable)"

        for lag, te in lag_values:
            rows.append({
                "source": src,
                "target": tgt,
                "lag": lag,
                "te": te,
                "te_norm": te / te_lag1,
                "half_life": half_life,
                "category": category,
                "direction": direction,
                "lagged_corr": lagged_corr,
            })

    return pd.DataFrame(rows)
