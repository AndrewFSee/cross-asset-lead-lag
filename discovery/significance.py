"""Statistical significance testing for transfer entropy and mutual information."""

from __future__ import annotations

import logging
from typing import Dict, Literal

import numpy as np

from discovery.transfer_entropy import transfer_entropy_knn

logger = logging.getLogger(__name__)


def bootstrap_te_significance(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    k: int = 5,
    history_len: int = 3,
    block_size: int = 10,
) -> Dict[str, object]:
    """Block bootstrap significance test for transfer entropy.

    Generates bootstrap null distribution by resampling blocks of the source
    series (preserving autocorrelation), then tests whether the observed TE
    is significantly greater than the null.

    Args:
        source: Source time series array.
        target: Target time series array.
        lag: Prediction lag.
        n_bootstraps: Number of bootstrap replicates.
        alpha: Significance level.
        k: KNN neighbors for TE estimator.
        history_len: Embedding history length.
        block_size: Block size for block bootstrap.

    Returns:
        Dict with keys: te_observed, p_value, is_significant, ci_lower, ci_upper.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    te_observed = transfer_entropy_knn(source, target, lag=lag, k=k, history_len=history_len)

    n = len(source)
    bootstrap_tes = np.zeros(n_bootstraps)

    for b in range(n_bootstraps):
        # Block bootstrap: randomly sample blocks from source
        n_blocks = int(np.ceil(n / block_size))
        block_starts = np.random.randint(0, max(1, n - block_size), size=n_blocks)
        shuffled_source = np.concatenate([source[s : s + block_size] for s in block_starts])[:n]
        bootstrap_tes[b] = transfer_entropy_knn(
            shuffled_source, target, lag=lag, k=k, history_len=history_len
        )

    p_value = float(np.mean(bootstrap_tes >= te_observed))
    ci_lower = float(np.percentile(bootstrap_tes, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_tes, 100 * (1 - alpha / 2)))
    is_significant = p_value < alpha

    logger.info(
        "Bootstrap TE significance: TE=%.4f, p=%.4f, significant=%s",
        te_observed,
        p_value,
        is_significant,
    )

    return {
        "te_observed": te_observed,
        "p_value": p_value,
        "is_significant": is_significant,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def surrogate_significance(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_surrogates: int = 1000,
    method: Literal["shuffle", "phase"] = "shuffle",
    k: int = 5,
    history_len: int = 3,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Surrogate-based significance test for transfer entropy.

    Generates surrogate time series by shuffling or phase-randomizing the
    source, then builds a null distribution of TE values.

    Args:
        source: Source time series array.
        target: Target time series array.
        lag: Prediction lag.
        n_surrogates: Number of surrogate replicates.
        method: 'shuffle' (random permutation) or 'phase' (FFT phase randomization).
        k: KNN neighbors for TE estimator.
        history_len: Embedding history length.
        alpha: Significance level.

    Returns:
        Dict with keys: te_observed, p_value, is_significant, surrogate_mean, surrogate_std.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    te_observed = transfer_entropy_knn(source, target, lag=lag, k=k, history_len=history_len)

    surrogate_tes = np.zeros(n_surrogates)

    for s in range(n_surrogates):
        if method == "shuffle":
            surrogate_source = np.random.permutation(source)
        elif method == "phase":
            # Phase randomization preserves power spectrum
            n = len(source)
            fft = np.fft.rfft(source)
            random_phases = np.exp(2j * np.pi * np.random.random(len(fft)))
            surrogate_source = np.fft.irfft(fft * random_phases, n=n)
        else:
            raise ValueError(f"Unknown surrogate method: {method}")

        surrogate_tes[s] = transfer_entropy_knn(
            surrogate_source, target, lag=lag, k=k, history_len=history_len
        )

    p_value = float(np.mean(surrogate_tes >= te_observed))
    is_significant = p_value < alpha

    return {
        "te_observed": te_observed,
        "p_value": p_value,
        "is_significant": is_significant,
        "surrogate_mean": float(np.mean(surrogate_tes)),
        "surrogate_std": float(np.std(surrogate_tes)),
    }
