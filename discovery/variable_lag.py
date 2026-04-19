"""Variable-lag transfer entropy / lead-lag discovery.

Classical TE fixes a single lag `τ` for every (source, target) pair. In
markets the optimal lag varies by pair — e.g. HY OAS → SPX may peak at
lag 1, DXY → EURUSD at lag 3, VIX → XLU at lag 5.

This module picks the per-pair optimal lag via effective TE across a
candidate grid and reports a simple half-sample stability test so that
transient / spurious peaks can be filtered out.

References:
    Amornbunchornvej et al. "Variable-Lag Granger Causality and Transfer
    Entropy for Time Series Analysis." ACM TKDD 2021.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from discovery.transfer_entropy import (
    effective_transfer_entropy,
    transfer_entropy_knn,
)

logger = logging.getLogger(__name__)


def best_lag_effective_te(
    source: np.ndarray,
    target: np.ndarray,
    candidate_lags: List[int],
    k: int = 5,
    history_len: int = 3,
    n_surrogates: int = 30,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """Return the lag in `candidate_lags` with highest effective TE, plus
    a half-sample stability measure.

    Stability is computed by splitting the series in half and recomputing
    *raw* TE (skipping the expensive surrogate step) at the best lag on
    each half. We return the coefficient of variation — lower is more
    stable.

    Args:
        source: Source series (1-D).
        target: Target series (1-D).
        candidate_lags: Lags to evaluate.
        k, history_len: KSG parameters.
        n_surrogates: Surrogates per candidate lag. Scales cost linearly.
        random_state: RNG seed.

    Returns:
        Dict with keys:
            best_lag, te_effective, p_value,
            stability_cv (coefficient of variation across halves),
            all_lags (dict lag -> te_effective).
    """
    all_lags: Dict[int, float] = {}
    p_values: Dict[int, float] = {}
    for lag in candidate_lags:
        r = effective_transfer_entropy(
            source, target, lag=lag, k=k, history_len=history_len,
            n_surrogates=n_surrogates, random_state=random_state,
        )
        all_lags[lag] = r["te_effective"]
        p_values[lag] = r["p_value"]

    best_lag = int(max(all_lags, key=all_lags.get))
    best_te = float(all_lags[best_lag])
    best_p = float(p_values[best_lag])

    # Half-sample stability at best_lag
    n = len(source)
    half = n // 2
    if half >= max(50, 3 * k):
        te_h1 = transfer_entropy_knn(
            source[:half], target[:half],
            lag=best_lag, k=k, history_len=history_len,
        )
        te_h2 = transfer_entropy_knn(
            source[half:], target[half:],
            lag=best_lag, k=k, history_len=history_len,
        )
        mu = 0.5 * (te_h1 + te_h2)
        if mu > 1e-10:
            cv = float(abs(te_h1 - te_h2) / (2.0 * mu))
        else:
            cv = float("nan")
    else:
        cv = float("nan")

    return {
        "best_lag": best_lag,
        "te_effective": best_te,
        "p_value": best_p,
        "stability_cv": cv,
        "all_lags": all_lags,
    }


def compute_variable_lag_matrix(
    returns_dict: Dict[str, np.ndarray],
    candidate_lags: Optional[List[int]] = None,
    k: int = 5,
    history_len: int = 3,
    n_surrogates: int = 30,
    target_subset: Optional[List[str]] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Pairwise variable-lag effective TE.

    Returns a tidy DataFrame rather than a square matrix so that the
    per-pair best_lag, p_value, and stability fit naturally alongside the
    effective TE score.

    Args:
        returns_dict: Dict of asset_name -> 1-D return array.
        candidate_lags: Lags to probe. Defaults to [1, 2, 3, 5, 10].
        k, history_len, n_surrogates: Passed to `effective_transfer_entropy`.
        target_subset: If given, restrict followers to this list. Useful
            for equity-only followers. Leaders = all assets in dict.
        random_state: RNG seed.

    Returns:
        DataFrame with columns:
            source, target, best_lag, te_effective, p_value, stability_cv.
    """
    if candidate_lags is None:
        candidate_lags = [1, 2, 3, 5, 10]

    assets = list(returns_dict.keys())
    followers = target_subset if target_subset is not None else assets
    rows = []
    for src in assets:
        for tgt in followers:
            if src == tgt or tgt not in returns_dict:
                continue
            try:
                r = best_lag_effective_te(
                    returns_dict[src], returns_dict[tgt],
                    candidate_lags=candidate_lags,
                    k=k, history_len=history_len,
                    n_surrogates=n_surrogates,
                    random_state=random_state,
                )
            except Exception as exc:
                logger.warning("variable-lag TE failed for %s→%s: %s", src, tgt, exc)
                continue
            rows.append({
                "source": src,
                "target": tgt,
                "best_lag": r["best_lag"],
                "te_effective": r["te_effective"],
                "p_value": r["p_value"],
                "stability_cv": r["stability_cv"],
            })
    return pd.DataFrame(rows)
