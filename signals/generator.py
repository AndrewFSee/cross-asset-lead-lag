"""Lead-lag signal generation and Bayesian model averaging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeadSignal:
    """A single lead-lag signal.

    Attributes:
        leader: Name of the leading asset.
        follower: Name of the following asset.
        lag: Lag in periods.
        te_score: Transfer entropy score.
        regime: Current market regime index.
        coefficient: VAR coefficient from the leading asset to the follower.
        leader_return: Most recent return of the leading asset.
    """

    leader: str
    follower: str
    lag: int
    te_score: float
    regime: int
    coefficient: float
    leader_return: float

    @property
    def raw_signal(self) -> float:
        """Expected return contribution from this signal.

        Returns:
            Product of coefficient and leader return.
        """
        return float(self.coefficient * self.leader_return)

    @property
    def confidence(self) -> float:
        """Signal confidence normalized to [0, 1].

        Returns:
            Sigmoid-transformed TE score as a confidence measure.
        """
        return float(1.0 / (1.0 + np.exp(-self.te_score * 10)))


def generate_signals(
    te_matrix: pd.DataFrame,
    ms_var_model,
    latest_returns: pd.Series,
    regime: int,
    te_threshold: float = 0.01,
    top_k: int = 10,
) -> List[LeadSignal]:
    """Generate lead-lag signals from TE matrix and MS-VAR coefficients.

    Args:
        te_matrix: Transfer entropy matrix (rows=source, cols=target).
        ms_var_model: Fitted MarkovSwitchingVAR instance.
        latest_returns: Most recent return for each asset.
        regime: Current regime index.
        te_threshold: Minimum TE score to generate a signal.
        top_k: Maximum number of signals to return.

    Returns:
        List of LeadSignal instances sorted by TE score descending.
    """
    signals: List[LeadSignal] = []

    assets = te_matrix.index.tolist()
    regime_coefs = ms_var_model.get_regime_coefficients()
    B_regime = regime_coefs.get(regime, list(regime_coefs.values())[0])

    # B_regime shape: (n_vars, n_vars*n_lags+1)
    # Coefficient for asset i→j is in row j, columns 1..n_vars (lag 1)

    for i, leader in enumerate(assets):
        for j, follower in enumerate(assets):
            if i == j:
                continue
            te_score = float(te_matrix.iloc[i, j])
            if te_score < te_threshold:
                continue

            # Extract lag-1 coefficient for leader→follower
            if i < B_regime.shape[1] - 1 and j < B_regime.shape[0]:
                coef = float(B_regime[j, 1 + i])
            else:
                coef = 0.0

            leader_ret = float(latest_returns.get(leader, 0.0))

            signals.append(
                LeadSignal(
                    leader=leader,
                    follower=follower,
                    lag=1,
                    te_score=te_score,
                    regime=regime,
                    coefficient=coef,
                    leader_return=leader_ret,
                )
            )

    signals.sort(key=lambda s: s.te_score, reverse=True)
    return signals[:top_k]


def bayesian_model_average(
    signals: List[LeadSignal],
) -> Dict[str, Dict]:
    """Aggregate signals per follower via Bayesian model averaging.

    Each leader is treated as a separate model. The BMA weights are
    proportional to the TE score of each leader→follower pair.

    Args:
        signals: List of LeadSignal instances.

    Returns:
        Dict mapping follower name to dict with keys:
        expected_return, confidence, n_leaders, dominant_leader,
        leaders (list of dicts with name, weight, signal).
    """
    from collections import defaultdict  # noqa: PLC0415

    follower_groups: Dict[str, List[LeadSignal]] = defaultdict(list)
    for sig in signals:
        follower_groups[sig.follower].append(sig)

    result: Dict[str, Dict] = {}

    for follower, group in follower_groups.items():
        te_scores = np.array([s.te_score for s in group])
        raw_signals = np.array([s.raw_signal for s in group])

        # BMA weights proportional to TE score
        total_te = te_scores.sum()
        if total_te > 1e-10:
            weights = te_scores / total_te
        else:
            weights = np.full(len(group), 1.0 / len(group))

        expected_return = float(np.dot(weights, raw_signals))
        avg_confidence = float(np.dot(weights, [s.confidence for s in group]))
        dominant_idx = int(np.argmax(weights))

        result[follower] = {
            "expected_return": expected_return,
            "confidence": avg_confidence,
            "n_leaders": len(group),
            "dominant_leader": group[dominant_idx].leader,
            "leaders": [
                {
                    "name": s.leader,
                    "weight": float(w),
                    "signal": float(s.raw_signal),
                    "te_score": float(s.te_score),
                }
                for s, w in zip(group, weights)
            ],
        }
        logger.debug(
            "BMA for %s: expected_return=%.4f, n_leaders=%d",
            follower,
            expected_return,
            len(group),
        )

    return result


def regime_conditional_te_weights(
    per_regime_te: Mapping[int, pd.DataFrame],
    current_regime: int,
    followers: List[str],
    te_threshold: float = 0.01,
    top_k_per_follower: int = 3,
    fallback_regime: Optional[int] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """Select leader→follower pairs whose TE is significant in *this* regime.

    The idea (from regime-conditional causality literature): a pair that
    only expresses its edge in a stress regime should not be traded in a
    calm regime, and vice versa. Given a dict of per-regime TE matrices,
    pick the top-k leaders for each follower *using only the current
    regime's matrix*. If no regime-specific matrix is available we fall
    back to ``fallback_regime`` (or return an empty mapping).

    Args:
        per_regime_te: Mapping ``regime_label → TE DataFrame`` where rows
            are source (leader) and columns are target (follower) assets.
        current_regime: The regime label active at the decision bar.
        followers: Which targets we care about.
        te_threshold: Minimum per-regime TE to include a pair.
        top_k_per_follower: Cap on leaders retained per follower.
        fallback_regime: If ``current_regime`` is missing, use this one.

    Returns:
        ``{follower: [{leader, te, weight}, …]}`` with weights
        normalised to sum to 1 within each follower.
    """
    te = per_regime_te.get(current_regime)
    if te is None and fallback_regime is not None:
        te = per_regime_te.get(fallback_regime)
    if te is None:
        return {}

    out: Dict[str, List[Dict[str, float]]] = {}
    for fol in followers:
        if fol not in te.columns:
            continue
        col = te[fol].drop(labels=[fol], errors="ignore")
        col = col[col >= te_threshold]
        if col.empty:
            continue
        top = col.nlargest(top_k_per_follower)
        total = float(top.sum()) or 1.0
        out[fol] = [
            {"leader": str(name), "te": float(val), "weight": float(val) / total}
            for name, val in top.items()
        ]
    return out


def compute_per_regime_te(
    returns: pd.DataFrame,
    regime_labels: np.ndarray,
    te_fn,
    min_bars_per_regime: int = 120,
    **te_kwargs,
) -> Dict[int, pd.DataFrame]:
    """Build a TE matrix per regime by slicing returns on ``regime_labels``.

    Regimes with fewer than ``min_bars_per_regime`` observations are
    skipped (a TE matrix on 40 bars is noise). ``te_fn`` must accept a
    ``Dict[str, np.ndarray]`` and return a tidy DataFrame with columns
    ``source, target, te`` (the existing ``compute_te_matrix``).

    Args:
        returns: Shape (T, N) returns panel aligned with ``regime_labels``.
        regime_labels: Array of length T with regime id per bar.
        te_fn: Callable used to compute TE per regime slice.
        min_bars_per_regime: Skip sparse regimes.
        **te_kwargs: Forwarded to ``te_fn``.

    Returns:
        Dict ``regime_id → TE DataFrame`` (source × target).
    """
    from collections import Counter

    if len(regime_labels) != len(returns):
        raise ValueError("regime_labels length must match returns")
    out: Dict[int, pd.DataFrame] = {}
    counts = Counter(int(r) for r in regime_labels)
    for regime, n in counts.items():
        if n < min_bars_per_regime:
            logger.debug("skip regime %d with only %d bars", regime, n)
            continue
        mask = regime_labels == regime
        sub = returns.loc[mask]
        series_dict = {c: sub[c].to_numpy() for c in sub.columns}
        tidy = te_fn(series_dict, **te_kwargs)
        # Pivot to (source × target)
        if isinstance(tidy, pd.DataFrame) and {"source", "target", "te"}.issubset(tidy.columns):
            mat = tidy.pivot(index="source", columns="target", values="te").fillna(0.0)
        else:
            mat = tidy  # already a matrix
        out[int(regime)] = mat
    return out
