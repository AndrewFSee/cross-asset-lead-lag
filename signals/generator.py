"""Lead-lag signal generation and Bayesian model averaging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

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
