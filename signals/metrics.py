"""Backtest-evaluation metrics: deflated Sharpe, PBO, bootstrap CIs.

These are complements, not replacements, for the raw Sharpe / Sortino /
drawdown reporting in `signals.backtest`. Their purpose is to answer
"how much of this looks real, after accounting for multiple trials and
selection bias?"

References
----------
López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality."
Bailey, D. H., Borwein, J., López de Prado, M., Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting."
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def _euler_gamma() -> float:
    return 0.5772156649015329


def expected_maximum_sharpe(n_trials: int) -> float:
    """Expected max of `n_trials` i.i.d. N(0,1) Sharpe estimates.

    Closed-form from Bailey & López de Prado (2014), eq. (1).
    """
    if n_trials < 2:
        return 0.0
    gamma = _euler_gamma()
    a = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(a + b)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
) -> dict:
    """Deflated Sharpe ratio (DSR) of López de Prado 2014.

    Accounts for:
      - **Selection bias**: `n_trials` configurations tried on the same data.
      - **Non-normality**: higher-moment corrections (skew, kurtosis) to
        the SR standard error.
      - **Sample length**: shorter histories have wider SR sampling.

    The DSR is the probability that the true SR exceeds zero *after*
    deflating for the expected maximum across `n_trials`. A DSR ≥ 0.95 is
    the common publication threshold.

    Args:
        sharpe: Observed Sharpe ratio (annualised or not — consistent
            with `sharpe_std` used).
        n_obs: Number of return observations (T).
        n_trials: Number of independent strategy configurations tested.
        skew, kurtosis: Sample skewness and kurtosis (not excess) of
            the return stream. Defaults assume normal (γ₁=0, γ₂=3).
        sharpe_std: Std of SR estimates across trials — if unknown, 1.0
            yields the default "single-trial" deflation.

    Returns:
        Dict with keys:
            expected_max_sharpe, deflated_sharpe, p_value, passes_95.
    """
    if n_obs < 2:
        return {
            "expected_max_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "p_value": 1.0,
            "passes_95": False,
        }

    sr0 = sharpe_std * expected_maximum_sharpe(max(n_trials, 2))
    sr_std_error = math.sqrt(
        (1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe * sharpe)
        / max(n_obs - 1, 1)
    )
    if sr_std_error <= 0.0:
        dsr = 0.0
    else:
        dsr = float(stats.norm.cdf((sharpe - sr0) / sr_std_error))

    return {
        "expected_max_sharpe": float(sr0),
        "deflated_sharpe": float(dsr),
        "p_value": float(1.0 - dsr),
        "passes_95": bool(dsr >= 0.95),
    }


def probability_of_backtest_overfitting(
    in_sample_performance: Sequence[Sequence[float]],
) -> dict:
    """Probability of Backtest Overfitting (PBO) via combinatorial paths.

    Bailey et al. 2017. Given an (T, n_trials) matrix of per-bar returns
    for `n_trials` strategy configurations, partition time into `S`
    blocks (S is inferred: largest even S ≤ min(16, T//30)). For each
    way of splitting the S blocks into equal in-sample (IS) and
    out-of-sample (OOS) halves, rank trials by IS Sharpe and compute
    the relative OOS rank of the IS winner. PBO is the fraction of
    splits where the IS winner falls in the bottom half of OOS.

    A PBO ≥ 0.5 means the top IS strategy is *worse than median* OOS as
    often as not — a strong signal of overfitting.

    Args:
        in_sample_performance: Shape (T, n_trials). Each column is the
            per-bar return of one strategy configuration on the SAME
            calendar. Rows must line up across columns.

    Returns:
        Dict with: pbo (float in [0, 1]), n_splits (number of IS/OOS
        partitions evaluated).
    """
    R = np.asarray(in_sample_performance, dtype=float)
    if R.ndim != 2 or R.shape[0] < 60 or R.shape[1] < 2:
        return {"pbo": float("nan"), "n_splits": 0}

    from itertools import combinations

    T, M = R.shape
    S = min(16, (T // 30) * 2)
    if S < 4 or S % 2 != 0:
        S = max(4, S - (S % 2))
    bounds = np.linspace(0, T, S + 1, dtype=int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(S)]

    losses = []
    for is_combo in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_combo])
        is_mu = R[is_idx].mean(axis=0)
        is_sd = R[is_idx].std(axis=0, ddof=1) + 1e-12
        oos_mu = R[oos_idx].mean(axis=0)
        oos_sd = R[oos_idx].std(axis=0, ddof=1) + 1e-12
        is_sr = is_mu / is_sd
        oos_sr = oos_mu / oos_sd
        is_winner = int(np.argmax(is_sr))
        # OOS rank of that strategy (1 = best, M = worst)
        oos_rank = int((oos_sr > oos_sr[is_winner]).sum()) + 1
        # Logit loss: λ = log(rank / (M + 1 - rank)) — positive means
        # below-median, which counts toward PBO.
        lam = math.log(oos_rank / max(M + 1 - oos_rank, 1))
        losses.append(lam > 0)

    pbo = float(np.mean(losses)) if losses else float("nan")
    return {"pbo": pbo, "n_splits": len(losses)}


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    annualization: float = 252.0,
    random_state: int | None = 0,
) -> dict:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Uses non-overlapping random blocks of length √T (Politis & Romano
    style). For daily data over 10 years this is ~50-day blocks.

    Args:
        returns: 1-D sequence of per-bar returns.
        n_boot: Bootstrap replicates.
        ci: Two-sided confidence level.
        annualization: Factor for SR annualisation (252 daily, 52 weekly).
        random_state: RNG seed.

    Returns:
        Dict with sharpe, lower, upper, annualization.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 30:
        return {"sharpe": float("nan"), "lower": float("nan"),
                "upper": float("nan"), "annualization": annualization}
    block = max(5, int(round(np.sqrt(T))))
    n_blocks = T // block
    rng = np.random.default_rng(random_state)

    sr_boot = np.zeros(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([r[p * block:(p + 1) * block] for p in picks])
        sd = sample.std(ddof=1)
        sr_boot[i] = (sample.mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0

    lo = float(np.quantile(sr_boot, 0.5 - ci / 2))
    hi = float(np.quantile(sr_boot, 0.5 + ci / 2))
    point = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(annualization)
    return {
        "sharpe": float(point),
        "lower": lo,
        "upper": hi,
        "annualization": float(annualization),
    }
