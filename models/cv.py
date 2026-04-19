"""Purged and combinatorial-purged cross-validation for financial ML.

Implements the CV scheme from López de Prado, *Advances in Financial
Machine Learning* (2018), Ch. 7, and the combinatorial variant that
extends to N≥5 test groups to mitigate backtest overfitting.

The key invariants:

* **Purge** — any training sample whose label horizon overlaps a test
  sample's horizon is removed from training. For a forecasting horizon
  of `horizon` bars, this purges `horizon` bars before and after each
  test group.
* **Embargo** — a buffer of `embargo` bars immediately *after* a test
  group is also removed from training, to prevent autocorrelation
  leakage for features that depend on forward look-windows.

This module deliberately has no sklearn dependency on the split objects
— it yields `(train_idx, test_idx)` numpy arrays so callers can wire
into any estimator / scoring loop.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np


def _contiguous_groups(n: int, n_splits: int) -> List[np.ndarray]:
    """Partition range(n) into `n_splits` ~equal contiguous groups."""
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]


def purged_kfold_split(
    n_samples: int,
    n_splits: int = 5,
    horizon: int = 1,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold: contiguous test groups with purge+embargo in train.

    Args:
        n_samples: Number of observations (e.g. len(returns)).
        n_splits: Number of folds.
        horizon: Forecast/label horizon in bars — any training sample
            whose label overlaps a test sample is purged. Use the max
            prediction lag of your model.
        embargo: Extra bars removed immediately after each test group to
            absorb autocorrelation from slow-decay features.

    Yields:
        (train_idx, test_idx) numpy arrays.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    groups = _contiguous_groups(n_samples, n_splits)
    all_idx = np.arange(n_samples)
    for test in groups:
        if len(test) == 0:
            continue
        test_start, test_end = test[0], test[-1] + 1
        purge_start = max(0, test_start - horizon)
        purge_end = min(n_samples, test_end + horizon + embargo)
        mask = np.ones(n_samples, dtype=bool)
        mask[purge_start:purge_end] = False
        train = all_idx[mask]
        yield train, test


def combinatorial_purged_kfold_split(
    n_samples: int,
    n_splits: int = 6,
    n_test_groups: int = 2,
    horizon: int = 1,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Combinatorial Purged K-Fold (CPCV) of López de Prado 2018.

    Partition the series into `n_splits` contiguous groups, then for
    every combination of `n_test_groups` groups treat their union as
    the test set and the rest (after purge+embargo around each test
    group) as the train set. Yields `C(n_splits, n_test_groups)` folds.

    With `n_splits=6, n_test_groups=2` we get 15 folds, which is enough
    to generate many synthetic backtest paths for PBO / deflated-Sharpe
    calculations while still leaving ~66% of the data in training.

    Args:
        n_samples: Number of observations.
        n_splits: Number of contiguous groups.
        n_test_groups: How many groups to reserve for testing per fold.
        horizon: Forecast horizon (see `purged_kfold_split`).
        embargo: Embargo bars after each test group.

    Yields:
        (train_idx, test_idx) arrays. `test_idx` is the sorted union of
        the chosen test groups.
    """
    if n_test_groups < 1 or n_test_groups >= n_splits:
        raise ValueError("0 < n_test_groups < n_splits required")
    groups = _contiguous_groups(n_samples, n_splits)
    all_idx = np.arange(n_samples)
    for combo in combinations(range(n_splits), n_test_groups):
        test_parts = [groups[i] for i in combo]
        test = np.concatenate(test_parts)
        test.sort()
        mask = np.ones(n_samples, dtype=bool)
        for part in test_parts:
            start, end = part[0], part[-1] + 1
            purge_start = max(0, start - horizon)
            purge_end = min(n_samples, end + horizon + embargo)
            mask[purge_start:purge_end] = False
        train = all_idx[mask]
        yield train, test
