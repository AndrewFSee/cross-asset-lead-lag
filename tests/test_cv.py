"""Tests for purged K-fold and CPCV splitters."""

from __future__ import annotations

from math import comb

import numpy as np

from models.cv import combinatorial_purged_kfold_split, purged_kfold_split


class TestPurgedKFold:
    def test_all_samples_covered_as_test_exactly_once(self):
        n = 1000
        seen_test = np.zeros(n, dtype=int)
        for _train, test in purged_kfold_split(n, n_splits=5, horizon=1, embargo=0):
            seen_test[test] += 1
        assert (seen_test == 1).all()

    def test_train_and_test_disjoint(self):
        n = 500
        for train, test in purged_kfold_split(n, n_splits=5, horizon=3, embargo=2):
            assert np.intersect1d(train, test).size == 0

    def test_purge_buffer_around_test(self):
        n = 200
        horizon = 5
        embargo = 3
        for train, test in purged_kfold_split(n, n_splits=4, horizon=horizon, embargo=embargo):
            t_start, t_end = test[0], test[-1]
            # No train index within [t_start - horizon, t_end + horizon + embargo]
            forbidden_lo = max(0, t_start - horizon)
            forbidden_hi = min(n, t_end + 1 + horizon + embargo)
            forbidden = set(range(forbidden_lo, forbidden_hi))
            assert not (forbidden & set(train.tolist()))

    def test_invalid_params_raise(self):
        import pytest

        with pytest.raises(ValueError):
            list(purged_kfold_split(100, n_splits=1))
        with pytest.raises(ValueError):
            list(purged_kfold_split(100, n_splits=5, horizon=0))


class TestCombinatorialPurged:
    def test_correct_number_of_folds(self):
        n = 600
        folds = list(combinatorial_purged_kfold_split(n, n_splits=6, n_test_groups=2))
        assert len(folds) == comb(6, 2)

    def test_all_folds_disjoint(self):
        n = 600
        for train, test in combinatorial_purged_kfold_split(
            n, n_splits=6, n_test_groups=2, horizon=2, embargo=1,
        ):
            assert np.intersect1d(train, test).size == 0

    def test_test_set_size_scales_with_n_test_groups(self):
        n = 1200
        small = next(iter(combinatorial_purged_kfold_split(n, n_splits=6, n_test_groups=1)))
        big = next(iter(combinatorial_purged_kfold_split(n, n_splits=6, n_test_groups=3)))
        assert len(big[1]) > len(small[1])

    def test_invalid_n_test_groups(self):
        import pytest

        with pytest.raises(ValueError):
            list(combinatorial_purged_kfold_split(100, n_splits=5, n_test_groups=0))
        with pytest.raises(ValueError):
            list(combinatorial_purged_kfold_split(100, n_splits=5, n_test_groups=5))
