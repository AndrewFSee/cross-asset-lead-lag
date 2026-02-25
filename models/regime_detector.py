"""Regime classification using Gaussian HMM or MS-VAR smoothed probabilities."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regimes using Gaussian HMM or MS-VAR output.

    Args:
        n_regimes: Number of regimes to detect.
        method: 'hmm' for standalone Gaussian HMM or 'ms_var' to use smoothed
            probabilities from an already-fitted MarkovSwitchingVAR.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        method: str = "hmm",
    ) -> None:
        self.n_regimes = n_regimes
        self.method = method
        self._model = None
        self._regime_labels: Optional[np.ndarray] = None
        self._dates: Optional[pd.DatetimeIndex] = None
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> "RegimeDetector":
        """Fit the regime detector.

        Args:
            features: Feature matrix of shape (T, n_features).
            dates: Optional date index for the time series.

        Returns:
            Self.
        """
        features = np.asarray(features, dtype=float)

        # Drop NaN rows
        mask = np.isfinite(features).all(axis=1) if features.ndim > 1 else np.isfinite(features)
        features = features[mask]
        if dates is not None:
            dates = dates[mask]

        if self.method == "hmm":
            try:
                from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42,
                )
                model.fit(features)
                self._model = model
                self._regime_labels = model.predict(features)
            except ImportError:
                logger.warning("hmmlearn not installed; falling back to K-means")
                from sklearn.cluster import KMeans  # noqa: PLC0415

                model = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
                self._regime_labels = model.fit_predict(features)
                self._model = model
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'hmm'.")

        self._features = features
        self._dates = dates
        self._fitted = True
        return self

    def predict_regime(self, features: np.ndarray) -> int:
        """Predict the regime for a new observation.

        Args:
            features: Feature vector or matrix (1+ rows).

        Returns:
            Predicted regime index (most recent).
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector must be fitted first")

        features = np.atleast_2d(np.asarray(features, dtype=float))

        if self.method == "hmm" and hasattr(self._model, "predict"):
            labels = self._model.predict(features)
            return int(labels[-1])
        elif hasattr(self._model, "predict"):
            labels = self._model.predict(features)
            return int(labels[-1])
        return 0

    def regime_history(self) -> pd.Series:
        """Return the full history of regime labels as a Series.

        Returns:
            Series of integer regime labels, indexed by date if available.
        """
        if not self._fitted or self._regime_labels is None:
            raise RuntimeError("RegimeDetector must be fitted first")

        if self._dates is not None:
            return pd.Series(self._regime_labels, index=self._dates, name="regime")
        return pd.Series(self._regime_labels, name="regime")

    def regime_summary(self) -> Dict[int, Dict[str, float]]:
        """Return per-regime statistics.

        Returns:
            Dict mapping regime index to dict with 'mean', 'std', 'n_obs', 'pct'.
        """
        if not self._fitted or self._regime_labels is None:
            raise RuntimeError("RegimeDetector must be fitted first")

        summary = {}
        features = self._features
        T = len(self._regime_labels)

        for r in range(self.n_regimes):
            mask = self._regime_labels == r
            n_obs = int(mask.sum())
            if n_obs > 0:
                mean_r = float(np.mean(features[mask]))
                std_r = float(np.std(features[mask]))
            else:
                mean_r = std_r = 0.0
            summary[r] = {
                "mean": mean_r,
                "std": std_r,
                "n_obs": n_obs,
                "pct": n_obs / T,
            }

        return summary
