"""Markov-Switching VAR (MS-VAR) model with EM estimation.

Reference:
    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2).

    Krolzig, H.-M. (1997). Markov-Switching Vector Autoregressions.
    Lecture Notes in Economics and Mathematical Systems, Springer.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class MarkovSwitchingVAR:
    """Markov-Switching Vector Autoregression with 2+ regimes.

    Estimated via the EM algorithm using Hamilton filter (E-step) and
    Kim smoother for smoothed probabilities, with M-step updating
    regime-specific VAR coefficients and covariance matrices.

    Args:
        n_vars: Number of variables in the VAR system.
        n_lags: Number of VAR lags.
        n_regimes: Number of regimes.
    """

    def __init__(
        self,
        n_vars: int,
        n_lags: int = 2,
        n_regimes: int = 2,
    ) -> None:
        self.n_vars = n_vars
        self.n_lags = n_lags
        self.n_regimes = n_regimes

        # Estimated parameters
        self.B: np.ndarray = np.zeros((n_regimes, n_vars, n_vars * n_lags + 1))
        self.Sigma: np.ndarray = np.array([np.eye(n_vars) for _ in range(n_regimes)])
        self.P: np.ndarray = np.full((n_regimes, n_regimes), 1.0 / n_regimes)  # Transition
        self.pi: np.ndarray = np.full(n_regimes, 1.0 / n_regimes)  # Initial probs

        self._smoothed_probs: Optional[np.ndarray] = None
        self._filtered_probs: Optional[np.ndarray] = None
        self._log_likelihood: float = -np.inf
        self._fitted: bool = False

    def _initialize(self, Y: np.ndarray) -> None:
        """Initialize parameters using K-means clustering on returns.

        Args:
            Y: Data matrix of shape (T, n_vars).
        """
        from sklearn.cluster import KMeans  # noqa: PLC0415

        km = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
        labels = km.fit_predict(Y)

        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() > self.n_vars * self.n_lags + 1:
                self.Sigma[r] = np.cov(Y[mask].T) + 1e-6 * np.eye(self.n_vars)
            else:
                self.Sigma[r] = np.eye(self.n_vars)

        # Ergodic transition matrix
        self.P = np.full((self.n_regimes, self.n_regimes), 0.1 / (self.n_regimes - 1))
        np.fill_diagonal(self.P, 0.9)

        # Random VAR coefficients (small)
        self.B = np.random.randn(self.n_regimes, self.n_vars, self.n_vars * self.n_lags + 1) * 0.01
        self.pi = np.full(self.n_regimes, 1.0 / self.n_regimes)

    def _build_lagged_matrix(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build lagged regressor matrix.

        Args:
            Y: Data matrix of shape (T, n_vars).

        Returns:
            Tuple (Y_t, X_t) where Y_t shape (T-p, n_vars) and
            X_t shape (T-p, n_vars*p+1).
        """
        T = len(Y)
        p = self.n_lags
        T_eff = T - p

        Y_t = Y[p:]  # (T_eff, n_vars)
        X_t = np.zeros((T_eff, self.n_vars * p + 1))
        X_t[:, 0] = 1.0  # Intercept

        for lag in range(1, p + 1):
            start_col = 1 + (lag - 1) * self.n_vars
            end_col = start_col + self.n_vars
            X_t[:, start_col:end_col] = Y[p - lag : T - lag]

        return Y_t, X_t

    def _mvn_logpdf(self, Y_t: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Multivariate normal log-density for each observation.

        Args:
            Y_t: Observations of shape (T, n_vars).
            mean: Mean vector of shape (n_vars,).
            cov: Covariance matrix of shape (n_vars, n_vars).

        Returns:
            Log-densities of shape (T,).
        """
        try:
            log_dens = multivariate_normal.logpdf(Y_t, mean=mean, cov=cov, allow_singular=True)
        except Exception:
            log_dens = np.full(len(Y_t), -500.0)
        return np.atleast_1d(log_dens)

    def _hamilton_filter(
        self, Y_t: np.ndarray, X_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Hamilton (1989) filter for regime probabilities.

        Args:
            Y_t: Dependent variable (T_eff, n_vars).
            X_t: Lagged regressors (T_eff, n_vars*p+1).

        Returns:
            Tuple of (filtered_probs, predicted_probs, log_likelihood)
            Each prob array: shape (T_eff, n_regimes).
        """
        T = len(Y_t)
        R = self.n_regimes
        filtered = np.zeros((T, R))
        predicted = np.zeros((T, R))
        log_lik = 0.0

        prob_t = self.pi.copy()

        for t in range(T):
            predicted[t] = prob_t

            # Compute regime-conditional densities
            eta = np.zeros(R)
            for r in range(R):
                mean_r = X_t[t] @ self.B[r].T  # (n_vars,)
                eta[r] = np.exp(
                    np.clip(
                        self._mvn_logpdf(Y_t[t : t + 1], mean_r, self.Sigma[r])[0],
                        -500,
                        0,
                    )
                )

            joint = predicted[t] * eta
            total = joint.sum()
            if total < 1e-300:
                total = 1e-300
            filtered[t] = joint / total
            log_lik += np.log(total)

            # Predict next period
            prob_t = self.P.T @ filtered[t]
            prob_t = np.clip(prob_t, 1e-10, 1.0)
            prob_t /= prob_t.sum()

        return filtered, predicted, log_lik

    def _kim_smoother(self, filtered: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Kim (1994) smoother for smoothed regime probabilities.

        Args:
            filtered: Filtered probs (T, n_regimes).
            predicted: Predicted probs (T, n_regimes).

        Returns:
            Smoothed probabilities (T, n_regimes).
        """
        T, R = filtered.shape
        smoothed = np.zeros((T, R))
        smoothed[-1] = filtered[-1]

        for t in range(T - 2, -1, -1):
            for r in range(R):
                # Sum over next-period states
                num = 0.0
                for s in range(R):
                    if predicted[t + 1, s] > 1e-300:
                        num += self.P[r, s] * smoothed[t + 1, s] / predicted[t + 1, s]
                smoothed[t, r] = filtered[t, r] * num

            # Normalize
            row_sum = smoothed[t].sum()
            if row_sum > 1e-300:
                smoothed[t] /= row_sum
            else:
                smoothed[t] = filtered[t]

        return smoothed

    def _m_step(
        self,
        Y_t: np.ndarray,
        X_t: np.ndarray,
        smoothed: np.ndarray,
        filtered: np.ndarray,
    ) -> None:
        """M-step: update parameters given smoothed/filtered probabilities.

        Args:
            Y_t: Observations (T_eff, n_vars).
            X_t: Lagged regressors (T_eff, n_vars*p+1).
            smoothed: Smoothed probabilities (T_eff, n_regimes).
            filtered: Filtered probabilities (T_eff, n_regimes).
        """
        T, R = smoothed.shape

        # Update transition matrix
        for r in range(R):
            for s in range(R):
                num = sum(
                    filtered[t, r]
                    * self.P[r, s]
                    * smoothed[t + 1, s]
                    / max(1e-300, (self.P.T @ filtered[t]).sum())
                    for t in range(T - 1)
                )
                self.P[r, s] = num

            # Normalize row
            row_sum = self.P[r].sum()
            if row_sum > 1e-10:
                self.P[r] /= row_sum
            else:
                self.P[r] = np.full(R, 1.0 / R)

        # Update initial distribution
        self.pi = smoothed[0] / smoothed[0].sum()

        # Update regime-specific VAR coefficients and covariances
        for r in range(R):
            w = smoothed[:, r]  # (T,)
            W = np.diag(w)

            # WLS: B_r = (X'WX)^{-1} X'WY
            XtW = X_t.T @ W  # (k, T)
            XtWX = XtW @ X_t  # (k, k)
            XtWY = XtW @ Y_t  # (k, n_vars)

            reg = 1e-6 * np.eye(XtWX.shape[0])
            try:
                B_r = np.linalg.solve(XtWX + reg, XtWY)  # (k, n_vars)
                self.B[r] = B_r.T
            except np.linalg.LinAlgError:
                pass

            # Update covariance
            resid = Y_t - X_t @ self.B[r].T  # (T, n_vars)
            w_sum = max(w.sum(), 1e-10)
            Sigma_r = (resid.T @ W @ resid) / w_sum
            self.Sigma[r] = Sigma_r + 1e-6 * np.eye(self.n_vars)

    def fit(
        self,
        Y: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> "MarkovSwitchingVAR":
        """Fit the MS-VAR via EM algorithm.

        Args:
            Y: Data matrix (T, n_vars). Must have T > n_lags.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood.

        Returns:
            Self.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._initialize(Y)
        Y_t, X_t = self._build_lagged_matrix(Y)

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step
            filtered, predicted, log_lik = self._hamilton_filter(Y_t, X_t)
            smoothed = self._kim_smoother(filtered, predicted)

            # M-step
            self._m_step(Y_t, X_t, smoothed, filtered)

            delta = log_lik - prev_ll
            logger.debug("EM iter %d: log-lik=%.4f, delta=%.6f", iteration, log_lik, delta)

            if abs(delta) < tol and iteration > 5:
                logger.info("MS-VAR converged at iteration %d", iteration)
                break
            prev_ll = log_lik

        # Final E-step to get final probabilities
        filtered, predicted, log_lik = self._hamilton_filter(Y_t, X_t)
        smoothed = self._kim_smoother(filtered, predicted)

        self._filtered_probs = filtered
        self._smoothed_probs = smoothed
        self._log_likelihood = log_lik
        self._Y_t = Y_t
        self._X_t = X_t
        self._fitted = True

        return self

    def forecast(
        self,
        Y: np.ndarray,
        horizon: int = 1,
    ) -> np.ndarray:
        """Regime-weighted forecast.

        Args:
            Y: Recent data to condition on (at least n_lags rows).
            horizon: Forecast horizon (only 1 is exact; >1 is iterated).

        Returns:
            Forecast array of shape (horizon, n_vars).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        Y = np.asarray(Y, dtype=float)
        p = self.n_lags

        forecasts = []
        Y_ext = Y[-p:].copy()

        for h in range(horizon):
            # Build regressor
            x_t = np.zeros(self.n_vars * p + 1)
            x_t[0] = 1.0
            for lag in range(1, p + 1):
                start = 1 + (lag - 1) * self.n_vars
                x_t[start : start + self.n_vars] = Y_ext[-lag]

            # Regime probabilities (use last filtered if available)
            if self._filtered_probs is not None:
                regime_probs = self._filtered_probs[-1]
            else:
                regime_probs = self.pi

            # Weighted forecast
            fc = sum(regime_probs[r] * (x_t @ self.B[r].T) for r in range(self.n_regimes))
            forecasts.append(fc)
            Y_ext = np.vstack([Y_ext[1:], fc])

        return np.array(forecasts)

    def get_regime_coefficients(self) -> Dict[int, np.ndarray]:
        """Return VAR coefficients for each regime.

        Returns:
            Dict mapping regime index to coefficient matrix of shape
            (n_vars, n_vars*n_lags+1).
        """
        return {r: self.B[r].copy() for r in range(self.n_regimes)}

    def get_current_regime(self) -> int:
        """Return the most likely current regime.

        Returns:
            Regime index with highest filtered probability.
        """
        if self._filtered_probs is None:
            return 0
        return int(np.argmax(self._filtered_probs[-1]))

    @property
    def smoothed_probs(self) -> Optional[np.ndarray]:
        """Smoothed regime probabilities of shape (T, n_regimes)."""
        return self._smoothed_probs

    @property
    def log_likelihood(self) -> float:
        """Final log-likelihood value."""
        return self._log_likelihood
