"""Lasso-penalized VAR for high-dimensional lead-lag discovery."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC

logger = logging.getLogger(__name__)


class LassoVAR:
    """Lasso-penalized VAR model.

    Fits a separate Lasso regression for each variable in the system,
    using cross-validation or information criteria to select lambda.

    Args:
        n_lags: Number of VAR lags.
        alpha: Regularization parameter. If None, selected via BIC.
        criterion: 'bic' or 'aic' for automatic alpha selection (ignored if alpha given).
        max_iter: Maximum iterations for Lasso solver.
    """

    def __init__(
        self,
        n_lags: int = 5,
        alpha: Optional[float] = None,
        criterion: str = "bic",
        max_iter: int = 10000,
    ) -> None:
        self.n_lags = n_lags
        self.alpha = alpha
        self.criterion = criterion
        self.max_iter = max_iter

        self._coefs: List[np.ndarray] = []
        self._intercepts: List[float] = []
        self._alphas: List[float] = []
        self._asset_names: List[str] = []
        self._fitted = False

    def _build_lagged_matrix(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build lagged regressor matrix for VAR.

        Args:
            Y: Data matrix of shape (T, n_vars).

        Returns:
            Tuple (Y_t, X_t) where Y_t.shape=(T-p, n_vars) and
            X_t.shape=(T-p, n_vars*p).
        """
        T, n_vars = Y.shape
        p = self.n_lags
        T_eff = T - p

        Y_t = Y[p:]
        X_t = np.zeros((T_eff, n_vars * p))

        for lag in range(1, p + 1):
            start_col = (lag - 1) * n_vars
            X_t[:, start_col : start_col + n_vars] = Y[p - lag : T - lag]

        return Y_t, X_t

    def fit(self, Y: np.ndarray, asset_names: Optional[List[str]] = None) -> "LassoVAR":
        """Fit Lasso-VAR model.

        Args:
            Y: Data matrix of shape (T, n_vars).
            asset_names: Optional list of asset names for result labeling.

        Returns:
            Self.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        T, n_vars = Y.shape

        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n_vars)]
        self._asset_names = asset_names

        Y_t, X_t = self._build_lagged_matrix(Y)

        self._coefs = []
        self._intercepts = []
        self._alphas = []

        for j in range(n_vars):
            y_j = Y_t[:, j]

            if self.alpha is not None:
                from sklearn.linear_model import Lasso  # noqa: PLC0415

                model = Lasso(alpha=self.alpha, max_iter=self.max_iter, fit_intercept=True)
            else:
                # Select alpha via BIC or AIC using LassoLarsIC
                model = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter)

            model.fit(X_t, y_j)
            self._coefs.append(model.coef_.copy())
            self._intercepts.append(float(model.intercept_))
            alpha_val = model.alpha_ if hasattr(model, "alpha_") else (self.alpha or 0.0)
            self._alphas.append(float(alpha_val))
            logger.debug(
                "Fitted Lasso-VAR for %s: alpha=%.4f, nonzero=%d",
                asset_names[j],
                alpha_val,
                np.sum(model.coef_ != 0),
            )

        self._n_vars = n_vars
        self._fitted = True
        return self

    def predict(self, Y: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Generate iterated forecasts.

        Args:
            Y: Recent data to condition on (at least n_lags rows).
            horizon: Forecast horizon.

        Returns:
            Forecast array of shape (horizon, n_vars).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")

        Y = np.asarray(Y, dtype=float)
        n_vars = self._n_vars
        p = self.n_lags
        Y_ext = Y[-p:].copy()

        forecasts = []
        for _ in range(horizon):
            x_t = np.concatenate([Y_ext[-lag] for lag in range(1, p + 1)])
            fc = np.array(
                [float(self._intercepts[j] + x_t @ self._coefs[j]) for j in range(n_vars)]
            )
            forecasts.append(fc)
            Y_ext = np.vstack([Y_ext[1:], fc])

        return np.array(forecasts)

    def get_lead_lag_matrix(self) -> pd.DataFrame:
        """Return the lead-lag coefficient matrix.

        Returns a summary of which assets have non-zero lagged coefficients
        on which other assets (at any lag). Rows = leaders, Columns = followers.

        Returns:
            DataFrame of shape (n_vars, n_vars) with mean absolute coefficient.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before inspecting coefficients")

        n_vars = self._n_vars
        p = self.n_lags
        lead_lag = np.zeros((n_vars, n_vars))

        for j in range(n_vars):
            coef = self._coefs[j]  # (n_vars * p,)
            for lag in range(1, p + 1):
                start = (lag - 1) * n_vars
                lag_coefs = coef[start : start + n_vars]
                for i in range(n_vars):
                    lead_lag[i, j] += abs(lag_coefs[i])

        return pd.DataFrame(
            lead_lag / p,
            index=self._asset_names,
            columns=self._asset_names,
        )
