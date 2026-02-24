"""Walk-forward backtesting harness for lead-lag signals."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Walk-forward backtesting engine.

    Trains on an expanding or rolling window, generates signals via a
    user-provided function, and computes realized portfolio returns.

    Args:
        returns: DataFrame of asset returns (rows=dates, columns=assets).
        signal_func: Callable that takes a returns slice and returns a dict
            mapping asset name to target weight (or a pd.Series of weights).
        initial_window: Initial training window size (observations).
        step_size: Number of observations per rebalancing step.
        rebalance_freq: How often to rebalance (every N days).
        rolling: If True, use a rolling (fixed-size) window; else expanding.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        signal_func: Callable,
        initial_window: int = 252,
        step_size: int = 21,
        rebalance_freq: int = 5,
        rolling: bool = False,
    ) -> None:
        self.returns = returns
        self.signal_func = signal_func
        self.initial_window = initial_window
        self.step_size = step_size
        self.rebalance_freq = rebalance_freq
        self.rolling = rolling

        self._portfolio_returns: Optional[pd.Series] = None
        self._weights_history: Optional[pd.DataFrame] = None
        self._run_complete = False

    def run(self) -> "WalkForwardBacktest":
        """Execute the walk-forward backtest.

        Returns:
            Self.
        """
        returns = self.returns.dropna(how="all")
        T = len(returns)
        assets = returns.columns.tolist()

        port_returns = []
        weights_list = []
        dates = []

        current_weights = pd.Series(np.zeros(len(assets)), index=assets)

        for t in range(self.initial_window, T, self.step_size):
            # Training window
            if self.rolling:
                train_slice = returns.iloc[t - self.initial_window : t]
            else:
                train_slice = returns.iloc[:t]

            # Generate new weights
            try:
                new_weights = self.signal_func(train_slice)
                if isinstance(new_weights, dict):
                    new_weights = pd.Series(new_weights).reindex(assets).fillna(0.0)
                elif isinstance(new_weights, pd.Series):
                    new_weights = new_weights.reindex(assets).fillna(0.0)
                else:
                    new_weights = pd.Series(np.zeros(len(assets)), index=assets)
            except Exception as exc:
                logger.warning("Signal function failed at t=%d: %s", t, exc)
                new_weights = pd.Series(np.zeros(len(assets)), index=assets)

            current_weights = new_weights

            # Apply weights to out-of-sample period
            oos_end = min(t + self.step_size, T)
            for oos_t in range(t, oos_end):
                day_ret = returns.iloc[oos_t]
                port_ret = float((current_weights * day_ret).sum())
                port_returns.append(port_ret)
                dates.append(returns.index[oos_t])
                weights_list.append(current_weights.values.copy())

        self._portfolio_returns = pd.Series(port_returns, index=pd.DatetimeIndex(dates))
        self._weights_history = pd.DataFrame(
            weights_list,
            index=pd.DatetimeIndex(dates),
            columns=assets,
        )
        self._run_complete = True
        return self

    def compute_metrics(self) -> Dict[str, float]:
        """Compute backtest performance metrics.

        Returns:
            Dict with Sharpe, Sortino, max_drawdown, calmar, hit_rate,
            avg_turnover, total_return, annual_return.
        """
        if not self._run_complete:
            raise RuntimeError("Run backtest first with .run()")

        r = self._portfolio_returns.dropna()
        if len(r) == 0:
            return {}

        annualization = 252.0
        mean_ret = r.mean()
        std_ret = r.std()
        downside_std = r[r < 0].std() if (r < 0).any() else 1e-10

        sharpe = float(mean_ret / std_ret * np.sqrt(annualization)) if std_ret > 1e-10 else 0.0
        sortino = (
            float(mean_ret / downside_std * np.sqrt(annualization)) if downside_std > 1e-10 else 0.0
        )

        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        annual_ret = float((1 + mean_ret) ** annualization - 1)
        total_ret = float(cum.iloc[-1] - 1)
        calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        hit_rate = float((r > 0).mean())

        # Average turnover (sum of absolute weight changes per period)
        if self._weights_history is not None and len(self._weights_history) > 1:
            wdiff = self._weights_history.diff().abs().sum(axis=1)
            avg_turnover = float(wdiff.mean())
        else:
            avg_turnover = 0.0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "avg_turnover": avg_turnover,
            "total_return": total_ret,
            "annual_return": annual_ret,
        }

    def equity_curve(self) -> pd.Series:
        """Return cumulative equity curve.

        Returns:
            Series of cumulative returns (starts at 1.0).
        """
        if not self._run_complete:
            raise RuntimeError("Run backtest first with .run()")
        r = self._portfolio_returns.fillna(0.0)
        return (1 + r).cumprod()

    def drawdown_series(self) -> pd.Series:
        """Return drawdown series.

        Returns:
            Series of drawdown values (<= 0).
        """
        cum = self.equity_curve()
        rolling_max = cum.cummax()
        return (cum - rolling_max) / rolling_max

    def monthly_returns(self) -> pd.DataFrame:
        """Return monthly returns pivot table.

        Returns:
            DataFrame with years as rows and months as columns.
        """
        if not self._run_complete:
            raise RuntimeError("Run backtest first with .run()")
        r = self._portfolio_returns
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if len(monthly) == 0:
            return pd.DataFrame()
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        return monthly_df.pivot(index="year", columns="month", values="return")
