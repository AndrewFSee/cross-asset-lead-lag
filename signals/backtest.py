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
    user-provided function, and computes realized portfolio returns net
    of transaction costs.

    Args:
        returns: DataFrame of asset returns (rows=dates, columns=assets).
        signal_func: Callable that takes a returns slice and returns a dict
            mapping asset name to target weight (or a pd.Series of weights).
        initial_window: Initial training window size (observations).
        step_size: Number of observations per rebalancing step.
        rebalance_freq: How often to rebalance (every N days).
        rolling: If True, use a rolling (fixed-size) window; else expanding.
        tc_bps: Round-trip transaction cost in basis points applied to
            per-side turnover. Each rebalance day we subtract
            (tc_bps / 2 / 10_000) * sum(|Δweights|) from the portfolio
            return — so a 100%-turnover rebalance costs exactly tc_bps bps.
        execution_lag: Number of bars between signal generation and first
            realized return. With execution_lag=1 (default), signals made
            from data through close(t-1) trade from close(t) onward —
            reflecting a realistic "decide today, execute next close" flow.
            Set to 0 to reproduce the (optimistic) same-close behaviour.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        signal_func: Callable,
        initial_window: int = 252,
        step_size: int = 21,
        rebalance_freq: int = 5,
        rolling: bool = False,
        tc_bps: float = 1.0,
        execution_lag: int = 1,
    ) -> None:
        self.returns = returns
        self.signal_func = signal_func
        self.initial_window = initial_window
        self.step_size = step_size
        self.rebalance_freq = rebalance_freq
        self.rolling = rolling
        self.tc_bps = float(tc_bps)
        self.execution_lag = int(execution_lag)

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
        tc_per_unit = self.tc_bps / 2.0 / 10_000.0  # per-side cost per unit of |Δw|

        for t in range(self.initial_window, T, self.step_size):
            # Training window — uses data strictly before time t
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

            # Transaction cost on the *change* from existing book
            turnover = float((new_weights - current_weights).abs().sum())
            rebalance_cost = tc_per_unit * turnover

            current_weights = new_weights

            # Apply weights to out-of-sample period. execution_lag shifts the
            # first realised return by N bars to reflect "decide at close(t-1),
            # execute at close(t-1+lag)" — lag=1 is the realistic default.
            oos_start = t + self.execution_lag
            oos_end = min(t + self.step_size + self.execution_lag, T)
            first_bar = True
            for oos_t in range(oos_start, oos_end):
                day_ret = returns.iloc[oos_t]
                port_ret = float((current_weights * day_ret).sum())
                if first_bar:
                    port_ret -= rebalance_cost
                    first_bar = False
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
