"""Signals module for lead-lag signal generation, portfolio construction, and backtesting."""

from signals.backtest import WalkForwardBacktest
from signals.generator import LeadSignal, bayesian_model_average, generate_signals
from signals.portfolio import apply_constraints, kelly_sizing, risk_parity_weights

__all__ = [
    "LeadSignal",
    "generate_signals",
    "bayesian_model_average",
    "risk_parity_weights",
    "kelly_sizing",
    "apply_constraints",
    "WalkForwardBacktest",
]
