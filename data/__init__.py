"""Data module for cross-asset lead-lag discovery engine."""

from data.ingestion import fetch_all_data
from data.preprocessing import (
    align_calendars,
    handle_missing,
    stationarity_check,
    winsorize_returns,
)
from data.returns import build_returns_panel

__all__ = [
    "fetch_all_data",
    "stationarity_check",
    "winsorize_returns",
    "handle_missing",
    "align_calendars",
    "build_returns_panel",
]
