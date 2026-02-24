"""Pydantic settings for the cross-asset lead-lag discovery engine."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Attributes:
        fred_api_key: FRED API key for macroeconomic data.
        openai_api_key: OpenAI API key for LLM narrative generation.
        slack_webhook_url: Slack webhook URL for alerts.
        data_cache_dir: Directory for local parquet cache.
        default_start_date: Default start date for data fetching.
        te_lags: Transfer entropy lag values.
        te_k_neighbors: Number of neighbors for KNN-based TE estimator.
        ms_var_n_regimes: Number of regimes for Markov-Switching VAR.
        ms_var_n_lags: Number of lags for MS-VAR.
        neural_gc_hidden_dim: Hidden dimension for neural GC LSTM.
        neural_gc_history_len: History length for neural GC.
        neural_gc_epochs: Training epochs for neural GC.
        bootstrap_n_samples: Number of bootstrap samples for significance.
        significance_alpha: Significance level for hypothesis tests.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    fred_api_key: str = Field(default="", description="FRED API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    slack_webhook_url: str = Field(default="", description="Slack webhook URL")
    data_cache_dir: str = Field(default="data/cache", description="Data cache directory")
    default_start_date: str = Field(default="2005-01-01", description="Default start date")
    te_lags: List[int] = Field(default=[1, 2, 3, 5, 10, 20], description="TE lag values")
    te_k_neighbors: int = Field(default=5, description="KNN neighbors for TE")
    ms_var_n_regimes: int = Field(default=2, description="Number of MS-VAR regimes")
    ms_var_n_lags: int = Field(default=2, description="Number of MS-VAR lags")
    neural_gc_hidden_dim: int = Field(default=64, description="Neural GC hidden dim")
    neural_gc_history_len: int = Field(default=20, description="Neural GC history length")
    neural_gc_epochs: int = Field(default=200, description="Neural GC training epochs")
    bootstrap_n_samples: int = Field(default=1000, description="Bootstrap sample count")
    significance_alpha: float = Field(default=0.05, description="Significance level")

    @property
    def cache_path(self) -> Path:
        """Return the cache directory as a Path object."""
        return Path(self.data_cache_dir)
