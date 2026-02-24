"""Data ingestion module for fetching market and macro data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "universe.yaml"


def _load_universe() -> Dict[str, Any]:
    """Load asset universe configuration from YAML file."""
    with open(_UNIVERSE_PATH) as f:
        return yaml.safe_load(f)


def _fetch_yahoo_series(ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.Series:
    """Fetch a single price series from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Price series with date index.
    """
    import yfinance as yf  # noqa: PLC0415

    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    close = data["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    close.name = ticker
    return close


def _fetch_fred_series(series_id: str, start_date: str, api_key: str) -> pd.Series:
    """Fetch a single series from FRED.

    Args:
        series_id: FRED series identifier.
        start_date: Start date string (YYYY-MM-DD).
        api_key: FRED API key.

    Returns:
        Series with date index.
    """
    from fredapi import Fred  # noqa: PLC0415

    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start_date)
    series.name = series_id
    return series


def fetch_all_data(
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    fred_api_key: str = "",
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch all data defined in the universe configuration.

    Downloads equity/commodity/fx/crypto prices from Yahoo Finance and
    rates/credit/vol/macro series from FRED. Results are cached as parquet
    files and returned as a dict of DataFrames keyed by asset class.

    Args:
        start_date: Start date for data fetching.
        end_date: End date for data fetching. Defaults to today.
        fred_api_key: FRED API key. Falls back to settings if empty.
        cache_dir: Directory for parquet cache.
        use_cache: Whether to use cached data if available.

    Returns:
        Dict mapping asset class names to DataFrames of price/level series.
    """
    try:
        from config.settings import Settings  # noqa: PLC0415

        settings = Settings()
        if not fred_api_key:
            fred_api_key = settings.fred_api_key
        if not end_date:
            end_date = None
    except Exception:
        pass

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    universe = _load_universe()
    result: Dict[str, pd.DataFrame] = {}

    # ── Equities (Yahoo Finance) ─────────────────────────────────────────────
    equity_cache = cache_path / "equities.parquet"
    if use_cache and equity_cache.exists():
        logger.info("Loading equities from cache")
        result["equities"] = pd.read_parquet(equity_cache)
    else:
        eq_series = {}
        eq_universe = universe.get("equities", {})
        for subgroup in ("indices", "sectors"):
            for name, cfg in eq_universe.get(subgroup, {}).items():
                try:
                    s = _fetch_yahoo_series(cfg["ticker"], start_date, end_date)
                    eq_series[name] = s
                    logger.info("Fetched equity %s (%s)", name, cfg["ticker"])
                except Exception as exc:
                    logger.warning("Failed to fetch equity %s: %s", name, exc)
        if eq_series:
            df = pd.DataFrame(eq_series)
            df.to_parquet(equity_cache)
            result["equities"] = df

    # ── Commodities (Yahoo Finance) ──────────────────────────────────────────
    commodities_cache = cache_path / "commodities.parquet"
    if use_cache and commodities_cache.exists():
        logger.info("Loading commodities from cache")
        result["commodities"] = pd.read_parquet(commodities_cache)
    else:
        com_series = {}
        for name, cfg in universe.get("commodities", {}).items():
            try:
                s = _fetch_yahoo_series(cfg["ticker"], start_date, end_date)
                com_series[name] = s
                logger.info("Fetched commodity %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch commodity %s: %s", name, exc)
        if com_series:
            df = pd.DataFrame(com_series)
            df.to_parquet(commodities_cache)
            result["commodities"] = df

    # ── FX (Yahoo Finance) ───────────────────────────────────────────────────
    fx_cache = cache_path / "fx.parquet"
    if use_cache and fx_cache.exists():
        logger.info("Loading FX from cache")
        result["fx"] = pd.read_parquet(fx_cache)
    else:
        fx_series = {}
        for name, cfg in universe.get("fx", {}).items():
            try:
                s = _fetch_yahoo_series(cfg["ticker"], start_date, end_date)
                fx_series[name] = s
                logger.info("Fetched FX %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch FX %s: %s", name, exc)
        if fx_series:
            df = pd.DataFrame(fx_series)
            df.to_parquet(fx_cache)
            result["fx"] = df

    # ── Volatility (Yahoo + FRED) ────────────────────────────────────────────
    vol_cache = cache_path / "volatility.parquet"
    if use_cache and vol_cache.exists():
        logger.info("Loading volatility from cache")
        result["volatility"] = pd.read_parquet(vol_cache)
    else:
        vol_series = {}
        for name, cfg in universe.get("volatility", {}).items():
            try:
                if cfg.get("source") == "yahoo":
                    s = _fetch_yahoo_series(cfg["ticker"], start_date, end_date)
                else:
                    if not fred_api_key:
                        logger.warning("No FRED API key; skipping %s", name)
                        continue
                    s = _fetch_fred_series(cfg["series_id"], start_date, fred_api_key)
                vol_series[name] = s
                logger.info("Fetched volatility %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch volatility %s: %s", name, exc)
        if vol_series:
            df = pd.DataFrame(vol_series)
            df.to_parquet(vol_cache)
            result["volatility"] = df

    # ── Crypto (Yahoo Finance) ───────────────────────────────────────────────
    crypto_cache = cache_path / "crypto.parquet"
    if use_cache and crypto_cache.exists():
        logger.info("Loading crypto from cache")
        result["crypto"] = pd.read_parquet(crypto_cache)
    else:
        crypto_series = {}
        for name, cfg in universe.get("crypto", {}).items():
            try:
                s = _fetch_yahoo_series(cfg["ticker"], start_date, end_date)
                crypto_series[name] = s
                logger.info("Fetched crypto %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch crypto %s: %s", name, exc)
        if crypto_series:
            df = pd.DataFrame(crypto_series)
            df.to_parquet(crypto_cache)
            result["crypto"] = df

    # ── Rates (FRED) ─────────────────────────────────────────────────────────
    rates_cache = cache_path / "rates.parquet"
    if use_cache and rates_cache.exists():
        logger.info("Loading rates from cache")
        result["rates"] = pd.read_parquet(rates_cache)
    elif fred_api_key:
        rates_series = {}
        for name, cfg in universe.get("rates", {}).items():
            try:
                s = _fetch_fred_series(cfg["series_id"], start_date, fred_api_key)
                rates_series[name] = s
                logger.info("Fetched rate %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch rate %s: %s", name, exc)
        if rates_series:
            df = pd.DataFrame(rates_series)
            df.to_parquet(rates_cache)
            result["rates"] = df
    else:
        logger.warning("No FRED API key provided; skipping rates data")

    # ── Credit (FRED) ────────────────────────────────────────────────────────
    credit_cache = cache_path / "credit.parquet"
    if use_cache and credit_cache.exists():
        logger.info("Loading credit from cache")
        result["credit"] = pd.read_parquet(credit_cache)
    elif fred_api_key:
        credit_series = {}
        for name, cfg in universe.get("credit", {}).items():
            try:
                s = _fetch_fred_series(cfg["series_id"], start_date, fred_api_key)
                credit_series[name] = s
                logger.info("Fetched credit %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch credit %s: %s", name, exc)
        if credit_series:
            df = pd.DataFrame(credit_series)
            df.to_parquet(credit_cache)
            result["credit"] = df
    else:
        logger.warning("No FRED API key provided; skipping credit data")

    # ── Macro (FRED) ─────────────────────────────────────────────────────────
    macro_cache = cache_path / "macro.parquet"
    if use_cache and macro_cache.exists():
        logger.info("Loading macro from cache")
        result["macro"] = pd.read_parquet(macro_cache)
    elif fred_api_key:
        macro_series = {}
        for name, cfg in universe.get("macro", {}).items():
            try:
                s = _fetch_fred_series(cfg["series_id"], start_date, fred_api_key)
                macro_series[name] = s
                logger.info("Fetched macro %s", name)
            except Exception as exc:
                logger.warning("Failed to fetch macro %s: %s", name, exc)
        if macro_series:
            df = pd.DataFrame(macro_series)
            df.to_parquet(macro_cache)
            result["macro"] = df
    else:
        logger.warning("No FRED API key provided; skipping macro data")

    logger.info("Data fetch complete. Asset classes loaded: %s", list(result.keys()))
    return result
