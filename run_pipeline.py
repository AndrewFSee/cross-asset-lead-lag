"""End-to-end pipeline for the cross-asset lead-lag discovery engine.

Usage:
    python run_pipeline.py                  # Full pipeline (requires FRED_API_KEY)
    python run_pipeline.py --skip-fetch     # Skip data fetch, use cached data
    python run_pipeline.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Asset Lead-Lag Discovery Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch, use cache")
    parser.add_argument("--te-only", action="store_true", help="Run only TE discovery")
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3, 5], help="TE lag values")
    parser.add_argument("--n-regimes", type=int, default=2, help="Number of MS-VAR regimes")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument(
        "--recent-window", type=int, default=750,
        help="Use only the last N observations for TE/MI (0 = all data)",
    )
    args = parser.parse_args()

    from config.settings import Settings

    settings = Settings()

    # Ensure output directory exists
    output_dir = Path(settings.data_cache_dir).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data Ingestion ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)

    cache_dir = settings.data_cache_dir
    if args.skip_fetch:
        logger.info("Loading cached data from %s", cache_dir)
        data_dict = _load_cached_data(cache_dir)
        if not data_dict:
            logger.error("No cached data found. Run without --skip-fetch first.")
            sys.exit(1)
    else:
        if not settings.fred_api_key:
            logger.warning("No FRED_API_KEY set. FRED series (rates, credit, macro) will be skipped.")
        from data.ingestion import fetch_all_data

        data_dict = fetch_all_data(
            start_date=settings.default_start_date,
            fred_api_key=settings.fred_api_key,
            cache_dir=cache_dir,
        )

    logger.info("Loaded asset classes: %s", list(data_dict.keys()))
    for ac, df in data_dict.items():
        if df is not None and not df.empty:
            logger.info("  %s: %d rows × %d columns", ac, len(df), len(df.columns))

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing & Returns Panel")
    logger.info("=" * 60)

    from data.preprocessing import align_calendars, winsorize_returns
    from data.returns import build_returns_panel

    aligned = align_calendars(data_dict)
    panel = build_returns_panel(aligned)

    if panel.empty:
        logger.error("Returns panel is empty. Check data ingestion.")
        sys.exit(1)

    panel = winsorize_returns(panel, n_sigma=4.0)
    panel = panel.dropna(how="all").fillna(0.0)

    logger.info("Returns panel: %d dates × %d assets", len(panel), len(panel.columns))
    logger.info("Date range: %s to %s", panel.index.min().date(), panel.index.max().date())
    panel.to_parquet(output_dir / "returns_panel.parquet")

    # ── Step 3: Transfer Entropy Discovery ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Transfer Entropy Discovery")
    logger.info("=" * 60)

    from discovery.transfer_entropy import compute_te_matrix

    # Optionally trim to recent window for performance
    te_panel = panel
    if args.recent_window and args.recent_window > 0 and len(panel) > args.recent_window:
        te_panel = panel.iloc[-args.recent_window :]
        logger.info(
            "Trimmed to last %d observations (%s to %s) for TE computation",
            args.recent_window, te_panel.index.min().date(), te_panel.index.max().date(),
        )

    returns_dict = {col: te_panel[col].values for col in te_panel.columns}
    te_matrices = compute_te_matrix(returns_dict, lags=args.lags, k=settings.te_k_neighbors)

    for lag, te_df in te_matrices.items():
        te_df.to_parquet(output_dir / f"te_matrix_lag{lag}.parquet")
        top_pairs = _top_te_pairs(te_df, n=5)
        logger.info("Top TE pairs (lag=%d):", lag)
        for src, tgt, val in top_pairs:
            logger.info("  %s → %s : %.4f", src, tgt, val)

    # ── Step 3b: TE Decay Profiles ───────────────────────────────────────────
    logger.info("Computing TE decay profiles for top pairs...")
    from discovery.transfer_entropy import compute_te_decay

    te_lag1 = te_matrices.get(args.lags[0], list(te_matrices.values())[0])
    top_n_pairs = _top_te_pairs(te_lag1, n=40)
    pair_tuples = [(src, tgt) for src, tgt, _ in top_n_pairs]
    decay_lags = [1, 2, 3, 5, 10, 20]
    decay_df = compute_te_decay(
        returns_dict, pair_tuples, lags=decay_lags, k=settings.te_k_neighbors,
    )
    decay_df.to_parquet(output_dir / "te_decay.parquet", index=False)

    for cat in decay_df["category"].unique():
        subset = decay_df[decay_df["category"] == cat].drop_duplicates(subset=["source", "target"])
        pairs_str = ", ".join(f"{r.source}→{r.target}" for _, r in subset.iterrows())
        logger.info("  %s: %s", cat, pairs_str)

    if args.te_only:
        logger.info("TE-only mode. Pipeline complete.")
        return

    # ── Step 4: Lasso VAR ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Lasso-Penalized VAR")
    logger.info("=" * 60)

    from models.lasso_var import LassoVAR

    lasso = LassoVAR(n_lags=5)
    lasso.fit(panel.values, asset_names=panel.columns.tolist())
    ll_matrix = lasso.get_lead_lag_matrix()

    top_lasso = _top_te_pairs(ll_matrix, n=5)
    ll_matrix.to_parquet(output_dir / "lasso_var_matrix.parquet")
    logger.info("Top Lasso-VAR lead-lag pairs:")
    for src, tgt, val in top_lasso:
        logger.info("  %s → %s : %.4f", src, tgt, val)

    # ── Step 5: Markov-Switching VAR ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Markov-Switching VAR")
    logger.info("=" * 60)

    # Select top assets by TE for MS-VAR (limit dimensionality)
    te_lag1 = te_matrices.get(1, list(te_matrices.values())[0])
    top_assets = _select_top_assets(te_lag1, max_assets=8)
    sub_panel = panel[top_assets].dropna()

    from models.ms_var import MarkovSwitchingVAR

    ms_var = MarkovSwitchingVAR(
        n_vars=len(top_assets),
        n_lags=settings.ms_var_n_lags,
        n_regimes=args.n_regimes,
    )
    ms_var.fit(sub_panel.values, max_iter=50)

    current_regime = ms_var.get_current_regime()
    logger.info("Current regime: %d", current_regime)
    logger.info("Log-likelihood: %.2f", ms_var.log_likelihood)

    # ── Step 6: Regime Detection ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Regime Detection")
    logger.info("=" * 60)

    from models.regime_detector import RegimeDetector

    # Build interpretable regime features from the full returns panel
    regime_features = _build_regime_features(panel)
    logger.info("Regime features: %s (%d obs)", list(regime_features.columns), len(regime_features))

    detector = RegimeDetector(n_regimes=args.n_regimes, method="hmm")
    detector.fit(regime_features.values, dates=regime_features.index)

    summary = detector.regime_summary()
    regime_labels = pd.Series(
        detector._regime_labels, index=regime_features.index, name="regime",
    )
    regime_labels.to_frame().to_parquet(output_dir / "regime_labels.parquet")
    for r, stats in summary.items():
        logger.info(
            "Regime %d: mean=%.4f, std=%.4f, obs=%d (%.1f%%)",
            r, stats["mean"], stats["std"], stats["n_obs"], stats["pct"] * 100,
        )

    # ── Step 7: Walk-Forward Backtest (optional) ─────────────────────────────
    if args.backtest:
        logger.info("=" * 60)
        logger.info("STEP 7: Walk-Forward Backtest")
        logger.info("=" * 60)

        from signals.backtest import WalkForwardBacktest

        # Use the recent-window subset for backtest to avoid overflow
        bt_panel = te_panel if args.recent_window and args.recent_window > 0 else panel

        # --- Strategy A: Inverse-vol benchmark (risk parity, no signals) ---
        def benchmark_signal_func(train_returns: pd.DataFrame) -> dict:
            """Inverse-volatility weighted benchmark — no lead-lag signals."""
            vols = train_returns.iloc[-252:].std()
            vols = vols.replace(0, np.nan).dropna()
            if vols.empty:
                n = len(train_returns.columns)
                return {col: 1.0 / n for col in train_returns.columns}
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()
            return weights.to_dict()

        bt_bench = WalkForwardBacktest(
            returns=bt_panel,
            signal_func=benchmark_signal_func,
            initial_window=252,
            step_size=21,
        )
        bt_bench.run()
        bench_metrics = bt_bench.compute_metrics()

        logger.info("Benchmark (inverse-vol) metrics:")
        for k, v in bench_metrics.items():
            logger.info("  %s: %.4f", k, v)

        bt_bench.equity_curve().to_frame("equity").to_parquet(
            output_dir / "backtest_benchmark_equity.parquet"
        )
        pd.Series(bench_metrics).to_frame("value").to_parquet(
            output_dir / "backtest_benchmark_metrics.parquet"
        )

        # --- Strategy B: Lead-lag signal strategy ---
        # Pre-compute the TE matrix for pair selection (already available)
        _te_lag1 = te_matrices.get(1, list(te_matrices.values())[0])

        # Identify tradable follower assets (equities only — not FX/credit/macro
        # which react before you can trade)
        EQUITY_ASSETS = {
            "SPX", "NDX", "RTY", "XLB", "XLC", "XLE", "XLF", "XLI",
            "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        }

        # Build the top lead-lag pairs: leader → equity follower
        _ll_pairs = []
        for src in _te_lag1.index:
            for tgt in _te_lag1.columns:
                if src != tgt and tgt in EQUITY_ASSETS:
                    _ll_pairs.append((src, tgt, float(_te_lag1.loc[src, tgt])))
        _ll_pairs.sort(key=lambda x: x[2], reverse=True)
        _ll_pairs = _ll_pairs[:30]  # top 30 leader→equity pairs

        logger.info("Lead-lag pairs for backtest (%d):", len(_ll_pairs))
        for src, tgt, te_val in _ll_pairs[:10]:
            logger.info("  %s → %s : TE=%.4f", src, tgt, te_val)

        # TE weights for aggregation (normalized per follower)
        _te_weights = {}
        for src, tgt, te_val in _ll_pairs:
            _te_weights.setdefault(tgt, []).append((src, te_val))

        def leadlag_signal_func(train_returns: pd.DataFrame) -> dict:
            """Lead-lag signal: use yesterday's leader returns to predict
            follower direction today. Size by lagged beta × TE weight."""
            assets = train_returns.columns.tolist()
            n = len(train_returns)

            # Need at least 60 days to estimate correlations
            if n < 60:
                return {a: 0.0 for a in assets}

            # Yesterday's leader returns (last row of training data)
            leader_rets = train_returns.iloc[-1]

            # Compute trailing lagged correlations and vols (lookback = min(252, n-1))
            lb = min(252, n - 1)
            leader_slice = train_returns.iloc[-(lb + 1):-1]  # day t
            follower_slice = train_returns.iloc[-lb:]         # day t+1
            stds = train_returns.iloc[-lb:].std()

            # For each follower, aggregate predicted return across its leaders
            predictions = {}
            for follower, leader_list in _te_weights.items():
                if follower not in assets:
                    continue
                pred = 0.0
                te_sum = 0.0
                for leader, te_wt in leader_list:
                    if leader not in assets:
                        continue
                    # Lagged correlation: corr(leader_t, follower_{t+1})
                    s = leader_slice[leader].values
                    f = follower_slice[follower].values
                    min_len = min(len(s), len(f))
                    if min_len < 30:
                        continue
                    s, f = s[-min_len:], f[-min_len:]
                    ss, fs = np.std(s), np.std(f)
                    if ss < 1e-10 or fs < 1e-10:
                        continue
                    corr = np.corrcoef(s, f)[0, 1]
                    beta = corr * (fs / ss)

                    # Signal = beta × leader's yesterday return, weighted by TE
                    pred += te_wt * beta * float(leader_rets.get(leader, 0.0))
                    te_sum += te_wt

                if te_sum > 0:
                    predictions[follower] = pred / te_sum  # TE-weighted avg

            if not predictions:
                return {a: 0.0 for a in assets}

            # Convert predictions to weights:
            # Long if predicted > 0, short if predicted < 0
            # Size proportional to |prediction|, then normalize
            raw_weights = pd.Series(predictions)
            gross = raw_weights.abs().sum()
            if gross < 1e-12:
                return {a: 0.0 for a in assets}

            # Target gross leverage = 1.0, cap any single position at 15%
            weights = raw_weights / gross
            weights = weights.clip(-0.15, 0.15)
            # Re-normalize to gross leverage 1.0
            gross2 = weights.abs().sum()
            if gross2 > 1e-12:
                weights = weights / gross2

            return weights.reindex(assets).fillna(0.0).to_dict()

        bt_ll = WalkForwardBacktest(
            returns=bt_panel,
            signal_func=leadlag_signal_func,
            initial_window=252,
            step_size=1,  # daily rebalance — signals change every day
        )
        bt_ll.run()
        ll_metrics = bt_ll.compute_metrics()

        logger.info("Lead-lag signal strategy metrics:")
        for k, v in ll_metrics.items():
            logger.info("  %s: %.4f", k, v)

        bt_ll.equity_curve().to_frame("equity").to_parquet(
            output_dir / "backtest_equity.parquet"
        )
        pd.Series(ll_metrics).to_frame("value").to_parquet(
            output_dir / "backtest_metrics.parquet"
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


def _load_cached_data(cache_dir: str) -> dict:
    """Load all cached parquet files."""
    cache_path = Path(cache_dir)
    result = {}
    for parquet in cache_path.glob("*.parquet"):
        name = parquet.stem
        result[name] = pd.read_parquet(parquet)
    return result


def _top_te_pairs(matrix: pd.DataFrame, n: int = 5) -> list:
    """Extract top-N (source, target, value) from a square matrix."""
    pairs = []
    for src in matrix.index:
        for tgt in matrix.columns:
            if src != tgt:
                pairs.append((src, tgt, float(matrix.loc[src, tgt])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n]


def _select_top_assets(te_matrix: pd.DataFrame, max_assets: int = 8) -> list:
    """Select assets with highest total TE (in + out) for MS-VAR."""
    total_te = te_matrix.sum(axis=0) + te_matrix.sum(axis=1)
    return total_te.nlargest(max_assets).index.tolist()


def _build_regime_features(panel: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Build a low-dimensional feature matrix for HMM regime detection.

    Uses interpretable market regime indicators:
      - SPX realized volatility (21-day rolling std, annualized)
      - Credit stress (HY_OAS returns, or IG_OAS if HY unavailable)
      - Yield curve slope (SPREAD_2s10s returns, if available)
      - VIX level proxy (VIX returns)

    All features are z-scored so the HMM sees comparable scales.
    """
    features = pd.DataFrame(index=panel.index)

    # 1. Realized equity vol (annualized rolling std of SPX returns)
    if "SPX" in panel.columns:
        features["spx_rvol"] = panel["SPX"].rolling(vol_window).std() * np.sqrt(252)

    # 2. Credit stress
    for credit_col in ["HY_OAS", "IG_OAS", "BBB_OAS"]:
        if credit_col in panel.columns:
            features["credit_stress"] = panel[credit_col].rolling(vol_window).mean()
            break

    # 3. Yield curve slope
    if "SPREAD_2s10s" in panel.columns:
        features["curve_slope"] = panel["SPREAD_2s10s"].rolling(vol_window).mean()

    # 4. VIX level proxy
    if "VIX" in panel.columns:
        features["vix_level"] = panel["VIX"].rolling(vol_window).mean()

    # Drop warm-up NaNs, z-score each feature
    features = features.dropna()
    if features.empty:
        return features

    for col in features.columns:
        mu, sigma = features[col].mean(), features[col].std()
        if sigma > 0:
            features[col] = (features[col] - mu) / sigma

    return features


if __name__ == "__main__":
    main()
