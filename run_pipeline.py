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
    parser.add_argument(
        "--tc-bps", type=float, default=1.0,
        help="Round-trip transaction cost in basis points (per 100%% turnover).",
    )
    parser.add_argument(
        "--execution-lag", type=int, default=1,
        help="Bars between signal and execution (1 = next-close, 0 = same-close).",
    )
    parser.add_argument(
        "--te-refresh-freq", type=int, default=63,
        help="Bars between walk-forward TE refreshes (smaller = fresher, slower).",
    )
    parser.add_argument(
        "--te-lookback", type=int, default=750,
        help="Size of the in-loop TE training slice used at each refresh.",
    )
    parser.add_argument(
        "--tc-sweep", nargs="+", type=float, default=None,
        help="After main backtest, re-run over these TC values (bps) and "
             "write backtest_tc_sensitivity.parquet for the dashboard.",
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
            tc_bps=args.tc_bps,
            execution_lag=args.execution_lag,
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
        # Identify tradable follower assets (equities only — not FX/credit/macro
        # which react before you can trade)
        EQUITY_ASSETS = {
            "SPX", "NDX", "RTY", "XLB", "XLC", "XLE", "XLF", "XLI",
            "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        }

        # Closure-based state for walk-forward TE refresh. The signal function
        # recomputes TE leader→follower pairs every `te_refresh_freq` bars
        # using ONLY the current training slice — no information from dates
        # at or after the rebalance point leaks into pair selection.
        from discovery.transfer_entropy import compute_te_matrix

        te_state = {
            "last_refresh_n": -10**9,
            "te_weights": {},
            "refresh_count": 0,
        }

        def _refresh_te_weights(train_returns: pd.DataFrame) -> dict:
            lookback = min(args.te_lookback, len(train_returns))
            te_slice = train_returns.iloc[-lookback:]
            # Restrict TE computation to (all leaders) × (equity followers)
            # by slicing the returns dict to only the equity target columns
            # during matrix assembly. compute_te_matrix computes full NxN but
            # we only read rows with tgt in EQUITY_ASSETS, so the cost is
            # already close to optimal for our use case — we keep the call
            # simple here for readability.
            rdict = {c: te_slice[c].values for c in te_slice.columns}
            te_mats = compute_te_matrix(
                rdict, lags=[1], k=settings.te_k_neighbors,
            )
            te_lag1_local = te_mats[1]

            pairs = []
            for src in te_lag1_local.index:
                for tgt in te_lag1_local.columns:
                    if src != tgt and tgt in EQUITY_ASSETS:
                        pairs.append((src, tgt, float(te_lag1_local.loc[src, tgt])))
            pairs.sort(key=lambda x: x[2], reverse=True)
            pairs = pairs[:30]  # top 30 leader→equity pairs

            weights_map: dict = {}
            for src, tgt, te_val in pairs:
                weights_map.setdefault(tgt, []).append((src, te_val))
            return weights_map

        def leadlag_signal_func(train_returns: pd.DataFrame) -> dict:
            """Walk-forward lead-lag signal.

            Uses only information in `train_returns` (which the backtest
            harness guarantees is strictly prior to the current rebalance).
            TE pair selection is refreshed every `--te-refresh-freq` bars;
            lagged beta is re-estimated every call from the training slice.
            """
            assets = train_returns.columns.tolist()
            n = len(train_returns)

            # Need at least 252 days before we can estimate anything useful
            if n < 252:
                return {a: 0.0 for a in assets}

            # Refresh the TE pair map on a slow cadence
            if n - te_state["last_refresh_n"] >= args.te_refresh_freq:
                te_state["te_weights"] = _refresh_te_weights(train_returns)
                te_state["last_refresh_n"] = n
                te_state["refresh_count"] += 1
                logger.info(
                    "TE refresh #%d at n=%d (%d follower groups, %d total pairs)",
                    te_state["refresh_count"], n,
                    len(te_state["te_weights"]),
                    sum(len(v) for v in te_state["te_weights"].values()),
                )

            te_weights_live = te_state["te_weights"]
            if not te_weights_live:
                return {a: 0.0 for a in assets}

            # Most recent leader returns (last row of training data — this is
            # close(t-1), known at decision time)
            leader_rets = train_returns.iloc[-1]

            # Estimate trailing lagged beta from a 252-bar window inside the
            # training slice. leader[k] pairs with follower[k+1]; both slices
            # live strictly inside train_returns, so no lookahead.
            lb = min(252, n - 1)
            leader_slice = train_returns.iloc[-(lb + 1):-1]
            follower_slice = train_returns.iloc[-lb:]

            predictions = {}
            for follower, leader_list in te_weights_live.items():
                if follower not in assets:
                    continue
                pred = 0.0
                te_sum = 0.0
                for leader, te_wt in leader_list:
                    if leader not in assets:
                        continue
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
                    pred += te_wt * beta * float(leader_rets.get(leader, 0.0))
                    te_sum += te_wt

                if te_sum > 0:
                    predictions[follower] = pred / te_sum

            if not predictions:
                return {a: 0.0 for a in assets}

            raw_weights = pd.Series(predictions)
            gross = raw_weights.abs().sum()
            if gross < 1e-12:
                return {a: 0.0 for a in assets}

            weights = raw_weights / gross
            weights = weights.clip(-0.15, 0.15)
            gross2 = weights.abs().sum()
            if gross2 > 1e-12:
                weights = weights / gross2

            return weights.reindex(assets).fillna(0.0).to_dict()

        bt_ll = WalkForwardBacktest(
            returns=bt_panel,
            signal_func=leadlag_signal_func,
            initial_window=252,
            step_size=1,  # daily rebalance — signals change every day
            tc_bps=args.tc_bps,
            execution_lag=args.execution_lag,
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

        if args.tc_sweep:
            logger.info("TC sensitivity sweep over %s bps", args.tc_sweep)
            sweep_rows = []
            for tc in args.tc_sweep:
                bt_sweep = WalkForwardBacktest(
                    returns=bt_panel,
                    signal_func=leadlag_signal_func,
                    initial_window=252,
                    step_size=1,
                    tc_bps=float(tc),
                    execution_lag=args.execution_lag,
                )
                bt_sweep.run()
                m = bt_sweep.compute_metrics()
                sweep_rows.append({
                    "tc_bps": float(tc),
                    "sharpe": float(m.get("sharpe", 0.0)),
                    "total_return": float(m.get("total_return", 0.0)),
                    "max_drawdown": float(m.get("max_drawdown", 0.0)),
                })
            pd.DataFrame(sweep_rows).to_parquet(
                output_dir / "backtest_tc_sensitivity.parquet"
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
