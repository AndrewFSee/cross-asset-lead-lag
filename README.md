# Cross-Asset Stochastic Lead-Lag Discovery Engine

[![CI](https://github.com/AndrewFSee/cross-asset-lead-lag/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewFSee/cross-asset-lead-lag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready system for discovering and exploiting lead-lag relationships across
asset classes (equities, fixed income, credit, commodities, FX, crypto) using
information-theoretic measures and neural Granger causality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  Yahoo Finance (equities, FX, commodities, crypto, vol indices)     │
│  FRED (rates, credit spreads, macro indicators)                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INGESTION & PREPROCESSING                     │
│  data/ingestion.py     → fetch_all_data()                          │
│  data/preprocessing.py → stationarity, winsorize, align calendars  │
│  data/returns.py       → build_returns_panel()                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DISCOVERY ENGINE                                 │
│  discovery/transfer_entropy.py  → KSG-based TE estimator           │
│  discovery/neural_granger.py    → ComponentLSTM + ablation         │
│  discovery/time_lagged_mi.py    → KSG mutual information           │
│  discovery/significance.py      → bootstrap + surrogate tests      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODELS                                        │
│  models/ms_var.py         → Markov-Switching VAR (EM algorithm)    │
│  models/lasso_var.py      → Lasso-penalized VAR (BIC selection)    │
│  models/regime_detector.py → Gaussian HMM regime classification    │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SIGNALS & BACKTEST                            │
│  signals/generator.py  → LeadSignal + Bayesian model averaging     │
│  signals/portfolio.py  → risk parity, Kelly sizing, constraints    │
│  signals/backtest.py   → WalkForwardBacktest + metrics             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT & MONITORING                               │
│  agent/monitor.py   → structural break detection (z-score)         │
│  agent/narrator.py  → LLM narrative generation (OpenAI)            │
│  agent/alerts.py    → Slack + email alerting                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DASHBOARD                                     │
│  dashboard/app.py              → Streamlit entry point             │
│  dashboard/pages/network_graph → TE network visualization          │
│  dashboard/pages/regime_panel  → regime probabilities              │
│  dashboard/pages/signal_monitor → live signals                     │
│  dashboard/pages/backtest_results → performance metrics            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/AndrewFSee/cross-asset-lead-lag.git
cd cross-asset-lead-lag

# 2. Install the package and development dependencies
pip install -e ".[dev]"

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your FRED_API_KEY (free at https://fred.stlouisfed.org/docs/api/api_key.html)

# 4. Fetch market data (requires FRED_API_KEY)
make data

# 5. Run the notebooks
jupyter notebook notebooks/

# 6. Launch the interactive dashboard
make dashboard
```

---

## Data Sources

| Asset Class | Source | Series Count | Description |
|-------------|--------|--------------|-------------|
| Equity Indices | Yahoo Finance | 3 | SPX, NDX, RTY |
| Equity Sectors | Yahoo Finance | 11 | XLF, XLE, XLI, XLB, XLK, XLU, XLP, XLV, XLY, XLRE, XLC |
| Rates | FRED | 8 | 2Y/5Y/10Y/30Y Treasuries, 2s10s, real yields, breakevens |
| Credit | FRED | 4 | HY OAS, IG OAS, BBB OAS, CCC OAS |
| Commodities | Yahoo Finance | 5 | WTI, Gold, Copper, Natural Gas, Silver |
| FX | Yahoo Finance | 5 | DXY, EUR/USD, USD/JPY, AUD/USD, USD/CNH |
| Volatility | Yahoo Finance / FRED | 3 | VIX, MOVE, OVX |
| Crypto | Yahoo Finance | 2 | BTC-USD, ETH-USD |
| Macro | FRED | 5 | ISM PMI, Initial Claims, UMich Sentiment, NFCI, Fed Funds |

All data sources are **free** (Yahoo Finance + FRED). Get a FRED API key at
https://fred.stlouisfed.org/docs/api/api_key.html.

---

## Known Lead-Lag Relationships to Validate Against

The following economically motivated lead-lag relationships serve as ground-truth
validation for the discovery engine:

| Leader | Follower | Direction | Rationale |
|--------|----------|-----------|-----------|
| COPPER | XLI | + | Copper demand anticipates industrial activity |
| HY_OAS | SPX | - | HY spread widening precedes equity sell-offs |
| VIX | HY_OAS | + | Equity volatility spikes precede credit stress |
| INITIAL_CLAIMS | SPX | - | Rising jobless claims signal economic weakness |
| SPREAD_2s10s | XLF | + | Steeper yield curve precedes bank profitability |
| DXY | GOLD | - | Dollar strength inversely leads gold |
| WTI | XLE | + | Oil prices lead energy sector returns |
| BTC | ETH | + | Bitcoin moves tend to precede Ethereum moves |

---

## Module Documentation

### `data/`
- **`ingestion.py`**: Downloads market data from Yahoo Finance (yfinance) and macro data from FRED. Caches results as parquet files in `data/cache/`. Handles individual series failures gracefully.
- **`preprocessing.py`**: ADF + KPSS stationarity tests, Winsorization at ±n_sigma, forward-fill missing data, calendar alignment (crypto 7d vs equities 5d).
- **`returns.py`**: Builds a unified returns panel. Applies log-returns to prices, first-differences to yields/spreads, forward-fills macro data before differencing.

### `discovery/`
- **`transfer_entropy.py`**: KSG estimator (Kraskov et al. 2004) for transfer entropy. Builds joint/marginal embeddings with configurable history length and lag.
- **`neural_granger.py`**: ComponentLSTM (Tank et al. 2021) with per-asset embeddings. Ablation testing reveals Granger causal influence.
- **`time_lagged_mi.py`**: KSG mutual information at multiple lags for symmetric pairwise dependency measurement.
- **`significance.py`**: Block bootstrap and phase-randomization surrogate tests for statistical significance.

### `models/`
- **`ms_var.py`**: Markov-Switching VAR estimated via EM with Hamilton filter + Kim smoother. Regime-conditional VAR coefficients capture how lead-lag relationships change across market states.
- **`lasso_var.py`**: High-dimensional VAR with Lasso regularization. BIC/AIC automatic lambda selection. `get_lead_lag_matrix()` reveals sparse causal structure.
- **`regime_detector.py`**: Standalone Gaussian HMM for regime classification from any feature set.

### `signals/`
- **`generator.py`**: `LeadSignal` dataclass, signal generation from TE + MS-VAR, Bayesian model averaging across multiple leaders per follower.
- **`portfolio.py`**: Risk parity (equal risk contribution via SLSQP), Kelly sizing (regularized matrix inversion), position/sector constraints.
- **`backtest.py`**: Walk-forward backtest with expanding/rolling windows. Metrics: Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover.

### `agent/`
- **`monitor.py`**: Rolling TE monitoring with z-score spike/decay detection and regime transition alerts.
- **`narrator.py`**: LLM narrative generation via OpenAI API. Gracefully degrades if API key is missing.
- **`alerts.py`**: Slack webhook and SMTP email alerting.

### `dashboard/`
- **`app.py`**: Streamlit multi-page app with sidebar navigation.
- **`pages/`**: Network graph (NetworkX + Plotly), regime panel, signal monitor, backtest results.
- **`components/charts.py`**: Reusable Plotly chart helpers.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Load data, check coverage, visualize correlations, stationarity |
| `02_transfer_entropy.ipynb` | Compute TE matrix, heatmap, identify strongest leads |
| `03_neural_granger.ipynb` | Train neural GC model, compare with linear TE |
| `04_regime_switching_var.ipynb` | Fit MS-VAR, visualize regime probabilities |
| `05_signal_backtest.ipynb` | Generate signals, walk-forward backtest, analyze results |
| `06_dashboard_prototype.ipynb` | Prototype dashboard components inline |

---

## Academic References

- Schreiber, T. (2000). "Measuring Information Transfer." *Physical Review Letters*, 85(2).
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). "Estimating Mutual Information." *Physical Review E*, 69(6).
- Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. (2021). "Neural Granger Causality." *IEEE TPAMI*.
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2).
- Billio, M., Getmansky, M., Lo, A.W., & Pelizzon, L. (2012). "Econometric Measures of Connectedness and Systemic Risk." *Journal of Financial Economics*, 104(3).

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run: `make lint && make test`
5. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.
