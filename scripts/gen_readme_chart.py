"""Generate backtest equity curve chart for README."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

output_dir = Path("data/outputs")
eq = pd.read_parquet(output_dir / "backtest_equity.parquet")["equity"]
bench = pd.read_parquet(output_dir / "backtest_benchmark_equity.parquet")["equity"]
metrics = pd.read_parquet(output_dir / "backtest_metrics.parquet")["value"].to_dict()
bench_metrics = pd.read_parquet(output_dir / "backtest_benchmark_metrics.parquet")["value"].to_dict()

# Drawdowns
dd = (eq - eq.cummax()) / eq.cummax()
bench_dd = (bench - bench.cummax()) / bench.cummax()

# Dark theme
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 5.5), gridspec_kw={"height_ratios": [2.5, 1]},
    sharex=True,
)
fig.patch.set_facecolor("#0d1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

# Equity curve
ax1.plot(eq.index, eq.values, color="#26a69a", linewidth=1.8, label="Lead-Lag Signals")
ax1.plot(bench.index, bench.values, color="#7e57c2", linewidth=1.8, linestyle=":", label="Benchmark (Inv-Vol)")
ax1.set_ylabel("Growth of $1", color="#8b949e", fontsize=10)
ax1.set_title("Walk-Forward Backtest: Lead-Lag Signals vs Passive Benchmark",
              color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.6, edgecolor="#30363d")
ax1.grid(axis="y", color="#21262d", linewidth=0.5)

# Metric annotations — bottom-right to avoid legend overlap
s = metrics
b = bench_metrics
ax1.text(0.99, 0.05,
         f"Strategy:   Sharpe {s['sharpe']:.2f}  |  Return {s['total_return']:.1%}  |  Max DD {s['max_drawdown']:.1%}\n"
         f"Benchmark:  Sharpe {b['sharpe']:.2f}  |  Return {b['total_return']:.1%}  |  Max DD {b['max_drawdown']:.1%}",
         transform=ax1.transAxes, fontsize=9, color="#8b949e", family="monospace",
         verticalalignment="bottom", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#30363d", alpha=0.9))

# Drawdown
ax2.fill_between(dd.index, dd.values, 0, color="#e53935", alpha=0.5, label="Signal DD")
ax2.fill_between(bench_dd.index, bench_dd.values, 0, color="#ff9800", alpha=0.35, label="Bench DD")
ax2.set_ylabel("Drawdown", color="#8b949e", fontsize=10)
ax2.legend(loc="lower left", fontsize=8, framealpha=0.3)
ax2.grid(axis="y", color="#21262d", linewidth=0.5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=0, ha="center")

fig.tight_layout(h_pad=0.5)
Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/backtest_equity_curve.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none")
print("Saved docs/backtest_equity_curve.png")
plt.close()
