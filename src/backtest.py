import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def portfolio_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    weights: monthly DataFrame (index = month-end), columns = tickers
    prices:  daily DataFrame, columns = tickers
    Returns daily portfolio returns, rebalanced at each month-end.
    """
    daily_ret = prices.pct_change().dropna(how="all")
    port_rets = []
    months = weights.index

    for i, month_end in enumerate(months[:-1]):
        next_month_end = months[i + 1]
        w = weights.loc[month_end]
        mask = (daily_ret.index > month_end) & (daily_ret.index <= next_month_end)
        period = daily_ret.loc[mask, w.index]
        port_rets.append(period.fillna(0).dot(w))

    return pd.concat(port_rets).sort_index()


def compute_stats(returns: pd.Series, freq: int = 252) -> dict:
    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol

    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    return {
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Max Drawdown":    dd.min(),
    }


def alpha_tstat(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """
    OLS: r_port = alpha + beta * r_bench + eps
    Returns (alpha annualized, t-stat of alpha).
    Uses correct SE: s² * (X'X)^{-1}[0,0]  with s² = RSS/(n-2).
    """
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    df.columns = ["port", "bench"]
    n = len(df)
    x = df["bench"].values
    y = df["port"].values
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_daily = beta[0]
    residuals = y - X @ beta
    s2 = (residuals @ residuals) / (n - 2)
    # Var(alpha_hat) = s2 * (X'X)^{-1}[0,0] = s2 * (1/n + xbar^2 / sum((x-xbar)^2))
    se_alpha = np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / np.sum((x - x.mean()) ** 2)))
    return alpha_daily * 252, alpha_daily / se_alpha


def portfolio_returns_with_costs(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    spread_bps: float = 10.0,
) -> pd.Series:
    """
    Like portfolio_returns() but deducts estimated transaction costs.

    Cost = (one-way turnover) × spread_bps / 10_000, applied on the first
    day of each holding period.  One-way turnover = Σ|Δw| / 2.
    """
    daily_ret = prices.pct_change().dropna(how="all")
    port_rets = []
    months = weights.index
    prev_w = pd.Series(0.0, index=weights.columns)

    for i, month_end in enumerate(months[:-1]):
        next_month_end = months[i + 1]
        w = weights.loc[month_end].fillna(0.0)

        turnover = (w - prev_w).abs().sum() / 2.0
        cost = turnover * spread_bps / 10_000.0

        mask = (daily_ret.index > month_end) & (daily_ret.index <= next_month_end)
        period = daily_ret.loc[mask, w.index].fillna(0.0).dot(w)

        if len(period) > 0:
            period = period.copy()
            period.iloc[0] -= cost

        port_rets.append(period)
        prev_w = w

    return pd.concat(port_rets).sort_index()


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame,
                 label: str = "Strategy") -> tuple[pd.Series, dict]:
    RESULTS_DIR.mkdir(exist_ok=True)

    port_ret = portfolio_returns(weights, prices)
    stats = compute_stats(port_ret)

    bench_ret = prices.pct_change().dropna(how="all").reindex(port_ret.index).mean(axis=1)
    alpha_ann, t_stat = alpha_tstat(port_ret, bench_ret)
    stats["Alpha (ann.)"] = alpha_ann
    stats["Alpha t-stat"] = t_stat

    _print_stats(label, stats)
    return port_ret, stats


def _print_stats(label: str, stats: dict) -> None:
    print(f"\n===== {label} =====")
    for k, v in stats.items():
        if k in ("Sharpe Ratio", "Alpha t-stat"):
            print(f"  {k:<22}: {v:+.3f}")
        else:
            print(f"  {k:<22}: {v:+.2%}")


def compare_strategies(
    results: dict[str, tuple[pd.Series, dict]],
    title: str = "12-1 Momentum Factor Comparison  (2015–2025)",
    output_name: str = "momentum_comparison.png",
) -> None:
    """
    results: OrderedDict of {label: (daily_returns, stats_dict)}
    Prints a side-by-side table and saves a comparison chart.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- printed table ---
    labels = list(results.keys())
    metric_keys = list(next(iter(results.values()))[1].keys())
    col_w = 22

    header = f"{'Metric':<{col_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print("\n" + "=" * len(header))
    print("  STRATEGY COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in metric_keys:
        row = f"  {k:<{col_w - 2}}"
        for lb in labels:
            v = results[lb][1][k]
            if k in ("Sharpe Ratio", "Alpha t-stat"):
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{v:>{col_w - 1}.2%} "
        print(row)
    print("=" * len(header))

    # --- comparison chart ---
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1.2]})

    ax_ret, ax_dd = axes

    for idx, (lb, (rets, stats)) in enumerate(results.items()):
        cum = (1 + rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        c = colors[idx % len(colors)]
        sharpe = stats["Sharpe Ratio"]
        mdd = stats["Max Drawdown"]
        ax_ret.plot(cum.index, cum.values, label=f"{lb}  (Sharpe {sharpe:.2f})",
                    color=c, linewidth=1.6)
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.4, color=c, label=f"{lb}  (MDD {mdd:.1%})")

    ax_ret.set_title(title, fontsize=12, fontweight="bold")
    ax_ret.set_ylabel("Cumulative Return")
    ax_ret.legend(fontsize=9)
    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ret.grid(alpha=0.25)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(fontsize=9)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.grid(alpha=0.25)

    plt.tight_layout()
    out = RESULTS_DIR / output_name
    plt.savefig(out, dpi=150)
    print(f"\nComparison chart saved → {out}")
    plt.close()
