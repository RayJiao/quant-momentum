"""
Full cross-universe momentum factor comparison (2015-2025)

Strategy                            Universe     Sector-Neutral  Costs   Benchmark
────────────────────────────────────────────────────────────────────────────────────
S&P 500 Top-50  (baseline)          SP500 top-50  No              0 bps   SPY
S&P 500 Full    (v1)                SP500 full    Yes             0 bps   SPY
R2K Raw         (v2 baseline)       Russell 2000  Yes            10 bps   IWM
R2K Vol-Adj     (v2 improved)       Russell 2000  Yes            10 bps   IWM

Outputs saved to results/:
  momentum_comparison_full.png   — cumulative return + drawdown chart
  comparison_report.txt          — Sharpe / alpha t-stat / MDD table
"""

import yfinance as yf
import pandas as pd
import numpy as np

from src.utils import (
    DATA_DIR, RESULTS_DIR,
    load_prices, download_prices,
    load_sp500_prices, download_sp500_prices,
    load_russell2000_prices, download_russell2000_prices,
)
from src.factor import (
    compute_momentum,
    compute_vol_adjusted_momentum,
    apply_liquidity_filter,
    sector_neutralize,
    build_long_short_weights,
)
from src.backtest import (
    portfolio_returns,
    portfolio_returns_with_costs,
    compute_stats,
    alpha_tstat,
    compare_strategies,
    _print_stats,
)

START, END = "2015-01-01", "2025-12-31"

# ── helpers ──────────────────────────────────────────────────────────────────

def fetch_benchmark(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    return raw["Close"].squeeze().pct_change().dropna()


def run_strategy(
    label: str,
    factor: pd.DataFrame,
    sectors: pd.Series,
    daily_prices: pd.DataFrame,
    benchmark: pd.Series,
    spread_bps: float = 0.0,
) -> tuple[pd.Series, dict]:
    """Sector-neutralise → quintile weights → returns → stats."""
    factor_neutral = sector_neutralize(factor, sectors)
    weights        = build_long_short_weights(factor_neutral)

    if spread_bps > 0:
        rets = portfolio_returns_with_costs(weights, daily_prices, spread_bps=spread_bps)
    else:
        rets = portfolio_returns(weights, daily_prices)

    stats = compute_stats(rets)
    alpha_ann, t_stat = alpha_tstat(rets, benchmark)
    stats["Alpha (ann.)"] = alpha_ann
    stats["Alpha t-stat"] = t_stat

    _print_stats(label, stats)
    return rets, stats


def save_text_report(results: dict, path) -> None:
    labels = list(results.keys())
    metric_keys = list(next(iter(results.values()))[1].keys())
    col_w = 22

    lines = [
        "=" * (20 + col_w * len(labels)),
        "  MOMENTUM FACTOR COMPARISON REPORT  (2015-2025)",
        "  Universe comparison: S&P 500 vs Russell 2000",
        "=" * (20 + col_w * len(labels)),
        "",
        f"{'Metric':<20}" + "".join(f"{lb:>{col_w}}" for lb in labels),
        "-" * (20 + col_w * len(labels)),
    ]
    for k in metric_keys:
        row = f"  {k:<18}"
        for lb in labels:
            v = results[lb][1][k]
            if k in ("Sharpe Ratio", "Alpha t-stat"):
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{v:>{col_w - 1}.2%} "
        lines.append(row)
    lines.append("=" * (20 + col_w * len(labels)))
    lines.append("")
    lines.append("Notes:")
    lines.append("  • Sector neutralization: OLS residual on GICS sector dummies (monthly)")
    lines.append("  • Transaction costs applied at month-end rebalance (one-way)")
    lines.append("  • R2K liquidity filter: trailing 63-day avg volume ≥ 100k shares")
    lines.append("  • Vol-adj factor: 12-1 momentum / trailing 6M annualized vol")
    lines.append("  • Benchmarks: SPY for S&P 500 strategies, IWM for R2K strategies")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nText report saved → {path}")


# ── pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. S&P 500 Top-50 ────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}\nSTEP 1 — S&P 500 Top-50  (baseline, no sector neutral)\n{sep}")
    if (DATA_DIR / "prices.parquet").exists():
        prices_top50 = load_prices()
        print(f"Loaded top-50: {prices_top50.shape}")
    else:
        prices_top50 = download_prices(start=START, end=END)

    # No real sector data for the top-50 → treat as one sector (within-sector
    # demeaning becomes a no-op since every stock is in the same bucket).
    sectors_top50 = pd.Series("All", index=prices_top50.columns, name="sector")

    # ── 2. S&P 500 Full ──────────────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 2 — S&P 500 Full + Sector Neutral  (v1)\n{sep}")
    if (DATA_DIR / "prices_sp500.parquet").exists():
        prices_sp500, sectors_sp500 = load_sp500_prices()
        print(f"Loaded S&P 500: {prices_sp500.shape}, "
              f"{sectors_sp500.nunique()} GICS sectors")
    else:
        prices_sp500, sectors_sp500 = download_sp500_prices(start=START, end=END)

    # ── 3. Russell 2000 ──────────────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 3 — Russell 2000  (v2)\n{sep}")
    if (DATA_DIR / "prices_r2k.parquet").exists():
        prices_r2k, volume_r2k, sectors_r2k = load_russell2000_prices()
        print(f"Loaded R2K: {prices_r2k.shape}, "
              f"{sectors_r2k.nunique()} sectors")
    else:
        print("Downloading R2K universe — this takes ~10-15 min (one-time only)…")
        prices_r2k, volume_r2k, sectors_r2k = download_russell2000_prices(
            start=START, end=END
        )

    close_liquid = apply_liquidity_filter(
        prices_r2k, volume_r2k, min_avg_volume=100_000
    )
    n_liquid_avg = close_liquid.notna().sum(axis=1).mean()
    print(f"Avg liquid stocks per month: {n_liquid_avg:.0f} / {prices_r2k.shape[1]}")

    # ── 4. Benchmarks ────────────────────────────────────────────────────────
    print(f"\n{sep}\nDownloading benchmarks (SPY, IWM)…\n{sep}")
    spy = fetch_benchmark("SPY")
    iwm = fetch_benchmark("IWM")

    # ── 5. Run strategies ────────────────────────────────────────────────────
    print(f"\n{sep}\nRUNNING ALL STRATEGIES\n{sep}")

    results = {}

    results["SP500 Top-50\n(no sector neut)"] = run_strategy(
        "S&P 500 Top-50  (no sector neutral)",
        compute_momentum(prices_top50),
        sectors_top50,
        prices_top50,
        spy,
        spread_bps=0.0,
    )

    results["SP500 Full\n(sector-neutral)"] = run_strategy(
        "S&P 500 Full  (sector-neutral, v1)",
        compute_momentum(prices_sp500),
        sectors_sp500,
        prices_sp500,
        spy,
        spread_bps=0.0,
    )

    results["R2K Raw\n(sector-neut, 10bps)"] = run_strategy(
        "R2K Raw  (sector-neutral, 10 bps, v2)",
        compute_momentum(close_liquid),
        sectors_r2k,
        prices_r2k,
        iwm,
        spread_bps=10.0,
    )

    results["R2K Vol-Adj\n(sector-neut, 10bps)"] = run_strategy(
        "R2K Vol-Adj  (sector-neutral, 10 bps, v2)",
        compute_vol_adjusted_momentum(close_liquid),
        sectors_r2k,
        prices_r2k,
        iwm,
        spread_bps=10.0,
    )

    # ── 6. Outputs ───────────────────────────────────────────────────────────
    compare_strategies(
        results,
        title="Momentum Factor: S&P 500 vs Russell 2000  (2015–2025)",
        output_name="momentum_comparison_full.png",
    )

    save_text_report(results, RESULTS_DIR / "comparison_report.txt")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
