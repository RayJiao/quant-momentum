"""
v2: Russell 2000 momentum factor pipeline

Improvements over v1 (S&P 500 top-50):
  Universe  : IWM constituents (~2000 small-cap stocks)
  Filter    : liquidity — avg daily volume ≥ 100k shares (trailing 63 days)
  Factor    : volatility-adjusted 12-1 month momentum  (return / 6M ann. vol)
  Costs     : 10 bps one-way transaction cost at each monthly rebalance
  Benchmark : IWM ETF (vs. equal-weight universe used in v1)
"""

from pathlib import Path

import yfinance as yf
import pandas as pd

from src.utils import (
    DATA_DIR, RESULTS_DIR,
    download_russell2000_prices,
    load_russell2000_prices,
)
from src.factor import (
    compute_momentum,
    compute_vol_adjusted_momentum,
    apply_liquidity_filter,
    sector_neutralize,
    build_long_short_weights,
)
from src.backtest import (
    compute_stats,
    alpha_tstat,
    portfolio_returns_with_costs,
    compare_strategies,
    _print_stats,
)

START = "2015-01-01"
END   = "2025-12-31"


def get_iwm_daily_returns() -> pd.Series:
    raw = yf.download("IWM", start=START, end=END, auto_adjust=True, progress=False)
    return raw["Close"].squeeze().pct_change().dropna()


def build_strategy(name: str, factor: pd.DataFrame, sectors: pd.Series,
                   close: pd.DataFrame, spread_bps: float,
                   benchmark: pd.Series) -> tuple[pd.Series, dict]:
    factor_neutral = sector_neutralize(factor, sectors)
    weights = build_long_short_weights(factor_neutral)
    port_ret = portfolio_returns_with_costs(weights, close, spread_bps=spread_bps)
    stats = compute_stats(port_ret)
    alpha_ann, t_stat = alpha_tstat(port_ret, benchmark)
    stats["Alpha (ann.)"] = alpha_ann
    stats["Alpha t-stat"] = t_stat
    _print_stats(name, stats)
    return port_ret, stats


def main() -> None:
    # ── 1. Data ────────────────────────────────────────────────────────────
    price_file = DATA_DIR / "prices_r2k.parquet"
    if price_file.exists():
        print("Loading cached Russell 2000 data …")
        close, volume, sectors = load_russell2000_prices()
    else:
        print("Downloading Russell 2000 universe (one-time, ~10–15 min) …")
        close, volume, sectors = download_russell2000_prices(start=START, end=END)

    print(f"Universe: {close.shape[1]} stocks × {close.shape[0]} trading days\n")

    # ── 2. Liquidity filter ────────────────────────────────────────────────
    # Prices become NaN for stock-months below the volume threshold;
    # those stocks automatically receive 0 weight in the factor.
    close_liquid = apply_liquidity_filter(close, volume, min_avg_volume=100_000)

    # ── 3. IWM benchmark ──────────────────────────────────────────────────
    print("Downloading IWM benchmark …")
    iwm_ret = get_iwm_daily_returns()

    # ── 4. Strategies ─────────────────────────────────────────────────────
    results = {}

    # 4a. Raw 12-1 momentum, no costs  (same logic as v1, just on R2K)
    mom_raw = compute_momentum(close_liquid)
    results["R2K Raw  (0 bps)"] = build_strategy(
        "R2K Raw Momentum (0 bps)", mom_raw, sectors, close,
        spread_bps=0.0, benchmark=iwm_ret,
    )

    # 4b. Raw 12-1 momentum, 10 bps costs
    results["R2K Raw (10 bps)"] = build_strategy(
        "R2K Raw Momentum (10 bps)", mom_raw, sectors, close,
        spread_bps=10.0, benchmark=iwm_ret,
    )

    # 4c. Vol-adjusted momentum, 10 bps costs  (main v2 improvement)
    mom_vol = compute_vol_adjusted_momentum(close_liquid)
    results["R2K Vol-Adj (10 bps)"] = build_strategy(
        "R2K Vol-Adj Momentum (10 bps)", mom_vol, sectors, close,
        spread_bps=10.0, benchmark=iwm_ret,
    )

    # ── 5. Comparison chart ───────────────────────────────────────────────
    compare_strategies(
        results,
        title="12-1 Momentum Factor: Russell 2000  (2015–2025)",
    )

    # Save chart under a v2-specific name
    src = RESULTS_DIR / "momentum_comparison.png"
    dst = RESULTS_DIR / "momentum_comparison_v2.png"
    if src.exists():
        src.rename(dst)
        print(f"Chart saved → {dst}")


if __name__ == "__main__":
    main()
