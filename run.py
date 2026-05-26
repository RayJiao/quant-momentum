import sys
sys.path.insert(0, "src")

from utils import download_prices, load_prices, DATA_DIR
from factor import compute_momentum, build_long_short_weights
from backtest import run_backtest
from pathlib import Path


def main():
    price_path = DATA_DIR / "prices.parquet"
    if not price_path.exists():
        print("Downloading price data...")
        prices = download_prices()
    else:
        print("Loading cached price data...")
        prices = load_prices()

    print(f"Prices: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\nComputing 12-1 momentum factor...")
    mom = compute_momentum(prices, lookback=12, skip=1)

    print("Building long/short weights (top/bottom quintile)...")
    weights = build_long_short_weights(mom)

    print("Running backtest...")
    port_ret, stats = run_backtest(weights, prices)


if __name__ == "__main__":
    main()
