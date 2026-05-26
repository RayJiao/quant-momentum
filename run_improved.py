import sys
sys.path.insert(0, "src")

from utils import (
    load_prices, download_prices,
    load_sp500_prices, download_sp500_prices,
    DATA_DIR,
)
from factor import compute_momentum, build_long_short_weights, sector_neutralize
from backtest import run_backtest, compare_strategies


def main():
    # ── 1. Original strategy: top-50 universe, no sector neutralization ──────
    print("\n" + "─" * 60)
    print("STEP 1 — Original strategy (top-50, no sector neutral)")
    print("─" * 60)
    if (DATA_DIR / "prices.parquet").exists():
        prices_orig = load_prices()
        print(f"Loaded cached top-50 prices: {prices_orig.shape}")
    else:
        prices_orig = download_prices()

    mom_orig   = compute_momentum(prices_orig)
    w_orig     = build_long_short_weights(mom_orig)
    ret_orig, stats_orig = run_backtest(w_orig, prices_orig, label="Original (Top-50)")

    # ── 2. Improved strategy: full S&P 500 + sector neutralization ───────────
    print("\n" + "─" * 60)
    print("STEP 2 — Improved strategy (S&P 500, sector-neutral)")
    print("─" * 60)
    if (DATA_DIR / "prices_sp500.parquet").exists():
        prices_sp500, sectors = load_sp500_prices()
        print(f"Loaded cached S&P 500 prices: {prices_sp500.shape}")
        print(f"Sectors: {sectors.nunique()} GICS sectors across {len(sectors)} tickers")
    else:
        prices_sp500, sectors = download_sp500_prices()

    mom_raw    = compute_momentum(prices_sp500)
    mom_neut   = sector_neutralize(mom_raw, sectors)
    w_improved = build_long_short_weights(mom_neut)
    ret_improved, stats_improved = run_backtest(
        w_improved, prices_sp500, label="Improved (S&P500 + Sector-Neutral)"
    )

    # ── 3. Side-by-side comparison ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3 — Comparison")
    print("─" * 60)
    compare_strategies({
        "Original (Top-50)":               (ret_orig,     stats_orig),
        "Improved (S&P500 + Sector-Neut)": (ret_improved, stats_improved),
    })


if __name__ == "__main__":
    main()
