# Momentum Factor Research

Systematic implementation of the 12-1 month price momentum factor across large-cap (S&P 500) and small-cap (S&P 600 / Russell 2000 proxy) US equities. Covers signal construction, sector neutralization, volatility adjustment, and long-short portfolio backtesting over the 2015–2025 period.

---

## Results

Four strategies were backtested from January 2015 to December 2025 using daily OHLCV data from Yahoo Finance. Alpha is estimated via OLS regression against the universe benchmark (SPY for S&P 500 strategies, IWM for small-cap).

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Alpha (ann.) | Alpha t-stat |
|---|---|---|---|---|---|---|
| S&P 500 Top-50 · no sector neut (baseline) | +1.70% | 20.73% | +0.082 | -42.89% | -0.07% | -0.011 |
| S&P 500 Full · sector-neutral (v1) | -0.28% | 14.20% | -0.020 | -39.69% | +0.58% | +0.135 |
| S&P 600 Raw · sector-neutral · 10 bps (v2) | -5.17% | 14.65% | -0.353 | -56.40% | -3.95% | -0.903 |
| S&P 600 Vol-Adj · sector-neutral · 10 bps (v2) | -1.38% | 12.97% | **-0.106** | -40.84% | -0.50% | -0.127 |

Chart: `results/momentum_comparison_full.png`

---

## Conclusion

The 12-1 momentum factor produced weak or negative returns across all four strategy variants over 2015–2025. No alpha t-statistic exceeded the conventional |2.0| significance threshold. Two structural breaks drove this:

1. **2020 COVID crash** — momentum portfolios held the pre-crash winners (energy, value) into the crash and missed the sharp tech-led recovery; the factor suffered its worst recorded drawdown.
2. **2022 growth-to-value rotation** — elevated interest rates triggered a rapid reversal of the 2020–2021 momentum winners (high-multiple tech/growth), compressing factor returns further.

The volatility-adjusted variant (v2 Vol-Adj) meaningfully improved over the raw small-cap strategy: Sharpe -0.106 vs -0.353, Max Drawdown -41% vs -56%, and annualized volatility reduced by 170 bps. This confirms that down-weighting high-volatility stocks before ranking helps in turbulent small-cap universes, but is not sufficient to produce positive alpha in a hostile macro regime for momentum.

---

## Methodology

### Signal Definition

**12-1 month momentum** — the cumulative price return from month *t-12* to month *t-1*, skipping the most recent month to avoid the short-term reversal effect documented by Jegadeesh (1990):

```
MOM(t) = Price(t-1) / Price(t-13) - 1
```

Prices are sampled at calendar month-end. All returns use split- and dividend-adjusted closing prices.

### Sector Neutralization

At each month-end, the raw factor is residualized on GICS sector dummies via OLS — equivalent to subtracting the within-sector mean from each stock's factor value. This removes passive sector-level bets so the portfolio reflects pure stock-selection skill:

```
MOM_neutral(i,t) = MOM(i,t) - mean(MOM(sector_i, t))
```

Sectors: 11 GICS sectors from Wikipedia (S&P 500) and S&P 600 constituent pages.

### Volatility Adjustment (v2)

The sector-neutralized factor is divided by each stock's trailing 6-month annualized return volatility, computed from monthly returns. This produces a Sharpe-ratio-style signal that down-weights high-volatility small-cap stocks whose momentum may not be persistent:

```
MOM_vol_adj(i,t) = MOM_neutral(i,t) / Vol_6M(i,t)   [clipped to ±10]
```

### Portfolio Construction

Each month, stocks are sorted into quintiles on the (neutralized) factor. The portfolio goes long the top quintile and short the bottom quintile, equal-weighted within each leg. Weights are rebalanced at every month-end.

### Liquidity Filter (v2 only)

Small-cap stocks with a trailing 63-day average daily volume below 100,000 shares are excluded from the factor universe each month. This avoids microstructure noise and unrealistic fill assumptions.

### Transaction Costs

S&P 500 strategies: 0 bps (large-cap, negligible spread).  
S&P 600 / Russell 2000 strategies: 10 bps one-way, applied on the first day of each holding period based on realized monthly turnover.

---

## File Structure

```
quant-momentum/
├── src/
│   ├── utils.py          # data download (yfinance) and parquet caching
│   ├── factor.py         # momentum signal construction and sector neutralization
│   └── backtest.py       # portfolio return engine, stats, alpha t-stat, charts
│
├── run.py                # v0: S&P 500 top-50, raw momentum
├── run_improved.py       # v1: full S&P 500 + sector neutralization
├── run_v2.py             # v2: S&P 600 small-cap, vol-adjusted + sector neutral
├── run_comparison.py     # unified 4-strategy comparison (entry point)
├── visualize_nn.py       # bonus: live neural network training visualization
│
├── data/
│   ├── prices.parquet         # S&P 500 top-50 daily close (2015-2025)
│   ├── prices_sp500.parquet   # full S&P 500 daily close
│   ├── sectors_sp500.parquet  # GICS sector labels
│   ├── prices_r2k.parquet     # S&P 600 small-cap daily close
│   ├── volume_r2k.parquet     # S&P 600 daily volume (for liquidity filter)
│   └── sectors_r2k.parquet    # GICS sector labels
│
└── results/
    ├── momentum_comparison_full.png   # cumulative return + drawdown chart
    └── comparison_report.txt          # full metrics table
```

---

## Quickstart

```bash
pip install yfinance pandas numpy matplotlib lxml

# Run the full 4-strategy comparison (downloads data on first run, ~5 min)
python run_comparison.py

# Run only the small-cap v2 pipeline
python run_v2.py

# Interactive neural network training visualization (bonus)
python visualize_nn.py
```

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| Python | 3.11 | runtime |
| pandas | latest | data wrangling, resampling, parquet I/O |
| numpy | latest | factor math, OLS backprop |
| yfinance | latest | OHLCV data download |
| matplotlib | latest | cumulative return, drawdown, weight heatmap charts |
| lxml | latest | Wikipedia HTML table parsing |

---

## Known Limitations

- **Survivorship bias** — constituent lists are scraped from Wikipedia at a single point in time (current membership). Stocks that were added or removed during 2015–2025 may introduce look-ahead bias in the universe.
- **Point-in-time sectors** — GICS sector assignments reflect current classifications; historical reclassifications are not tracked.
- **Transaction costs** — 10 bps is a conservative estimate; actual small-cap bid-ask spreads vary widely by stock and period.
- **Single signal** — pure price momentum only; no fundamental or earnings-revision signals.
- **Short-side feasibility** — the short leg assumes all stocks are borrowable at zero cost, which is unrealistic for small-cap names.

## Next Steps

- **Add earnings momentum** (SUE signal) alongside price momentum for a composite factor
- **Conditional momentum** — scale factor exposure by VIX regime or recent market return to avoid momentum crashes
- **Point-in-time universe** — use historical index snapshots (e.g., CRSP) to eliminate survivorship bias
- **Risk-model neutralization** — neutralize to a full Barra-style factor model (size, value, beta) instead of sectors only
- **Transaction cost model** — use bid-ask spread estimates from TAQ data instead of a flat bps assumption
