import pandas as pd
import numpy as np


def compute_momentum(prices: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """12-1 month momentum: cumulative return over [t-12, t-1], skipping last month."""
    monthly = prices.resample("ME").last()
    mom = monthly.shift(skip).pct_change(lookback)
    return mom


def sector_neutralize(factor: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """
    Cross-sectionally residualize the factor on sector dummies each month.
    Equivalent to within-sector demeaning (OLS residuals from sector-dummy regression).

    factor:  monthly DataFrame, index=dates, columns=tickers
    sectors: pd.Series, index=tickers, values=sector strings
    """
    result = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)

    for date, row in factor.iterrows():
        valid = row.dropna()
        if len(valid) < 20:
            continue
        sec = sectors.reindex(valid.index).dropna()
        aligned = valid.reindex(sec.index)
        # Subtract sector mean from each stock (≡ OLS residual on sector dummies)
        sector_mean = aligned.groupby(sec).transform("mean")
        result.loc[date, sec.index] = (aligned - sector_mean).values

    return result


def apply_liquidity_filter(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    min_avg_volume: int = 100_000,
    lookback_days: int = 63,
) -> pd.DataFrame:
    """
    Return daily prices with illiquid stock-months set to NaN.

    Liquidity is determined at each month-end using trailing `lookback_days`-day
    average daily volume. Forward-filled to all days in the following month so
    the factor uses only prior-month liquidity (no look-ahead).
    """
    # Trailing average daily volume (daily frequency)
    avg_vol = volume.rolling(lookback_days, min_periods=21).mean()
    # Sample at month-end to get one liquidity reading per month
    monthly_avg_vol = avg_vol.resample("ME").last()
    liquid_mask_monthly = monthly_avg_vol >= min_avg_volume
    # Expand back to daily: forward-fill so each month's days inherit the prior month-end reading
    daily_mask = liquid_mask_monthly.reindex(prices.index, method="ffill")
    return prices.where(daily_mask)


def compute_vol_adjusted_momentum(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    vol_window: int = 6,
) -> pd.DataFrame:
    """
    Volatility-adjusted 12-1 month momentum: raw momentum / trailing annualized vol.

    More robust than raw momentum in high-volatility small-cap universes because
    it down-weights stocks with outsized recent swings that may not persist.
    """
    monthly = prices.resample("ME").last()
    monthly_ret = monthly.pct_change()

    mom = monthly.shift(skip).pct_change(lookback)

    # Trailing annualized vol over vol_window months, shifted by skip (no look-ahead)
    trailing_vol = (
        monthly_ret.shift(skip).rolling(vol_window, min_periods=3).std() * np.sqrt(12)
    )
    trailing_vol = trailing_vol.replace(0, np.nan)

    return (mom / trailing_vol).clip(-10, 10)


def build_long_short_weights(factor: pd.DataFrame, n_quantile: int = 5) -> pd.DataFrame:
    """Long top quintile, short bottom quintile, equal-weighted within each leg."""
    def row_weights(row):
        valid = row.dropna()
        if len(valid) < n_quantile * 2:
            return pd.Series(0.0, index=row.index)
        q_low = valid.quantile(1 / n_quantile)
        q_high = valid.quantile(1 - 1 / n_quantile)
        w = pd.Series(0.0, index=row.index)
        longs = valid[valid >= q_high]
        shorts = valid[valid <= q_low]
        if len(longs) > 0:
            w[longs.index] = 1.0 / len(longs)
        if len(shorts) > 0:
            w[shorts.index] = -1.0 / len(shorts)
        return w

    return factor.apply(row_weights, axis=1)
