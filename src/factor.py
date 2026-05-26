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
