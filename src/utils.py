import io
import ssl
import urllib.request

import yfinance as yf
import pandas as pd
from pathlib import Path

SP500_TOP50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "LLY",
    "V", "XOM", "UNH", "AVGO", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "NFLX", "PEP", "KO", "TMO", "ACN", "MCD",
    "LIN", "WMT", "CSCO", "ABT", "DHR", "IBM", "GE", "TXN", "PM", "ADBE",
    "CAT", "QCOM", "MS", "GS", "INTU", "SPGI", "RTX", "NEE", "HON", "AMGN",
]

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def download_prices(tickers=SP500_TOP50, start="2015-01-01", end="2025-12-31"):
    DATA_DIR.mkdir(exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    close = raw["Close"].dropna(how="all")
    path = DATA_DIR / "prices.parquet"
    close.to_parquet(path)
    print(f"Saved {close.shape[1]} tickers × {close.shape[0]} days → {path}")
    return close


def load_prices():
    return pd.read_parquet(DATA_DIR / "prices.parquet")


def get_sp500_constituents() -> tuple[list[str], pd.Series]:
    """
    Scrape current S&P 500 constituents + GICS sectors from Wikipedia.
    Returns (tickers, sectors) where sectors is a pd.Series ticker->sector.
    """
    import io
    import ssl
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # macOS Python 3.x often ships without root certs — bypass verification for this read-only scrape
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        html = resp.read().decode("utf-8")
    table = pd.read_html(io.StringIO(html))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    sectors = (
        table.set_index("Symbol")["GICS Sector"]
        .rename("sector")
    )
    sectors.index = sectors.index.str.replace(".", "-", regex=False)
    return tickers, sectors


def download_sp500_prices(start="2015-01-01", end="2025-12-31") -> tuple[pd.DataFrame, pd.Series]:
    DATA_DIR.mkdir(exist_ok=True)
    tickers, sectors = get_sp500_constituents()
    print(f"Downloading {len(tickers)} S&P 500 tickers …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    close = raw["Close"].dropna(how="all")
    # Keep only tickers that have sector info and price data
    common = close.columns.intersection(sectors.index)
    close = close[common]
    sectors = sectors.reindex(common)
    close.to_parquet(DATA_DIR / "prices_sp500.parquet")
    sectors.to_frame().to_parquet(DATA_DIR / "sectors_sp500.parquet")
    print(f"Saved {close.shape[1]} tickers × {close.shape[0]} days → data/prices_sp500.parquet")
    return close, sectors


def load_sp500_prices() -> tuple[pd.DataFrame, pd.Series]:
    close = pd.read_parquet(DATA_DIR / "prices_sp500.parquet")
    sectors = pd.read_parquet(DATA_DIR / "sectors_sp500.parquet")["sector"]
    return close, sectors


# ---------------------------------------------------------------------------
# Russell 2000 universe (IWM constituents)
# ---------------------------------------------------------------------------

def get_russell2000_tickers() -> tuple[list[str], pd.Series]:
    """
    Return small-cap universe tickers + GICS sectors.

    Uses the S&P 600 constituent list from Wikipedia as the small-cap proxy
    (603 stocks, same GICS sector taxonomy, equivalent factor-research universe).
    iShares IWM now requires JavaScript so the direct CSV endpoint is unusable.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        html = resp.read().decode("utf-8")

    table = pd.read_html(io.StringIO(html))[0]
    table["Symbol"] = table["Symbol"].str.strip().str.replace(".", "-", regex=False)
    tickers = table["Symbol"].dropna().tolist()
    sectors = table.set_index("Symbol")["GICS Sector"].rename("sector")
    return tickers, sectors


def download_russell2000_prices(
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    batch_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Download close prices and daily volume for the Russell 2000 universe.
    Returns (close, volume, sectors).
    Saves three parquet files: prices_r2k, volume_r2k, sectors_r2k.
    """
    DATA_DIR.mkdir(exist_ok=True)
    tickers, sectors = get_russell2000_tickers()
    print(f"Small-cap universe (S&P 600 proxy): {len(tickers)} tickers")

    all_close, all_vol = [], []
    n_batches = (len(tickers) - 1) // batch_size + 1

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{n_batches}  ({len(batch)} tickers)…", flush=True)
        raw = yf.download(
            batch, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            all_close.append(raw["Close"])
            all_vol.append(raw["Volume"])
        else:
            # Single-ticker fallback
            ticker = batch[0]
            all_close.append(raw[["Close"]].rename(columns={"Close": ticker}))
            all_vol.append(raw[["Volume"]].rename(columns={"Volume": ticker}))

    close = pd.concat(all_close, axis=1).dropna(how="all")
    volume = pd.concat(all_vol, axis=1).dropna(how="all")

    common = close.columns.intersection(sectors.index)
    close = close[common]
    volume = volume[common]
    sectors = sectors.reindex(common)

    close.to_parquet(DATA_DIR / "prices_r2k.parquet")
    volume.to_parquet(DATA_DIR / "volume_r2k.parquet")
    sectors.to_frame().to_parquet(DATA_DIR / "sectors_r2k.parquet")
    print(
        f"Saved {close.shape[1]} tickers × {close.shape[0]} days "
        f"→ data/prices_r2k.parquet"
    )
    return close, volume, sectors


def load_russell2000_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    close = pd.read_parquet(DATA_DIR / "prices_r2k.parquet")
    volume = pd.read_parquet(DATA_DIR / "volume_r2k.parquet")
    sectors = pd.read_parquet(DATA_DIR / "sectors_r2k.parquet")["sector"]
    return close, volume, sectors
