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
