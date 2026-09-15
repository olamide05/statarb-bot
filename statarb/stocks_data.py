"""
Equities OHLCV data access via yfinance, for the same cointegration
pairs-trading pipeline used for crypto. Kept as a separate module (not
folded into data.py) because the two sources are genuinely different:
different API, different symbol format (plain tickers, no quote suffix),
different trading calendar (weekdays/market-hours only vs. crypto's 24/7),
and -- unlike ccxt's exchange-agnostic interface -- yfinance is a single,
specific, occasionally-flaky data source.

Known reliability caveat (be aware of this before assuming a failure is a
bug): Yahoo Finance requires a "cookie/crumb" handshake that yfinance
performs internally, and it can fail from datacenter/cloud IPs (including
GitHub Actions runners) with connection errors or empty results -- this is
the equities-data equivalent of the Binance-451-on-GitHub-Actions issue
this project already hit once. If `scan`/`backtest` with `asset_class:
stock` returns no data or every symbol fails, that's the first thing to
check, not a bug in this file. Stooq (https://stooq.com/q/d/l/?s=<ticker>.us
&i=d) is a plain-CSV, no-auth alternative for DAILY bars specifically if
yfinance ever needs to be swapped out the way Binance was.

Returns the same OHLCV DataFrame shape as statarb.data (timestamp, open,
high, low, close, volume, all real numeric dtypes -- see the dtype-upcast
bug that hit the crypto loader; this module builds numeric-dtyped frames
from the start rather than repeating that mistake) so it's a drop-in for
pairs.py/backtest.py, which only ever operate on a plain wide price
DataFrame and don't know or care where the data came from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_NUMERIC_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# yfinance interval strings that are safe to use with a long lookback.
# Intraday intervals (1m/5m/15m/1h etc.) are capped by Yahoo to the last
# ~60 days of history regardless of what you ask for, which is too little
# for a meaningful cointegration test window -- so this module (and
# config_stocks.yaml) sticks to daily bars. Swap `timeframe` to an
# intraday value only if you also shrink lookback_days accordingly.
_SUPPORTED_INTERVALS = {"1d", "1wk", "1mo"}


def _empty_ohlcv_frame() -> pd.DataFrame:
    """Same rationale as statarb.data._empty_ohlcv_frame: give every column
    a real dtype up front so an empty result never silently upcasts a later
    pd.concat to dtype=object (the bug that broke the crypto loader)."""
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        **{c: pd.Series(dtype="float64") for c in _NUMERIC_OHLCV_COLUMNS},
    })


def _cache_path(cache_dir: str, ticker: str, timeframe: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"yfinance_{ticker}_{timeframe}.parquet"


def fetch_stock_history(ticker: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    """Pull daily OHLCV for one ticker via yfinance. No API key needed."""
    if timeframe not in _SUPPORTED_INTERVALS:
        raise ValueError(
            f"stocks_data only supports {sorted(_SUPPORTED_INTERVALS)} timeframes "
            f"(got {timeframe!r}) -- intraday history from yfinance is too short "
            f"(~60 days) for a meaningful train/test window."
        )
    import yfinance as yf  # imported lazily so crypto-only users never need it installed

    period_days = max(lookback_days + 10, 30)  # pad a little for weekends/holidays
    raw = yf.Ticker(ticker).history(period=f"{period_days}d", interval=timeframe, auto_adjust=True)

    if raw is None or raw.empty:
        return _empty_ohlcv_frame()

    df = raw.reset_index()
    # yfinance names the index column "Date" for daily/weekly/monthly bars,
    # "Datetime" for intraday -- normalize either to our "timestamp".
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(columns={
        date_col: "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[OHLCV_COLUMNS]
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(timestamp=ts)
    df[_NUMERIC_OHLCV_COLUMNS] = df[_NUMERIC_OHLCV_COLUMNS].astype("float64")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def load_stock_history(
    ticker: str,
    timeframe: str,
    lookback_days: int,
    cache_dir: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch (or load from cache) OHLCV history for one ticker."""
    path = _cache_path(cache_dir, ticker, timeframe)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)

    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        cached[_NUMERIC_OHLCV_COLUMNS] = cached[_NUMERIC_OHLCV_COLUMNS].astype("float64")
        # Two conditions, both required: the cache has to reach back far
        # enough to cover the *requested* lookback_days (not just whatever
        # lookback_days some earlier call happened to use -- a cache built
        # for a shorter window must NOT be silently served for a longer
        # one, that would quietly truncate the result), and it has to be
        # recent enough to trust (daily bars close once a day, so same-day
        # reuse is fine -- no need to refetch the tail every run the way
        # the crypto loader does for its every-few-minutes bars).
        covers_requested_window = not cached.empty and cached["timestamp"].min() <= cutoff
        is_fresh = not cached.empty and cached["timestamp"].max().date() >= \
            (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).date()
        if covers_requested_window and is_fresh:
            return cached[cached["timestamp"] >= cutoff].reset_index(drop=True)

    fresh = fetch_stock_history(ticker, timeframe, lookback_days)
    if use_cache and not fresh.empty:
        fresh.to_parquet(path, index=False)
    return fresh


def load_stock_universe_prices(
    universe: List[str],
    timeframe: str,
    lookback_days: int,
    cache_dir: str,
    use_cache: bool = True,
    field: str = "close",
) -> pd.DataFrame:
    """Same contract as statarb.data.load_universe_prices: a wide DataFrame
    of `field` prices, columns=tickers, aligned on timestamp. Drop-in for
    scan_pairs/backtest_portfolio, which don't distinguish crypto vs stock."""
    series: Dict[str, pd.Series] = {}
    failures: List[str] = []
    for ticker in universe:
        hist = load_stock_history(ticker, timeframe, lookback_days, cache_dir, use_cache)
        if hist.empty:
            failures.append(ticker)
            continue
        series[ticker] = hist.set_index("timestamp")[field]

    if not series:
        raise RuntimeError(
            "No data fetched for any symbol in the stock universe "
            f"{universe}. If this is running on a cloud/CI runner, Yahoo "
            "Finance's cookie/crumb handshake is a known failure point "
            "there -- see the module docstring in stocks_data.py for a "
            "no-auth fallback (Stooq, daily bars only)."
        )
    if failures:
        import sys
        print(f"Warning: no data for {failures} (skipped, not in the result)", file=sys.stderr)

    prices = pd.DataFrame(series)
    prices = prices.sort_index().ffill().dropna(how="any")
    return prices
