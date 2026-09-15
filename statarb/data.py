"""
OHLCV data access via ccxt, public endpoints only (no API key needed).

Handles pagination for historical fetches and caches results to parquet so
repeated scans/backtests over the same window don't re-hit the exchange.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_NUMERIC_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _empty_ohlcv_frame() -> pd.DataFrame:
    """An empty OHLCV frame with real (non-object) dtypes per column.

    A bare `pd.DataFrame(columns=OHLCV_COLUMNS)` defaults every column to
    dtype=object. Concatenating that with a real, numeric-dtyped frame later
    (e.g. `pd.concat([cached, fresh])` when there's no cache yet) can upcast
    the WHOLE result to object -- on pandas 3.x this reliably happens -- which
    then breaks statsmodels' OLS with a cryptic `ufunc 'isfinite' not
    supported for the input types` deep inside cointegration/hedge-ratio
    fitting. Giving the empty frame real dtypes up front avoids that.
    """
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        **{c: pd.Series(dtype="float64") for c in _NUMERIC_OHLCV_COLUMNS},
    })


def get_exchange(exchange_id: str = "binance"):
    """Instantiate a ccxt exchange for public market data only."""
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown ccxt exchange id: {exchange_id!r}")
    klass = getattr(ccxt, exchange_id)
    ex = klass({"enableRateLimit": True})
    return ex


def _cache_path(cache_dir: str, exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exchange_id}_{safe_symbol}_{timeframe}.parquet"


def fetch_ohlcv_history(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    limit: int = 1000,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Page through exchange.fetch_ohlcv from since_ms to until_ms (default: now)."""
    until_ms = until_ms or exchange.milliseconds()
    all_rows: List[list] = []
    cursor = since_ms
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    while cursor < until_ms:
        for attempt in range(max_retries):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            # exchange isn't advancing; avoid an infinite loop
            break
        cursor = last_ts + tf_ms
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        return _empty_ohlcv_frame()

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[_NUMERIC_OHLCV_COLUMNS] = df[_NUMERIC_OHLCV_COLUMNS].astype("float64")
    df = df[(df["timestamp"] >= pd.to_datetime(since_ms, unit="ms", utc=True))]
    if until_ms:
        df = df[df["timestamp"] <= pd.to_datetime(until_ms, unit="ms", utc=True)]
    return df.reset_index(drop=True)


def load_symbol_history(
    symbol: str,
    exchange_id: str,
    timeframe: str,
    lookback_days: int,
    cache_dir: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch (or load from cache) OHLCV history for one symbol, refreshing the tail."""
    path = _cache_path(cache_dir, exchange_id, symbol, timeframe)
    exchange = get_exchange(exchange_id)
    now_ms = exchange.milliseconds()
    since_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000

    cached = _empty_ohlcv_frame()
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        cached[_NUMERIC_OHLCV_COLUMNS] = cached[_NUMERIC_OHLCV_COLUMNS].astype("float64")

    if not cached.empty and cached["timestamp"].min() <= pd.to_datetime(since_ms, unit="ms", utc=True):
        # cache already covers the requested window at the front; only pull the fresh tail
        fetch_since = int(cached["timestamp"].max().timestamp() * 1000)
    else:
        fetch_since = since_ms

    fresh = fetch_ohlcv_history(exchange, symbol, timeframe, fetch_since, now_ms)
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    combined = combined[combined["timestamp"] >= pd.to_datetime(since_ms, unit="ms", utc=True)].reset_index(drop=True)

    if use_cache:
        combined.to_parquet(path, index=False)

    return combined


def load_universe_prices(
    universe: List[str],
    quote: str,
    exchange_id: str,
    timeframe: str,
    lookback_days: int,
    cache_dir: str,
    use_cache: bool = True,
    field: str = "close",
) -> pd.DataFrame:
    """Return a wide DataFrame of `field` prices, columns=universe symbols, aligned on timestamp."""
    series: Dict[str, pd.Series] = {}
    for coin in universe:
        symbol = f"{coin}/{quote}"
        hist = load_symbol_history(symbol, exchange_id, timeframe, lookback_days, cache_dir, use_cache)
        if hist.empty:
            continue
        series[coin] = hist.set_index("timestamp")[field]

    if not series:
        raise RuntimeError("No data fetched for any symbol in the universe.")

    prices = pd.DataFrame(series)
    prices = prices.sort_index().ffill().dropna(how="any")
    return prices
