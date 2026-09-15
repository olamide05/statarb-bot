"""
Regression test for a real bug found while running the bot for the first
time against live data: `load_symbol_history` built its "no cache yet"
placeholder as a bare `pd.DataFrame(columns=OHLCV_COLUMNS)`, which defaults
every column to dtype=object. Concatenating that with a real, numeric-dtyped
fetch result upcasts the WHOLE combined frame to object on pandas 3.x,
which then breaks statsmodels' OLS deep inside hedge-ratio fitting with a
cryptic `ufunc 'isfinite' not supported for the input types` error.

The existing smoke test (test_smoke.py) never touches statarb/data.py at
all -- it feeds the pipeline synthetic in-memory prices directly -- so it
never exercised this path. This test uses a fake ccxt-shaped exchange (no
network) to cover the actual data-loading code that broke.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data import load_symbol_history
from statarb.pairs import fit_hedge_ratio


class FakeExchange:
    """Minimal stand-in for a ccxt exchange, just enough for fetch_ohlcv_history."""

    def __init__(self, n_candles: int = 200, timeframe_minutes: int = 60):
        self.rateLimit = 0  # skip real sleeps in tests
        self._tf_minutes = timeframe_minutes
        tf_ms = timeframe_minutes * 60 * 1000
        self._now = 1_700_000_000_000 + n_candles * tf_ms
        start = self._now - n_candles * tf_ms
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 0.5, n_candles))
        self.candles = [
            [start + i * tf_ms, float(prices[i]), float(prices[i]) + 1, float(prices[i]) - 1,
             float(prices[i]), 10.0]
            for i in range(n_candles)
        ]

    def milliseconds(self):
        return self._now

    def parse_timeframe(self, tf):
        return self._tf_minutes * 60

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=1000):
        rows = [c for c in self.candles if c[0] >= (since or 0)]
        return rows[:limit]


def test_load_symbol_history_first_run_has_numeric_dtypes(tmp_path, monkeypatch):
    """The exact scenario that broke: fresh cache_dir, no parquet cache yet."""
    fake = FakeExchange(n_candles=100)
    monkeypatch.setattr("statarb.data.get_exchange", lambda exchange_id: fake)

    cache_dir = tmp_path / "cache"
    hist = load_symbol_history("BTC/USDT", "binance", "1h", lookback_days=5, cache_dir=str(cache_dir))

    assert not hist.empty
    for col in ["open", "high", "low", "close", "volume"]:
        assert hist[col].dtype == np.float64, f"{col} has dtype {hist[col].dtype}, expected float64"

    # this is the exact call that raised "ufunc 'isfinite' not supported" before the fix
    beta, alpha = fit_hedge_ratio(hist["close"], hist["close"] * 0.9 + 1)
    assert np.isfinite(beta) and np.isfinite(alpha)


def test_load_symbol_history_second_run_reuses_cache_with_numeric_dtypes(tmp_path, monkeypatch):
    """Second call should hit the parquet cache path (a different code path) and still be numeric."""
    fake = FakeExchange(n_candles=100)
    monkeypatch.setattr("statarb.data.get_exchange", lambda exchange_id: fake)
    cache_dir = tmp_path / "cache"

    first = load_symbol_history("BTC/USDT", "binance", "1h", lookback_days=5, cache_dir=str(cache_dir))
    second = load_symbol_history("BTC/USDT", "binance", "1h", lookback_days=5, cache_dir=str(cache_dir))

    assert not second.empty
    for col in ["open", "high", "low", "close", "volume"]:
        assert second[col].dtype == np.float64
    assert len(second) >= len(first)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
