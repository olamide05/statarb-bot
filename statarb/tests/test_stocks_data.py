"""
Regression-style test for statarb/stocks_data.py, mirroring test_data.py's
approach for the crypto loader: build the OHLCV frame with a fake data
source (no network, no real yfinance calls) and assert real numeric dtypes
all the way through to fit_hedge_ratio, since that's exactly the kind of
silent object-dtype upcast that broke the crypto loader in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.pairs import fit_hedge_ratio
from statarb.stocks_data import load_stock_history, load_stock_universe_prices


def _fake_yf_history(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Shape-compatible stand-in for yf.Ticker(...).history(...)'s return:
    a DataFrame indexed by tz-aware Datetime with Open/High/Low/Close/Volume."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=n_days, freq="B")  # business days
    prices = 100 + np.cumsum(rng.normal(0, 1, n_days))
    return pd.DataFrame({
        "Open": prices, "High": prices + 1, "Low": prices - 1,
        "Close": prices, "Volume": 1_000_000.0,
    }, index=idx.rename("Date"))


class _FakeTicker:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def history(self, *args, **kwargs):
        return self._df.copy()


def test_load_stock_history_has_numeric_dtypes(tmp_path, monkeypatch):
    fake_df = _fake_yf_history(n_days=300)
    monkeypatch.setattr(
        "yfinance.Ticker", lambda ticker: _FakeTicker(fake_df),
    )

    cache_dir = tmp_path / "cache"
    hist = load_stock_history("AAPL", "1d", lookback_days=250, cache_dir=str(cache_dir))

    assert not hist.empty
    for col in ["open", "high", "low", "close", "volume"]:
        assert hist[col].dtype == np.float64, f"{col} has dtype {hist[col].dtype}, expected float64"

    # same crash this exact pattern caused in the crypto loader before the dtype fix
    beta, alpha = fit_hedge_ratio(hist["close"], hist["close"] * 0.9 + 1)
    assert np.isfinite(beta) and np.isfinite(alpha)


def test_load_stock_history_rejects_unsupported_intraday_timeframe(tmp_path):
    with pytest.raises(ValueError, match="only supports"):
        load_stock_history("AAPL", "5m", lookback_days=5, cache_dir=str(tmp_path))


def test_load_stock_universe_prices_builds_wide_frame(tmp_path, monkeypatch):
    tickers = {"AAPL": _fake_yf_history(300, seed=1), "MSFT": _fake_yf_history(300, seed=2)}
    monkeypatch.setattr(
        "yfinance.Ticker", lambda ticker: _FakeTicker(tickers[ticker]),
    )

    prices = load_stock_universe_prices(
        list(tickers), "1d", lookback_days=250, cache_dir=str(tmp_path / "cache"),
    )

    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert not prices.empty
    assert prices.isna().sum().sum() == 0  # dropna(how="any") should leave no gaps


def test_load_stock_universe_prices_raises_clearly_when_every_symbol_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yfinance.Ticker", lambda ticker: _FakeTicker(pd.DataFrame()),
    )
    with pytest.raises(RuntimeError, match="No data fetched"):
        load_stock_universe_prices(["AAPL"], "1d", lookback_days=250, cache_dir=str(tmp_path))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
