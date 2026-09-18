"""
Tests for statarb/flow.py -- the exchange trade-flow signal (the free,
already-working-infrastructure alternative to on-chain whale-wallet
tracking; see that module's docstring for why). No network: uses a fake
ccxt-shaped exchange exposing fetch_trades(), same pattern as
test_data.py's FakeExchange for fetch_ohlcv.
"""
from __future__ import annotations

import pandas as pd

from statarb.flow import compute_flow_snapshot, fetch_recent_trades, flow_entry_veto, log_flow_snapshot


class FakeTradesExchange:
    """Minimal ccxt-shaped stand-in exposing only fetch_trades."""

    def __init__(self, trades: list):
        self._trades = trades

    def fetch_trades(self, symbol, limit=500):
        return self._trades[:limit]


def _trade(ts_ms, side, price, amount):
    return {"timestamp": ts_ms, "side": side, "price": price, "amount": amount, "cost": price * amount}


def test_fetch_recent_trades_shapes_and_sorts_by_time():
    raw = [
        _trade(2000, "sell", 100.0, 1.0),
        _trade(1000, "buy", 99.0, 2.0),
    ]
    exchange = FakeTradesExchange(raw)
    df = fetch_recent_trades(exchange, "BTC/USDT", limit=10)
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert df.iloc[0]["side"] == "buy"
    assert df.iloc[0]["cost"] == 198.0


def test_fetch_recent_trades_handles_empty_response():
    exchange = FakeTradesExchange([])
    df = fetch_recent_trades(exchange, "BTC/USDT")
    assert df.empty
    assert list(df.columns) == ["timestamp", "side", "price", "amount", "cost"]


def test_compute_flow_snapshot_imbalance_and_large_trade_detection():
    # 9 small trades (cost=10 each, mixed sides) + 1 much larger buy (cost=1000)
    raw = [_trade(1000 + i, "buy" if i % 2 == 0 else "sell", 10.0, 1.0) for i in range(9)]
    raw.append(_trade(2000, "buy", 1000.0, 1.0))
    exchange = FakeTradesExchange(raw)
    trades = fetch_recent_trades(exchange, "BTC/USDT", limit=100)

    snap = compute_flow_snapshot("BTC/USDT", trades, large_trade_pctile=0.95)
    assert snap is not None
    assert snap.n_trades == 10
    assert snap.large_trade_count >= 1
    # the one big trade dominates buy_notional, so imbalance should be strongly positive
    assert snap.imbalance > 0.5
    assert snap.large_trade_net_notional > 0  # the flagged large trade(s) are net buys here


def test_compute_flow_snapshot_returns_none_for_empty_trades():
    empty = pd.DataFrame(columns=["timestamp", "side", "price", "amount", "cost"])
    assert compute_flow_snapshot("BTC/USDT", empty) is None


def test_compute_flow_snapshot_balanced_flow_has_near_zero_imbalance():
    raw = [_trade(1000 + i, "buy" if i % 2 == 0 else "sell", 100.0, 1.0) for i in range(20)]
    exchange = FakeTradesExchange(raw)
    trades = fetch_recent_trades(exchange, "ETH/USDT", limit=100)
    snap = compute_flow_snapshot("ETH/USDT", trades)
    assert abs(snap.imbalance) < 0.15


def test_log_flow_snapshot_writes_header_then_appends(tmp_path):
    raw = [_trade(1000, "buy", 100.0, 1.0), _trade(2000, "sell", 100.0, 1.0)]
    exchange = FakeTradesExchange(raw)
    trades = fetch_recent_trades(exchange, "BTC/USDT")
    snap = compute_flow_snapshot("BTC/USDT", trades)

    log_dir = tmp_path / "paper_logs"
    log_flow_snapshot(str(log_dir), snap)
    log_flow_snapshot(str(log_dir), snap)  # second call should append, not duplicate the header

    path = log_dir / "flow_BTC-USDT.csv"
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("timestamp,symbol")
    assert len(lines) == 3  # header + 2 data rows


def test_flow_entry_veto_off_with_no_snapshot():
    assert flow_entry_veto(None, "enter_long") is False


def test_flow_entry_veto_blocks_long_entry_against_heavy_selling():
    raw = [_trade(1000 + i, "sell", 100.0, 10.0) for i in range(10)]
    exchange = FakeTradesExchange(raw)
    trades = fetch_recent_trades(exchange, "BTC/USDT")
    snap = compute_flow_snapshot("BTC/USDT", trades)
    assert flow_entry_veto(snap, "enter_long", veto_imbalance=0.6) is True
    assert flow_entry_veto(snap, "enter_short", veto_imbalance=0.6) is False


def test_flow_entry_veto_allows_entry_when_flow_is_neutral():
    raw = [_trade(1000 + i, "buy" if i % 2 == 0 else "sell", 100.0, 1.0) for i in range(10)]
    exchange = FakeTradesExchange(raw)
    trades = fetch_recent_trades(exchange, "BTC/USDT")
    snap = compute_flow_snapshot("BTC/USDT", trades)
    assert flow_entry_veto(snap, "enter_long", veto_imbalance=0.6) is False
    assert flow_entry_veto(snap, "enter_short", veto_imbalance=0.6) is False
