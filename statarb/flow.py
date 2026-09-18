"""
Exchange trade-flow signal: a free, real-time proxy for "big trader
activity" on the crypto side, built entirely from ccxt's public
fetch_trades() endpoint. No new dependency, no new account/API key, reuses
the exact same exchange connection paper trading already has open.

Why this instead of on-chain whale-wallet tracking (what was actually asked
for): checked the current state of that market before writing any code
here (see chat) rather than assuming it's still true from training data.
Whale Alert -- historically the standard option for this -- no longer
appears to offer a free tier at all: plans now start at $29.95/mo for
websocket alerts, and real-time REST API access is $699/mo. Etherscan has
also been cutting back free *programmatic* API access on several chains
recently, and even its most generous free tier only covers Ethereum-family
chains, never Bitcoin -- so it can't cover both legs of the BTC/ETH pair
this project actually trades. An exchange's own recent public trade tape is
a legitimate, well-established proxy for the same underlying thing (large,
aggressive positioning by big players) -- just measured at the exchange
rather than at the wallet level -- and it's free, symmetric across BTC and
ETH, and doesn't add a new external dependency with its own uptime/rate-
limit risk on top of the ones this project already has to work around
(Binance's 451 on GitHub Actions, etc).

Honest limitation, stated up front, the same way the OOS-selection gap was:
ccxt's fetch_trades() only returns RECENT trades (recent minutes-to-hours
of tape, exchange-dependent) -- there is no historical trade-by-trade tape
to backtest this against, because nothing has ever captured it before now.
This CANNOT be backtested the way the price-based cointegration signal can.
It can only be logged going forward (log_flow_snapshot, wired into
paper.py's poll loop) and evaluated honestly once enough real history has
accumulated -- the same posture already applied to paper trading itself.
That's why flow_filter_enabled defaults to False in config.py: the signal
is logged and visible from day one, but doesn't touch trade decisions until
there's actual evidence it helps rather than hurts.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("statarb.flow")


@dataclass
class FlowSnapshot:
    symbol: str
    timestamp: pd.Timestamp
    n_trades: int
    buy_notional: float
    sell_notional: float
    large_trade_notional_threshold: float
    large_trade_count: int
    large_trade_net_notional: float  # signed: large buys - large sells, in quote currency

    @property
    def imbalance(self) -> float:
        """(buy - sell) / total notional, in [-1, 1]. Positive = net aggressive buying."""
        total = self.buy_notional + self.sell_notional
        return (self.buy_notional - self.sell_notional) / total if total > 0 else 0.0


def fetch_recent_trades(exchange, symbol: str, limit: int = 500) -> pd.DataFrame:
    """Wrap ccxt's public fetch_trades (no API key required). Returns
    columns [timestamp, side, price, amount, cost] sorted ascending by time.
    `side` can come back missing/unknown on some exchanges/trade types --
    those rows just won't count toward buy_notional or sell_notional."""
    raw = exchange.fetch_trades(symbol, limit=limit)
    if not raw:
        return pd.DataFrame(columns=["timestamp", "side", "price", "amount", "cost"])
    rows = [{
        "timestamp": pd.Timestamp(t["timestamp"], unit="ms", tz="UTC"),
        "side": t.get("side") or "unknown",
        "price": float(t["price"]),
        "amount": float(t["amount"]),
        "cost": float(t.get("cost") or t["price"] * t["amount"]),
    } for t in raw]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def compute_flow_snapshot(symbol: str, trades: pd.DataFrame, large_trade_pctile: float = 0.95) -> Optional[FlowSnapshot]:
    """Summarize a batch of recent trades into one point-in-time snapshot.
    Trades at/above `large_trade_pctile` of notional size WITHIN THIS BATCH
    are flagged as 'large' -- a relative, self-calibrating threshold (no
    fixed dollar cutoff to tune/go stale) that stands in for "big trader"
    activity. Returns None if there were no trades to summarize."""
    if trades.empty:
        return None
    buys = trades[trades["side"] == "buy"]
    sells = trades[trades["side"] == "sell"]
    threshold = float(trades["cost"].quantile(large_trade_pctile)) if len(trades) >= 10 else float(trades["cost"].max())
    large = trades[trades["cost"] >= threshold]
    large_buy = large[large["side"] == "buy"]["cost"].sum()
    large_sell = large[large["side"] == "sell"]["cost"].sum()
    return FlowSnapshot(
        symbol=symbol,
        timestamp=trades["timestamp"].max(),
        n_trades=len(trades),
        buy_notional=float(buys["cost"].sum()),
        sell_notional=float(sells["cost"].sum()),
        large_trade_notional_threshold=threshold,
        large_trade_count=int(len(large)),
        large_trade_net_notional=float(large_buy - large_sell),
    )


_FLOW_LOG_COLUMNS = [
    "timestamp", "symbol", "n_trades", "buy_notional", "sell_notional",
    "imbalance", "large_trade_threshold", "large_trade_count", "large_trade_net_notional",
]


def log_flow_snapshot(log_dir: str, snapshot: FlowSnapshot) -> None:
    """Append one snapshot to <log_dir>/flow_<SYMBOL>.csv, writing a header
    on first use. This is the only way this signal ever becomes evaluable --
    see module docstring: there's no historical tape to backtest against,
    only what gets logged from here forward."""
    path = Path(log_dir) / f"flow_{snapshot.symbol.replace('/', '-')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(_FLOW_LOG_COLUMNS)
        w.writerow([
            snapshot.timestamp, snapshot.symbol, snapshot.n_trades,
            round(snapshot.buy_notional, 2), round(snapshot.sell_notional, 2),
            round(snapshot.imbalance, 4), round(snapshot.large_trade_notional_threshold, 2),
            snapshot.large_trade_count, round(snapshot.large_trade_net_notional, 2),
        ])


def flow_entry_veto(snapshot: Optional[FlowSnapshot], direction: str, veto_imbalance: float = 0.6) -> bool:
    """OFF by default (config.flow_filter_enabled) -- not enough logged
    history yet to know whether this actually helps, see module docstring.
    When enabled: True means "skip this entry", for one specific, stateable
    hypothesis -- entering a mean-reversion trade straight into a strong,
    opposing directional flow imbalance is riskier than entering when flow
    is neutral or aligned. Untested; that's exactly why it stays opt-in
    until there's logged evidence either way.

    `direction` is "enter_long" or "enter_short" as returned by
    signal.next_action. LONG_SPREAD is a bet that symbol_a is undervalued
    relative to symbol_b, so heavy net aggressive SELLING on symbol_a right
    now (negative imbalance) is the opposing case; SHORT_SPREAD is the
    mirror image."""
    if snapshot is None:
        return False
    if direction == "enter_long":
        return snapshot.imbalance <= -veto_imbalance
    if direction == "enter_short":
        return snapshot.imbalance >= veto_imbalance
    return False
