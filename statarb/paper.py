"""
Paper trading loop.

Polls the exchange for newly-closed candles on the configured timeframe,
appends them to an in-memory price history, refits the hedge ratio on a
schedule, and runs the exact same signal/state-machine logic as the
backtester. All trades are SIMULATED and logged to CSV + stdout; no order
is ever sent to an exchange.

Safety: real order execution is intentionally not implemented anywhere in
this module. `run_paper_trading(..., live=True)` requires an additional
`confirm_live_trading=True` before it will even consider proceeding, and at
that point it still raises NotImplementedError -- see README.md for what
would need to be built before that could ever change.

State persistence: equity, open position, and the fitted model are saved to
a small JSON file (`<paper_log_dir>/state_<A>_<B>.json`) after every poll
and reloaded on startup. This lets the loop be run as many short-lived CLI
invocations spread over hours/days (e.g. from a scheduler) instead of one
long-running process, without losing track of an open position or resetting
equity back to starting_capital each time. Delete the state file to force a
fresh start.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .backtest import _size_position, _trade_cost
from .config import Config
from .data import get_exchange
from .pairs import PairModel, compute_spread, fit_pair
from .signal import Position, SignalState, apply_action, next_action, rolling_zscore

logger = logging.getLogger("statarb.paper")


def assert_safe_mode(live: bool, confirm_live_trading: bool) -> None:
    """The 'deliberate extra step' safety gate. See module docstring."""
    if not live:
        return
    if not confirm_live_trading:
        raise SystemExit(
            "Refusing to start: --live was passed without --confirm-live-trading.\n"
            "This extra flag exists so live trading can never be switched on by accident."
        )
    raise NotImplementedError(
        "Live order execution is intentionally NOT implemented in this scaffold.\n"
        "Passing --live --confirm-live-trading gets you here, and no further: there is no\n"
        "exchange-authenticated order path in this codebase. See README.md, section\n"
        "'Before this should ever touch real money', for what would need to be built first."
    )


class PaperTrader:
    def __init__(self, config: Config, coin_a: str, coin_b: str):
        self.config = config
        self.exchange = get_exchange(config.exchange)
        self.coin_a, self.coin_b = coin_a, coin_b
        self.market_a = f"{coin_a}/{config.quote}"
        self.market_b = f"{coin_b}/{config.quote}"

        self.history = pd.DataFrame(columns=["A", "B"])
        self.history.index.name = "timestamp"
        self.model: Optional[PairModel] = None
        self.state = SignalState(Position.FLAT)
        self.open_trade: Optional[dict] = None
        self.equity = config.starting_capital
        self.bars_since_fit = 0
        self._last_ts_a: Optional[int] = None

        log_dir = Path(config.paper_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = log_dir / f"trades_{coin_a}_{coin_b}.csv"
        self.state_path = log_dir / f"state_{coin_a}_{coin_b}.json"
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "event", "time", "direction", "z", "price_a", "price_b",
                    "shares_a", "shares_b", "pnl", "equity",
                ])

    def _save_state(self):
        state = {
            "version": 1,
            "equity": self.equity,
            "position": self.state.position.name,
            "open_trade": self.open_trade,
            "bars_since_fit": self.bars_since_fit,
            "last_ts_a": self._last_ts_a,
            "model": None if self.model is None else {
                "hedge_ratio": self.model.hedge_ratio,
                "alpha": self.model.alpha,
                "spread_mean": self.model.spread_mean,
                "spread_std": self.model.spread_std,
            },
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        # numpy floats sneak into equity/shares/prices; `default=float` covers them
        tmp.write_text(json.dumps(state, indent=2, default=float))
        tmp.replace(self.state_path)  # atomic on POSIX and Windows

    def _load_state(self) -> bool:
        """Returns True if prior state was found and restored."""
        if not self.state_path.exists():
            return False
        try:
            state = json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read state file %s (%s); starting fresh.", self.state_path, e)
            return False
        self.equity = state.get("equity", self.equity)
        self.state = SignalState(Position[state.get("position", "FLAT")])
        self.open_trade = state.get("open_trade")
        self.bars_since_fit = state.get("bars_since_fit", 0)
        self._last_ts_a = state.get("last_ts_a")
        m = state.get("model")
        if m:
            self.model = PairModel(self.coin_a, self.coin_b, m["hedge_ratio"], m["alpha"],
                                    m["spread_mean"], m["spread_std"])
        return True

    def _log_event(self, event: str, t, direction: str, z: float, price_a: float, price_b: float,
                    shares_a: float = 0.0, shares_b: float = 0.0, pnl: float = 0.0):
        with open(self.trade_log_path, "a", newline="") as f:
            csv.writer(f).writerow([event, t, direction, round(z, 4) if z == z else "", price_a, price_b,
                                     shares_a, shares_b, round(pnl, 2), round(self.equity, 2)])
        logger.info(
            "%s | %s dir=%s z=%.3f price_a=%.4f price_b=%.4f pnl=%.2f equity=%.2f",
            event, t, direction, z if z == z else float("nan"), price_a, price_b, pnl, self.equity,
        )

    def bootstrap(self):
        """Prefill history with enough closed candles to fit the first model."""
        from .data import load_symbol_history
        # config.lookback_days should be large enough to cover train_window + zscore_window
        # bars at the configured timeframe -- the caller is responsible for that (the CLI
        # warns if it looks too small); we just pull whatever lookback_days gives us.
        hist_a = load_symbol_history(self.market_a, self.config.exchange, self.config.timeframe,
                                      self.config.lookback_days, self.config.cache_dir)
        hist_b = load_symbol_history(self.market_b, self.config.exchange, self.config.timeframe,
                                      self.config.lookback_days, self.config.cache_dir)
        a = hist_a.set_index("timestamp")["close"].rename("A")
        b = hist_b.set_index("timestamp")["close"].rename("B")
        df = pd.concat([a, b], axis=1).ffill().dropna()
        self.history = df.tail(self.config.train_window + self.config.zscore_window)
        if len(self.history) < self.config.train_window:
            raise RuntimeError(
                f"Only {len(self.history)} bars available to bootstrap; need train_window="
                f"{self.config.train_window}. Increase lookback_days."
            )

        resumed = self._load_state()
        if resumed and self.model is not None:
            logger.info(
                "Resumed %s/%s from saved state: equity=%.2f position=%s open_trade=%s bars_since_fit=%d",
                self.coin_a, self.coin_b, self.equity, self.state.position.name,
                "yes" if self.open_trade else "no", self.bars_since_fit,
            )
        else:
            if self._last_ts_a is None:
                self._last_ts_a = int(self.history.index[-1].value // 1_000_000)
            self._refit()
            self._save_state()
            logger.info("Bootstrapped %s/%s fresh with %d bars, hedge_ratio=%.4f", self.coin_a, self.coin_b,
                        len(self.history), self.model.hedge_ratio)

    def _refit(self):
        train = self.history.tail(self.config.train_window)
        self.model = fit_pair(self.coin_a, self.coin_b, train["A"], train["B"])
        self.bars_since_fit = 0

    def _fetch_new_closed_bars(self, limit: int = 500):
        """Return every CLOSED candle strictly after the last processed timestamp, in
        chronological order -- NOT just the latest one. If polling is infrequent (e.g. a
        scheduler firing every few hours rather than a continuously-running process),
        several candles can close between polls; only ever looking at "the latest" would
        silently skip every candle in between instead of evaluating the signal on each of
        them, which would badly undersample what a real continuously-running bot would see.
        """
        tf_ms = self.exchange.parse_timeframe(self.config.timeframe) * 1000
        now_ms = self.exchange.milliseconds()
        since = (self._last_ts_a + 1) if self._last_ts_a is not None else (now_ms - 5 * tf_ms)

        oh_a = self.exchange.fetch_ohlcv(self.market_a, timeframe=self.config.timeframe, since=since, limit=limit)
        oh_b = self.exchange.fetch_ohlcv(self.market_b, timeframe=self.config.timeframe, since=since, limit=limit)
        if not oh_a or not oh_b:
            return []

        # keep only fully closed candles (drop the still-forming last one)
        closed_a = {c[0]: c[4] for c in oh_a if c[0] + tf_ms <= now_ms}
        closed_b = {c[0]: c[4] for c in oh_b if c[0] + tf_ms <= now_ms}
        common_ts = sorted(
            t for t in closed_a
            if t in closed_b and (self._last_ts_a is None or t > self._last_ts_a)
        )
        return [(pd.Timestamp(t, unit="ms", tz="UTC"), closed_a[t], closed_b[t]) for t in common_ts]

    def _process_bar(self, t, price_a: float, price_b: float):
        """Evaluate the signal and state machine for exactly one closed bar, then persist."""
        self.history.loc[t] = [price_a, price_b]
        self.history = self.history.tail(self.config.train_window + self.config.zscore_window * 2)
        self.bars_since_fit += 1

        if self.model is None or self.bars_since_fit >= self.config.test_window:
            self._refit()

        ctx = self.history.tail(self.config.zscore_window * 2)
        spread = compute_spread(ctx["A"], ctx["B"], self.model)
        z_series = rolling_zscore(spread, self.config.zscore_window, self.config.min_periods_zscore)
        z = float(z_series.iloc[-1])

        action = next_action(self.state, z, self.config.entry_z, self.config.exit_z, self.config.stop_z)

        if action in ("enter_long", "enter_short") and self.open_trade is None:
            new_state = apply_action(self.state, action)
            alloc = self.equity * self.config.risk_per_pair_pct
            shares_a, shares_b = _size_position(new_state.position, self.model.hedge_ratio, price_a, price_b, alloc)
            notional = abs(shares_a) * price_a + abs(shares_b) * price_b
            cost = _trade_cost(notional, self.config.fee_bps, self.config.slippage_bps)
            self.equity -= cost
            self.open_trade = {
                "direction": "long_spread" if new_state.position == Position.LONG_SPREAD else "short_spread",
                "entry_price_a": price_a, "entry_price_b": price_b,
                "shares_a": shares_a, "shares_b": shares_b, "entry_cost": cost,
            }
            self.state = new_state
            self._log_event("ENTER", t, self.open_trade["direction"], z, price_a, price_b, shares_a, shares_b)

        elif action == "exit" and self.open_trade is not None:
            ot = self.open_trade
            notional = abs(ot["shares_a"]) * price_a + abs(ot["shares_b"]) * price_b
            cost = _trade_cost(notional, self.config.fee_bps, self.config.slippage_bps)
            gross = ot["shares_a"] * (price_a - ot["entry_price_a"]) + ot["shares_b"] * (price_b - ot["entry_price_b"])
            self.equity += gross - cost
            self._log_event("EXIT", t, ot["direction"], z, price_a, price_b, ot["shares_a"], ot["shares_b"],
                             pnl=gross - cost - ot["entry_cost"])
            self.open_trade = None
            self.state = SignalState(Position.FLAT)
        else:
            unrealized = 0.0
            if self.open_trade is not None:
                ot = self.open_trade
                unrealized = ot["shares_a"] * (price_a - ot["entry_price_a"]) + ot["shares_b"] * (price_b - ot["entry_price_b"])
            self._log_event("HOLD", t, self.state.position.name, z, price_a, price_b, pnl=unrealized)

        self._last_ts_a = int(t.value // 1_000_000)
        self._save_state()  # save after every single bar, not just every poll -- a poll can
        # now cover many bars, and we don't want to lose progress on already-processed ones
        # if the process gets killed mid-catch-up (e.g. a scheduler's hard timeout).

    def poll_once(self):
        bars = self._fetch_new_closed_bars()
        if not bars:
            logger.debug("No new closed bars for %s/%s yet.", self.coin_a, self.coin_b)
            return
        if len(bars) > 1:
            logger.info("Catching up on %d closed bars for %s/%s since last poll.",
                        len(bars), self.coin_a, self.coin_b)
        for t, price_a, price_b in bars:
            self._process_bar(t, price_a, price_b)

    def run(self, iterations: Optional[int] = None):
        self.bootstrap()
        i = 0
        while iterations is None or i < iterations:
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error during poll; continuing after sleep.")
            i += 1
            if iterations is None or i < iterations:
                time.sleep(self.config.poll_interval_sec)


def run_paper_trading(
    config: Config,
    coin_a: str,
    coin_b: str,
    live: bool = False,
    confirm_live_trading: bool = False,
    iterations: Optional[int] = None,
):
    assert_safe_mode(live, confirm_live_trading)
    trader = PaperTrader(config, coin_a, coin_b)
    trader.run(iterations=iterations)
    return trader
