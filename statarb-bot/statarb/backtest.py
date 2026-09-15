"""
Walk-forward backtester.

The hedge ratio and spread mean/std are fit ONLY on a rolling train window,
then held fixed while trading the following, unseen test window. The window
then rolls forward by test_window bars and refits. This is deliberately the
only backtest mode offered here: fitting the hedge ratio on the same data
you trade (in-sample) overstates performance and is not exposed by this CLI.

Position sizing is dollar-neutral in the statistical sense: at entry we size
shares_a and shares_b so that shares_b = -hedge_ratio * shares_a (that's the
hedge implied by the fitted spread = price_a - hedge_ratio * price_b), with
total notional (|leg_a| + |leg_b|) equal to risk_per_pair_pct * equity.
Fees + slippage (bps) are charged on both legs, on entry and on exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .pairs import PairModel, compute_spread, fit_pair
from .signal import Position, SignalState, apply_action, next_action, rolling_zscore

TIMEFRAME_TO_BARS_PER_YEAR = {
    "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2, "1h": 365 * 24, "2h": 365 * 12, "4h": 365 * 6,
    "6h": 365 * 4, "12h": 365 * 2, "1d": 365,
}


def bars_per_year(timeframe: str) -> float:
    return TIMEFRAME_TO_BARS_PER_YEAR.get(timeframe, 365 * 24)


@dataclass
class Trade:
    symbol_a: str
    symbol_b: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_z: float
    exit_z: float
    exit_reason: str
    shares_a: float
    shares_b: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    gross_pnl: float
    costs: float
    net_pnl: float


@dataclass
class BacktestResult:
    symbol_a: str
    symbol_b: str
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _trade_cost(notional: float, fee_bps: float, slippage_bps: float) -> float:
    return notional * (fee_bps + slippage_bps) / 10_000.0


def _size_position(direction: Position, hedge_ratio: float, price_a: float, price_b: float, alloc: float):
    """Return (shares_a, shares_b) such that shares_b = -hedge_ratio*shares_a
    and total notional |shares_a*price_a| + |shares_b*price_b| == alloc."""
    denom = price_a + abs(hedge_ratio) * price_b
    if denom <= 0:
        return 0.0, 0.0
    n = alloc / denom
    if direction == Position.LONG_SPREAD:
        shares_a = n
        shares_b = -hedge_ratio * n
    elif direction == Position.SHORT_SPREAD:
        shares_a = -n
        shares_b = hedge_ratio * n
    else:
        shares_a = shares_b = 0.0
    return shares_a, shares_b


def walk_forward_backtest(
    prices: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    config: Config,
) -> BacktestResult:
    y = prices[symbol_a]
    x = prices[symbol_b]
    n = len(prices)
    idx = prices.index

    equity = config.starting_capital
    equity_curve = pd.Series(index=idx, dtype=float)
    equity_curve.iloc[0] = equity

    trades: List[Trade] = []
    state = SignalState(Position.FLAT)
    open_trade: Optional[dict] = None

    start = 0
    train_w, test_w = config.train_window, config.test_window
    if n < train_w + 10:
        raise ValueError(
            f"Not enough history ({n} bars) for train_window={train_w}. "
            "Lower train_window or pull more lookback_days."
        )

    last_filled = 0
    while start + train_w < n:
        train_slice = slice(start, start + train_w)
        test_end = min(start + train_w + test_w, n)
        test_slice = slice(start + train_w, test_end)

        train_y, train_x = y.iloc[train_slice], x.iloc[train_slice]
        model: PairModel = fit_pair(symbol_a, symbol_b, train_y, train_x)

        # compute spread/z-score over train-tail + test so the rolling window
        # has causal warm-up data without ever looking into the future
        warm_start = max(0, start + train_w - config.zscore_window)
        ctx_slice = slice(warm_start, test_end)
        spread_ctx = compute_spread(y.iloc[ctx_slice], x.iloc[ctx_slice], model)
        z_ctx = rolling_zscore(spread_ctx, config.zscore_window, config.min_periods_zscore)

        test_positions = range(start + train_w, test_end)
        for pos in test_positions:
            t = idx[pos]
            z = z_ctx.loc[t] if t in z_ctx.index else np.nan
            price_a, price_b = y.iloc[pos], x.iloc[pos]

            action = next_action(state, z, config.entry_z, config.exit_z, config.stop_z)

            if action in ("enter_long", "enter_short") and open_trade is None:
                new_state = apply_action(state, action)
                alloc = equity * config.risk_per_pair_pct
                shares_a, shares_b = _size_position(new_state.position, model.hedge_ratio, price_a, price_b, alloc)
                entry_notional = abs(shares_a) * price_a + abs(shares_b) * price_b
                cost = _trade_cost(entry_notional, config.fee_bps, config.slippage_bps)
                equity -= cost
                open_trade = {
                    "direction": "long_spread" if new_state.position == Position.LONG_SPREAD else "short_spread",
                    "entry_time": t, "entry_z": float(z),
                    "shares_a": shares_a, "shares_b": shares_b,
                    "entry_price_a": price_a, "entry_price_b": price_b,
                    "entry_cost": cost,
                }
                state = new_state

            elif action == "exit" and open_trade is not None:
                exit_notional = abs(open_trade["shares_a"]) * price_a + abs(open_trade["shares_b"]) * price_b
                cost = _trade_cost(exit_notional, config.fee_bps, config.slippage_bps)
                gross = (
                    open_trade["shares_a"] * (price_a - open_trade["entry_price_a"])
                    + open_trade["shares_b"] * (price_b - open_trade["entry_price_b"])
                )
                total_cost = open_trade["entry_cost"] + cost
                net = gross - total_cost
                # entry_cost was already deducted from equity when the trade opened;
                # here we only add the price-change gain/loss and the exit cost
                equity += gross - cost
                trades.append(Trade(
                    symbol_a=symbol_a, symbol_b=symbol_b,
                    direction=open_trade["direction"],
                    entry_time=open_trade["entry_time"], exit_time=t,
                    entry_z=open_trade["entry_z"], exit_z=float(z),
                    exit_reason="stop_loss" if abs(z) >= config.stop_z else "mean_revert",
                    shares_a=open_trade["shares_a"], shares_b=open_trade["shares_b"],
                    entry_price_a=open_trade["entry_price_a"], entry_price_b=open_trade["entry_price_b"],
                    exit_price_a=price_a, exit_price_b=price_b,
                    gross_pnl=gross, costs=total_cost, net_pnl=net,
                ))
                open_trade = None
                state = SignalState(Position.FLAT)

            # mark-to-market equity for this bar (unrealized PnL if a position is open)
            unrealized = 0.0
            if open_trade is not None:
                unrealized = (
                    open_trade["shares_a"] * (price_a - open_trade["entry_price_a"])
                    + open_trade["shares_b"] * (price_b - open_trade["entry_price_b"])
                )
            equity_curve.loc[t] = equity + unrealized
            last_filled = pos

        # force-close any open position at the window boundary before refitting;
        # the hedge ratio is about to change so carrying the position forward
        # would silently apply a new model to an old trade
        if open_trade is not None:
            t = idx[test_end - 1]
            price_a, price_b = y.iloc[test_end - 1], x.iloc[test_end - 1]
            exit_notional = abs(open_trade["shares_a"]) * price_a + abs(open_trade["shares_b"]) * price_b
            cost = _trade_cost(exit_notional, config.fee_bps, config.slippage_bps)
            gross = (
                open_trade["shares_a"] * (price_a - open_trade["entry_price_a"])
                + open_trade["shares_b"] * (price_b - open_trade["entry_price_b"])
            )
            net = gross - open_trade["entry_cost"] - cost
            equity = equity - open_trade["entry_cost"] + gross - cost
            z_final = z_ctx.iloc[-1] if len(z_ctx) else float("nan")
            trades.append(Trade(
                symbol_a=symbol_a, symbol_b=symbol_b,
                direction=open_trade["direction"],
                entry_time=open_trade["entry_time"], exit_time=t,
                entry_z=open_trade["entry_z"], exit_z=float(z_final),
                exit_reason="window_end",
                shares_a=open_trade["shares_a"], shares_b=open_trade["shares_b"],
                entry_price_a=open_trade["entry_price_a"], entry_price_b=open_trade["entry_price_b"],
                exit_price_a=price_a, exit_price_b=price_b,
                gross_pnl=gross, costs=open_trade["entry_cost"] + cost, net_pnl=net,
            ))
            open_trade = None
            state = SignalState(Position.FLAT)
            equity_curve.loc[t] = equity

        start += test_w

    equity_curve = equity_curve.ffill().dropna()
    metrics = compute_metrics(equity_curve, trades, config.timeframe)
    return BacktestResult(symbol_a=symbol_a, symbol_b=symbol_b, equity_curve=equity_curve, trades=trades, metrics=metrics)


def compute_metrics(equity_curve: pd.Series, trades: List[Trade], timeframe: str) -> dict:
    if len(equity_curve) < 2:
        return {"total_return_pct": 0.0, "sharpe": float("nan"), "max_drawdown_pct": 0.0,
                "num_trades": len(trades), "win_rate_pct": float("nan")}

    rets = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    bpy = bars_per_year(timeframe)
    sharpe = float("nan")
    if rets.std(ddof=0) > 0:
        sharpe = (rets.mean() / rets.std(ddof=0)) * np.sqrt(bpy)

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_dd = drawdown.min()

    wins = [t for t in trades if t.net_pnl > 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else float("nan")

    return {
        "total_return_pct": round(total_return * 100, 3),
        "sharpe": round(sharpe, 3) if sharpe == sharpe else None,
        "max_drawdown_pct": round(max_dd * 100, 3),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2) if win_rate == win_rate else None,
        "final_equity": round(float(equity_curve.iloc[-1]), 2),
    }


def backtest_portfolio(
    prices: pd.DataFrame,
    pairs: List[tuple],
    config: Config,
) -> dict:
    """Run walk_forward_backtest independently per pair, then combine equity
    curves into one portfolio curve (sum of each pair's PnL series, seeded at
    starting_capital). See README for the caveats of this simplification."""
    results = {}
    combined_pnl: Optional[pd.Series] = None
    for a, b in pairs:
        res = walk_forward_backtest(prices, a, b, config)
        results[f"{a}/{b}"] = res
        pnl_series = res.equity_curve - config.starting_capital
        combined_pnl = pnl_series if combined_pnl is None else combined_pnl.add(pnl_series, fill_value=0.0)

    portfolio_equity = (combined_pnl + config.starting_capital) if combined_pnl is not None else pd.Series(dtype=float)
    portfolio_metrics = compute_metrics(portfolio_equity, [t for r in results.values() for t in r.trades], config.timeframe)
    return {"pairs": results, "portfolio_equity": portfolio_equity, "portfolio_metrics": portfolio_metrics}
