"""
Z-score signal and position state machine on top of a fitted PairModel's
spread. Kept separate from backtest.py so the exact same functions drive
both the backtester and the paper-trading loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Causal rolling z-score: at each bar, uses only that bar and earlier ones."""
    mean = spread.rolling(window=window, min_periods=min_periods).mean()
    std = spread.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (spread - mean) / std.replace(0.0, np.nan)
    return z


class Position(Enum):
    FLAT = 0
    LONG_SPREAD = 1   # long symbol_a, short hedge_ratio * symbol_b
    SHORT_SPREAD = -1  # short symbol_a, long hedge_ratio * symbol_b


@dataclass
class SignalState:
    position: Position = Position.FLAT


def next_action(state: SignalState, z: float, entry_z: float, exit_z: float, stop_z: float) -> Optional[str]:
    """
    Pure state-transition function: given the current position and the latest
    z-score, decide what to do. Returns one of:
      "enter_long"  - open LONG_SPREAD from flat
      "enter_short" - open SHORT_SPREAD from flat
      "exit"        - close an open position (mean reversion or stop-loss)
      None          - hold
    Does not mutate `state`; caller applies the transition after acting on it.
    """
    if np.isnan(z):
        return None

    if state.position == Position.FLAT:
        if z <= -entry_z:
            return "enter_long"
        if z >= entry_z:
            return "enter_short"
        return None

    # in a position: check stop-loss first, then mean-reversion exit
    if abs(z) >= stop_z:
        return "exit"
    if state.position == Position.LONG_SPREAD and z >= -exit_z:
        return "exit"
    if state.position == Position.SHORT_SPREAD and z <= exit_z:
        return "exit"
    return None


def apply_action(state: SignalState, action: Optional[str]) -> SignalState:
    if action == "enter_long":
        return SignalState(Position.LONG_SPREAD)
    if action == "enter_short":
        return SignalState(Position.SHORT_SPREAD)
    if action == "exit":
        return SignalState(Position.FLAT)
    return state
