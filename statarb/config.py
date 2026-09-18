"""
Tunable configuration for the statarb pipeline.

Loaded from a YAML file (see config.yaml for the shipped defaults) and/or
overridden by CLI flags. Everything a user is likely to want to tune lives
here rather than being hardcoded in the algorithm modules.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    # --- Data ---
    asset_class: str = "crypto"        # "crypto" (ccxt) or "stock" (yfinance) -- picks the
                                        # data loader in cli.py. Everything below `universe`
                                        # means the same thing for either: `universe` is coin
                                        # symbols for crypto (combined with `quote`) or plain
                                        # tickers for stocks (`quote`/`exchange` are ignored).
    exchange: str = "binance"          # any ccxt id; public OHLCV only, no API key required.
                                        # Ignored when asset_class == "stock".
    quote: str = "USDT"                # quote currency all universe symbols are traded against.
                                        # Ignored when asset_class == "stock".
    universe: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "LTC", "AVAX",
    ])
    timeframe: str = "1h"              # ccxt timeframe string for crypto; for stocks only
                                        # "1d"/"1wk"/"1mo" are supported (see stocks_data.py --
                                        # yfinance intraday history is capped too short to be
                                        # useful for a train/test window).
    lookback_days: int = 180           # history to pull for `scan` / `backtest`
    cache_dir: str = ".cache/ohlcv"    # local cache of fetched OHLCV (parquet)

    # When > 0, `backtest` (with no explicit --pair) picks pairs using an
    # EARLIER window of this many days, then backtests ONLY the later
    # `lookback_days` window -- which those picked pairs never influenced.
    # Default 0 preserves the original behavior (scan and backtest share
    # one window) for backward compatibility -- crypto configs don't set
    # this. See pairs.split_for_oos_pair_selection for why it matters.
    selection_lookback_days: int = 0

    # --- Pair discovery (Engle-Granger cointegration) ---
    coint_pvalue_threshold: float = 0.05

    # --- Walk-forward backtest / signal windows (in bars, not days) ---
    train_window: int = 720            # bars used to fit hedge ratio each roll
    test_window: int = 168             # bars traded out-of-sample before refit
    zscore_window: int = 168           # trailing window for the rolling z-score
    min_periods_zscore: int = 48       # min bars before a z-score is considered valid

    # --- Signal thresholds ---
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0                # hard stop-loss on the spread z-score

    # --- Costs (applied per leg, per trade, i.e. paid on entry AND exit) ---
    fee_bps: float = 10.0              # taker fee, basis points of notional
    slippage_bps: float = 5.0          # assumed slippage, basis points of notional

    # --- Capital / risk ---
    starting_capital: float = 10_000.0
    risk_per_pair_pct: float = 0.10    # fraction of capital allocated to one open pair position
    max_concurrent_pairs: int = 3      # top-N pairs (by p-value) traded together in `backtest`/`paper`

    # --- Paper trading ---
    poll_interval_sec: int = 60
    paper_log_dir: str = "paper_logs"

    # --- Adaptive pair scoring (see adaptive.py) ---
    # Blends each candidate pair's cointegration p-value with its own
    # trailing realized Sharpe from past out-of-sample windows (0 = no
    # track record yet -> falls back to neutral, i.e. pure p-value for that
    # pair). Only used by the `walkforward` CLI command right now, not by
    # `backtest`/`paper` -- see README for why this needs to prove itself
    # across multiple OOS windows before being trusted as the default.
    adaptive_performance_weight: float = 0.35

    # --- Crypto trade-flow ("big trader activity") monitoring, crypto only ---
    # See flow.py's module docstring for what this is (an exchange trade-
    # tape proxy, NOT on-chain wallet tracking) and why. log_flow_signal
    # just records snapshots to paper_logs/flow_<SYMBOL>.csv every poll --
    # pure monitoring, doesn't touch trading. flow_filter_enabled additionally
    # lets a strongly-opposing flow imbalance skip a trade entry -- defaults
    # to False because there isn't enough logged history yet to know if that
    # helps; flip it on once there's real evidence, not before.
    log_flow_signal: bool = False
    flow_filter_enabled: bool = False
    flow_large_trade_pctile: float = 0.95
    flow_veto_imbalance: float = 0.6

    # --- Safety ---
    # dry_run is the ONLY supported mode: this scaffold never places real
    # orders. The flag and the confirm string below exist so that the CLI
    # has the "deliberate extra step" interface the spec calls for, wired
    # up in cli.py / paper.py, even though flipping it currently raises
    # NotImplementedError rather than silently trading live.
    dry_run: bool = True

    @staticmethod
    def load(path: Optional[str] = None) -> "Config":
        """Load config from YAML, falling back to defaults for any missing keys."""
        if path is None:
            return Config()
        data = yaml.safe_load(Path(path).read_text()) or {}
        known_fields = {f.name for f in dataclasses.fields(Config)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")
        return Config(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
