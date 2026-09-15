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
    exchange: str = "binance"          # any ccxt id; public OHLCV only, no API key required
    quote: str = "USDT"                # quote currency all universe symbols are traded against
    universe: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "LTC", "AVAX",
    ])
    timeframe: str = "1h"              # ccxt timeframe string
    lookback_days: int = 180           # history to pull for `scan` / `backtest`
    cache_dir: str = ".cache/ohlcv"    # local cache of fetched OHLCV (parquet)

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
