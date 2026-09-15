"""
Synthetic price-series generator for the smoke test (and for anyone who
wants to sanity-check the pipeline without hitting an exchange).

Generates a cointegrated pair by construction: x follows a geometric
(log-space) random walk so it stays strictly positive over long series,
the spread is a stationary mean-reverting AR(1) process, and
y = hedge_ratio * x + spread + offset. Also exposes an independent-walk
generator for a non-cointegrated control case.

Note: an earlier version used an additive random walk clipped at a price
floor (np.maximum(x, 1.0)); over long series a driftless additive walk
will wander non-positive and the clip produced long flat stretches
(zero-variance windows), which broke OLS fitting downstream. Geometric
(multiplicative) walks avoid that by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_cointegrated_pair(
    n: int = 2000,
    seed: int = 42,
    hedge_ratio: float = 1.5,
    offset: float = 50.0,
    x_start: float = 100.0,
    x_log_vol: float = 0.01,
    spread_mean_reversion: float = 0.05,
    spread_noise_std: float = 0.5,
    freq: str = "h",
) -> pd.DataFrame:
    """Return a DataFrame with columns ['A', 'B'] (A is cointegrated with B)."""
    rng = np.random.default_rng(seed)

    log_x_steps = rng.normal(0, x_log_vol, n)
    x = x_start * np.exp(np.cumsum(log_x_steps))

    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - spread_mean_reversion) + rng.normal(0, spread_noise_std)

    y = hedge_ratio * x + spread + offset

    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"A": y, "B": x}, index=idx)


def generate_independent_walks(
    n: int = 2000,
    seed: int = 7,
    start: float = 100.0,
    log_vol: float = 0.01,
    freq: str = "h",
) -> pd.DataFrame:
    """Two independent geometric random walks -- should generally NOT show cointegration."""
    rng = np.random.default_rng(seed)
    a = start * np.exp(np.cumsum(rng.normal(0, log_vol, n)))
    b = start * np.exp(np.cumsum(rng.normal(0, log_vol, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"A": a, "B": b}, index=idx)
