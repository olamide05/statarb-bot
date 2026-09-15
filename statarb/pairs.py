"""
Pair discovery via Engle-Granger cointegration testing.

For every candidate pair (a, b) in the universe we regress a on b (OLS) to
get a hedge ratio, then run the Engle-Granger test on the residual to check
whether that spread is stationary (i.e. the two series are cointegrated).
Pairs are ranked by p-value: lower = stronger evidence of a stable,
mean-reverting relationship.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass
class PairModel:
    """Result of fitting the hedge ratio / spread relationship on a train window."""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    alpha: float
    spread_mean: float
    spread_std: float


def engle_granger_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger two-step cointegration test p-value for y ~ x."""
    y, x = y.align(x, join="inner")
    if len(y) < 20:
        return 1.0
    _, pvalue, _ = coint(y.values, x.values)
    return float(pvalue)


def fit_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS y = alpha + beta*x on a train window. Returns (beta, alpha)."""
    y, x = y.align(x, join="inner")
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha, beta = model.params[0], model.params[1]
    return float(beta), float(alpha)


def fit_pair(symbol_a: str, symbol_b: str, train_y: pd.Series, train_x: pd.Series) -> PairModel:
    """Fit hedge ratio and spread stats on a train window only (no lookahead)."""
    beta, alpha = fit_hedge_ratio(train_y, train_x)
    spread = train_y - (alpha + beta * train_x)
    return PairModel(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        hedge_ratio=beta,
        alpha=alpha,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=0)) or 1e-8,
    )


def compute_spread(y: pd.Series, x: pd.Series, model: PairModel) -> pd.Series:
    """Apply a fitted PairModel's hedge ratio to (possibly out-of-sample) prices."""
    y, x = y.align(x, join="inner")
    return y - (model.alpha + model.hedge_ratio * x)


def scan_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Test every unordered pair of symbols for cointegration over the full
    price history given. Returns a DataFrame ranked by p-value ascending,
    including pairs above the threshold (flagged via `cointegrated`) so the
    caller can inspect near-misses too.
    """
    symbols = symbols or list(prices.columns)
    rows = []
    for a, b in itertools.combinations(symbols, 2):
        y, x = prices[a], prices[b]
        pvalue = engle_granger_pvalue(y, x)
        beta, alpha = fit_hedge_ratio(y, x)
        spread = y - (alpha + beta * x)
        rows.append({
            "symbol_a": a,
            "symbol_b": b,
            "pvalue": pvalue,
            "hedge_ratio": beta,
            "alpha": alpha,
            "spread_std": float(spread.std(ddof=0)),
            "cointegrated": pvalue < pvalue_threshold,
        })
    result = pd.DataFrame(rows).sort_values("pvalue", ascending=True).reset_index(drop=True)
    return result
