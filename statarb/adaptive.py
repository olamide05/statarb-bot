"""
Adaptive pair scoring: pair selection that learns from its OWN realized
out-of-sample track record over time, instead of ranking candidate pairs by
cointegration p-value alone.

Why this exists: the OOS pair-selection fix (see pairs.split_for_oos_pair_selection)
made selection honest about lookahead -- pairs are picked without seeing the
data they're graded on -- but it doesn't address a separate risk flagged
after the first real stocks backtest: with dozens of candidate pairs tested
at p<0.05, some will look cointegrated by chance alone (multiple-testing
false discovery), with no real economic relationship behind them. A single
OOS window can't tell a real relationship from a lucky one -- but a pair's
performance ACROSS several sequential OOS windows can start to: a real
relationship should keep paying off; a fluke usually won't repeat.

Design choice, deliberately: this is NOT a black-box ML model predicting
prices. It's an interpretable re-ranking of scan_pairs()' own output, using
only each pair's own past realized performance (never anything from the
window currently being selected for). That keeps it auditable and hard to
silently overfit, and keeps the "prove it before trusting it" bar this
project holds everything else to -- see compare_adaptive_vs_static, which
runs the exact same walk-forward with performance_weight=0 (pure p-value,
identical to the pre-existing behavior) side by side with performance_weight>0
so you can see whether the adaptive version is actually better, not just
different.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import backtest_portfolio
from .config import Config
from .pairs import scan_pairs

logger = logging.getLogger("statarb.adaptive")


def _pair_key(symbol_a: str, symbol_b: str) -> Tuple[str, str]:
    return tuple(sorted((symbol_a, symbol_b)))  # type: ignore[return-value]


@dataclass
class PairPerformanceHistory:
    """Trailing realized-Sharpe history per pair, built up ONE walk-forward
    window at a time. record() is only ever called by run_walk_forward_selection
    AFTER a window's backtest has completed -- so at the moment a pair is
    scored for window i, its history only contains windows < i. That's what
    keeps this out-of-sample: a pair's future performance never leaks into
    its own selection."""

    _history: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)

    def record(self, symbol_a: str, symbol_b: str, sharpe: Optional[float]) -> None:
        if sharpe is None or sharpe != sharpe:  # None or NaN
            return
        self._history.setdefault(_pair_key(symbol_a, symbol_b), []).append(float(sharpe))

    def trailing_ewma(self, symbol_a: str, symbol_b: str, span: int = 3) -> Optional[float]:
        """Exponentially-weighted mean of this pair's past realized Sharpes,
        recent windows weighted more -- None if the pair has never been
        traded before (cold start; see score_pairs for how that's handled)."""
        vals = self._history.get(_pair_key(symbol_a, symbol_b))
        if not vals:
            return None
        return float(pd.Series(vals).ewm(span=span, adjust=False).mean().iloc[-1])

    def n_observations(self, symbol_a: str, symbol_b: str) -> int:
        return len(self._history.get(_pair_key(symbol_a, symbol_b), []))


def score_pairs(
    ranked: pd.DataFrame,
    history: PairPerformanceHistory,
    performance_weight: float = 0.35,
) -> pd.DataFrame:
    """Re-rank scan_pairs()-shaped output (needs symbol_a/symbol_b/pvalue
    columns) by blending statistical evidence with trailing realized
    performance. Returns a copy sorted best-first (ascending score).

    performance_weight=0 reproduces the original p-value-only ranking
    exactly (this is the 'static' baseline compare_adaptive_vs_static runs
    against) -- pvalue_rank is unchanged, performance_rank contributes 0.
    A pair with no track record yet gets the MEDIAN performance rank among
    pairs that do have one (neutral, not punished) so new/untested pairs
    aren't locked out just for lacking history; with zero pairs having any
    history at all (e.g. window 0 of any run), this collapses to pure
    p-value ranking regardless of performance_weight.

    Ranks (not raw values) are blended rather than raw p-value + raw Sharpe,
    since those are on incommensurate scales and combining them directly
    would make the relative weighting depend arbitrarily on units.
    """
    df = ranked.copy().reset_index(drop=True)
    df["pvalue_rank"] = df["pvalue"].rank(method="average")

    trailing = df.apply(lambda r: history.trailing_ewma(r["symbol_a"], r["symbol_b"]), axis=1)
    df["trailing_sharpe"] = trailing

    perf_rank_raw = df["trailing_sharpe"].rank(method="average", ascending=False)
    if perf_rank_raw.notna().any():
        perf_rank = perf_rank_raw.fillna(perf_rank_raw.median())
    else:
        perf_rank = df["pvalue_rank"]  # nobody has history yet -- nothing to blend

    df["performance_rank"] = perf_rank
    df["score"] = (1 - performance_weight) * df["pvalue_rank"] + performance_weight * df["performance_rank"]
    return df.sort_values("score", ascending=True).reset_index(drop=True)


def walk_forward_windows(
    prices: pd.DataFrame,
    selection_days: int,
    test_days: int,
    n_windows: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield up to n_windows (selection, test) slices, walking forward
    through time: window i's selection covers the `selection_days`
    immediately before its test period, and test period i+1 starts exactly
    where test period i ended (so test periods never overlap each other,
    same non-lookahead guarantee as pairs.split_for_oos_pair_selection,
    just repeated several times across history instead of once).

    Returns fewer than n_windows if the price history runs out -- callers
    should treat that as informational, not an error (a short window is
    still a valid, if smaller, out-of-sample test)."""
    if prices.empty:
        return []
    idx = prices.index
    windows: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    test_start = idx.min() + pd.Timedelta(days=selection_days)
    for _ in range(n_windows):
        test_end = test_start + pd.Timedelta(days=test_days)
        sel_start = test_start - pd.Timedelta(days=selection_days)
        selection = prices[(prices.index >= sel_start) & (prices.index <= test_start)]
        test = prices[(prices.index > test_start) & (prices.index <= test_end)]
        if selection.empty or test.empty:
            break
        windows.append((selection, test))
        test_start = test_end
    return windows


@dataclass
class WalkForwardWindowResult:
    window_index: int
    selection_start: pd.Timestamp
    selection_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_pairs: List[Tuple[str, str]]
    portfolio: Optional[dict]  # backtest_portfolio() output, None if no pairs were cointegrated


def run_walk_forward_selection(
    prices: pd.DataFrame,
    config: Config,
    n_windows: int,
    top_n: int,
    adaptive: bool,
    performance_weight: float = 0.35,
) -> List[WalkForwardWindowResult]:
    """Walk forward through n_windows sequential out-of-sample periods.
    Each window: scan for cointegrated pairs on the selection slice, pick
    the top_n (by p-value if adaptive=False, by score_pairs' blended score
    if adaptive=True), backtest ONLY the held-out test slice, then --
    adaptive mode only -- record each chosen pair's realized Sharpe from
    THIS window into the running history, so it can influence window i+1's
    scoring. adaptive=False never touches history at all, which is what
    makes it a clean baseline: identical mechanics, just no learning."""
    windows = walk_forward_windows(prices, config.selection_lookback_days, config.lookback_days, n_windows)
    if len(windows) < n_windows:
        logger.warning(
            "Requested %d walk-forward windows but only %d fit in the available price history "
            "(%s..%s). Pull more lookback_days or reduce --windows.",
            n_windows, len(windows), prices.index.min(), prices.index.max(),
        )

    history = PairPerformanceHistory()
    results: List[WalkForwardWindowResult] = []
    for i, (selection, test) in enumerate(windows):
        ranked = scan_pairs(selection, pvalue_threshold=config.coint_pvalue_threshold)
        cointegrated = ranked[ranked["cointegrated"]]

        if cointegrated.empty:
            results.append(WalkForwardWindowResult(
                i, selection.index.min(), selection.index.max(),
                test.index.min(), test.index.max(), [], None,
            ))
            continue

        if adaptive:
            scored = score_pairs(cointegrated, history, performance_weight)
            chosen = list(scored.head(top_n)[["symbol_a", "symbol_b"]].itertuples(index=False, name=None))
        else:
            chosen = list(cointegrated.head(top_n)[["symbol_a", "symbol_b"]].itertuples(index=False, name=None))

        portfolio = backtest_portfolio(test, chosen, config)

        if adaptive:
            for a, b in chosen:
                res = portfolio["pairs"].get(f"{a}/{b}")
                if res is not None:
                    history.record(a, b, res.metrics.get("sharpe"))

        results.append(WalkForwardWindowResult(
            i, selection.index.min(), selection.index.max(),
            test.index.min(), test.index.max(), chosen, portfolio,
        ))
    return results


def summarize_walk_forward(results: List[WalkForwardWindowResult]) -> dict:
    """Aggregate metrics across all windows. Windows are separate, non-
    contiguous backtests (each restarts at config.starting_capital), so
    there's no single real equity curve to report -- cumulative_return_pct
    approximates one by COMPOUNDING each window's own return sequentially,
    which is a reasonable estimate of "what if you kept re-investing the
    same capital each window" but is not a substitute for an actual
    continuous backtest. num_trades/win_rate_pct are pooled across all
    windows' real trades (those numbers ARE exact, not approximated)."""
    trades = []
    per_window_returns = []
    sharpes = []
    for r in results:
        if r.portfolio is None:
            continue
        trades.extend(t for res in r.portfolio["pairs"].values() for t in res.trades)
        pm = r.portfolio["portfolio_metrics"]
        if pm.get("total_return_pct") is not None:
            per_window_returns.append(pm["total_return_pct"] / 100.0)
        if pm.get("sharpe") is not None:
            sharpes.append(pm["sharpe"])

    cumulative = 1.0
    for r in per_window_returns:
        cumulative *= (1.0 + r)
    cumulative_return_pct = round((cumulative - 1.0) * 100, 3)

    wins = [t for t in trades if t.net_pnl > 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else float("nan")

    return {
        "windows_with_trades": len(per_window_returns),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2) if win_rate == win_rate else None,
        "avg_window_sharpe": round(float(np.mean(sharpes)), 3) if sharpes else None,
        "cumulative_return_pct": cumulative_return_pct,
    }
