"""
Tests for statarb/adaptive.py -- the adaptive pair-scoring feature added to
address a gap flagged after the first real OOS stocks backtest: a single
out-of-sample window can't distinguish a real cointegration relationship
from a multiple-testing false positive, but a pair's performance across
SEVERAL sequential OOS windows starts to. See adaptive.py's module
docstring for the full rationale.
"""
from __future__ import annotations

import pandas as pd

from statarb.adaptive import (
    PairPerformanceHistory,
    run_walk_forward_selection,
    score_pairs,
    summarize_walk_forward,
    walk_forward_windows,
)
from statarb.config import Config
from statarb.synthetic import generate_cointegrated_pair, generate_independent_walks


def test_score_pairs_with_zero_performance_weight_reproduces_pvalue_ranking():
    ranked = pd.DataFrame({
        "symbol_a": ["A", "C"], "symbol_b": ["B", "D"],
        "pvalue": [0.01, 0.03], "cointegrated": [True, True],
    })
    history = PairPerformanceHistory()
    history.record("C", "D", -1.5)  # even with a terrible track record...

    scored = score_pairs(ranked, history, performance_weight=0.0)
    # ...performance_weight=0 must ignore it entirely and keep the pure p-value order
    assert list(zip(scored["symbol_a"], scored["symbol_b"])) == [("A", "B"), ("C", "D")]


def test_score_pairs_lets_strong_trailing_performance_overtake_a_better_pvalue():
    ranked = pd.DataFrame({
        "symbol_a": ["A", "C"], "symbol_b": ["B", "D"],
        "pvalue": [0.01, 0.03],  # A/B has the statistically stronger p-value
        "cointegrated": [True, True],
    })
    history = PairPerformanceHistory()
    history.record("A", "B", -1.8)  # but has performed badly out-of-sample before
    history.record("A", "B", -1.6)
    history.record("C", "D", 2.1)   # while C/D has performed well
    history.record("C", "D", 2.4)

    scored = score_pairs(ranked, history, performance_weight=0.6)
    # with enough weight on trailing performance, the pair with the worse p-value
    # but the better track record should now rank first
    assert (scored.iloc[0]["symbol_a"], scored.iloc[0]["symbol_b"]) == ("C", "D")


def test_score_pairs_gives_untested_pair_a_neutral_not_punished_rank():
    ranked = pd.DataFrame({
        "symbol_a": ["A", "C", "E"], "symbol_b": ["B", "D", "F"],
        "pvalue": [0.04, 0.03, 0.02],
        "cointegrated": [True, True, True],
    })
    history = PairPerformanceHistory()
    history.record("C", "D", 3.0)   # C/D: great track record
    history.record("E", "F", -3.0)  # E/F: terrible track record
    # A/B: no history at all -- should land BETWEEN the great and terrible pair,
    # not be treated as worst-of-all just for being new
    scored = score_pairs(ranked, history, performance_weight=0.8)
    order = list(zip(scored["symbol_a"], scored["symbol_b"]))
    assert order.index(("A", "B")) < order.index(("E", "F")), (
        "an untested pair should not rank below one with a genuinely bad track record"
    )


def test_walk_forward_windows_are_sequential_and_non_overlapping():
    prices = generate_cointegrated_pair(n=400, seed=0, freq="D")
    windows = walk_forward_windows(prices, selection_days=100, test_days=50, n_windows=3)
    assert len(windows) == 3
    prev_test_end = None
    for selection, test in windows:
        assert selection.index.max() <= test.index.min()
        if prev_test_end is not None:
            assert test.index.min() > prev_test_end
        prev_test_end = test.index.max()


def test_walk_forward_windows_returns_fewer_when_history_runs_out():
    prices = generate_cointegrated_pair(n=150, seed=0, freq="D")
    # 150 days only fits ~1 window at selection=100/test=50; asking for 5 should
    # come back short rather than crash
    windows = walk_forward_windows(prices, selection_days=100, test_days=50, n_windows=5)
    assert 0 < len(windows) < 5


def _small_config(**overrides) -> Config:
    base = dict(
        selection_lookback_days=100, lookback_days=50,
        train_window=15, test_window=10, zscore_window=10, min_periods_zscore=5,
        entry_z=1.5, exit_z=0.5, stop_z=4.0,
        timeframe="1d", coint_pvalue_threshold=0.05,
        max_concurrent_pairs=2,
    )
    base.update(overrides)
    return Config(**base)


def _mixed_universe(n=400):
    coint = generate_cointegrated_pair(n=n, seed=1, hedge_ratio=1.5, freq="D")
    noise = generate_independent_walks(n=n, seed=2, freq="D")
    prices = coint.rename(columns={"A": "A", "B": "B"}).copy()
    prices["C"] = noise["A"].values
    prices["D"] = noise["B"].values
    return prices


def test_run_walk_forward_selection_static_smoke():
    prices = _mixed_universe()
    config = _small_config()
    results = run_walk_forward_selection(prices, config, n_windows=3, top_n=2, adaptive=False)
    assert len(results) == 3
    # the genuinely cointegrated pair should get picked at least once
    all_chosen = {pair for r in results for pair in r.chosen_pairs}
    assert ("A", "B") in all_chosen


def test_run_walk_forward_selection_adaptive_builds_history_and_summarizes():
    prices = _mixed_universe()
    config = _small_config()
    results = run_walk_forward_selection(prices, config, n_windows=3, top_n=2, adaptive=True, performance_weight=0.5)
    assert len(results) == 3

    summary = summarize_walk_forward(results)
    assert summary["windows_with_trades"] >= 0
    assert isinstance(summary["num_trades"], int)
    # cumulative_return_pct should be a real, finite number, not NaN/None, when trades happened
    if summary["num_trades"] > 0:
        assert summary["cumulative_return_pct"] == summary["cumulative_return_pct"]  # not NaN


def test_summarize_walk_forward_handles_all_windows_with_no_cointegrated_pairs():
    from statarb.adaptive import WalkForwardWindowResult
    now = pd.Timestamp("2024-01-01", tz="UTC")
    empty_results = [
        WalkForwardWindowResult(0, now, now, now, now, [], None),
        WalkForwardWindowResult(1, now, now, now, now, [], None),
    ]
    summary = summarize_walk_forward(empty_results)
    assert summary["num_trades"] == 0
    assert summary["cumulative_return_pct"] == 0.0
