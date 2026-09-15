"""
Synthetic-data smoke test for the full pipeline: cointegration -> signal ->
walk-forward backtest. Uses no network / exchange connection at all, so it
can run anywhere, including CI.

Run with: pytest statarb/tests/test_smoke.py -v
Or standalone: python -m statarb.tests.test_smoke
"""
from __future__ import annotations

import math

from statarb.backtest import walk_forward_backtest
from statarb.config import Config
from statarb.pairs import engle_granger_pvalue, fit_hedge_ratio, scan_pairs
from statarb.signal import rolling_zscore
from statarb.synthetic import generate_cointegrated_pair, generate_independent_walks


def test_cointegration_detects_synthetic_pair():
    prices = generate_cointegrated_pair(n=1500, seed=42, hedge_ratio=1.5)
    pvalue = engle_granger_pvalue(prices["A"], prices["B"])
    assert pvalue < 0.05, f"Expected the constructed-cointegrated pair to test significant, got p={pvalue}"

    beta, alpha = fit_hedge_ratio(prices["A"], prices["B"])
    assert math.isclose(beta, 1.5, rel_tol=0.15), f"Recovered hedge ratio {beta} far from true 1.5"


def test_scan_pairs_runs_and_ranks():
    coint = generate_cointegrated_pair(n=1000, seed=1, hedge_ratio=2.0)
    indep = generate_independent_walks(n=1000, seed=2)
    # build a 3-symbol universe: A,B cointegrated; C independent of both
    prices = coint.copy()
    prices["C"] = indep["A"].values
    result = scan_pairs(prices, pvalue_threshold=0.05)
    assert len(result) == 3  # C(3,2) = 3 pairs
    assert set(result.columns) >= {"symbol_a", "symbol_b", "pvalue", "hedge_ratio", "cointegrated"}
    # the true cointegrated pair should rank first (lowest p-value)
    top = result.iloc[0]
    assert {top["symbol_a"], top["symbol_b"]} == {"A", "B"}


def test_rolling_zscore_shape_and_nan_warmup():
    prices = generate_cointegrated_pair(n=500, seed=3)
    spread = prices["A"] - 1.5 * prices["B"]
    z = rolling_zscore(spread, window=50, min_periods=20)
    assert len(z) == len(spread)
    assert z.iloc[:19].isna().all()  # not enough data yet
    assert z.iloc[100:].notna().all()  # fully warmed up


def test_walk_forward_backtest_runs_end_to_end():
    prices = generate_cointegrated_pair(n=2500, seed=42, hedge_ratio=1.5, spread_noise_std=0.6)
    config = Config(
        train_window=300, test_window=100, zscore_window=100, min_periods_zscore=30,
        entry_z=1.5, exit_z=0.5, stop_z=4.0,
        fee_bps=10, slippage_bps=5, starting_capital=10_000.0, risk_per_pair_pct=0.2,
        timeframe="1h",
    )
    result = walk_forward_backtest(prices, "A", "B", config)

    assert len(result.equity_curve) > 0
    assert result.equity_curve.isna().sum() == 0
    # equity should never be wildly negative-infinite / nan out of a sane synthetic run
    assert result.equity_curve.iloc[-1] > 0

    required_metrics = {"total_return_pct", "sharpe", "max_drawdown_pct", "num_trades", "win_rate_pct", "final_equity"}
    assert required_metrics.issubset(result.metrics.keys())

    # a mean-reverting synthetic spread over 2500 bars with these thresholds should trade at least once
    assert result.metrics["num_trades"] >= 1

    trades_df = result.trades_frame()
    assert len(trades_df) == result.metrics["num_trades"]
    if len(trades_df):
        assert (trades_df["costs"] >= 0).all()


def test_walk_forward_backtest_handles_non_cointegrated_pair_without_crashing():
    prices = generate_independent_walks(n=1500, seed=99)
    config = Config(train_window=300, test_window=100, zscore_window=100, min_periods_zscore=30,
                     timeframe="1h")
    # should not raise even though the pair isn't really cointegrated -- it just may
    # trade poorly / rarely, which is exactly what the walk-forward backtest should reveal
    result = walk_forward_backtest(prices, "A", "B", config)
    assert len(result.equity_curve) > 0


if __name__ == "__main__":
    tests = [
        test_cointegration_detects_synthetic_pair,
        test_scan_pairs_runs_and_ranks,
        test_rolling_zscore_shape_and_nan_warmup,
        test_walk_forward_backtest_runs_end_to_end,
        test_walk_forward_backtest_handles_non_cointegrated_pair_without_crashing,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("\nAll smoke tests passed.")
