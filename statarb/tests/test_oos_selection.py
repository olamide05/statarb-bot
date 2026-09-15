"""
Tests for split_for_oos_pair_selection (statarb/pairs.py) -- the fix for a
real methodology bug found after the first stocks backtest: `backtest`
without --pair used to scan for cointegrated pairs and backtest them on
the SAME price window, which is a form of selection bias (picking winners
using full-sample knowledge, then "testing" only the winners). This
verifies the split actually produces two non-overlapping windows with the
later one strictly held out from the earlier one, and that scanning only
the selection half finds the synthetic pair the way the full smoke test
already expects scan_pairs to.
"""
from __future__ import annotations

import pandas as pd

from statarb.pairs import scan_pairs, split_for_oos_pair_selection
from statarb.synthetic import generate_cointegrated_pair


def test_split_produces_nonoverlapping_windows_with_test_strictly_after_selection():
    prices = generate_cointegrated_pair(n=1000, seed=0, freq="D")

    selection, test = split_for_oos_pair_selection(prices, test_days=300)

    assert not selection.empty and not test.empty
    assert selection.index.max() < test.index.min(), "selection window must end strictly before test window starts"
    assert len(selection) + len(test) == len(prices)
    # the test window should cover roughly the requested number of days
    assert (test.index.max() - test.index.min()).days <= 300


def test_split_selection_half_alone_still_detects_the_synthetic_pair():
    prices = generate_cointegrated_pair(n=1000, seed=0, freq="D")

    selection, _ = split_for_oos_pair_selection(prices, test_days=300)
    result = scan_pairs(selection, pvalue_threshold=0.05)

    assert result.iloc[0]["cointegrated"], (
        "the known-cointegrated synthetic pair should still be detected "
        "using only the (smaller) selection window"
    )


def test_split_handles_empty_input_without_crashing():
    empty = pd.DataFrame(columns=["A", "B"])
    selection, test = split_for_oos_pair_selection(empty, test_days=100)
    assert selection.empty and test.empty
