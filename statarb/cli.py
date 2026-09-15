"""
CLI entry point: scan / backtest / paper.

Usage:
    python -m statarb.cli scan --config config.yaml
    python -m statarb.cli backtest --config config.yaml --top-n 3
    python -m statarb.cli backtest --config config.yaml --pair BTC ETH
    python -m statarb.cli paper --config config.yaml --pair BTC ETH
    python -m statarb.cli paper --config config.yaml --pair BTC ETH --iterations 5   # for testing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .backtest import backtest_portfolio
from .config import Config
from .data import load_universe_prices
from .pairs import scan_pairs
from .paper import run_paper_trading


def _load_prices(config: Config) -> pd.DataFrame:
    """Dispatch to the crypto (ccxt) or stock (yfinance) data loader based
    on config.asset_class. scan_pairs/backtest_portfolio don't care which
    one produced the prices -- they just operate on the resulting wide
    DataFrame -- so this is the only place asset_class needs a branch."""
    if config.asset_class == "stock":
        from .stocks_data import load_stock_universe_prices
        return load_stock_universe_prices(
            config.universe, config.timeframe, config.lookback_days, config.cache_dir,
        )
    elif config.asset_class == "crypto":
        return load_universe_prices(
            config.universe, config.quote, config.exchange, config.timeframe,
            config.lookback_days, config.cache_dir,
        )
    else:
        raise ValueError(f"Unknown asset_class {config.asset_class!r}, expected 'crypto' or 'stock'.")


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_config(args) -> Config:
    return Config.load(args.config)


def cmd_scan(args):
    config = _load_config(args)
    source = "yfinance" if config.asset_class == "stock" else config.exchange
    print(f"Fetching {config.lookback_days}d of {config.timeframe} data for {len(config.universe)} "
          f"{config.asset_class} symbols from {source} ...", file=sys.stderr)
    prices = _load_prices(config)
    result = scan_pairs(prices, pvalue_threshold=config.coint_pvalue_threshold)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(result.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved full ranking to {args.out}", file=sys.stderr)
    n_hits = int(result["cointegrated"].sum())
    print(f"\n{n_hits}/{len(result)} candidate pairs cointegrated at p < {config.coint_pvalue_threshold}",
          file=sys.stderr)


def cmd_backtest(args):
    config = _load_config(args)
    prices = _load_prices(config)

    if args.pair:
        pairs = [tuple(args.pair)]
    else:
        ranked = scan_pairs(prices, pvalue_threshold=config.coint_pvalue_threshold)
        cointegrated = ranked[ranked["cointegrated"]]
        top_n = args.top_n or config.max_concurrent_pairs
        if cointegrated.empty:
            print("No cointegrated pairs found at the configured p-value threshold; "
                  "try --pair to force one, or loosen coint_pvalue_threshold.", file=sys.stderr)
            sys.exit(1)
        pairs = list(cointegrated.head(top_n)[["symbol_a", "symbol_b"]].itertuples(index=False, name=None))
        print(f"Backtesting top {len(pairs)} cointegrated pair(s): {pairs}", file=sys.stderr)

    portfolio = backtest_portfolio(prices, pairs, config)

    for name, res in portfolio["pairs"].items():
        print(f"\n=== {name} ===")
        for k, v in res.metrics.items():
            print(f"  {k}: {v}")
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = name.replace("/", "-")
            res.equity_curve.to_csv(out_dir / f"equity_{safe}.csv", header=["equity"])
            res.trades_frame().to_csv(out_dir / f"trades_{safe}.csv", index=False)

    print("\n=== Portfolio ===")
    for k, v in portfolio["portfolio_metrics"].items():
        print(f"  {k}: {v}")
    if args.out_dir:
        portfolio["portfolio_equity"].to_csv(Path(args.out_dir) / "equity_portfolio.csv", header=["equity"])
        print(f"\nSaved equity curves + trade logs to {args.out_dir}", file=sys.stderr)


def cmd_paper(args):
    config = _load_config(args)
    if config.asset_class == "stock":
        print("`paper` only supports asset_class: crypto for now -- the live polling loop is "
              "ccxt-specific (continuous 24/7 bars) and doesn't yet handle stock market hours/"
              "holidays. Use `scan`/`backtest` for stocks; paper-trading them is a reasonable "
              "next step to add later.", file=sys.stderr)
        sys.exit(1)
    if not args.pair:
        print("`paper` requires --pair COIN_A COIN_B (paper trading runs one pair at a time in this scaffold).",
              file=sys.stderr)
        sys.exit(1)
    coin_a, coin_b = args.pair
    print(f"Starting PAPER trading for {coin_a}/{coin_b} (dry_run={not args.live}). "
          f"No real orders will ever be placed by this command.", file=sys.stderr)
    run_paper_trading(
        config, coin_a, coin_b,
        live=args.live, confirm_live_trading=args.confirm_live_trading,
        iterations=args.iterations,
    )


def build_parser() -> argparse.ArgumentParser:
    # --config/-v are on a shared "parent" parser included in both the top-level parser
    # AND every subcommand parser, so both of these work identically:
    #   statarb --config config.yaml paper --pair BTC ETH
    #   statarb paper --config config.yaml --pair BTC ETH
    # (argparse does NOT do this automatically for a parent-only option placed after a
    # subcommand -- omitting this was a bug that made the second, more natural-looking
    # form fail with "unrecognized arguments".)
    # default=SUPPRESS (not None/False) is required here: with the same dest defined on
    # both the parent and every subparser, an ordinary default on the subparser would
    # silently overwrite a value already set from before the subcommand (e.g.
    # `statarb --config x.yaml paper ...`) back to the default when parsing the
    # subcommand's own arguments. SUPPRESS means "don't touch the namespace if this
    # wasn't given here" -- main() fills in the real defaults afterward.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                         help="Path to a YAML config file (defaults to built-in defaults).")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    p = argparse.ArgumentParser(prog="statarb", description="Crypto statistical arbitrage pairs-trading scaffold.",
                                 parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Rank universe pairs by Engle-Granger cointegration p-value.", parents=[common])
    s.add_argument("--out", default=None, help="Optional CSV path to save the full ranking.")
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("backtest", help="Walk-forward backtest one pair or the top cointegrated pairs.",
                        parents=[common])
    b.add_argument("--pair", nargs=2, metavar=("COIN_A", "COIN_B"), default=None,
                    help="Force a specific pair, e.g. --pair BTC ETH. Default: top cointegrated pairs from scan.")
    b.add_argument("--top-n", type=int, default=None, help="How many top pairs to backtest (default: config.max_concurrent_pairs).")
    b.add_argument("--out-dir", default=None, help="Directory to write equity curve + trade log CSVs.")
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="Run the (simulated) paper-trading loop for one pair.", parents=[common])
    pp.add_argument("--pair", nargs=2, metavar=("COIN_A", "COIN_B"), required=False, default=None)
    pp.add_argument("--iterations", type=int, default=None,
                     help="Stop after N polls instead of looping forever (useful for testing).")
    pp.add_argument("--live", action="store_true",
                     help="Attempt to disable dry-run. Requires --confirm-live-trading too, and even then "
                          "raises NotImplementedError -- see README. Never enables real order placement.")
    pp.add_argument("--confirm-live-trading", action="store_true",
                     help="Required alongside --live as a deliberate second step. Still not enough on its "
                          "own to place real orders -- that code path does not exist in this scaffold.")
    pp.set_defaults(func=cmd_paper)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # fill in real defaults for the SUPPRESS-ed shared options (see build_parser)
    if not hasattr(args, "config"):
        args.config = None
    if not hasattr(args, "verbose"):
        args.verbose = False
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
