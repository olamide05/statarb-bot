# statarb-bot

A crypto statistical arbitrage (cointegration pairs trading) scaffold: find
two coins that historically move together, trade the spread between them
when it diverges, exit when it reverts. Market-neutral by construction —
the bet is on the *relationship* between two prices, not on either one's
direction.

This is a **backtesting + paper-trading scaffold**, not a live-money bot.
Real order execution is not implemented anywhere in this codebase — see
[Before this should ever touch real money](#before-this-should-ever-touch-real-money).

## How it works

1. **Pair discovery** (`scan`): pull OHLCV history for a universe of coins
   (all quoted in USDT by default) via [ccxt](https://github.com/ccxt/ccxt)
   against Binance's public market-data endpoints (no API key needed), then
   run the Engle-Granger two-step cointegration test (`statsmodels`) on every
   pair. Pairs are ranked by p-value; a low p-value is evidence the spread
   between the two prices is stationary (mean-reverting) rather than a
   random walk.

2. **Signal**: for a given pair, fit `price_a = alpha + hedge_ratio * price_b`
   by OLS, and define `spread = price_a - hedge_ratio * price_b`. The signal
   is the rolling z-score of that spread. Enter a position when `|z|` crosses
   an entry threshold (the spread has diverged unusually far from its recent
   mean); exit when `z` reverts back near zero (or hits a stop-loss z).

3. **Walk-forward backtest** (`backtest`): the hedge ratio and spread
   mean/std are fit **only on a rolling train window**, then held fixed
   while trades are simulated on the following, unseen test window. The
   window then rolls forward and refits. This is the *only* backtest mode
   offered — there's no in-sample mode, because fitting the hedge ratio on
   the same data you trade overstates performance (the model gets to "see"
   the exact relationship it's about to trade against). Every simulated
   trade pays fees + slippage (in basis points, both legs, entry and exit).

4. **Paper trading** (`paper`): the same signal logic runs against live
   polled prices, refitting the hedge ratio on the same schedule as the
   backtest, and logs every simulated entry/exit to CSV + stdout. No order
   is ever sent to an exchange — see the safety section below.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .   (installs the `statarb` console command)
```

Requires Python 3.9+. No exchange API key is needed for anything in this
repo — only public market-data endpoints are used.

## Usage

All commands read tunable parameters from `config.yaml` (or your own copy)
via `--config`; omit it to use the built-in defaults in `statarb/config.py`.

```bash
# Rank the configured universe by cointegration p-value
python -m statarb.cli scan --config config.yaml --out pairs.csv

# Walk-forward backtest the top cointegrated pairs (default: top max_concurrent_pairs)
python -m statarb.cli backtest --config config.yaml --out-dir results/

# Walk-forward backtest one specific pair, ignoring the scan ranking
python -m statarb.cli backtest --config config.yaml --pair BTC ETH --out-dir results/

# Paper-trade one pair (simulated only; Ctrl+C to stop, or use --iterations for a bounded run)
python -m statarb.cli paper --config config.yaml --pair BTC ETH --iterations 5
```

If you installed with `pip install -e .`, drop the `python -m statarb.cli`
prefix and run `statarb scan`, `statarb backtest`, `statarb paper` directly.

### Running `paper` across many short sessions (not one long process)

`paper` saves equity, the open position (if any), and the fitted hedge
ratio to `<paper_log_dir>/state_<A>_<B>.json` after every poll, and reloads
it on startup. That means you don't have to keep one process running for a
week straight -- you can invoke `statarb paper --iterations N` repeatedly
(from cron, a scheduler, or just by hand) and it picks up exactly where the
last run left off, instead of resetting equity to `starting_capital` and
forgetting any open position every time. Delete the state file to force a
clean start.

`config_paper.yaml` is a second config, tuned for this: same universe/costs/
thresholds as `config.yaml`, but on 5-minute candles instead of 1-hour ones,
so a short check-in actually has a chance of seeing a new candle close (and
therefore real signal movement) rather than mostly polling into a still-open
hourly bar. Use `config.yaml` (1h) for `scan`/`backtest`; use
`config_paper.yaml` for `paper`:

```bash
python -m statarb.cli paper --config config_paper.yaml --pair BTC ETH --iterations 6
```

### Config knobs (`config.yaml`)

| Key | What it controls |
|---|---|
| `universe`, `quote`, `exchange`, `timeframe`, `lookback_days` | what data gets pulled |
| `coint_pvalue_threshold` | cutoff for "cointegrated" in `scan` |
| `train_window` / `test_window` | walk-forward fit / trade window sizes, in bars |
| `zscore_window`, `min_periods_zscore` | rolling window for the z-score signal |
| `entry_z`, `exit_z`, `stop_z` | signal thresholds |
| `fee_bps`, `slippage_bps` | per-leg, per-trade cost assumptions |
| `starting_capital`, `risk_per_pair_pct`, `max_concurrent_pairs` | sizing |
| `poll_interval_sec`, `paper_log_dir` | paper-trading loop behavior |

## Stocks (`config_stocks.yaml`)

`scan` and `backtest` also work on a stock universe, using the same
Engle-Granger cointegration + walk-forward backtest pipeline -- just a
different data source (`yfinance` instead of `ccxt`) and a different
default universe (ten liquid large caps instead of eight coins):

```bash
python -m statarb.cli scan --config config_stocks.yaml
python -m statarb.cli backtest --config config_stocks.yaml --pair AAPL MSFT
```

Set `asset_class: stock` in a config to switch `scan`/`backtest` onto this
path; `universe` becomes plain tickers (`quote`/`exchange` are ignored).
Only daily/weekly/monthly bars are supported for stocks -- yfinance caps
intraday history at ~60 days, too short for a meaningful train/test window,
so `timeframe` must be `1d`, `1wk`, or `1mo`.

A few things worth knowing before reading much into results here:
- **`paper` does not support stocks yet.** The live polling loop assumes
  continuous 24/7 bars (crypto) and doesn't handle market hours, weekends,
  or holidays -- running it against a stock config exits with an error
  rather than silently doing something wrong. Paper-trading stocks
  properly is a reasonable next step, not yet built.
- **Cross-asset-class pairs (a stock vs. a crypto) are not something this
  project does, on purpose.** They trade on different calendars, currencies,
  and are driven by essentially unrelated fundamentals -- any cointegration
  `scan` found between them would almost certainly be spurious, not a real
  relationship. Stocks get their own universe and their own `scan`/`backtest`
  run, deliberately kept separate from the crypto side.
- **yfinance is a known-flaky dependency in cloud/CI environments.** Yahoo
  requires a cookie/crumb handshake that can fail outright from datacenter
  IPs (GitHub Actions included) -- the equities-data equivalent of the
  Binance-451 issue this project already hit for crypto (see above). If
  `scan --config config_stocks.yaml` comes back with no data, that's the
  first thing to check, not a bug in this project. `statarb/stocks_data.py`
  has a no-auth fallback noted in its docstring (Stooq, daily bars only) if
  yfinance needs to be swapped out the way Binance was.
- **The cost assumptions in `config_stocks.yaml` (`fee_bps: 2.0`,
  `slippage_bps: 2.0`) are a starting guess, not a researched number** --
  equities are typically cheaper to trade than crypto, but the real numbers
  depend entirely on your actual broker. Update them before the backtest
  numbers mean anything.

## Running via GitHub Actions (instead of your own laptop)

`paper` doesn't need to run on your laptop at all -- `.github/workflows/paper-trading.yml`
runs one poll every 15 minutes on GitHub's own runners, which (a) don't depend
on your laptop being on, and (b) have normal outbound internet, so they reach
Binance directly. The state files under `paper_logs/` (the same JSON files
`paper` already writes so it can resume across separate invocations) get
committed back to the repo after every run, so state survives even though
each run is a brand-new, short-lived machine.

**What "more frequent" actually buys you here:** the polling logic already
backfills every closed candle since the last check, not just the latest one
(see `_fetch_new_closed_bars` in `statarb/paper.py`), so infrequent polling
was never silently *losing* data or making the simulation less accurate.
What more frequent polling buys is *promptness* -- the paper trade fires
closer to when a real bot would have fired, and the equity curve reflects
in-between price action instead of only what's true at each check-in.
15 minutes vs. 4 hours is a real improvement on that front; it isn't fixing
an accuracy bug.

**One-time setup:**

1. Create a new repo on GitHub (from your machine or the GitHub website).
   A **public** repo is worth considering: this project never uses an API
   key or secret (it only reads public Binance market data), so there's
   nothing sensitive to expose, and public repos get unlimited free Actions
   minutes. A private repo works too, but the free tier is 2,000 Actions
   minutes/month, and a run every 15 minutes (~96/day) will eat into that
   faster than you'd expect -- widen the cron interval (e.g. `*/30` or
   hourly) if you keep it private.
2. Push this project to it:
   ```bash
   cd statarb-bot
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. On GitHub: **Settings -> Actions -> General -> Workflow permissions ->
   "Read and write permissions"**, then Save. This lets the workflow's
   built-in token push the state-update commits; without it the commit step
   fails with a permissions error. No secrets or API keys need to be added
   anywhere.
4. That's it -- the workflow starts running on its own schedule. Check the
   **Actions** tab on GitHub to watch runs, or click "Run workflow" there to
   trigger one immediately instead of waiting for the next scheduled tick.

**Worth knowing:**
- It still only ever simulates trades -- same `dry_run` safety gate as
  running locally, nothing about moving this to GitHub changes that.
- Every run adds a commit to `paper_logs/`. That's expected and is how state
  persists; if the commit history bothers you, that's a good sign to
  eventually swap state storage for something outside git (a small database,
  a gist, etc.) -- not necessary for this to keep working.
- If you keep the laptop-based scheduled runs going at the same time, you'll
  end up with two independent state files drifting apart (one on your
  machine, one in the repo) rather than one continuous history -- probably
  worth turning the laptop one off once you've confirmed the Actions runs
  are working.
- Only `BTC ETH` is wired up in the workflow as shipped. To paper-trade more
  pairs, either duplicate the job step with different `--pair` values, or
  see `scan`/`backtest` above to pick pairs first.

## Smoke test (no exchange connection needed)

```bash
python -m pytest statarb/tests/test_smoke.py -v
# or: python -m statarb.tests.test_smoke
```

This generates a synthetic cointegrated pair (a geometric random walk plus a
mean-reverting AR(1) spread, by construction) and a synthetic *non*-
cointegrated pair (two independent random walks), then runs the full
pipeline — cointegration test, hedge-ratio fit, rolling z-score, walk-forward
backtest with costs — against both, asserting each stage produces sane,
non-crashing output. It touches no network.

## What's simplified / not production-ready

Being upfront about this matters more than it looking finished:

- **Pair selection uses one full-history scan, not walk-forward.** `scan`
  (and the pair selection inside `backtest` when you don't force `--pair`)
  tests cointegration once over the whole lookback window. The *trading*
  signal (hedge ratio, spread mean/std) **is** refit walk-forward and never
  peeks at test-window data — but which pairs get selected in the first
  place is not re-validated on a rolling basis. A pair that was cointegrated
  historically can stop being so; a production version would re-screen the
  universe periodically and be able to drop/rotate pairs mid-run.
- **Position sizing is simplistic.** Each pair position is sized to a fixed
  fraction of capital (`risk_per_pair_pct`), split across the two legs by
  the fitted hedge ratio. There's no volatility targeting, no correlation-
  aware portfolio sizing across concurrently-held pairs, and no dynamic
  capital reallocation — the portfolio backtest just sums independently-run
  per-pair PnL series.
- **Costs are a flat bps assumption**, not a real order-book/slippage model.
  Real fills depend on book depth, your order size, and market impact,
  especially in an adverse-selection scenario like paper-fake position
  exits.
- **Paper trading polls REST endpoints, not a websocket feed**, and only
  acts on newly-closed candles (same timeframe as the backtest) rather than
  tick data. That keeps its logic consistent with the backtest but means it
  reacts once per bar close, not intraday.
- **No portfolio-level risk controls**: no max drawdown kill-switch, no
  correlation caps across pairs, no handling of an exchange delisting a
  symbol mid-backtest, no handling of funding-rate carry if you were to
  extend this to perpetual futures instead of spot.
- **Cointegration can and does break.** Engle-Granger on a rolling window is
  a reasonable, well-understood starting point, but crypto pairs' long-run
  relationships are not stable the way, say, index-arbitrage relationships
  can be. Treat every backtest number as "how this specific rule would have
  done on this specific history," not a forecast.
- **The Engle-Granger test also has known limitations** — sensitivity to
  which series is the regressor vs. regressand, weak power with strongly
  autocorrelated series, and no correction for testing many pairs at once
  (a Johansen test or multiple-testing correction would be the natural next
  step for a more rigorous version).

## Before this should ever touch real money

This scaffold intentionally has **no code path that can place a real
order** — not behind a flag, not behind an env var, nowhere. `statarb paper
--live` requires an additional `--confirm-live-trading` flag as a
deliberate second step, and even then it raises `NotImplementedError`
pointing back here. Concretely, before turning any of this into a live bot,
you'd want at minimum:

1. **Much longer paper-trading validation** — weeks to months, across
   different market regimes, comparing paper fills/PnL to what the backtest
   predicted for the same period.
2. **Exchange authentication + order routing**, built with care around API
   key scoping (trade-only, no withdrawal permission), idempotent order
   placement, and reconciliation between what you think your position is
   and what the exchange says it is.
3. **Real risk controls**: hard per-pair and portfolio max-loss limits, a
   kill switch, alerting on stale data / dropped connections / model
   staleness, and a manual override to flatten all positions.
4. **Realistic execution cost modeling** — actual order book depth and
   expected slippage for your real position sizes, not a flat bps guess.
5. **A start with real capital you can afford to lose entirely**, sized
   far below what the backtest "optimal" sizing suggests, precisely because
   backtests (even walk-forward ones) tend to overstate live performance.
6. **Ongoing monitoring for regime change** — a periodic re-screen of the
   pair universe, and a defined process for what happens when a previously-
   cointegrated pair stops behaving that way.

None of that is in this repo yet. Treat everything here as a well-scoped
research and paper-trading tool, not an investment product.

## Project layout

```
statarb/
  config.py      # Config dataclass + YAML loader
  data.py        # ccxt OHLCV fetch + parquet cache
  pairs.py       # Engle-Granger cointegration test + hedge ratio fit
  signal.py      # rolling z-score + entry/exit state machine
  backtest.py    # walk-forward backtest engine + metrics
  paper.py       # simulated live paper-trading loop
  synthetic.py   # synthetic data generator (used by the smoke test)
  cli.py         # scan / backtest / paper CLI
  tests/
    test_smoke.py
config.yaml      # default tunable parameters
requirements.txt
pyproject.toml
```
