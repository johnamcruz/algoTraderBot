# algoTraderBot — multi-strategy AI futures bot (3-min, TopstepX)

A live TopstepX bot that trades **mechanical entries graded by AI**, with a
**reinforcement-learned trailing exit** — for 3-min futures (`NQ`, `ES`, `RTY`,
`YM`, `GC`).

> ⚠️ **Educational — live mode places LIVE orders.** Run it on a
> practice/evaluation account first. (Backtests place **no** orders, but still
> need credentials — contract specs are fetched from the broker API.)

---

## Getting started

### 1. Requirements

- **Python 3.10+** and **git** (dependencies install the public
  [`futures_foundation`](https://github.com/johnamcruz/Futures-Foundation-Model)
  library from GitHub).
- **Internet** — downloads the ~45 MB `amazon/chronos-bolt-tiny` checkpoint on
  first run, and the bot reads contract specs from the broker API at startup.
- A **TopstepX account + API key** — required for **both live and backtest**.
  Tick size / tick value come from the broker API (`/Contract/search`); there is
  no offline mode. Backtests still use **local CSV bars** — only the contract
  specs come from the API.

### 2. Install

```bash
git clone https://github.com/johnamcruz/algoTraderBot.git
cd algoTraderBot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> **macOS note:** if you hit a segfault (the torch/xgboost OpenMP clash), prefix
> commands with `KMP_DUPLICATE_LIB_OK=TRUE`, e.g.
> `KMP_DUPLICATE_LIB_OK=TRUE python bot.py …`.

### 3. Add your credentials (`.env`)

Credentials live in a **gitignored `.env`** file. Copy the template:

```bash
cp .env.example .env
```

Open `.env` and fill in your TopstepX details:

```ini
TOPSTEPX_USERNAME=your_login      # your TopstepX login
TOPSTEPX_API_KEY=your_api_key     # API KEY from the dashboard — NOT your password
TOPSTEPX_ACCOUNT=                 # blank = first tradable account, or an id/name
```

- Get the **API key** from your TopstepX dashboard (Settings → API). It is a
  key, not your account password.
- Leave `TOPSTEPX_ACCOUNT` blank to use your first tradable account — the bot
  prints every account on startup, so you can copy an **id or name** here to pin
  a specific one (do this to be sure it's your practice account).
- Real environment variables (`export TOPSTEPX_API_KEY=…`) override `.env` if
  set — handy for CI or secrets managers.

**Verify the install** with a short backtest (uses your creds for the contract
spec; the first run also downloads the Chronos checkpoint, so give it a minute):

```bash
python bot.py --backtest --symbol NQ --start 2026-05-23 --end 2026-05-25
```

You should see candles/signals scroll past and a `BACKTEST NQ | trades=… win=…`
summary at the end. If you get that, you're ready.

### 4. Run (live)

```bash
python bot.py                          # config defaults
python bot.py --strategy ema           # run one strategy
python bot.py --strategy ema keltner bos   # run several (highest proba wins)
python bot.py --size 3                  # fixed: 3 contracts per trade
python bot.py --risk 500                # risk-based: ~$500 risked per trade
python bot.py --risk 500 --max-contracts 5
python bot.py --proba-floor 0.45        # only take entries graded ≥ 0.45 confidence
```

**Strategies** (`--strategy` overrides `config.ACTIVE_STRATEGIES`; pick one or
several — when more than one fires on a bar, the highest-proba signal wins):

| name | entry |
|---|---|
| `supertrend` | SuperTrend flip (period 10, mult 3.0) |
| `ema` | 9/20 EMA crossover, gated to ADX ≥ 18 |
| `keltner` | Keltner-channel breakout, gated to ADX ≥ 20 |
| `bos` | break of the last confirmed swing (break of structure) |
| `orb` | 15-min opening-range breakout (09:30 ET), gated to ADX ≥ 18 |

On startup the bot prints your tradable accounts and a banner —
`✅ <account> | <contract> | 3-min | [ema] | conf≥0.35 | exit: PPO stop-reprice |
size: fixed 1`. **Confirm it picked your practice/eval account.** From there,
every candle, every graded signal (with a `TAKE`/`skip` flag), each entry, and
every stop move are logged to the console **and** `log/bot.log`. Stop with
`Ctrl-C`.

Pick strategies and exit shaping in `config.py`:

```python
ACTIVE_STRATEGIES = ["ema"]      # any of: supertrend, ema, keltner, bos, orb (or --strategy)
PROBA_FLOOR       = 0.35         # min entry confidence (or pass --proba-floor)
ACTIVATE_R        = 2.0          # hold the initial 1R stop until +2R, then trail
GIVEBACK_R        = 0.75         # once trailing, give back ≤ this R from the peak
USE_PPO_EXIT      = True         # False → simple fixed-RR bracket instead
```

**Position sizing** — use a **fixed size** *or* **risk-based sizing** (not both).
With `--risk` (or `config.RISK_PER_TRADE`), contracts are sized from the stop so
each trade risks roughly the same dollars:

```
contracts = min(MAX_CONTRACTS, max(1, floor(risk_$ / (stop_ticks × tick_value))))
```

`tick_value` is read from the broker contract (`/Contract/search` → `tickValue`,
e.g. NQ ≈ $5/tick, MNQ ≈ $0.50/tick). A tighter stop ⇒ more contracts, a wider
stop ⇒ fewer — so dollar risk stays roughly constant, which pairs naturally with
the 0.5×ATR stop. **Micros** (MNQ, MES, MGC, M2K, MYM) just work — same models
and bars as their parent, sized at the micro's smaller `tickValue`.

### 5. Backtest (local bars, live specs)

Backtesting runs the **exact live logic** — same strategies, grading, and PPO
trailing exit — over a local CSV, with a simulated broker filling
entries/stops/trailing against history. Contract specs (tick size / value) are
still looked up from the broker API, so credentials are required.

```bash
# one month of NQ
python bot.py --backtest --symbol NQ --start 2026-05-01 --end 2026-06-01

# a micro and a different ticker
python bot.py --backtest --symbol MNQ --start 2026-05-01 --end 2026-06-01
python bot.py --backtest --symbol ES --risk 500
```

- `--symbol` reads `data/<symbol>_3min.csv` (ships with NQ, ES, RTY, YM, GC);
  **micros use their parent's bars** (MNQ → NQ) at the micro's tick value.
- `--start` / `--end` are `YYYY-MM-DD` (**start inclusive, end exclusive**); omit
  either to run from the warmup point / to the end of the file.
- `--size`, `--risk`, `--proba-floor` and the `config.py` knobs all apply, so you
  can A/B a setting by re-running.

It prints a summary — trades, win rate, mean/sum R, profit factor, plus MFE and
per-strategy / per-exit breakdowns — and writes every trade to
`log/backtest_<symbol>.csv`. Entries fill at the signal bar's close; when a bar
straddles both stop and target the stop is assumed first. Grading embeds each
signal through Chronos, so longer ranges take a few minutes.

---

## How it works

```
each bar ─► every active strategy detects its entry
        ─► its model grades the signal  →  proba = P(win)
        ─► best signal with proba ≥ floor is taken (highest proba wins)
        ─► one shared Chronos embedding per bar feeds every strategy's grade
        ─► PPO policy trails the stop bar-by-bar until exit
```

Each strategy is a thin signal generator paired with its own Chronos+XGBoost
model; the model decides *which* signals to take, and a PPO policy decides *when
to get out*. The entry models are trained on multiple 3-min futures (NQ, ES, RTY,
YM, GC) and generalize across them; the framework is ticker- and broker-agnostic.

- **Entry** — when flat, every active strategy gets a chance to `detect()` +
  `grade()`. Signals with `proba ≥ PROBA_FLOOR` are candidates; the **highest
  proba wins** (one position per contract). The trade enters at market with a
  protective stop at `0.5×ATR(20)` — exactly how the models scored the trade.
- **Exit** — each bar the PPO policy (`models/rl_trail_exit/`) reads the open
  trade's R-state (unrealized R, MFE, stop distance, ATR/risk, time, momentum) —
  **strategy-agnostic**, it never sees how the trade was entered, so one policy
  fits every strategy on the standard 0.5×ATR(20) stop. It computes a
  trailing-stop level and **reprices the live stop to it via `/Order/modify`**
  (ratcheting only in your favor). Two knobs shape it:
  **`ACTIVATE_R`** (hold the initial 1R stop until the peak reaches +2R, so
  winners survive early pullbacks) and **`GIVEBACK_R`** (once trailing, the stop
  never sits more than 0.75R below the running peak). So a trade risks 1R, and
  once it's up +2R it locks in ≥ +1.25R and rides, giving back ≤ 0.75R from the
  best point. The policy forward-pass is pure numpy, so the bot never loads
  torch/SB3 next to xgboost.
- **Trailed-stop enforcement (intra-bar).** The peak is tracked from each bar's
  favorable extreme (not just the close), and if a bar's *unfavorable* wick
  crosses the trailed stop the bot **closes at market** (`/Position/closeContract`)
  rather than rely on a resting broker stop — which the broker rejects ("Invalid
  stop price") when a fast reversal puts the lock level on the wrong side of the
  market. This fixes a bug where a big winner could ride all the way back to the
  initial −1R stop because every stop-modify was rejected. Stops are also
  direction-aware tick-snapped (floor longs / ceil shorts) so rounding never
  lands them on the wrong side.

### Order safety (live)

Trading real orders has sharp edges; these are handled so a desync can't leave an
unmanaged position. Each is covered by a test (see [Tests](#tests)):

- **Signed bracket ticks.** SL/TP distances are signed relative to the fill — a
  long's stop is *negative* ticks (below), a short's *positive* (above), TP the
  mirror — clamped to the 4-tick broker minimum. (Fixes *"Invalid stop loss ticks
  (57). Ticks should be less than zero when longing."*)
- **No orphaned brackets.** A market close (`/Position/closeContract`) does **not**
  fire the OCO, so the broker leaves the protective stop working. `close_position`
  therefore sweeps and cancels every resting order for the contract — otherwise a
  stray stop could later fill and open a brand-new **naked** position the bot never
  opened (seen live: a +0.63R short close left its buy-stop, which filled into a
  naked long at the exact stop price).
- **Mid-session reconcile.** A flat account should have no resting orders, so on
  every flat bar the bot cancels any strays (an orphaned bracket, a missed exit, a
  manual order). This is the general safety net: it keeps the account in sync even
  if a close-time cancel was missed, and is what stops the naked-position scenario
  above from persisting.
- **ORB session gate.** The 09:30-ET opening range stays mathematically active
  until midnight ET; ORB entries are gated to the RTH window `[~09:45, 16:00)` ET
  (`ORB_CLOSE_MIN`) so the bot doesn't take stale overnight breakouts of the
  morning range.

## Architecture

Small, single-responsibility modules:

| file | responsibility |
|---|---|
| `bot.py` | entry point — `handle_bar` (detect → grade → enter → trail) + live loop + CLI |
| `config.py` | **all settings**: strategies, sizing, exit shaping, strategy params |
| `broker_base.py` | `BrokerClient` / `OrderRouter` — the broker **interface** |
| `broker.py` | `TopstepXClient` (a `BrokerClient`) over the ProjectX Gateway API + `make_broker()` |
| `sim_broker.py` | `SimBroker` (an `OrderRouter`) — fills/stops/trailing against a CSV for backtests |
| `backtest.py` | drives `handle_bar` over history with date-range selection |
| `indicators.py` | SuperTrend / ATR / ADX / EMA / Keltner / swings / opening range |
| `embedder.py` / `embed_worker.py` | warm Chronos embedding worker — model loaded once per session |
| `strategies/` | the pluggable strategies (one file each) + shared base |
| `exit_manager.py` | PPO trailing-stop management for an open position |
| `logsetup.py` | logging to `log/bot.log` |
| `trail_exit_env.py` | PPO training environment + torch-free policy loader |
| `train_ppo_exit.py` | trains the trailing-exit policy |
| `models/` | the entry models + the trailing-exit policy |

```
strategies/                 models/
  base.py   Strategy ABC       supertrend_chronos.joblib     entry model (SuperTrend)
  supertrend.py → supertrend   ema_cross_chronos.joblib      entry model (EMA cross)
  ema_cross.py  → ema          keltner_adx_chronos.joblib    entry model (Keltner)
  keltner.py    → keltner      bos_chronos.joblib            entry model (BOS)
  bos.py        → bos          orb_chronos.joblib            entry model (ORB)
  orb.py        → orb          ffm_feature_columns.json      FFM feature order
                               rl_trail_exit/ppo_trail_exit.npz   trailing-exit policy
```

The bot depends only on the public **`futures_foundation`** library (Chronos
embedding + the model head classes + indicator primitives) — no proprietary
code. The joblib bundles run **inference directly**; the FFM feature block is
computed live via `futures_foundation.features.derive_features`.

**Embeddings stay warm.** Chronos runs in a persistent subprocess
(`embed_worker.py`) that loads the model **once per session** — torch isolated
from xgboost, model never reloaded. A grade drops from ~3–4 s (cold reload each
call) to ~0.03 s, so backtests/retrains run in minutes and live signal bars are
near-instant. Falls back to the one-shot library path if the worker can't start.

**Adding a strategy** = one new file in `strategies/` implementing `_fired()` /
`_hand_features()`, plus its joblib model in `models/`, then register it in
`strategies/__init__.py`. The strategy-agnostic exit applies automatically — no
exit work per strategy. Five ship today (`supertrend`, `ema`, `keltner`, `bos`, `orb`).

**Adding a broker** = implement `BrokerClient` (`broker_base.py`) in a new module
and add a case to `broker.make_broker()` + `config.BROKER`. The bar loop, sizing,
and exit all go through that interface, so nothing else changes — e.g. a Rithmic
client would just provide the same account / market-data / order methods.

## Retrain the trailing exit (optional)

```bash
python bot.py --retrain-exit          # full retrain, then exit
python bot.py --retrain-exit --quick  # fast smoke retrain
python train_ppo_exit.py              # same thing, standalone
```

Catalogs a representative set of entry points in `data/NQ_3min.csv`, keeps the
ones the bot would enter (`proba ≥ 0.35`, cached in `proba_cache.npz`), simulates
each trade from the live **0.5×ATR(20) stop** with the `ACTIVATE_R`/`GIVEBACK_R`
shaping while the agent learns the trail, then benchmarks vs fixed-RR /
constant-trail baselines and writes the policy into `models/rl_trail_exit/`. The
exit is strategy-agnostic (it only sees the trade's R-state), so the same policy
serves every strategy. The printed holdout table is the source of truth for
current performance.

## Tests

```bash
pytest tests/                         # unit + end-to-end, ~1s
git config core.hooksPath .githooks   # once per clone: run tests on every commit
```

The versioned `.githooks/pre-commit` runs the suite before each commit and aborts
if anything fails (`git commit --no-verify` skips it). Enable it once per clone
with the command above.

No network, broker, or Chronos needed — everything runs against the `SimBroker`
and lightweight fakes. Coverage focuses on the order/exit money paths:

- `test_bracket_ticks` — SL/TP ticks signed by direction (the order-rejection fix)
- `test_close_cancels_brackets` / `test_no_orphan_orders` — no orphaned SL **or**
  TP: against a stateful gateway, after a market close, a give-back exit
  (`manage_trail`), or a flat reconcile, zero protective orders are left working
  for the contract (and other contracts are untouched)
- `test_reconcile` — a flat account is swept of stray orders every bar; an
  in-position bar never cancels its live protective stop (mid-session reconcile)
- `test_exit_manager` — PPO give-back: activation gate, give-back cap, and the
  intra-bar wick-cross that closes a winner at market instead of riding it back
- `test_orb_gate` — ORB only fires during the RTH window (no overnight breakouts)
- `test_indicators` — indicator correctness + strict causality (no look-ahead in
  EMA / Keltner / opening-range / confirmed swings — they feed every feature)
- `test_strategy_triggers` — each strategy's `_fired` fires long/short on the
  right pattern and respects its ADX/trend gate
- `test_detect_signal` — entry/stop/risk math (`risk = STOP_ATR×ATR`, stop on the
  correct side) that drives sizing and the exit
- `test_position_size` — fixed vs risk-based sizing, cap and 1-lot floor
- `test_config_micros` — micro→parent symbol mapping (MNQ→NQ, …)
- `test_broker_contract` — active front-month selection (rollover) and
  working-stop filtering, against a stubbed gateway
- `test_sim_broker` — stop/target/trailing fills, stop-before-target tie, close
- `test_e2e_trade` — full lifecycles through `backtest.drive` → `bot.handle_bar`
  (the real driver): long/short give-back winners, stop-out loser, fixed-RR
  target + stop, the highest-proba resolver, the proba floor, and
  reconstruct-on-restart

## Caveats

- **Scope**: entry models are trained on NQ/ES/RTY/YM/GC 3-min UTC bars and
  generalize across them; other tickers/timeframes are out of distribution until
  retrained.
- **Feature fidelity**: 68 of 76 FFM features are computed live; the 8
  session/time columns the current library doesn't emit are left NaN (XGBoost
  handles missing natively) — faithful but not bit-identical to training.
- **PPO exit**: one **strategy-agnostic** policy on the standard 0.5×ATR(20)
  stop — it sees only the trade's R-state, so it applies identically to every
  strategy (it trains on a representative catalog of NQ entry points).
  With `GIVEBACK_R = 0.75` the give-back cap dominates, so the exit is
  effectively a deterministic "trail 0.75R from peak" — loosen it to let
  trend-riding matter more, tighten it for consistency.
- **Exit mode**: default `USE_TRAILING_STOP = False` is the PPO-driven reprice
  (what it's trained for). `True` uses the ProjectX native trailing bracket
  (`type 5`); the PPO can only *tighten* that, so it mostly sits idle.
- **Contract rollover**: live trading always uses the broker's **active front
  month** (`/Contract/search`), and the bot re-resolves it once a day while flat,
  so a long-running session follows the quarterly roll to the new contract — and
  its clean warmup history — without a restart. The API is the source of truth;
  there is no roll calendar to drift. (Backtests use continuous CSV history, so
  they're unaffected.)
- **Online only**: the broker API is the single source of truth for contract
  specs (tick size / value) — there is no hard-coded fallback, so the bot needs
  credentials + connectivity at startup for both live and backtest. (The Chronos
  checkpoint itself is cached after the first download.)
