# algoTraderBot — multi-strategy AI futures bot (3-min, TopstepX)

A live TopstepX bot that trades **mechanical entries graded by AI**, with a
**reinforcement-learned trailing exit**. Each strategy is a thin signal
generator paired with its own Chronos+XGBoost model; the model decides *which*
signals to take, and a PPO policy decides *when to get out*.

```
each bar ─► every active strategy detects its entry (SuperTrend flip / EMA cross)
        ─► its model grades the signal  →  proba = P(win)
        ─► best signal with proba ≥ floor is taken (highest proba wins)
        ─► PPO policy trails the stop bar-by-bar until exit
```

The shipped **entry models are trained on multiple 3-min futures** — `NQ`, `ES`,
`RTY`, `YM`, and `GC` — so they generalize across those tickers. The framework
(strategies, sizing, trailing exit) is ticker-agnostic; other symbols (e.g.
`SI`, `CL`) run out of distribution until retrained on their own data.

> ⚠️ **Educational — places LIVE orders.** Run it on a practice/evaluation
> account first. 3-min bars; entry models trained on NQ/ES/RTY/YM/GC.

## Architecture

The bot is split into small, single-responsibility modules:

| file | responsibility |
|---|---|
| `bot.py` | entry point — `handle_bar` (detect → grade → enter → trail) + live loop + CLI |
| `config.py` | **all settings**: active strategies, exit flags, tick sizes (+ `.env` creds) |
| `broker.py` | `TopstepXClient` — REST wrapper over the ProjectX Gateway API |
| `sim_broker.py` | `SimBroker` — fills/stops/trailing against a CSV for backtests |
| `backtest.py` | drives `handle_bar` over history with date-range selection |
| `indicators.py` | SuperTrend / ATR / ADX (reused from `futures_foundation`) + EMA |
| `strategies/` | the pluggable strategies (one file each) + shared base |
| `exit_manager.py` | PPO trailing-stop management for an open position |
| `logsetup.py` | logging to `log/bot.log` (candles, proba, entries) |
| `trail_exit_env.py` | the PPO training environment + torch-free policy loader |
| `train_ppo_exit.py` | trains the trailing-exit policy |
| `precompute_proba.py` | grades flips to filter PPO training to real entries |
| `models/` | all trained models (see below) |

```
strategies/
  base.py          # Strategy ABC: detect() + grade() + Signal
  supertrend.py    # SuperTrendStrategy  → models/supertrend_chronos.joblib
  ema_cross.py     # EmaCrossStrategy    → models/ema_cross_chronos.joblib

models/
  supertrend_chronos.joblib    # entry model: SuperTrend flip selection
  ema_cross_chronos.joblib     # entry model: 9/20 EMA-cross selection
  ffm_feature_columns.json     # the 76 FFM feature names (parquet order)
  rl_trail_exit/
    ppo_trail_exit.npz         # the trailing-exit policy (loaded live)
    ppo_trail_exit_sb3.zip     # full SB3 model (for resuming training)
```

The bot depends only on the public **`futures_foundation`** library (Chronos
embedding + the model head classes + the indicator primitives) — no proprietary
code. The joblib bundles run **inference directly**; the FFM feature block is
computed live via `futures_foundation.features.derive_features` (no parquets).

### The Strategy interface

Every strategy inherits one small contract (`strategies/base.py`):

```python
class Strategy(ABC):
    name: str
    model_filename: str
    def detect(self, bars) -> Signal | None      # mechanical entry on the last bar
    def reference_line(self, bars) -> np.ndarray  # line the PPO exit trails against
    def _hand_features(self, bars, i, direction)  # the model's hand-crafted inputs
    def grade(self, bars, sig) -> (proba, r_hat)  # shared: embed + concat + heads
```

`detect()` returns a `Signal` (direction, entry, stop = 0.5×ATR(20), risk);
`grade()` runs the Chronos embedding + the strategy's XGBoost heads. Adding a new
strategy = one new file in `strategies/` + its joblib model.

**Two strategies ship today:**
- **SuperTrend** — enters on a SuperTrend flip (period 10, mult 3.0).
- **EMA cross** — enters on a 9/20 EMA crossover, gated to ADX ≥ 18.

## Configure & run

**1. Install** (pulls `futures_foundation` from GitHub, ~45 MB Chronos
checkpoint downloads on first run):

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your TopstepX credentials** — copy `.env.example` to `.env` (gitignored)
and fill it in:

```bash
cp .env.example .env
```
```ini
TOPSTEPX_USERNAME=your_login
TOPSTEPX_API_KEY=your_api_key     # the API KEY from the dashboard, not your password
TOPSTEPX_ACCOUNT=                 # blank = first tradable account, or set id/name
```

(Real environment variables override `.env`. Credentials are only needed for
live trading — backtesting works without them.)

**3. Pick your strategies & exit** in `config.py`:

```python
ACTIVE_STRATEGIES = ["supertrend", "ema"]   # one name, or both (highest proba wins)
PROBA_FLOOR       = 0.35                     # only take signals graded ≥ this
ACTIVATE_R        = 2.0                      # hold initial 1R stop until +2R, then trail
GIVEBACK_R        = 0.75                     # once trailing, give back ≤ this R from peak
USE_PPO_EXIT      = True                     # False → fixed-RR bracket instead
USE_TRAILING_STOP = False                    # False → PPO reprices the stop each bar
                                             #         (policy-driven, the default);
                                             # True  → broker-native trail (PPO can
                                             #         only tighten it)
```

**4. Run:**

```bash
python bot.py                      # uses config.SIZE contracts
python bot.py --size 3             # fixed: 3 contracts per trade
python bot.py --risk 500           # risk-based: ~$500 risked per trade
python bot.py --risk 500 --max-contracts 5
python bot.py --proba-floor 0.45   # only take entries graded ≥ 0.45 confidence
```

**Sizing** — use a fixed size *or* risk-based sizing (not both). With `--risk`
(or `config.RISK_PER_TRADE`), contracts are sized from the stop so each trade
risks ~that many dollars:

```
contracts = min(MAX_CONTRACTS,
                max(1, floor(risk_$ / (stop_ticks × tick_value))))
```

`tick_value` is derived from `config.POINT_VALUES` (e.g. NQ = $20/pt × 0.25 =
$5/tick). A tighter stop ⇒ more contracts, a wider stop ⇒ fewer, so dollar risk
per trade stays roughly constant (handy with the 0.5×ATR stop, which varies with
volatility).

It prints your tradable accounts on startup — make sure it picks the **practice**
one. Every candle, every strategy's proba (with a TAKE/skip flag), each entry,
and every trailing-stop move are logged via `log.info` to both the console and
`log/bot.log`.

## How entries & exits work

- **Entry** — when flat, every active strategy gets a chance to `detect()` +
  `grade()`. Signals with `proba ≥ PROBA_FLOOR` are candidates; the **highest
  proba wins** (one position per contract). The trade enters at market with a
  protective stop at `0.5×ATR(20)` — exactly how the models scored the trade.
- **Exit** — each bar the PPO policy (`models/rl_trail_exit/`) reads the open
  trade (unrealized R, MFE, ATR, momentum, distance from the strategy's reference
  line), computes a trailing-stop level, and **reprices the live stop to it via
  `/Order/modify`** (ratcheting only in your favor). The trail is shaped by two
  knobs: **`ACTIVATE_R`** (hold the initial 1R stop until the peak reaches +2R,
  so winners survive early pullbacks) and **`GIVEBACK_R`** (once trailing, the
  stop never sits more than 0.75R below the running peak). So a trade risks 1R,
  and once it's up +2R it locks in ≥ +1.25R and rides, giving back ≤ 0.75R from
  the best point. It's trained on the same 0.5×ATR(20) risk and the same shaping.
  If no policy is present it falls back to a fixed `RR` bracket. The policy
  forward-pass is pure numpy, so the bot never loads torch/SB3 next to xgboost.

## Backtest (no API, no credentials)

Run the **exact live logic** — same strategies, grading, and PPO trailing exit —
over a local CSV, with a simulated broker filling entries/stops/trailing against
history:

```bash
python bot.py --backtest --symbol NQ --start 2026-05-01 --end 2026-06-01
```

- `--symbol` reads `data/<symbol>_3min.csv`. The models are trained on NQ, ES,
  RTY, YM, GC; other symbols (SI, CL, …) are out of distribution.
- `--start` / `--end` are `YYYY-MM-DD` (start inclusive, end exclusive); omit to
  run from the warmup point to the end of file.

It prints a summary (trades, win rate, mean/sum R, profit factor) with per-strategy
and per-exit-reason breakdowns, and writes every trade to `log/backtest_<symbol>.csv`.
Entries fill at the signal bar's close; conservative fills assume the stop before
the target when a bar straddles both. Grading embeds each signal through Chronos,
so longer ranges take a while.

## Retrain the trailing exit (optional)

From the bot (convenient), or via the training script directly:

```bash
python bot.py --retrain-exit          # full retrain, then exit
python bot.py --retrain-exit --quick  # fast smoke retrain
python bot.py --retrain-exit --timesteps 1000000

python train_ppo_exit.py              # same thing, standalone
python train_ppo_exit.py --quick
```

Catalogs every SuperTrend flip in `data/NQ_3min.csv`, keeps only the ones the bot
would enter (`proba ≥ 0.35`, graded by the SuperTrend model and cached in
`proba_cache.npz`), simulates each trade from a **0.5×ATR(20) stop** (the live
entry risk) while the agent learns the trail, then benchmarks the policy vs
fixed-RR / constant-trail baselines and writes it into `models/rl_trail_exit/`.
The printed holdout table is the source of truth for current performance.

## Caveats

- **Scope**: the entry models are trained on NQ/ES/RTY/YM/GC 3-min UTC bars and
  generalize across them; other tickers or timeframes are out of distribution
  until retrained on their own data.
- **Feature fidelity**: 68 of the 76 FFM features are computed live; the 8
  session/time columns the current library doesn't emit are left NaN (XGBoost
  handles missing natively). The grade is faithful but not bit-identical to
  training.
- **PPO exit scope**: the trailing policy is trained on the live 0.5×ATR(20)
  stop, but only on **SuperTrend-flip** trades (the training env's catalog). It
  runs on every strategy live; for EMA entries the "distance from reference
  line" observation uses the slow EMA, which is slightly outside the training
  distribution. Retrain per-strategy for an exact match.
- **Exit mode**: the default `USE_TRAILING_STOP = False` is the PPO-driven trail
  (the policy reprices `stopPrice` each bar — what it's trained for). The
  alternative `True` uses the ProjectX native trailing bracket (`type 5`) +
  `/Order/modify` `trailPrice`; the PPO can only *tighten* it there, so it mostly
  sits idle. With `GIVEBACK_R = 0.75` the give-back cap dominates the exit (every
  trail multiple converges to the same result), so the policy is effectively a
  deterministic "trail 0.75R from peak". Loosen `GIVEBACK_R` to let the PPO and
  trend-riding matter more (higher mean-R, more give-back); tighten it for more
  consistency (higher capture, more trends cut short).
- **Internet** needed once for the Chronos checkpoint; offline after.
