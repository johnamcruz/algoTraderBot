# SuperTrend + Chronos Signal Head (NQ futures, 3-min)

A two-head XGBoost model that **grades SuperTrend flip events** on NQ
futures (3-min bars). For each flip candidate it outputs:

- **`proba`** — probability the flip becomes a winning trade (0..1)
- **`r_hat`** — predicted peak R-multiple the trade can reach (0..15)

It does **not** generate signals — you detect a SuperTrend flip (ATR-band
direction change) yourself, then call this model to decide whether the
flip is worth trading and how far it might run.

Trained on 95,421 flip signals, 2021-04-25 → 2026-05-04 (NQ 3-min, UTC).

This package also ships a **PPO trailing-exit** policy that replaces the
fixed 2R take-profit with a learned trailing stop — see
[the PPO trailing exit](#ppo-trailing-exit) below.

## What's in this package

| file | purpose |
|---|---|
| `signal_head.json` | XGBoost classifier (native format) → `proba` |
| `risk_head.json` | XGBoost regressor (native format) → `r_hat` |
| `metadata.json` | training spans, dims, holdout stats |
| `predict.py` | complete runnable inference script |
| `supertrend_ai_bot.py` | live TopstepX bot (entry grading + exit) |
| `trail_exit_env.py` | PPO trailing-exit env, simulator + numpy policy |
| `train_ppo_exit.py` | trains the trailing-exit policy from `data/NQ_3min.csv` |
| `precompute_proba.py` | grades every flip with the entry model (cached) |
| `ppo_trail_exit.npz` | trained trailing-exit policy (torch-free, loaded live) |
| `proba_cache.npz` | cached per-flip `proba` (so re-training is instant) |
| `data/NQ_3min.csv` | NQ 3-min OHLCV history used to train the exit |
| `requirements.txt` | python dependencies |

The models are in XGBoost's **native JSON format** — they load on any
platform/version with plain `xgboost`, no pickle, no custom classes.

## Setup (one time)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Verify everything works (uses synthetic data; first run downloads the
~45 MB `amazon/chronos-bolt-tiny` checkpoint from HuggingFace and caches
it):

```bash
python predict.py --demo
```

You should see a `proba = ...` / `r_hat = ...` line. That confirms the
full pipeline (Chronos embedding → XGBoost heads) is working.

## Running on real data

```bash
python predict.py --closes closes.csv --features features.csv
```

- `closes.csv` — one closing price per line, **at least 128 rows**,
  oldest → newest, 3-min NQ bars ending at the signal bar.
- `features.csv` — **78 values**, one per line (layout below). Unknown
  values can be written as `nan` — XGBoost handles missing natively
  (predictions degrade gracefully but accuracy is best with all 78).

Or call it from your own code:

```python
from predict import chronos_embedding, predict

emb = chronos_embedding(closes)        # last 128 closes -> (256,)
proba, r_hat = predict(emb, features)  # features: (78,)
```

## Input contract (what the model expects)

Feature vector = `concat([embedding_256, hand_crafted_78])` → shape (334,).

**1. `embedding_256`** — handled for you by `predict.py`:
`amazon/chronos-bolt-tiny` encoder over the **log** of the last 128
closes, masked-mean pooled over the hidden states. (The exact code is in
`chronos_embedding()` — if you re-implement, match it exactly.)

**2. `hand_crafted_78`** — computed at the signal bar:

| index | content |
|---|---|
| [0:76] | 76 engineered market features (fractional-differencing family: multi-horizon momentum/volatility/trend-state transforms of OHLCV) |
| [76] | `adx` — 14-period Average Directional Index |
| [77] | `adx_slope` — bar-over-bar change of adx |

The 76 engineered features come from a proprietary feature pipeline that
is not included. Practical options: (a) run with your own feature set in
those slots set to `nan` and rely on the embedding + adx (works, reduced
accuracy), or (b) contact the author about the feature pipeline.

**Preprocessing rule:** leave NaN as NaN; zero out ±inf
(`np.nan_to_num(x, nan=np.nan, posinf=0.0, neginf=0.0)`).

**r_hat inversion (already done in `predict.py`):** the risk head was
trained on `log1p(R)` — raw regressor output must be passed through
`clip(expm1(out), 0, 15)`.

## Using the outputs

Production operating rules that worked well:

- **Entry floor**: only trade flips with `proba >= 0.35`. The proba
  distribution is conservative (median ≈ 0.29, max ≈ 0.57) — 0.35–0.50
  is the useful band; you will not see 0.8s.
- **Sizing**: scale position size with proba (small base size below 0.40,
  larger only above it).
- **Take-profit**: `TP = clip(0.8 * r_hat, 1.5, 8.0)` in R-multiples.

## PPO trailing exit

By default the live bot exits at a fixed **2R** take-profit. The PPO
trailing exit replaces that with a learned **trailing stop**: the entry
logic is unchanged (SuperTrend flip graded by the Chronos+XGBoost head),
but once in a trade a small reinforcement-learning policy decides each bar
*how tightly to trail the stop*.

**What the agent controls.** Each bar it picks a trailing-stop distance
from a discrete set of ATR multiples (`[0.75, 1.0, 1.5, 2.0, 2.5, 3.5]`).
The stop only ever ratchets in your favor — it never loosens. A trade ends
on a stop hit, a max-hold timeout (80 bars = 4h), or end of data.

**What the agent sees** (7 inputs, all in R-multiples so longs and shorts
look identical): unrealized R, max favorable excursion, distance to the
current stop, ATR/initial-risk, time in trade, recent momentum, and
distance from the SuperTrend line. Reward each step is the change in
(un)realized R, so it telescopes to the trade's final realized R.

### Train it

```bash
pip install -r requirements.txt          # adds gymnasium + stable-baselines3
python train_ppo_exit.py                  # full train (~600k steps)
python train_ppo_exit.py --quick          # 20k-step smoke test
python train_ppo_exit.py --timesteps 1000000   # train harder
```

This catalogs every SuperTrend flip in `data/NQ_3min.csv`, **keeps only the
flips the live bot would actually enter** (`proba >= 0.35`, the same floor as
live trading — graded once and cached in `proba_cache.npz`), splits the last
10% of bars off as an untouched holdout, trains PPO on the rest, and prints a
holdout comparison of the PPO exit against the fixed-2R and constant-trail
baselines (same entries, different exits). Pass `--proba-floor 0` to train on
all flips instead. It writes two files:

- `ppo_trail_exit.npz` — the policy weights as plain dense layers. **This is
  what the live bot loads.** The forward pass runs in pure numpy, so the bot
  never imports torch/SB3 next to xgboost (which would risk the OpenMP
  segfault described under *Caveats*).
- `ppo_trail_exit_sb3.zip` — the full Stable-Baselines3 model, for resuming
  training or inspection.

### Run it live

`supertrend_ai_bot.py` picks the exit automatically. The **entry** is
unchanged either way — the bot still only takes flips with `proba >= 0.35`
(`PROBA_FLOOR`). What changes is the exit:

- If `ppo_trail_exit.npz` is present (and `USE_PPO_EXIT = True`, the default),
  the bot enters with a **protective stop only** at the SuperTrend line and
  the PPO manages it every bar — no fixed take-profit.
- If the file is missing, or you set `USE_PPO_EXIT = False`, it falls back to
  the original fixed-2R bracket. Nothing else changes.

**Two exit mechanisms** (config `USE_TRAILING_STOP`, both driven by the same
per-bar PPO decision and an `/Order/modify` call):

- `True` (default) — enter with a **broker-native trailing stop** that follows
  price tick-by-tick on its own; each bar the PPO *tightens* its follow
  distance. You get intra-bar protection between the bot's 3-min wake-ups plus
  bar-level policy control. The policy only ever ratchets tighter.
- `False` — enter with a **plain stop**; each bar the PPO reprices the stop
  level directly. No protection between bars, but no dependence on the broker's
  trailing-order semantics.

The live loop reconstructs its state from the broker if the bot is restarted
mid-trade (side/entry from the position, risk from the working stop), so a
restart won't strand an open position.

> ⚠️ The trailing exit issues **live stop-modify orders** (`/Order/modify`)
> every bar. Test it on a practice/evaluation account first.
>
> ⚠️ The native-trailing-stop path (`USE_TRAILING_STOP = True`) uses the
> ProjectX trailing-bracket type (`5`) and the `/Order/modify` `trailPrice`
> field — both confirmed in the ProjectX API docs. Two residual unknowns the
> docs don't pin down: whether `trailPrice` is a price distance (assumed here —
> we send `ticks * tickSize`) or a raw tick count, and whether modifying it
> re-anchors the trail. Verify on a practice account, or use
> `USE_TRAILING_STOP = False` (plain stop reprice) which relies only on the
> already-working `stopPrice` modify.

## Holdout performance (30 days out-of-sample)

- 1,406 flip signals, 28% base win rate
- At proba ≥ 0.50: 60% WR, mean +1.37R, profit factor 4.3 (small sample)

## Caveats

- **Scope**: trained on NQ 3-min UTC bars only. Other tickers or
  timeframes are out of distribution — expect degraded results.
- **torch + xgboost OpenMP conflict (macOS)**: loading both libraries in
  one process can segfault. `predict.py` already handles this — it
  computes the Chronos embedding in an isolated subprocess automatically.
  If you write your own integration, keep torch and xgboost in separate
  processes (or at minimum try `KMP_DUPLICATE_LIB_OK=TRUE`).
- Internet needed once (HuggingFace checkpoint download); offline after.
