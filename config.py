#!/usr/bin/env python3
"""config.py — all bot settings in one place.

Credentials can also be supplied via the TOPSTEPX_USERNAME / TOPSTEPX_API_KEY /
TOPSTEPX_ACCOUNT environment variables (those win over the values here).
"""
import os

try:
    from dotenv import load_dotenv
except ImportError:                       # dotenv optional (not needed for backtest)
    def load_dotenv(*a, **k):
        return False

HERE = os.path.dirname(os.path.abspath(__file__))

# ── credentials ────────────────────────────────────────────────────────
# Stored in a .env file (gitignored) — copy .env.example to .env and fill in.
#   TOPSTEPX_USERNAME=...   TOPSTEPX_API_KEY=...   TOPSTEPX_ACCOUNT=...
# Real environment variables still win over .env.
load_dotenv(os.path.join(HERE, ".env"))
TOPSTEPX_USERNAME = os.environ.get("TOPSTEPX_USERNAME", "")
TOPSTEPX_API_KEY  = os.environ.get("TOPSTEPX_API_KEY", "")
ACCOUNT = os.environ.get("TOPSTEPX_ACCOUNT", "")   # "" = first tradable account

# ── broker ─────────────────────────────────────────────────────────────
BROKER = "topstepx"        # which BrokerClient to use (see broker.make_broker)

# ── market / sizing ────────────────────────────────────────────────────
API_BASE = "https://api.topstepx.com/api"
SYMBOL = "NQ"
TIMEFRAME_MIN = 3
SIZE = 1

# Micro contracts trade the SAME bars as their full-size parent (so the models
# apply directly) at 1/10 the point value. Map each micro → its parent.
MICRO_PARENT = {"MNQ": "NQ", "MES": "ES", "M2K": "RTY", "MYM": "YM",
                "MGC": "GC", "MCL": "CL"}


def base_symbol(symbol: str) -> str:
    """The full-size parent for a micro (MNQ→NQ), else the symbol itself. Used
    for model feature derivation and choosing the data/<sym>_3min.csv file."""
    return MICRO_PARENT.get(symbol, symbol)


# Tick size and tick value are NOT hard-coded — they come from the broker API
# (/Contract/search → tickSize, tickValue) for both live and backtests.

# Full-size tickers the shipped entry models were trained on. Micros map to these
# via base_symbol(), so they're in-distribution too.
TRAINED_SYMBOLS = ("NQ", "ES", "RTY", "YM", "GC")

# Position sizing — use either a fixed SIZE or RISK_PER_TRADE (not both).
#   SIZE           fixed contracts per trade.
#   RISK_PER_TRADE if > 0, size from the stop instead:
#                  contracts = floor(RISK_PER_TRADE / (stop_ticks × tick_value)),
#                  clamped to [1, MAX_CONTRACTS]. 0 = use fixed SIZE.
RISK_PER_TRADE = 0.0       # $ risked per trade (0 = off)
MAX_CONTRACTS = 10         # cap on risk-sized contracts

# ── strategy selection ─────────────────────────────────────────────────
# Which strategies run. One name = single strategy; list both to run them
# together — when both fire on the same bar, the higher-proba signal is taken.
ACTIVE_STRATEGIES = ["ema"]                    # any of: supertrend, ema, keltner, bos
PROBA_FLOOR = 0.35          # enter only when a strategy grades its signal >= this

# ── shared trade definition (matches how the models scored trades) ─────
ATR_P     = 20              # ATR period for the protective stop
STOP_ATR  = 0.5            # stop = STOP_ATR * ATR(ATR_P) from entry
ADX_P     = 14
ADX_SLOPE = 5              # bars for the adx-slope feature
MIN_GAP   = 20             # (reference) min bars between signals
CTX       = 128            # Chronos context window

# SuperTrend strategy params
ST_PERIOD, ST_MULT = 10, 3.0
# EMA-cross strategy params
EMA_FAST, EMA_SLOW = 9, 20
SLOW_SLOPE_K = 5
ADX_GATE = 18.0            # only fire EMA crosses when ADX >= this (trend gate)
# Keltner-channel breakout strategy params
KC_LEN, KC_MULT, KC_ATR_P = 20, 1.5, 20
KC_ADX_THRESH = 20.0      # only fire Keltner breakouts when ADX >= this
KC_MID_SLOPE_K = 5
# Break-of-structure strategy params
SWING_K = 2               # fractal half-width for confirmed swings
# Opening-range-breakout strategy params
ORB_BARS = 5              # bars in the opening range (5 × 3-min = 15 min)
ORB_OPEN_MIN = 9 * 60 + 30   # session open = 09:30 in ORB_TZ (minutes from midnight)
ORB_TZ = "America/New_York"
ORB_ADX_GATE = 18.0       # only fire ORB breakouts when ADX >= this

# ── models ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(HERE, "models")
FFM_COLUMNS_PATH = os.path.join(MODELS_DIR, "ffm_feature_columns.json")

# ── exit ───────────────────────────────────────────────────────────────
# When a PPO policy is present the bot manages the exit with a trailing stop;
# otherwise it falls back to a fixed RR bracket.
USE_PPO_EXIT = True
# False (default) = PPO trailing: the policy reprices the stop each bar via
#   /Order/modify — this is what the policy is trained for, so the trail is fully
#   policy-driven (can loosen or tighten the trail distance).
# True = broker-native trailing stop that the PPO can only *tighten* — simpler
#   intra-bar protection, but the policy can't widen, so it mostly sits idle.
USE_TRAILING_STOP = False
POLICY_PATH = os.path.join(MODELS_DIR, "rl_trail_exit", "ppo_trail_exit.npz")
RR = 2.0                    # fixed-R take-profit fallback (no PPO policy)

# Trailing exit shape:
#   ACTIVATE_R — hold the initial stop (1R = STOP_ATR×ATR) until the trade's peak
#                reaches this many R; only then start trailing. Lets winners run
#                through early pullbacks before we protect.
#   GIVEBACK_R — once trailing, the stop never sits more than this many R below the
#                running peak (the PPO may trail tighter, never looser).
# e.g. ACTIVATE_R=2, GIVEBACK_R=0.75: risk 1R until +2R, then lock in ≥ +1.25R and
# ride, giving back at most 0.75R from the best point.
ACTIVATE_R = 2.0
GIVEBACK_R = 0.75
