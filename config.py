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

# ── market / sizing ────────────────────────────────────────────────────
API_BASE = "https://api.topstepx.com/api"
SYMBOL = "NQ"
TIMEFRAME_MIN = 3
SIZE = 1

# Tick sizes for backtesting (live mode reads tickSize from the broker).
TICK_SIZES = {"NQ": 0.25, "ES": 0.25, "RTY": 0.1, "YM": 1.0,
              "GC": 0.1, "SI": 0.005, "CL": 0.01}

# ── strategy selection ─────────────────────────────────────────────────
# Which strategies run. One name = single strategy; list both to run them
# together — when both fire on the same bar, the higher-proba signal is taken.
ACTIVE_STRATEGIES = ["ema"]                    # "supertrend" and/or "ema"
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
