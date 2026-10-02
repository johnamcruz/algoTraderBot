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
TIMEFRAME_MIN = 3                # bar interval in minutes (CLI: --timeframe)
TRAINED_TIMEFRAME_MIN = 3        # the interval the models/PPO were trained on; other
#                                 timeframes run but are out of distribution
SIZE = 1
BARS_WINDOW = 500                # bars of history per step — live fetch AND backtest
#                                 window, so both compute identical features
LIVE = False                     # set by the live loop (bot.run); False in backtests

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
ORB_CLOSE_MIN = 16 * 60   # stop firing ORB breakouts at 16:00 ET — the opening
#   range stays mathematically "active" until midnight ET, but a 09:30 range is
#   stale by the evening; gate entries to the RTH session [~09:45, 16:00) ET so
#   the bot doesn't take low-quality overnight breakouts on the morning range.

# Jev strategy (strategies/jev): a reasoning model IS the strategy. On every flat
# bar it reads all the context the six strategies have (their views, any signals
# with their model scores, their track records), the full market context and its
# own last decisions, and decides long / short / none. The six strategies never
# trade under jev. Its decision IS the entry — PROBA_FLOOR does not apply.
#
# The decider is swappable — "backend:model" here or JEV_DECIDER in .env:
#   jev:jev-latest                       TypeSafe's hosted Jev (needs TYPESAFE_AI_API_KEY)
#   mlx:mlx-community/Qwen3-0.6B-4bit    local model in-process on Apple silicon (mlx-lm)
#   openai:qwen3:4b                      any OpenAI-compatible server (Ollama, LM Studio,
#                                        mlx_lm.server, llama.cpp) at JEV_OPENAI_URL
JEV_DECIDER = os.environ.get("JEV_DECIDER", "jev:jev-latest")
TYPESAFE_AI_API_KEY = os.environ.get("TYPESAFE_AI_API_KEY", "")
JEV_API_BASE = "https://api.typesafe.ai/v1"
JEV_OPENAI_URL = os.environ.get("JEV_OPENAI_URL", "http://localhost:11434/v1")
JEV_OPENAI_KEY = os.environ.get("JEV_OPENAI_KEY", "")
JEV_LLM_THINK = os.environ.get("JEV_LLM_THINK", "1") != "0"   # chat models: reason
#                             before answering (0 = off; faster, for tiny models)
JEV_LLM_MAX_TOKENS = 2048   # chat models: reasoning + answer budget
JEV_TIMEOUT = 120.0         # seconds per chat-model decision (openai / mlx)
JEV_API_TIMEOUT = 5.0       # seconds per hosted-Jev call (it answers in ~100 ms)
JEV_RETRIES = 2             # retries on rate limit / overload / 5xx / connection
JEV_MAX_DECISION_SEC = 20.0 # live: if deciding took longer, skip the entry (the
#                             price it decided on is stale)
JEV_ONLY_ON_SIGNALS = False # True = ask only on bars where a strategy fired
#                             (fewer calls, e.g. for a slow local model)
JEV_STRATEGIES = []         # strategies whose context Jev gets; [] = every one
#                             with a model for the active timeframe
JEV_CACHE_PATH = os.path.join(HERE, "log", "jev_cache.jsonl")   # identical
#   requests are answered from here, so re-running a backtest makes no calls
# Track record (strategies/jev/memory.py): every strategy signal is followed until
# +TARGET_R, its stop, or MAX_BARS; Jev sees each strategy's last KEEP outcomes.
JEV_MEMORY_TARGET_R = 2.0
JEV_MEMORY_MAX_BARS = 20
JEV_MEMORY_KEEP = 20
JEV_MEMORY_BOOT_BARS = 400  # history replayed at startup to seed the record
JEV_HISTORY_KEEP = 8        # past decisions shown as worked examples (situation →
#                             signals → choice → how a long and a short would have
#                             done); quiet bars with no signal and no entry are skipped
JEV_RECENT_SETUP_BARS = 10  # show cisd_ote's last setup only if this recent

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
POLICY_PATH = os.path.join(HERE, "ppo_exit", "policies", "ppo_trail_exit.npz")
RR = 2.0                    # fixed-R take-profit fallback (no PPO policy)


def policy_path():
    """The PPO policy for the active timeframe: ppo_trail_exit.npz at the trained
    3-min default, ppo_trail_exit_<tf>min.npz otherwise (e.g. ..._1min.npz). The
    exit is retrained per timeframe — bar geometry (ATR, MAX_HOLD bars) differs."""
    base, ext = os.path.splitext(POLICY_PATH)
    return POLICY_PATH if TIMEFRAME_MIN == TRAINED_TIMEFRAME_MIN \
        else f"{base}_{TIMEFRAME_MIN}min{ext}"

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

# Per-timeframe exit shaping. The best ACTIVATE_R / GIVEBACK_R / STOP_ATR differ by
# timeframe (1-min vs 3-min), so they live in exit_configs.json keyed by minutes and
# are applied for the active timeframe. These knobs are read at runtime by BOTH the
# live exit (exit_manager) and the training sim (trail_exit_env), so training=live.
# Tune with optimize_exit.py; after changing a timeframe's config, retrain its policy
# (train_ppo_exit --timeframe N) so the PPO matches.
EXIT_CONFIGS_PATH = os.path.join(HERE, "ppo_exit", "exit_configs.json")


def apply_exit_config(tf=None):
    """Apply exit_configs.json[<tf>] to ACTIVATE_R / GIVEBACK_R / STOP_ATR. No-op
    (keeps the module defaults) if the file or the timeframe key is missing. Returns
    the applied dict, or None."""
    global ACTIVATE_R, GIVEBACK_R, STOP_ATR
    import json
    tf = TIMEFRAME_MIN if tf is None else tf
    try:
        with open(EXIT_CONFIGS_PATH) as f:
            cfg = json.load(f).get(str(tf))
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    if "ACTIVATE_R" in cfg:
        ACTIVATE_R = float(cfg["ACTIVATE_R"])
    if "GIVEBACK_R" in cfg:
        GIVEBACK_R = float(cfg["GIVEBACK_R"])
    if "STOP_ATR" in cfg:
        STOP_ATR = float(cfg["STOP_ATR"])
    return {"ACTIVATE_R": ACTIVATE_R, "GIVEBACK_R": GIVEBACK_R, "STOP_ATR": STOP_ATR}


apply_exit_config()             # apply the default timeframe's saved config at import
