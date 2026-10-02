#!/usr/bin/env python3
"""strategies/jev/state.py — what Jev sees, and the question it answers.

Jev IS the strategy: on each flat bar it decides long / short / none by itself.
The six mechanical strategies never trade — they are its context. The state is
consolidated so every fact appears ONCE, in the place it means most, with
readable names, units and labels:

  market       the shared market context, once: time, trend, momentum, volatility,
               volume, last bar, session, recent range, liquidity sweeps, higher
               timeframes (the FFM features every strategy model sees, relabeled,
               plus ADX / returns / recent closes)
  strategies   per strategy, only what is unique to it: its VIEW of the market,
               its SIGNAL if it fired this bar (with its model's win probability
               and the setup features that model scored that aren't already
               above), and its TRACK RECORD (how its recent signals played out)
  yourRecentPicks / recentDecisions   Jev's own record and its last decisions as
               worked examples (with how BOTH a long and a short would have done)

Everything is RELATIVE (ATR units vs the current close, time of day) — no absolute
prices, no dates, no raw volume — so Jev can't recognise a historical session. A
model's win probability is withheld (null, `modelInSample: true`) on bars inside
that model's training span, where it would be a leaked label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import indicators as ind
from strategies.base import FFM_COLS, adx_pair, ffm_block
from strategies.jev.memory import JEV

LONG, SHORT, NONE = "long", "short", "none"   # sides, and the no-trade option

# What each strategy is and when it tends to work — tells Jev how to read each view.
PLAYBOOK = {
    "supertrend": "SuperTrend(10, 3xATR) trend line; fires on a direction flip. "
                  "Trend-reversal signal: good when a new trend starts with rising "
                  "ADX, chops in sideways markets.",
    "ema": "9/20 EMA cross, fires only with ADX >= 18. Momentum continuation in an "
           "established trend; whipsaws when the EMAs are flat and tangled.",
    "keltner": "Keltner channel (20 EMA +/- 1.5 ATR); fires on a close outside it "
               "with ADX >= 20. Volatility-expansion breakout; fails in ranges "
               "where price mean-reverts back inside.",
    "bos": "Break of the last confirmed swing high/low. Market-structure "
           "continuation; fails on liquidity sweeps that reverse.",
    "orb": "15-min opening range from 09:30 ET; fires on a break during the NY "
           "session with ADX >= 18. Session breakout, strongest early in the day.",
    "cisd_ote": "CISD displacement on 12-min bars, then a pullback into the "
                "50-70.5% OTE fib zone (ICT/SMC). Mean-reversion pullback entry with "
                "a wide structural stop; best when the 1h EMA trend agrees.",
}

# Each strategy model's setup features (the tail of its _hand_features after the
# 76 FFM columns), in order. None = not shown because the same fact is already in
# `market` or the strategy's `view` (ADX, EMA spread, channel position, stop size,
# hour, 1h trend...). Signed by the trade's direction (positive = in its favor).
SETUP_FEATURES = {
    "supertrend": [None, None],                               # adx, adx slope
    "ema": [None, None, None, None, None],                    # = view + market.adx
    "keltner": [None, None, None, None],                      # = view + market.adx
    "bos": ["breakStrengthAtr", None, None, None, None],
    "orb": [None, "breakoutBeyondRangeAtr", "sessionGapAtr", "approachPosInRange",
            "breakBarVolumeVsRangeAvg", None, None],
    "cisd_ote": [None, "zoneHeightAtr", "zoneAge", "hadLiquiditySweep",
                 "displacement", "entryPosInZone", None, None, None, None, None],
}
_BOOL_SETUP = {"hadLiquiditySweep"}


def _r(x, nd=2):
    """Round a finite number for the state; non-finite → None (JSON null)."""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _session(tmin: int) -> str:
    if 570 <= tmin < 960:
        return "ny_rth"
    if 180 <= tmin < 570:
        return "london"
    if 960 <= tmin < 1080:
        return "ny_close"
    return "asia"


# ── the shared market context (FFM features, consolidated + relabeled) ─────

_CANDLE = {0: "doji", 1: "bull strength", 2: "bear strength", 3: "bull pin bar",
           4: "bear pin bar", 5: "mixed"}
_STRUCT = {1: "higher highs + higher lows", -1: "lower highs + lower lows",
           0: "mixed"}
_DIR3 = {1: "mostly up", -1: "mostly down", 0: "mixed"}
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _label(table, x):
    return table.get(int(x)) if np.isfinite(x) else None


def market_context(bars: pd.DataFrame, i: int) -> dict:
    """Everything shared about the market at bar i, each fact once."""
    f = dict(zip(FFM_COLS, ffm_block(bars, i)))
    c = bars["close"].to_numpy(float)
    a_arr = ind.atr(bars, config.ATR_P)
    a = float(a_arr[i]) if np.isfinite(a_arr[i]) and a_arr[i] > 0 else np.nan
    med_atr = np.nanmedian(a_arr[max(0, i - 99):i + 1])
    adx_i, adx_slope = adx_pair(bars, i)
    et = pd.Timestamp(bars["time"].iloc[i]).tz_convert(config.ORB_TZ)
    tmin = et.hour * 60 + et.minute

    def au(x):
        return _r(x / a) if np.isfinite(a) and np.isfinite(x) else None

    g = lambda k, nd=2: _r(f.get(k, np.nan), nd)          # noqa: E731
    flag = lambda k: bool(f.get(k, 0) > 0.5)              # noqa: E731
    sweep_dir = f.get("swp_tf_alignment", np.nan)
    htf_align = f.get("htf_tf_alignment", np.nan)
    return {
        "time": {"session": _session(tmin), "timeET": f"{et.hour:02d}:{et.minute:02d}",
                 "dayOfWeek": _DAYS[et.dayofweek]},
        "trend": {
            "adx": _r(adx_i, 1), "adxSlope5": _r(adx_slope, 1),
            "swingStructure": _label(_STRUCT, f.get("str_structure_state", np.nan)),
            "h1": {"moveFromHourOpenAtr": g("htf_1h_ret"),
                   "positionInHourRange": g("htf_1h_close_pos"),
                   "last3Hours": _label(_DIR3, f.get("htf_1h_structure", np.nan))},
            "h4": {"moveFrom4hOpenAtr": g("htf_4h_ret"),
                   "positionIn4hRange": g("htf_4h_close_pos")},
            "h1AndH4Agree": bool(htf_align > 0) if np.isfinite(htf_align) else None,
        },
        "momentum": {
            "returnsAtr": {f"last{n}Bars": au(c[i] - c[i - n]) if i >= n else None
                           for n in (1, 5, 20, 60)},
            # oldest..newest, each close minus the current close, in ATR
            "last30ClosesAtr": (" ".join(f"{(x - c[i]) / a:.1f}"
                                         for x in c[max(0, i - 29):i + 1])
                                if np.isfinite(a) else None),
            "speedRatio": g("momentum_speed_ratio"),
            "directionConsistency": g("dir_consistency"),
        },
        "volatility": {
            "atrBps": _r(a / c[i] * 1e4, 1),
            "atrVsMedian100": _r(a / med_atr) if np.isfinite(med_atr) and med_atr > 0 else None,
            "atrZscore": g("vty_atr_zscore"),
            "barRangeVsAvg5": g("vty_range_ratio_5"),
            "barRangeVsAvg20": g("vty_range_ratio_20"),
            "atrInstability": g("vty_atr_of_atr"),
            "realizedVolBps10": _r(f.get("vty_realized_10", np.nan) * 1e4, 1),
            "realizedVolBps20": _r(f.get("vty_realized_20", np.nan) * 1e4, 1),
        },
        "volume": {
            "vsAvg5": g("vol_ratio_5"), "vsAvg10": g("vol_ratio_10"),
            "vsAvg20": g("vol_ratio_20"), "changeVsPrevBar": g("vol_change"),
            "buyPressure5": g("vol_cum_signed_5"), "buyPressure20": g("vol_cum_signed_20"),
            "absorption": g("vol_absorption"), "confirmsMove": g("vol_momentum_align"),
        },
        "lastBar": {
            "type": _label(_CANDLE, f.get("candle_type", np.nan)),
            "rangeAtr": g("bar_range_atr"), "bodyAtr": g("bar_body_atr"),
            "upperWickAtr": g("bar_upper_wick_atr"), "lowerWickAtr": g("bar_lower_wick_atr"),
            "closePosInBar": g("vol_close_position"), "wickRejection": g("wick_rejection"),
            "engulfCount": g("engulf_count", 0), "sizeVsSessionAvg": g("bar_size_vs_session"),
        },
        "session": {"fromOpenAtr": g("sess_dist_from_open"),
                    "fromHighAtr": g("sess_dist_from_high"),
                    "fromLowAtr": g("sess_dist_from_low"),
                    "fromVwapAtr": g("sess_dist_from_vwap")},
        "range": {f"last{n}Bars": {"fromHighAtr": g(f"str_dist_from_high_{n}"),
                                   "fromLowAtr": g(f"str_dist_from_low_{n}"),
                                   "position": g(f"str_range_position_{n}")}
                  for n in (10, 20)},
        "liquiditySweeps": {
            **{tf: {"bullishActive": flag(f"swp_{tf}_bull_active"),
                    "bearishActive": flag(f"swp_{tf}_bear_active"),
                    "age": g(f"swp_{tf}_age_norm"), "sizeAtr": g(f"swp_{tf}_magnitude")}
               for tf in ("1h", "4h")},
            "direction": ({1: "bullish", -1: "bearish", 0: "none"}.get(int(sweep_dir))
                          if np.isfinite(sweep_dir) else None),
        },
    }


# ── per-strategy views (only what isn't already in `market`) ────────────────

def _view_supertrend(bars, c, i, au, adx_i, s):
    line, d = ind.supertrend(bars, config.ST_PERIOD, config.ST_MULT)
    flips = np.nonzero(d[1:] != d[:-1])[0]
    return {"bias": "up" if d[i] > 0 else "down",
            "lineDistAtr": au(c[i] - line[i]),
            "barsSinceFlip": int(i - 1 - flips[-1]) if len(flips) else None}


def _view_ema(bars, c, i, au, adx_i, s):
    ef, es = ind.ema(c, config.EMA_FAST), ind.ema(c, config.EMA_SLOW)
    k = config.SLOW_SLOPE_K
    return {"bias": "up" if ef[i] > es[i] else "down",
            "fastMinusSlowAtr": au(ef[i] - es[i]),
            "slowSlopeAtr": au(es[i] - es[i - k]) if i >= k else None,
            "priceVsSlowAtr": au(c[i] - es[i]),
            "gateOpen": bool(adx_i >= config.ADX_GATE)}


def _view_keltner(bars, c, i, au, adx_i, s):
    up, _mid, lo = ind.keltner_channel(bars, config.KC_LEN, config.KC_MULT,
                                       config.KC_ATR_P)
    pos = ("above" if c[i] > up[i] else "below" if c[i] < lo[i] else "inside") \
        if np.isfinite(up[i]) and np.isfinite(lo[i]) else None
    return {"priceVsChannel": pos, "gateOpen": bool(adx_i >= config.KC_ADX_THRESH)}


def _view_bos(bars, c, i, au, adx_i, s):
    sh, sl, shi, sli = ind.causal_swings(bars, config.SWING_K)
    return {"toSwingHighAtr": au(sh[i] - c[i]),
            "toSwingLowAtr": au(c[i] - sl[i]),
            "swingHighAgeBars": int(i - shi[i]) if shi[i] >= 0 else None,
            "swingLowAgeBars": int(i - sli[i]) if sli[i] >= 0 else None}


def _view_orb(bars, c, i, au, adx_i, s):
    oh, ol = ind.opening_range(bars, config.ORB_BARS, config.ORB_OPEN_MIN,
                               config.ORB_TZ)
    tmin = int(ind.et_minutes(bars, config.ORB_TZ)[i])
    if not (np.isfinite(oh[i]) and np.isfinite(ol[i])) or tmin >= config.ORB_CLOSE_MIN:
        return {"active": False}            # no range yet, or a stale morning range
    return {"active": True,
            "rangeAtr": au(oh[i] - ol[i]),
            "price": "above" if c[i] > oh[i] else "below" if c[i] < ol[i] else "inside",
            "gateOpen": bool(adx_i >= config.ORB_ADX_GATE)}


def _view_cisd_ote(bars, c, i, au, adx_i, s):
    p = getattr(s, "last_pipeline", None)      # from this bar's detect()
    if p is None:
        return {}
    _atr, _hour, htf_sign, htf_str, recs = p
    out = {"emaTrend1h": ("up" if htf_sign[i] > 0 else "down" if htf_sign[i] < 0
                          else None) if np.isfinite(htf_sign[i]) else None,
           "emaTrend1hStrengthBps": _r(htf_str[i] * 1e4, 1)}
    rec = next((r for r in reversed(recs) if r["exec_idx"] <= i), None)
    if rec is not None and i - rec["exec_idx"] <= config.JEV_RECENT_SETUP_BARS:
        zh = rec["fib_top"] - rec["fib_bot"]
        out["recentSetup"] = {
            "side": LONG if rec["is_long"] else SHORT,
            "barsAgo": int(i - rec["exec_idx"]),
            "priceInZone": _r((c[i] - rec["fib_bot"]) / zh) if zh > 0 else None}
    return out


VIEWS = {"supertrend": _view_supertrend, "ema": _view_ema,
         "keltner": _view_keltner, "bos": _view_bos, "orb": _view_orb,
         "cisd_ote": _view_cisd_ote}


def setup_features(s, bars, sig) -> dict | None:
    """The setup features its model scored this signal with that aren't shown
    elsewhere, named."""
    names = SETUP_FEATURES.get(s.name)
    if not names or not any(names) or not hasattr(s, "_hand_features"):
        return None
    tail = s._hand_features(bars, sig.bar_index, sig.direction)[len(FFM_COLS):]
    out = {}
    for n, v in zip(names, tail):
        if n is None:
            continue
        v = float(v)
        out[n] = bool(v > 0.5) if n in _BOOL_SETUP else _r(v)
    return out


def _in_sample(s, sig) -> bool:
    check = getattr(s, "in_sample", None)
    return bool(check and sig.bar_time is not None and check(sig.bar_time))


def build_state(bars: pd.DataFrame, subs, fired: dict, memory,
                history=None) -> dict:
    """Jev's view of the last closed bar. `subs` = the strategies whose context
    it gets, `fired` = {name: graded Signal} for those that fired on this bar,
    `memory` = the TrackRecord, `history` = its recent decisions rendered by
    DecisionLog (oldest → newest)."""
    c = bars["close"].to_numpy(float)
    i = len(c) - 1
    a_arr = ind.atr(bars, config.ATR_P)
    a = float(a_arr[i]) if np.isfinite(a_arr[i]) and a_arr[i] > 0 else np.nan

    def au(x):
        """Price distance → ATR units."""
        return _r(x / a) if np.isfinite(a) and np.isfinite(x) else None

    adx_i, _ = adx_pair(bars, i)
    strategies = {}
    for s in subs:
        view = VIEWS.get(s.name, lambda *a_: {})(bars, c, i, au, adx_i, s)
        sig = fired.get(s.name)
        signal = None
        if sig is not None:
            insample = _in_sample(s, sig)
            signal = {"side": LONG if sig.direction > 0 else SHORT,
                      "stopAtr": au(sig.risk),
                      "modelWinProb": None if insample else _r(sig.proba),
                      "modelExpectedR": None if insample else _r(sig.r_hat)}
            if insample:
                signal["modelInSample"] = True
            setup = setup_features(s, bars, sig)
            if setup:
                signal["setup"] = setup
        strategies[s.name] = {"view": view, "signal": signal,
                              "trackRecord": memory.summary(s.name)}

    return {
        "instrument": config.base_symbol(config.SYMBOL),
        "barMinutes": config.TIMEFRAME_MIN,
        "market": market_context(bars, i),
        "strategies": strategies,
        "yourRecentPicks": memory.summary(JEV),
        "recentDecisions": history or [],
    }


def situation(state: dict) -> dict:
    """A compact snapshot of a decision's market, kept in the decision log."""
    m, st = state["market"], state["strategies"]
    out = {"session": m["time"]["session"], "adx": m["trend"]["adx"],
           "last20BarsAtr": m["momentum"]["returnsAtr"]["last20Bars"],
           "swingStructure": m["trend"]["swingStructure"]}
    for name in ("supertrend", "ema"):
        if name in st:
            out[f"{name}Bias"] = st[name]["view"].get("bias")
    return out


def build_question(panel) -> dict:
    """Jev's decision for this bar: long, short, or none. `panel` = the strategy
    names whose context it gets, for the playbook."""
    target, horizon = config.JEV_MEMORY_TARGET_R, config.JEV_MEMORY_MAX_BARS
    tf, stop = config.TIMEFRAME_MIN, config.STOP_ATR
    breakeven = 1.0 / (1.0 + target)
    return {
        "type": "choice",
        "instructions": {
            "question": f"Should the bot go long, go short, or stay flat right now? "
                        f"A trade is a win if it reaches +{target:g}R before its -1R "
                        f"stop within the next {horizon} bars ({horizon * tf} "
                        f"minutes). Choose the side more likely to win, or none if "
                        f"neither is likely enough.",
            "goal": f"Intraday futures on {tf}-minute bars. Reward:risk is "
                    f"{target:g}:1, so a trade is worth taking when its chance of "
                    f"winning is above about {breakeven:.2f} (break-even, before "
                    "costs). Skipping costs nothing but misses winners.",
            "timing": "Decided at the close of the last bar; the market order fills "
                      f"at about that close with a stop {stop:g} ATR away (1R). "
                      "After entry a separate trailing policy manages the exit.",
            "inputs": "All prices are in ATR units relative to the current close; "
                      "times are US Eastern; null means not available. `market` is "
                      "the shared context (trend, momentum, volatility, volume, last "
                      "bar, session, recent range, liquidity sweeps, 1h/4h) — "
                      "positionIn*/position = 0 at the low, 1 at the high; "
                      "buyPressure is -0.5 (sellers) to +0.5 (buyers). `strategies` "
                      "is a panel of mechanical strategies that do NOT trade — they "
                      "are your evidence: each `view` is that strategy's own read of "
                      "the market; `signal` (only if it fired on this bar) is its "
                      "side, its own stop size, its trained model's probability that "
                      "its trade wins (`modelWinProb`, withheld when the model was "
                      "trained on this period) and expected R, plus setup details; "
                      "`trackRecord` is how its recent signals actually played out "
                      f"on the same +{target:g}R / -1R test (avgR is the best single "
                      f"summary; winRate = share that hit +{target:g}R; `last`: W hit "
                      "target, L stopped, F timed out, oldest to newest; pending = "
                      "not resolved yet). `yourRecentPicks` is that test applied to "
                      "your own entries, and `recentDecisions` are your last "
                      "decisions as worked examples: the situation, which signals "
                      "fired, your choice, and how a long AND a short would each have "
                      "turned out — so a skipped winner or a dodged loser is visible. "
                      "Use all of it: trade when the market context, the strategies' "
                      "views and signals, and what has been working recently line up.",
            "playbook": {n: PLAYBOOK[n] for n in panel if n in PLAYBOOK},
        },
        "criteria": {
            LONG: f"Buy at market now, stop {stop:g} ATR below.",
            SHORT: f"Sell short at market now, stop {stop:g} ATR above.",
            NONE: "Stay flat this bar.",
        },
    }
