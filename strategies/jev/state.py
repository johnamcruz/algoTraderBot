#!/usr/bin/env python3
"""strategies/jev/state.py — what Jev sees, and the question it answers.

Jev decides which entry to take: when any strategy fires, it picks one of the
fired entries (or none). Every strategy — fired or not — is its evidence, each
contributing
  • its current VIEW of the market (bias, key levels, whether its gate is open),
  • its live SIGNAL if it fired on this bar (side, its model's win prob, stop), and
  • its TRACK RECORD — how its recent signals actually played out (memory.py),
  • for a fired signal, the named SETUP features its model scored it with.
Plus shared market context, the full 76-feature FFM context every strategy model
sees, and Jev's own recent picks.

Everything is RELATIVE (ATR units vs the current close, time of day) — no absolute
prices and no dates, so Jev can't recognise a historical session and leak
hindsight into a backtest.
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
                "50-70.5% OTE fib zone (ICT/SMC). Mean-reversion pullback entry; "
                "best when the 1h trend (htfTrend) agrees.",
}


# The strategy-specific features each model scores a fired signal with (the tail
# of its _hand_features after the 76 FFM columns), named. Signed by the trade
# direction where the strategy signs them (positive = in the trade's favor).
SETUP_FEATURES = {
    "supertrend": ["adx", "adx_slope"],
    "ema": ["ema_spread_atr", "slow_slope_atr", "price_vs_slow_atr",
            "adx_div100", "adx_slope_div100"],
    "keltner": ["channel_pos", "mid_slope_atr", "adx_div100", "adx_slope_div100"],
    "bos": ["break_ext_atr", "swing_range_atr", "swing_age_log", "adx_div100",
            "adx_slope_div100"],
    "orb": ["range_size_atr", "breakout_ext_atr", "session_gap_atr",
            "approach_pos", "range_volume_ratio", "adx_div100", "adx_slope_div100"],
    "cisd_ote": ["direction", "zone_height_atr", "zone_age", "had_sweep",
                 "displacement", "entry_pos_in_zone", "stop_atr", "hour_frac",
                 "htf_trend_aligned", "htf_strength_aligned", "session_id"],
}

# FFM market features (the 76 every strategy model sees), grouped by family for
# readability. Dropped: vty_atr_raw (raw price points — leaks price scale) and the
# duplicate `_1h_structure` / `_daily_structure` columns.
_FFM_DROP = {"vty_atr_raw", "_1h_structure", "_daily_structure"}
_FFM_GROUPS = (("bar", "bar"), ("ret", "returns"), ("vol", "volume"),
               ("vty", "volatility"), ("sess", "session"), ("str", "structure"),
               ("swp", "liquiditySweeps"), ("htf", "higherTimeframe"),
               ("tmp", "time"))


def _sig4(x):
    """4 significant digits; non-finite → None."""
    x = float(x)
    return float(f"{x:.4g}") if np.isfinite(x) else None


def ffm_context(bars, i) -> dict:
    """The 76 FFM features at bar i, grouped by family."""
    out = {}
    for name, val in zip(FFM_COLS, ffm_block(bars, i)):
        if name in _FFM_DROP:
            continue
        group = next((g for p, g in _FFM_GROUPS if name.startswith(p + "_")),
                     "candle")
        out.setdefault(group, {})[name] = _sig4(val)
    return out


def setup_features(s, bars, sig) -> dict | None:
    """The named strategy-specific features its model scored this signal with."""
    names = SETUP_FEATURES.get(s.name)
    if not names or not hasattr(s, "_hand_features"):
        return None
    tail = s._hand_features(bars, sig.bar_index, sig.direction)[len(FFM_COLS):]
    return {n: _sig4(v) for n, v in zip(names, tail)}


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


# ── per-strategy views (computed every bar, fired or not) ───────────────────

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
    _up, mid, _lo = ind.keltner_channel(bars, config.KC_LEN, config.KC_MULT,
                                        config.KC_ATR_P)
    k = config.KC_MID_SLOPE_K
    pos = au(c[i] - mid[i])
    return {"channelPos": _r(pos / config.KC_MULT) if pos is not None else None,
            "midSlopeAtr": au(mid[i] - mid[i - k]) if i >= k else None,
            "gateOpen": bool(adx_i >= config.KC_ADX_THRESH)}


def _view_bos(bars, c, i, au, adx_i, s):
    sh, sl, shi, sli = ind.causal_swings(bars, config.SWING_K)
    return {"toSwingHighAtr": au(sh[i] - c[i]),
            "toSwingLowAtr": au(c[i] - sl[i]),
            "swingRangeAtr": au(sh[i] - sl[i]),
            "swingHighAgeBars": int(i - shi[i]) if shi[i] >= 0 else None,
            "swingLowAgeBars": int(i - sli[i]) if sli[i] >= 0 else None}


def _view_orb(bars, c, i, au, adx_i, s):
    oh, ol = ind.opening_range(bars, config.ORB_BARS, config.ORB_OPEN_MIN,
                               config.ORB_TZ)
    tmin = int(ind.et_minutes(bars, config.ORB_TZ)[i])
    if not (np.isfinite(oh[i]) and np.isfinite(ol[i])):
        return {"active": False}
    return {"active": bool(tmin < config.ORB_CLOSE_MIN),
            "rangeAtr": au(oh[i] - ol[i]),
            "price": "above" if c[i] > oh[i] else "below" if c[i] < ol[i] else "inside",
            "gateOpen": bool(adx_i >= config.ORB_ADX_GATE)}


def _view_cisd_ote(bars, c, i, au, adx_i, s):
    p = getattr(s, "last_pipeline", None)      # from this bar's detect()
    if p is None:
        return {}
    _atr, _hour, htf_sign, htf_str, recs = p
    out = {"htfTrend": ("up" if htf_sign[i] > 0 else "down" if htf_sign[i] < 0
                        else None) if np.isfinite(htf_sign[i]) else None,
           "htfStrengthBps": _r(htf_str[i] * 1e4, 1)}
    rec = next((r for r in reversed(recs) if r["exec_idx"] <= i), None)
    if rec is not None:
        zh = rec["fib_top"] - rec["fib_bot"]
        out["lastSetup"] = {
            "side": LONG if rec["is_long"] else SHORT,
            "barsAgo": int(i - rec["exec_idx"]),
            "priceInZone": _r((c[i] - rec["fib_bot"]) / zh) if zh > 0 else None}
    return out


VIEWS = {"supertrend": _view_supertrend, "ema": _view_ema,
         "keltner": _view_keltner, "bos": _view_bos, "orb": _view_orb,
         "cisd_ote": _view_cisd_ote}


def build_state(bars: pd.DataFrame, subs, fired: dict, memory) -> dict:
    """Jev's view of the last closed bar. `subs` = the strategies Jev listens to,
    `fired` = {name: graded Signal} for those that fired on this bar, `memory` =
    the TrackRecord of every strategy's recent signals and Jev's own picks."""
    c = bars["close"].to_numpy(float)
    v = bars["volume"].to_numpy(float)
    i = len(c) - 1
    a_arr = ind.atr(bars, config.ATR_P)
    a = float(a_arr[i]) if np.isfinite(a_arr[i]) and a_arr[i] > 0 else np.nan

    def au(x):
        """Price distance → ATR units."""
        return _r(x / a) if np.isfinite(a) and np.isfinite(x) else None

    adx_i, adx_slope = adx_pair(bars, i)
    tmin = int(ind.et_minutes(bars, config.ORB_TZ)[i])
    med_atr = np.nanmedian(a_arr[-100:])
    vol_avg = np.mean(v[-21:-1]) if i >= 20 else np.nan

    strategies = {}
    for s in subs:
        view = VIEWS.get(s.name, lambda *a_: {})(bars, c, i, au, adx_i, s)
        sig = fired.get(s.name)
        strategies[s.name] = {
            "view": view,
            "signal": None if sig is None else {
                "side": LONG if sig.direction > 0 else SHORT,
                "modelWinProb": _r(sig.proba),
                "modelExpectedR": _r(sig.r_hat),
                "stopAtr": au(sig.risk),
                "setup": setup_features(s, bars, sig)},
            "trackRecord": memory.summary(s.name),
        }

    return {
        "instrument": config.base_symbol(config.SYMBOL),
        "barMinutes": config.TIMEFRAME_MIN,
        "market": {
            "session": _session(tmin),
            "timeET": f"{tmin // 60:02d}:{tmin % 60:02d}",
            "adx": _r(adx_i, 1),
            "adxSlope": _r(adx_slope, 1),
            "atrBps": _r(a / c[i] * 1e4, 1),
            "atrVsMedian100": _r(a / med_atr) if np.isfinite(med_atr) and med_atr > 0 else None,
            "volumeVsAvg20": _r(v[i] / vol_avg) if np.isfinite(vol_avg) and vol_avg > 0 else None,
            "returnsAtr": {f"last{n}": au(c[i] - c[i - n]) if i >= n else None
                           for n in (1, 5, 20, 60)},
            # oldest..newest, each close minus the current close, in ATR
            "recentClosesAtr": (" ".join(f"{(x - c[i]) / a:.1f}" for x in c[-30:])
                                if np.isfinite(a) else None),
        },
        "context": ffm_context(bars, i),
        "strategies": strategies,
        "yourRecentPicks": memory.summary(JEV),
    }


def build_question(fired: dict, panel) -> dict:
    """Which entry to take: one option per fired strategy (its side + stop come
    with it), plus `none`. `panel` = every strategy name, for the playbook."""
    target = config.JEV_MEMORY_TARGET_R
    criteria = {
        name: f"Take {name}'s {LONG if sig.direction > 0 else SHORT} entry now. "
              f"{PLAYBOOK.get(name, '')}".strip()
        for name, sig in fired.items()}
    criteria[NONE] = ("Take no entry: the evidence doesn't favor any of the fired "
                      "signals, or conditions are poor.")
    return {
        "type": "choice",
        "instructions": {
            "question": "One or more strategies fired an entry on this bar. Which "
                        "entry should the bot take, if any?",
            "goal": f"Trade intraday futures on {config.TIMEFRAME_MIN}-minute bars. "
                    "The chosen entry fills at market on its own side with its own "
                    "protective stop (1R); a separate policy trails the stop after "
                    f"entry. Take an entry only when it is more likely to run "
                    f"+{target:g}R than to hit its stop. Taking none is free.",
            "inputs": "`strategies` is a panel of mechanical strategies — the ones "
                      "that fired AND the ones that did not. Each has a `view` of the "
                      "current market, a `signal` if it fired on this bar (with its "
                      "own trained model's win probability and the named `setup` features "
                      "that model scored), and a `trackRecord`: how "
                      f"its recent signals actually played out (winRate = share that "
                      f"reached +{target:g}R before the stop; `last` = W hit target, "
                      "L stopped, F timed out, oldest to newest). Use the whole panel "
                      "to judge market direction and context: favor entries whose "
                      "strategy is working in the current market and that the other "
                      "views agree with. `yourRecentPicks` is the same record for "
                      "your own past choices. `market` gives regime context in ATR "
                      "units; `context` is the full feature set every strategy "
                      "model sees (bar shape, returns, volume, volatility, session, "
                      "structure, liquidity sweeps, higher-timeframe trend).",
            "playbook": {n: PLAYBOOK[n] for n in panel if n in PLAYBOOK},
        },
        "criteria": criteria,
    }
