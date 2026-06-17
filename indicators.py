#!/usr/bin/env python3
"""indicators.py — signal indicators.

The SuperTrend / ATR / ADX primitives are reused directly from the public
`futures_foundation` library so the bot's live signals match exactly how the
models were trained. EMA is a trivial recursive (causal) helper.
"""
import numpy as np
import pandas as pd

from futures_foundation.chronos._primitives import (
    compute_adx, compute_atr, compute_supertrend,
)


def _hlc(bars: pd.DataFrame):
    return (bars["high"].to_numpy(float),
            bars["low"].to_numpy(float),
            bars["close"].to_numpy(float))


def atr(bars: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder ATR (float64[n], NaN before `period`)."""
    h, l, c = _hlc(bars)
    return compute_atr(h, l, c, period)


def supertrend(bars: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Returns (line, direction): direction +1 bull / -1 bear (a flip is a
    change between adjacent bars); line = the SuperTrend trailing level."""
    h, l, c = _hlc(bars)
    direction, line, _atr = compute_supertrend(h, l, c, period, mult)
    return np.asarray(line, dtype=float), np.asarray(direction, dtype=float)


def adx(bars: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ADX (float64[n], NaN early)."""
    h, l, c = _hlc(bars)
    return compute_adx(h, l, c, period)


def ema(close, span: int) -> np.ndarray:
    """Causal recursive EMA (adjust=False → strictly trailing)."""
    return pd.Series(np.asarray(close, dtype=float)).ewm(
        span=span, adjust=False).mean().to_numpy()


def keltner_channel(bars: pd.DataFrame, ema_len: int = 20, mult: float = 1.5,
                    atr_p: int = 20):
    """Causal Keltner channel → (upper, mid, lower). mid = EMA(close); band =
    mid ± mult × ATR(atr_p)."""
    mid = ema(bars["close"].to_numpy(float), ema_len)
    a = atr(bars, atr_p)
    return mid + mult * a, mid, mid - mult * a


def _et(bars: pd.DataFrame, tz: str):
    ts = pd.DatetimeIndex(bars["time"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert(tz)
    return et.normalize().asi8, (et.hour * 60 + et.minute).to_numpy()


def opening_range(bars: pd.DataFrame, orb_bars: int = 5, open_min: int = 570,
                  tz: str = "America/New_York"):
    """Causal opening range → (or_high, or_low). At bar i, the high/low of the
    first `orb_bars` bars at/after `open_min` (minutes from midnight, in `tz`) of
    i's session day — ACTIVE only from the bar AFTER that window closes (so the
    range is fully in the past). NaN before activation. Uses only bars ≤ i."""
    day, tmin = _et(bars, tz)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    n = len(h)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    cur_day = None
    oh = ol = np.nan
    count = 0
    done = False
    for i in range(n):
        if day[i] != cur_day:
            cur_day, oh, ol, count, done = day[i], np.nan, np.nan, 0, False
        if (not done) and tmin[i] >= open_min:
            oh = h[i] if count == 0 else max(oh, h[i])
            ol = l[i] if count == 0 else min(ol, l[i])
            count += 1
            if count >= orb_bars:
                done = True
        elif done:
            or_high[i], or_low[i] = oh, ol
    return or_high, or_low


def orb_extras(bars: pd.DataFrame, orb_bars: int = 5, open_min: int = 570,
               tz: str = "America/New_York"):
    """Causal per-day ORB context → (sess_open, prior_close, or_avg_vol):
    the session open (from `open_min` on), the prior session's last close, and
    the average volume of the opening-range bars (active after the range closes)."""
    day, tmin = _et(bars, tz)
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    v = bars["volume"].to_numpy(float)
    n = len(o)
    sess_open = np.full(n, np.nan)
    prior_close = np.full(n, np.nan)
    or_avg_vol = np.full(n, np.nan)
    cur_day = None
    day_open = prev_last_close = running_last_close = cur_or_vol = np.nan
    opened = False
    vol_sum, vol_cnt, or_done = 0.0, 0, False
    for i in range(n):
        if day[i] != cur_day:
            prev_last_close = running_last_close
            cur_day, day_open, opened = day[i], np.nan, False
            vol_sum, vol_cnt, or_done, cur_or_vol = 0.0, 0, False, np.nan
        if tmin[i] >= open_min:
            if not opened:
                day_open, opened = o[i], True
            if not or_done:
                vol_sum += v[i]
                vol_cnt += 1
                if vol_cnt >= orb_bars:
                    or_done, cur_or_vol = True, vol_sum / vol_cnt
        sess_open[i] = day_open if opened else np.nan
        prior_close[i] = prev_last_close
        or_avg_vol[i] = cur_or_vol
        running_last_close = c[i]
    return sess_open, prior_close, or_avg_vol


def causal_swings(bars: pd.DataFrame, k: int = 2):
    """Confirmation-lagged confirmed swings (strictly causal). Returns
    (sh, sl, shi, sli): at bar i, the most recent confirmed swing-high/low value
    and its centre-bar index whose confirmation bar (centre + k) is ≤ i; NaN / -1
    until the first confirmed swing. A fractal centred at j needs k strictly
    lower highs (or higher lows) on each side and is confirmed only at j + k."""
    h = bars["high"].to_numpy(np.float64)
    l = bars["low"].to_numpy(np.float64)
    n = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    shi = np.full(n, -1, np.int64)
    sli = np.full(n, -1, np.int64)
    cur_sh = cur_sl = np.nan
    cur_shi = cur_sli = -1
    for i in range(n):
        j = i - k                              # centre; bar j+k == i just closed
        if j - k >= 0:                         # full window [j-k, j+k] exists
            hj, lj = h[j], l[j]
            if np.all(hj > h[j - k:j]) and np.all(hj > h[j + 1:j + k + 1]):
                cur_sh, cur_shi = hj, j
            if np.all(lj < l[j - k:j]) and np.all(lj < l[j + 1:j + k + 1]):
                cur_sl, cur_sli = lj, j
        sh[i], sl[i], shi[i], sli[i] = cur_sh, cur_sl, cur_shi, cur_sli
    return sh, sl, shi, sli
