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
