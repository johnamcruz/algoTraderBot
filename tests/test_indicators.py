"""Indicator correctness + CAUSALITY. These feed both the entry triggers and the
model features, so a look-ahead bug here silently corrupts every signal. The key
invariant: a value at bar i must depend only on bars ≤ i."""
import numpy as np
import pandas as pd

import indicators as ind


def _bars(highs, lows, closes, tz="UTC", start="2026-01-01 00:00"):
    n = len(closes)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="3min", tz=tz),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": [100] * n,
    })


def test_ema_matches_pandas_ewm():
    c = np.array([10, 11, 12, 13, 14, 13, 12, 11, 10, 9], float)
    got = ind.ema(c, 3)
    want = pd.Series(c).ewm(span=3, adjust=False).mean().to_numpy()
    assert np.allclose(got, want)
    assert len(got) == len(c)


def test_keltner_bands_straddle_mid():
    n = 40
    closes = list(100 + np.sin(np.arange(n) / 3.0) * 2)
    bars = _bars([c + 1 for c in closes], [c - 1 for c in closes], closes)
    up, mid, lo = ind.keltner_channel(bars, 20, 1.5, 20)
    i = n - 1
    assert lo[i] < mid[i] < up[i]
    assert np.isclose(mid[i], ind.ema(np.array(closes), 20)[i])


def test_et_minutes_open_is_570():
    # 09:30 America/New_York → 570 minutes from midnight
    bars = _bars([1], [1], [1], tz="America/New_York", start="2026-06-15 09:30")
    assert ind.et_minutes(bars, "America/New_York")[0] == 570


def test_opening_range_inactive_then_active():
    # 8 bars from 09:30 ET; range = first 5 bars' high/low, active from bar 5 on
    highs = [101, 102, 103, 102, 101, 100, 100, 100]
    lows = [99, 98, 99, 99, 99, 99, 99, 99]
    closes = [100] * 8
    bars = _bars(highs, lows, closes, tz="America/New_York", start="2026-06-15 09:30")
    oh, ol = ind.opening_range(bars, orb_bars=5, open_min=570, tz="America/New_York")
    assert np.all(np.isnan(oh[:5]))                 # not active until the range closes
    assert oh[5] == 103 and ol[5] == 98             # high/low of the first 5 bars
    assert oh[7] == 103 and ol[7] == 98             # stays active through the day


def test_causal_swings_are_strictly_causal():
    # a clear swing high at index 3 (value 5); with k=2 it confirms at index 5
    highs = [1, 2, 3, 5, 3, 2, 1, 1, 1]
    lows = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    closes = [1, 2, 3, 4, 3, 2, 1, 1, 1]
    bars = _bars(highs, lows, closes)
    sh, _sl, shi, _sli = ind.causal_swings(bars, k=2)
    assert np.isnan(sh[4])                          # not yet confirmed (center+k = 5)
    assert sh[5] == 5 and shi[5] == 3               # confirmed at bar 5, points back to 3
    assert sh[6] == 5                               # stays the last confirmed swing high


def test_causal_swings_no_lookahead_on_prefix():
    # computing on a prefix must equal the prefix of the full computation
    highs = [1, 2, 3, 5, 3, 2, 1, 4, 6, 4, 2]
    lows = [0] * 11
    closes = [1, 2, 3, 4, 3, 2, 1, 3, 5, 3, 2]
    bars = _bars(highs, lows, closes)
    full_sh = ind.causal_swings(bars, k=2)[0]
    for cut in (6, 8, 10):
        pre_sh = ind.causal_swings(bars.iloc[:cut], k=2)[0]
        assert np.allclose(pre_sh, full_sh[:cut], equal_nan=True)


def test_atr_and_adx_finite_after_warmup():
    n = 60
    rng = np.arange(n)
    closes = list(100 + np.sin(rng / 4.0) * 5)
    bars = _bars([c + 2 for c in closes], [c - 2 for c in closes], closes)
    a = ind.atr(bars, 14)
    dx = ind.adx(bars, 14)
    assert np.isfinite(a[-1]) and a[-1] > 0
    assert np.isfinite(dx[-1]) and len(a) == len(dx) == n
