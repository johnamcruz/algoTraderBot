"""base.Strategy.detect — turns a fired direction into a Signal with the live
entry/stop/risk. This math drives BOTH position sizing (stop_ticks) and the PPO
exit (risk normalizes every observation), so it has to be exact:
    entry = signal-bar close,  risk = STOP_ATR × ATR(ATR_P),  stop = entry − dir×risk."""
import numpy as np
import pandas as pd

import config
import indicators as ind
import strategies.base as base
from strategies.base import Strategy


class _Fake(Strategy):
    name = "fake"

    def __init__(self, direction):
        super().__init__()
        self._dir = direction

    def _fired(self, bars):
        return self._dir

    def _hand_features(self, bars, i, direction):
        return np.zeros(1, np.float32)


def _bars(n=30, last_close=100.0):
    closes = [100.0] * (n - 1) + [last_close]
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC"),
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [1] * n,
    })


def test_no_signal_when_not_fired():
    class Silent(_Fake):
        def _fired(self, bars):
            return None
    assert Silent(0).detect(_bars()) is None


def test_long_signal_stop_below_entry(monkeypatch):
    config.STOP_ATR = 0.5
    bars = _bars(last_close=100.0)
    monkeypatch.setattr(base.ind, "atr", lambda b, p: np.full(len(b), 4.0))
    sig = _Fake(+1).detect(bars)
    assert sig.direction == 1 and sig.entry == 100.0
    assert sig.risk == 2.0                      # 0.5 × ATR(=4)
    assert sig.stop == 98.0                     # entry − risk (below)
    assert sig.bar_index == len(bars) - 1 and sig.strategy == "fake"


def test_short_signal_stop_above_entry(monkeypatch):
    config.STOP_ATR = 0.5
    bars = _bars(last_close=100.0)
    monkeypatch.setattr(base.ind, "atr", lambda b, p: np.full(len(b), 4.0))
    sig = _Fake(-1).detect(bars)
    assert sig.direction == -1 and sig.stop == 102.0      # above entry for a short
    assert sig.risk == 2.0


def test_rejects_nonpositive_atr(monkeypatch):
    bars = _bars()
    monkeypatch.setattr(base.ind, "atr", lambda b, p: np.zeros(len(b)))
    assert _Fake(+1).detect(bars) is None       # ATR=0 → no valid stop distance


def test_rejects_nan_atr(monkeypatch):
    bars = _bars()
    monkeypatch.setattr(base.ind, "atr", lambda b, p: np.full(len(b), np.nan))
    assert _Fake(+1).detect(bars) is None


def test_risk_scales_with_stop_atr(monkeypatch):
    config.STOP_ATR = 1.0
    bars = _bars()
    monkeypatch.setattr(base.ind, "atr", lambda b, p: np.full(len(b), 4.0))
    assert _Fake(+1).detect(bars).risk == 4.0   # 1.0 × ATR
