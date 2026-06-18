"""Each strategy's mechanical trigger (`_fired`) — does it fire long/short on the
right pattern, stay silent otherwise, and respect its ADX/trend gate? The indicator
MATH is covered in test_indicators; here we feed controlled indicator outputs so
the entry DECISION is deterministic (a wrong trigger = wrong or missed trades)."""
import numpy as np
import pandas as pd

import config
import strategies.bos as bos_mod
import strategies.ema_cross as ema_mod
import strategies.keltner as kc_mod
import strategies.supertrend as st_mod
from strategies.bos import BosStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.keltner import KeltnerAdxStrategy
from strategies.supertrend import SuperTrendStrategy


def _bars(n=6):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1] * n,
    })


# ── SuperTrend: fires only on a direction flip ─────────────────────────────

def test_supertrend_fires_on_flip(monkeypatch):
    bars = _bars()
    monkeypatch.setattr(st_mod.ind, "supertrend",
                        lambda *a, **k: (None, np.array([1, 1, 1, 1, -1, 1])))
    assert SuperTrendStrategy()._fired(bars) == 1          # flipped up on the last bar


def test_supertrend_silent_without_flip(monkeypatch):
    bars = _bars()
    monkeypatch.setattr(st_mod.ind, "supertrend",
                        lambda *a, **k: (None, np.array([1, 1, 1, 1, 1, 1])))
    assert SuperTrendStrategy()._fired(bars) is None


# ── EMA cross: fast crossing slow, gated by ADX ────────────────────────────

def _patch_ema(monkeypatch, fast, slow, adx):
    monkeypatch.setattr(ema_mod.ind, "ema",
                        lambda c, span: np.array(fast if span == config.EMA_FAST else slow))
    monkeypatch.setattr(ema_mod.ind, "adx", lambda *a, **k: np.array(adx))


def test_ema_cross_up(monkeypatch):
    config.ADX_GATE = 0.0
    _patch_ema(monkeypatch, fast=[9, 9, 9, 11], slow=[10, 10, 10, 10], adx=[25] * 4)
    assert EmaCrossStrategy()._fired(_bars(4)) == 1        # fast crossed above slow


def test_ema_cross_down(monkeypatch):
    config.ADX_GATE = 0.0
    _patch_ema(monkeypatch, fast=[11, 11, 11, 9], slow=[10, 10, 10, 10], adx=[25] * 4)
    assert EmaCrossStrategy()._fired(_bars(4)) == -1


def test_ema_cross_blocked_by_adx_gate(monkeypatch):
    config.ADX_GATE = 18.0
    _patch_ema(monkeypatch, fast=[9, 9, 9, 11], slow=[10, 10, 10, 10], adx=[5] * 4)
    assert EmaCrossStrategy()._fired(_bars(4)) is None     # below the trend gate


# ── Keltner: close breaking the band, gated by ADX ─────────────────────────

def _patch_kc(monkeypatch, up, lo, adx):
    monkeypatch.setattr(kc_mod.ind, "keltner_channel",
                        lambda *a, **k: (np.array(up), None, np.array(lo)))
    monkeypatch.setattr(kc_mod.ind, "adx", lambda *a, **k: np.array(adx))


def test_keltner_break_above(monkeypatch):
    config.KC_ADX_THRESH = 0.0
    bars = _bars(4)
    bars["close"] = [100.0, 100.0, 100.0, 106.0]          # pops above the upper band
    _patch_kc(monkeypatch, up=[105] * 4, lo=[95] * 4, adx=[30] * 4)
    assert KeltnerAdxStrategy()._fired(bars) == 1


def test_keltner_break_below(monkeypatch):
    config.KC_ADX_THRESH = 0.0
    bars = _bars(4)
    bars["close"] = [100.0, 100.0, 100.0, 94.0]
    _patch_kc(monkeypatch, up=[105] * 4, lo=[95] * 4, adx=[30] * 4)
    assert KeltnerAdxStrategy()._fired(bars) == -1


def test_keltner_no_break_is_silent(monkeypatch):
    config.KC_ADX_THRESH = 0.0
    bars = _bars(4)
    bars["close"] = [100.0, 100.0, 100.0, 102.0]          # still inside the channel
    _patch_kc(monkeypatch, up=[105] * 4, lo=[95] * 4, adx=[30] * 4)
    assert KeltnerAdxStrategy()._fired(bars) is None


# ── BOS: close crossing a confirmed swing ──────────────────────────────────

def test_bos_break_of_structure_up(monkeypatch):
    bars = _bars(4)
    bars["close"] = [100.0, 100.0, 100.0, 106.0]          # crosses above swing high 105
    monkeypatch.setattr(bos_mod.ind, "causal_swings",
                        lambda *a, **k: (np.array([105.0] * 4), np.array([95.0] * 4),
                                         np.array([-1] * 4), np.array([-1] * 4)))
    assert BosStrategy()._fired(bars) == 1


def test_bos_break_of_structure_down(monkeypatch):
    bars = _bars(4)
    bars["close"] = [100.0, 100.0, 100.0, 94.0]           # crosses below swing low 95
    monkeypatch.setattr(bos_mod.ind, "causal_swings",
                        lambda *a, **k: (np.array([105.0] * 4), np.array([95.0] * 4),
                                         np.array([-1] * 4), np.array([-1] * 4)))
    assert BosStrategy()._fired(bars) == -1
