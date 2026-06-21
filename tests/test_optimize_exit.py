"""Optuna exit-config scanner helpers. The objective replays the give-back sim
(TrailExitSim) per config and scores expectancy on a validation slice; these check
the metric/split math and that the replay is actually sensitive to the config knobs
(otherwise the scan is meaningless)."""
import numpy as np

import optimize_exit as oe
import trail_exit_env as tee


def test_metrics_basic():
    R = np.array([1.0, 1.0, -1.0, 2.0, -1.0])
    m = oe._metrics(R)
    assert m["n"] == 5
    assert abs(m["sumR"] - 2.0) < 1e-9
    assert abs(m["wr"] - 0.6) < 1e-9
    assert abs(m["pf"] - (4.0 / 2.0)) < 1e-9       # wins 4R / losses 2R


def test_metrics_empty():
    m = oe._metrics(np.empty(0))
    assert m["n"] == 0 and m["pf"] == 0.0


def test_split_is_time_ordered():
    # n_bars=100 → val = bars [60,80), test = [80,100]
    cat = np.array([[10, 1], [65, 1], [75, -1], [85, 1], [95, -1]])
    val, test = oe._split(cat, 100)
    assert [int(r[0]) for r in val] == [65, 75]
    assert [int(r[0]) for r in test] == [85, 95]


def _arr(close, high, low):
    n = len(close)
    return {"close": np.array(close, float), "high": np.array(high, float),
            "low": np.array(low, float), "atr": np.full(n, 1.0),
            "atr_stop": np.full(n, 4.0), "line": np.zeros(n), "direction": np.zeros(n)}


def test_realized_r_responds_to_giveback(monkeypatch):
    # a long that rallies to a new peak then reverses — the give-back width changes
    # where it exits, so the scanner's objective must move with GIVEBACK_R
    monkeypatch.setattr(tee, "STOP_ATR", 0.5)      # risk = 0.5 × atr_stop(4) = 2.0
    monkeypatch.setattr(tee, "ACTIVATE_R", 2.0)
    close = [100, 104, 106, 100]
    high = [100.5, 104.5, 106.5, 100.5]
    low = [99.5, 103.5, 105.5, 95.0]
    cat = np.array([[0, 1]])
    monkeypatch.setattr(tee, "GIVEBACK_R", 0.5)
    tight = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    monkeypatch.setattr(tee, "GIVEBACK_R", 1.5)
    loose = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    assert tight != loose                          # config genuinely drives the exit


def test_realized_r_responds_to_activate(monkeypatch):
    # below ACTIVATE_R the initial stop holds; raising it past the peak prevents the
    # trail from ever locking in — a different outcome
    monkeypatch.setattr(tee, "STOP_ATR", 0.5)
    monkeypatch.setattr(tee, "GIVEBACK_R", 0.75)
    close = [100, 104, 100, 98]
    high = [100.5, 104.5, 100.5, 98.5]
    low = [99.5, 103.5, 99.5, 97.5]
    cat = np.array([[0, 1]])
    monkeypatch.setattr(tee, "ACTIVATE_R", 1.0)    # activates (peak +2.25R ≥ 1)
    on = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    monkeypatch.setattr(tee, "ACTIVATE_R", 5.0)    # never activates → rides to stop
    off = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    assert on != off
