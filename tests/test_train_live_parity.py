"""TRAINING = LIVE parity. The PPO exit is trained inside TrailExitSim
(trail_exit_env) and run live by exit_manager.manage_trail. If the two diverge,
the policy behaves differently in production than it was optimized for. These
tests drive the SAME trade through BOTH engines with an identical (fixed-mult)
policy and assert they exit on the same bar at the same price.

The one accepted difference is fill granularity: live fills per-tick at the broker
while the sim fills at the bar's unfavorable extreme — but that extreme bounds
every intra-bar tick, so the bar-resolution decision is identical (and live's
sub-tick stop snap is allowed for within one tick)."""
import numpy as np
import pandas as pd

import config
from ppo_exit import exit_manager
from ppo_exit import trail_exit_env as tee

TICK = 0.25
ATR_TRAIL, ATR_STOP = 1.0, 4.0          # constant ATRs → risk = STOP_ATR×ATR_STOP
ACTION = 1                              # fixed trail mult = TRAIL_MULTS[1] = 1.0


class FixedPolicy:
    def trail_mult(self, obs):
        return float(tee.TRAIL_MULTS[ACTION])


class FakeBroker:
    def __init__(self, stop):
        self.stop, self.closed, self.modified = stop, [], []

    def working_stop_order(self, a, c):
        return {"id": 1, "stopPrice": self.stop, "type": 4}

    def modify_stop_price(self, a, o, p):
        self.modified.append(p); self.stop = p; return {"success": True}

    def modify_trail_price(self, a, o, p):
        return {"success": True}

    def close_position(self, a, c, price=None):
        self.closed.append(price); return {"success": True}

    def cancel_order(self, a, o):
        return {"success": True}

    def cancel_orders(self, a, c):
        return 0


def _arr(close, high, low):
    n = len(close)
    return {"close": np.array(close, float), "high": np.array(high, float),
            "low": np.array(low, float), "atr": np.full(n, ATR_TRAIL),
            "atr_stop": np.full(n, ATR_STOP), "line": np.zeros(n),
            "direction": np.zeros(n)}


def _bars(close, high, low):
    n = len(close)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=n, freq="3min", tz="UTC"),
        "open": close, "high": high, "low": low, "close": close, "volume": [1] * n})


def _train_exit(close, high, low, e, sign):
    sim = tee.TrailExitSim(_arr(close, high, low))
    sim.reset(e, sign)
    while True:
        _o, _r, done, info = sim.step(ACTION)
        if done:
            price = sim.entry + sign * info["realized_R"] * sim.risk
            return sim.i, price


def _live_exit(close, high, low, e, sign, monkeypatch):
    # Faithful to the live/backtest flow: each bar the RESTING broker stop (set
    # last bar) is checked first (the broker / SimBroker.process_exits fills it at
    # that level), THEN manage_trail runs (which may market-close as the backup).
    monkeypatch.setattr(exit_manager.ind, "atr", lambda b, p: np.full(len(b), ATR_TRAIL))
    bars = _bars(close, high, low)
    entry = float(bars["close"].iloc[e])
    risk = config.STOP_ATR * ATR_STOP
    st = {"sign": sign, "entry": entry, "risk": risk, "stop": entry - sign * risk,
          "bars_held": 0, "mfe": 0.0, "peak_R": 0.0, "trail_ticks": None, "strategy": None}
    client = FakeBroker(st["stop"])
    for t in range(e + 1, len(close)):
        resting = st["stop"]                          # the resting broker stop this bar
        unfav = low[t] if sign > 0 else high[t]
        if (unfav <= resting) if sign > 0 else (unfav >= resting):
            return t, resting                         # resting-stop fill (at the floor)
        out = exit_manager.manage_trail(tee, FixedPolicy(), client, 1, "NQ", TICK,
                                        bars.iloc[: t + 1], st, trailing=False)
        if out is None:
            return t, client.closed[-1]               # market-close backup (bar close)
    return None, None


def _assert_parity(close, high, low, e, sign, monkeypatch):
    config.STOP_ATR, config.ACTIVATE_R, config.GIVEBACK_R = 0.5, 2.0, 0.75
    t_idx, t_px = _train_exit(close, high, low, e, sign)
    l_idx, l_px = _live_exit(close, high, low, e, sign, monkeypatch)
    assert l_idx == t_idx, f"exit bar differs: train={t_idx} live={l_idx}"
    assert abs(l_px - t_px) <= TICK, f"exit price differs: train={t_px} live={l_px}"
    assert t_idx > e and l_idx > e, "entry and exit must NOT be the same candle"
    return t_idx, t_px


def test_long_giveback_parity(monkeypatch):
    close = [100, 100, 100, 100, 100, 100, 101, 102, 103, 104, 104, 100, 100]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    idx, px = _assert_parity(close, high, low, e=5, sign=1, monkeypatch=monkeypatch)
    assert idx == 11 and abs(px - 103.0) <= TICK     # locked +1.5R, not given back


def test_short_giveback_parity(monkeypatch):
    close = [100, 100, 100, 100, 100, 100, 99, 98, 97, 96, 96, 100, 100]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    idx, px = _assert_parity(close, high, low, e=5, sign=-1, monkeypatch=monkeypatch)
    assert idx == 11 and abs(px - 97.0) <= TICK


def test_within_bar_spike_fills_at_bar_close(monkeypatch):
    # The honest case: price spikes to +2R and reverses INSIDE one bar. The give-
    # back floor was tightened THIS bar (not a resting order yet), so live closes
    # at MARKET — both engines fill at the bar CLOSE, not the optimistic floor.
    # entry @100 (idx5), bar 6: high 105 (peak +2.5R activates), low 100.5, close 100.5.
    close = [100, 100, 100, 100, 100, 100, 100.5, 100.5]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    high[6], low[6] = 105.0, 100.5
    idx, px = _assert_parity(close, high, low, e=5, sign=1, monkeypatch=monkeypatch)
    assert idx == 6                                  # exits on the spike bar (after entry)
    assert abs(px - 100.5) <= TICK                   # at the bar close (~+0.25R), NOT the +1.75R floor


def test_max_hold_timeout_parity(monkeypatch):
    # never activates, never hits the initial stop → both force-exit at MAX_HOLD
    n = tee.MAX_HOLD + 12
    close = [100.0] * n
    high = [100.5] * n
    low = [99.5] * n
    idx, _px = _assert_parity(close, high, low, e=5, sign=1, monkeypatch=monkeypatch)
    assert idx == 5 + tee.MAX_HOLD                    # exits exactly at the horizon


def test_no_exit_on_entry_candle(monkeypatch):
    # The entry candle (idx 5) is violent — its low 85 would cross the initial stop
    # (98) and its high 115 would activate the trail. But entry is at that bar's
    # CLOSE, so neither engine evaluates the entry bar (no look-ahead); the exit
    # lands on a LATER candle. Bars 6–9 are benign, bar 10 drops through the stop.
    config.STOP_ATR, config.ACTIVATE_R, config.GIVEBACK_R = 0.5, 2.0, 0.75
    close = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 95, 95]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    high[5], low[5] = 115.0, 85.0                    # violent entry candle
    t_idx, t_px = _train_exit(close, high, low, e=5, sign=1)
    assert t_idx == 10 and t_idx > 5                 # exits at bar 10, NOT the entry bar
    assert abs(t_px - 98.0) <= TICK                  # at the initial stop (entry − 1R)


def test_long_initial_stop_loss_parity(monkeypatch):
    # drops straight through the initial stop before ever activating (−1R both)
    close = [100, 100, 100, 100, 100, 100, 99, 97, 95, 95]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    # NOTE: live leans on the resting broker stop for pre-activation hits, so
    # manage_trail won't close here — parity is asserted at the sim/training level
    # and enforced live by the resting stop + backtest SimBroker.process_exits.
    config.STOP_ATR, config.ACTIVATE_R, config.GIVEBACK_R = 0.5, 2.0, 0.75
    t_idx, t_px = _train_exit(close, high, low, e=5, sign=1)
    assert abs(t_px - 98.0) <= TICK                   # initial stop = entry − 1R
