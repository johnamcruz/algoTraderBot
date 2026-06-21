"""PPO give-back exit — the trailing-give-back fix.

manage_trail must: hold the initial stop until peak ≥ ACTIVATE_R; once trailing,
cap the stop at peak − GIVEBACK_R; and if the bar's UNFAVORABLE extreme crossed
the trailed stop, ENFORCE it by closing at market (rather than a modify the broker
could reject and leave a winner riding back to −1R)."""
import pandas as pd

import config
from ppo_exit import trail_exit_env as tee
from broker_base import POSITION_LONG
from ppo_exit.exit_manager import manage_trail, reconstruct_state

TICK = 0.25


class FakePolicy:
    def __init__(self, mult=1.0):
        self.mult = mult

    def trail_mult(self, obs):
        return self.mult


class FakeBroker:
    """Records modify / close / cancel calls; serves one working stop order."""

    def __init__(self, stop_price):
        self.stop = stop_price
        self.modified, self.closed, self.cancelled = [], [], []

    def working_stop_order(self, account_id, contract_id):
        return {"id": 1, "stopPrice": self.stop, "type": 4}

    def modify_stop_price(self, account_id, order_id, stop_price):
        self.modified.append(stop_price)
        self.stop = stop_price
        return {"success": True}

    def modify_trail_price(self, account_id, order_id, trail_price):
        return {"success": True}

    def close_position(self, account_id, contract_id, price=None):
        self.closed.append(price)
        return {"success": True}

    def cancel_order(self, account_id, order_id):
        self.cancelled.append(order_id)
        return {"success": True}


def _bars(last_hlc, n=14):
    """n bars: filler with a wide range (so ATR is finite) + a final (h,l,c) bar."""
    h, l, c = last_hlc
    highs = [101.0] * (n - 1) + [h]
    lows = [85.0] * (n - 1) + [l]
    closes = [90.0] * (n - 1) + [c]
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC"),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": [1] * n,
    })


def _short_state(peak_R=2.0):
    return {"sign": -1, "entry": 100.0, "risk": 10.0, "stop": 110.0,
            "bars_held": 4, "mfe": 2.0, "peak_R": peak_R, "trail_ticks": 40,
            "strategy": None}


def test_giveback_wick_cross_closes_at_market():
    # SHORT +2R, then the bar spikes back UP through the trailed give-back stop.
    # peak from low 78 → +2.2R; cap = 78 + 0.75R(7.5) = 85.5; bar high 92 ≥ 85.5.
    # The floor was tightened THIS bar (not resting), so it's a MARKET close — the
    # realistic fill is the bar CLOSE (90), not the optimistic floor (85.5).
    st = _short_state()
    client = FakeBroker(stop_price=110.0)
    out = manage_trail(tee, FakePolicy(), client, 1, "NQ", TICK,
                       _bars((92.0, 78.0, 90.0)), st, trailing=False)
    assert out is None                       # position closed
    assert len(client.closed) == 1           # via market close, not a modify
    assert abs(client.closed[0] - 90.0) < 0.25   # filled at the bar close (market)
    assert client.modified == []


def test_no_cross_reprices_stop():
    # SHORT in profit, quiet bar — no reversal through the stop → reprice, not close.
    st = _short_state()
    client = FakeBroker(stop_price=110.0)
    out = manage_trail(tee, FakePolicy(), client, 1, "NQ", TICK,
                       _bars((82.0, 82.0, 82.0)), st, trailing=False)
    assert out is st                         # still open
    assert client.closed == []
    assert len(client.modified) == 1 and client.modified[0] < 110.0   # tightened


def test_holds_initial_stop_below_activation():
    # peak only +0.5R (< ACTIVATE_R) → hold the initial stop, no trail action.
    config.ACTIVATE_R = 2.0
    st = _short_state(peak_R=0.5)
    client = FakeBroker(stop_price=110.0)
    out = manage_trail(tee, FakePolicy(), client, 1, "NQ", TICK,
                       _bars((99.0, 97.0, 98.0)), st, trailing=False)
    assert out is st
    assert client.closed == [] and client.modified == []
    assert st["peak_R"] < config.ACTIVATE_R


def test_reconstruct_state_from_open_position():
    pos = {"averagePrice": 100.0, "type": POSITION_LONG}
    client = FakeBroker(stop_price=90.0)        # 10pt below entry → risk 10, long
    st = reconstruct_state(client, 1, "NQ", pos, strategy=None)
    assert st["sign"] == 1 and st["entry"] == 100.0 and abs(st["risk"] - 10.0) < 1e-9
    assert st["stop"] == 90.0 and st["peak_R"] == 0.0


def test_reconstruct_state_rejects_wrong_side_stop():
    pos = {"averagePrice": 100.0, "type": POSITION_LONG}
    client = FakeBroker(stop_price=110.0)       # stop ABOVE entry for a long → invalid
    assert reconstruct_state(client, 1, "NQ", pos, strategy=None) is None
