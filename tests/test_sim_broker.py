"""SimBroker fills — the backtest broker the e2e tests (and `--backtest`) run on.
Verifies the OrderRouter surface: stop/target hits, the trailing ratchet, and the
market-close path used by the give-back exit."""
import pandas as pd

from broker_base import SIDE
from sim_broker import SimBroker

TICK = 0.25


def _sim(bars):
    """bars: list of (high, low, close). open == close for simplicity."""
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=len(bars), freq="3min", tz="UTC"),
        "open": [c for _, _, c in bars],
        "high": [h for h, _, _ in bars],
        "low": [l for _, l, _ in bars],
        "close": [c for _, _, c in bars],
        "volume": [1] * len(bars),
    })
    return SimBroker(df, TICK)


def _step(sim, i):
    sim.set_bar(i)
    sim.process_exits()


def test_long_stop_hit_is_minus_one_r():
    sim = _sim([(100, 100, 100), (101, 89, 95)])
    sim.set_bar(0)
    sim.place_market_with_stop(0, "NQ", side=SIDE["BUY"], size=1, stop_ticks=40)  # stop=90
    _step(sim, 1)                                          # low 89 ≤ 90
    assert sim.pos is None
    assert len(sim.trades) == 1
    t = sim.trades[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_long_target_hit_is_plus_r():
    sim = _sim([(100, 100, 100), (121, 99, 118)])
    sim.set_bar(0)
    sim.place_market_with_brackets(0, "NQ", side=SIDE["BUY"], size=1,
                                   stop_ticks=40, target_ticks=80)   # target=120
    _step(sim, 1)                                          # high 121 ≥ 120
    t = sim.trades[0]
    assert t.reason == "target" and abs(t.r - 2.0) < 1e-9


def test_short_stop_hit():
    sim = _sim([(100, 100, 100), (111, 100, 105)])
    sim.set_bar(0)
    sim.place_market_with_stop(0, "NQ", side=SIDE["SELL"], size=1, stop_ticks=40)  # stop=110
    _step(sim, 1)                                          # high 111 ≥ 110
    t = sim.trades[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_trailing_ratchets_only_up():
    sim = _sim([(100, 100, 100), (110, 104, 108), (112, 100, 101)])
    sim.set_bar(0)
    sim.place_market_with_trail(0, "NQ", side=SIDE["BUY"], size=1, trail_ticks=40)  # 10pt trail
    _step(sim, 1)                                          # best=110 → stop=100
    assert abs(sim.pos["stop"] - 100.0) < 1e-9
    _step(sim, 2)                                          # low 100 ≤ 100 → stop hit, breakeven
    assert sim.pos is None and abs(sim.trades[0].r) < 1e-9


def test_close_position_fills_at_hint_price_and_cancel_is_noop():
    sim = _sim([(100, 100, 100), (115, 100, 112)])
    sim.set_bar(0)
    sim.place_market_with_stop(0, "NQ", side=SIDE["BUY"], size=1, stop_ticks=40)
    sim.set_bar(1)
    assert sim.cancel_order(0, 1) == {"success": True}    # no-op in the sim
    sim.close_position(0, "NQ", price=110.0)              # give-back market close
    t = sim.trades[0]
    assert sim.pos is None
    assert t.reason == "trail" and abs(t.exit - 110.0) < 1e-9 and abs(t.r - 1.0) < 1e-9


def test_bar_touching_both_stop_and_target_assumes_stop_first():
    # conservative fill: if a bar's range spans both levels, the stop wins (−1R)
    sim = _sim([(100, 100, 100), (121, 89, 100)])     # high≥target(120) AND low≤stop(90)
    sim.set_bar(0)
    sim.place_market_with_brackets(0, "NQ", side=SIDE["BUY"], size=1,
                                   stop_ticks=40, target_ticks=80)
    _step(sim, 1)
    t = sim.trades[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_close_open_at_end_of_data():
    sim = _sim([(100, 100, 100), (108, 99, 105)])
    sim.set_bar(0)
    sim.place_market_with_stop(0, "NQ", side=SIDE["BUY"], size=1, stop_ticks=40)
    sim.set_bar(1)
    sim.close_open()
    assert sim.trades[0].reason == "eod" and abs(sim.trades[0].exit - 105.0) < 1e-9
