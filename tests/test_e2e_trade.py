"""End-to-end trade lifecycles through the REAL driver (backtest.drive →
bot.handle_bar) over a SimBroker — no network, broker, or Chronos. A fake strategy
supplies signals and a fake policy supplies the trail tightness, so the whole flow
runs: detect → grade → resolve → size → enter → exit.

Covers every branch: long give-back winner, short, stop-out loser, fixed-RR
(no policy), the multi-strategy resolver, the proba floor, and reconstruct-on-
restart (bot starts already in a position)."""
import types

import pandas as pd

import backtest
import bot
import config
from ppo_exit import trail_exit_env as tee
from broker_base import POSITION_LONG
from sim_broker import SimBroker

TICK = 0.25
WIN = 16              # small trailing window for the tests (real default is 500)


class FakePolicy:
    def trail_mult(self, obs):
        return 1.0


class FakeStrategy:
    """Fires one signal (first eligible flat bar), grading it at a fixed proba."""
    def __init__(self, name, direction, risk=10.0, proba=0.90):
        self.name, self.direction, self.risk, self.proba = name, direction, risk, proba
        self.fired = False

    def detect(self, bars):
        if self.fired or len(bars) < 12:
            return None
        self.fired = True
        entry = float(bars["close"].iloc[-1])
        return types.SimpleNamespace(
            direction=self.direction, entry=entry,
            stop=entry - self.direction * self.risk, risk=self.risk,
            bar_index=len(bars) - 1, bar_time=bars["time"].iloc[-1])

    def grade(self, bars, sig, emb=None):
        return self.proba, 5.0


def _df(closes, lead=2.0):
    """Bars from a close series; highs/lows offset so ATR is finite. `lead` is how
    far the high/low extends past the close (favorable extreme drives peak_R)."""
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=len(closes), freq="3min", tz="UTC"),
        "open": closes,
        "high": [c + lead for c in closes],
        "low": [c - lead for c in closes],
        "close": closes,
        "volume": [1] * len(closes),
    })


def _ctx(sim, strategies, *, policy, trailing=False):
    return types.SimpleNamespace(
        client=sim, account_id=0, contract_id="NQ", tick_size=TICK,
        tick_value=5.0, log_candles=False, policy=policy, tee=tee,
        strategies=strategies, trailing=trailing)


def _run(monkeypatch, closes, strategies, *, policy, trailing=False, start=WIN):
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    config.RISK_PER_TRADE = 0.0
    config.SIZE = 1
    config.MAX_CONTRACTS = 5
    config.PROBA_FLOOR = 0.35
    config.ACTIVATE_R = 2.0
    config.GIVEBACK_R = 0.75
    config.RR = 2.0
    sim = SimBroker(_df(closes), TICK)
    ctx = _ctx(sim, strategies, policy=policy, trailing=trailing)
    return backtest.drive(ctx, sim, _df(closes), start_idx=start, window=WIN)


# ── lifecycle branches ────────────────────────────────────────────────────

def test_long_giveback_winner_is_protected(monkeypatch):
    closes = [100.0] * 17 + [104, 108, 112, 116, 120, 116, 110, 104, 100, 98]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=FakePolicy())
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == 1 and t.r >= 1.0       # +2R winner not given back to a loss


def test_entry_and_exit_are_different_candles(monkeypatch):
    # the driver runs process_exits BEFORE entry each bar, so the entry candle is
    # never exit-checked: every trade's exit must be a strictly later candle.
    closes = [100.0] * 17 + [104, 108, 112, 116, 120, 116, 110, 104, 100, 98]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=FakePolicy())
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_time < t.exit_time           # not the same candle


def test_stop_out_also_exits_on_a_later_candle(monkeypatch):
    closes = [100.0] * 17 + [97, 92, 86, 84, 84]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=FakePolicy())
    assert trades and trades[0].entry_time < trades[0].exit_time


def test_short_giveback_winner_is_protected(monkeypatch):
    closes = [100.0] * 17 + [96, 92, 88, 84, 80, 84, 90, 96, 100, 102]
    trades = _run(monkeypatch, closes, [FakeStrategy("short", -1)], policy=FakePolicy())
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == -1 and t.r >= 1.0


def test_stop_out_loser_is_minus_one_r(monkeypatch):
    # long entry @100 (stop 90); price drops straight through before activation
    closes = [100.0] * 17 + [97, 92, 86, 84, 84]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=FakePolicy())
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_fixed_rr_target_win_without_policy(monkeypatch):
    # policy=None → handle_bar places an OCO bracket; price hits the 2R target
    closes = [100.0] * 17 + [106, 112, 118, 122, 122]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=None)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "target" and abs(t.r - config.RR) < 1e-9


def test_fixed_rr_stop_loss_without_policy(monkeypatch):
    closes = [100.0] * 17 + [97, 92, 86, 84, 84]
    trades = _run(monkeypatch, closes, [FakeStrategy("long", +1)], policy=None)
    assert len(trades) == 1
    assert trades[0].reason == "stop" and abs(trades[0].r + 1.0) < 1e-9


def test_resolver_picks_highest_proba(monkeypatch):
    # both fire on the same bar; the 0.90 signal must win over the 0.50 one
    closes = [100.0] * 17 + [104, 108, 112, 116, 120, 116, 110, 104, 100, 98]
    lo = FakeStrategy("lo", +1, proba=0.50)
    hi = FakeStrategy("hi", +1, proba=0.90)
    trades = _run(monkeypatch, closes, [lo, hi], policy=FakePolicy())
    assert len(trades) == 1
    assert trades[0].strategy == "hi"            # highest-proba strategy entered


def test_proba_floor_blocks_weak_signal(monkeypatch):
    closes = [100.0] * 30
    weak = FakeStrategy("weak", +1, proba=0.20)  # below PROBA_FLOOR (0.35)
    trades = _run(monkeypatch, closes, [weak], policy=FakePolicy())
    assert trades == []                          # no entry, no trade


def test_reconstruct_on_restart(monkeypatch):
    # Bot wakes up already in a SimBroker position with trade_state=None: it must
    # reconstruct state from the open position + working stop and then manage it.
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    config.USE_PPO_EXIT = True
    config.ACTIVATE_R = 2.0
    config.GIVEBACK_R = 0.75
    closes = [100.0] * 17 + [104, 108, 112, 116, 120, 116, 110, 104, 100, 98]
    sim = SimBroker(_df(closes), TICK)

    # pre-seed an OPEN long position (entry 100, stop 90) — as if mid-trade
    sim.set_bar(16)
    sim.place_market_with_stop(0, "NQ", side=0, size=1, stop_ticks=40)
    assert sim.open_position(0, "NQ") is not None

    # a strategy that never fires (we're testing management of a pre-existing pos)
    ctx = _ctx(sim, [FakeStrategy("idle", +1)], policy=FakePolicy())
    ctx.strategies[0].fired = True

    trade_state = None
    for i in range(17, len(closes)):
        sim.set_bar(i)
        sim.process_exits()
        if sim.pos is None:
            trade_state = None
        win = _df(closes).iloc[max(0, i - WIN + 1): i + 1]
        before = trade_state
        trade_state = bot.handle_bar(ctx, win, trade_state)
        if before is None and trade_state is not None:
            # state was reconstructed from the live position on this bar
            assert trade_state["entry"] == 100.0 and trade_state["sign"] == 1
    sim.close_open()

    assert len(sim.trades) == 1                  # the reconstructed trade was managed
    assert sim.trades[0].r >= 1.0                # and its winner protected
