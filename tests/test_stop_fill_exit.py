"""Broker-stop-fill exit logging — the observability fix.

When the broker's RESTING protective stop fills, the position is gone before the
bot can manage it (manage_trail never runs), so the exit was SILENT: a −1R loser
closed by the broker never showed up in the logs (seen live on MGC). The bot now
detects the in-position→flat transition and logs the exit, inferred from the level
the stop rested at."""
import logging

import pandas as pd

import bot
from ppo_exit import exit_manager as ex
from ppo_exit import trail_exit_env as tee


# ── stop_fill_exit math (pure) ─────────────────────────────────────────────

def test_long_initial_stop_is_minus_one_r():
    px, r = ex.stop_fill_exit({"sign": 1, "entry": 100.0, "risk": 10.0, "stop": 90.0})
    assert px == 90.0 and abs(r + 1.0) < 1e-9


def test_short_initial_stop_is_minus_one_r():
    px, r = ex.stop_fill_exit({"sign": -1, "entry": 100.0, "risk": 10.0, "stop": 110.0})
    assert px == 110.0 and abs(r + 1.0) < 1e-9


def test_trailed_long_stop_locks_positive_r():
    # a trailed stop above entry → a locked-in winner, not −1R
    px, r = ex.stop_fill_exit({"sign": 1, "entry": 100.0, "risk": 10.0, "stop": 112.5})
    assert px == 112.5 and abs(r - 1.25) < 1e-9


def test_trailed_short_stop_locks_positive_r():
    _px, r = ex.stop_fill_exit({"sign": -1, "entry": 100.0, "risk": 10.0, "stop": 87.5})
    assert abs(r - 1.25) < 1e-9


# ── handle_bar logs + clears on a silent broker stop fill ──────────────────

class _FlatClient:
    """Broker shows no open position; records reconcile sweeps."""
    def __init__(self):
        self.cancel_calls = []

    def open_position(self, account_id, contract_id):
        return None

    def cancel_orders(self, account_id, contract_id):
        self.cancel_calls.append((account_id, contract_id))
        return 0


def _ctx(client):
    import types
    return types.SimpleNamespace(
        client=client, account_id=7, contract_id="NQ", tick_size=0.25,
        tick_value=5.0, log_candles=False, policy=object(), tee=tee,
        strategies=[], trailing=False)


def _bars(n=20):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=n, freq="3min", tz="UTC"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1] * n,
    })


def _capture_bot_log():
    msgs = []

    class H(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    h = H()
    bot.log.addHandler(h)
    return msgs, h


def test_silent_broker_stop_fill_is_logged_and_cleared():
    client = _FlatClient()
    # we were holding a LONG with the stop trailed to +1.25R when the broker filled it
    st = {"sign": 1, "entry": 100.0, "risk": 10.0, "stop": 112.5,
          "bars_held": 7, "strategy": None}
    msgs, h = _capture_bot_log()
    try:
        out = bot.handle_bar(_ctx(client), _bars(), st)
    finally:
        bot.log.removeHandler(h)
    assert out is None                                   # state cleared
    assert client.cancel_calls == [(7, "NQ")]            # reconcile still ran
    exit_logs = [m for m in msgs if "broker stop filled" in m]
    assert len(exit_logs) == 1
    assert "+1.25R" in exit_logs[0] and "LONG" in exit_logs[0]


def test_flat_with_no_prior_trade_does_not_log_exit():
    client = _FlatClient()
    msgs, h = _capture_bot_log()
    try:
        out = bot.handle_bar(_ctx(client), _bars(), None)   # already flat, no trade
    finally:
        bot.log.removeHandler(h)
    assert out is None
    assert not any("broker stop filled" in m for m in msgs)   # nothing to report
