"""Mid-session reconcile — a flat account must have NO resting orders.

handle_bar sweeps stray working orders whenever it's flat, so an orphaned bracket
(or any desynced order) can't fill into an unmanaged naked position. While IN a
position it must NOT cancel (that would kill the live protective stop)."""
import types

import pandas as pd

import bot
from ppo_exit import trail_exit_env as tee


class RecordingClient:
    """Records cancel_orders calls; reports flat or in-position on demand."""
    def __init__(self, position=None, stray=0):
        self.position = position
        self.stray = stray
        self.cancel_calls = []

    def open_position(self, account_id, contract_id):
        return self.position

    def cancel_orders(self, account_id, contract_id):
        self.cancel_calls.append((account_id, contract_id))
        return self.stray


def _bars(n=20):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=n, freq="3min", tz="UTC"),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1] * n,
    })


def _ctx(client):
    return types.SimpleNamespace(
        client=client, account_id=7, contract_id="NQ", tick_size=0.25,
        tick_value=5.0, log_candles=False, policy=object(), tee=tee,
        strategies=[], trailing=False)          # no strategies → no entry, just reconcile


def test_flat_bar_cancels_stray_orders():
    client = RecordingClient(position=None, stray=2)
    out = bot.handle_bar(_ctx(client), _bars(), None)
    assert out is None
    assert client.cancel_calls == [(7, "NQ")]   # swept exactly once, for our contract


def test_in_position_does_not_cancel():
    pos = {"size": 1, "averagePrice": 100.0, "type": 1}
    client = RecordingClient(position=pos)
    # policy=None → in-position branch returns immediately without reconciling,
    # so the live bracket protecting the open trade is never cancelled.
    ctx = _ctx(client)
    ctx.policy = None
    bot.handle_bar(ctx, _bars(), None)
    assert client.cancel_calls == []
