"""No orphan orders — EVER. After a position is flattened by anything other than a
bracket fill, NO protective order (SL type 4, trailing-SL type 5, OR TP type 1) may
be left working for that contract — a survivor can later fill into a naked position.

Verified exhaustively against a STATEFUL gateway that actually removes cancelled
orders, across every path and shape: the market close, the give-back exit
(`manage_trail`, long and short), the flat reconcile, OCO-miss leftovers,
multi-order brackets, best-effort cancel failures, and the exact live incident
(a +0.63R short close leaving its buy-stop)."""
import types

import pandas as pd
import pytest

import bot
import broker
from ppo_exit import exit_manager
from ppo_exit import trail_exit_env as tee

SL, TP, TRAIL = 4, 1, 5          # ProjectX order types
WORKING, FILLED, CANCELLED = 1, 2, 0


class Gateway(broker.TopstepXClient):
    """In-memory ProjectX gateway: closeContract flattens, cancel removes an order,
    searchOpen returns only still-working orders. `fail_ids` make a cancel reject
    (to exercise the best-effort path)."""
    def __init__(self, orders, fail_ids=()):
        super().__init__("u", "k")
        self._token = "t"
        self.orders = orders
        self.fail_ids = set(fail_ids)
        self.flattened = []

    def _post(self, path, payload, auth=True):
        if path == "/Position/closeContract":
            self.flattened.append(payload["contractId"])
            return {"success": True}
        if path == "/Order/searchOpen":
            return {"orders": [o for o in self.orders if o["status"] == WORKING]}
        if path == "/Order/cancel":
            oid = payload["orderId"]
            if oid in self.fail_ids:
                return {"success": False, "errorMessage": "boom"}
            for o in self.orders:
                if o["id"] == oid:
                    o["status"] = CANCELLED
            return {"success": True}
        return {"success": True}

    def working(self, contract_id):
        return [o for o in self.orders
                if o["contractId"] == contract_id and o["status"] == WORKING]


def _order(id, contract, type_, status=WORKING):
    return {"id": id, "contractId": contract, "type": type_, "status": status}


# ── market close cancels every protective-order shape ──────────────────────

@pytest.mark.parametrize("orders, expect_cleared", [
    ([_order(1, "NQU6", SL)], [1]),                              # SL only
    ([_order(1, "NQU6", TP)], [1]),                              # TP only
    ([_order(1, "NQU6", TRAIL)], [1]),                           # trailing SL
    ([_order(1, "NQU6", SL), _order(2, "NQU6", TP)], [1, 2]),    # full bracket
    ([_order(1, "NQU6", SL), _order(2, "NQU6", SL)], [1, 2]),    # two stops
    ([_order(1, "NQU6", SL), _order(2, "NQU6", TP),
      _order(3, "NQU6", TRAIL)], [1, 2, 3]),                     # SL + TP + trail
])
def test_market_close_cancels_all_working_orders(orders, expect_cleared):
    gw = Gateway(list(orders))
    gw.close_position(7, "NQU6")
    assert gw.working("NQU6") == []
    assert all(o["status"] == CANCELLED for o in gw.orders if o["id"] in expect_cleared)


def test_other_contracts_are_never_touched():
    gw = Gateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP),
                  _order(9, "ESU6", SL), _order(8, "ESU6", TP)])
    gw.close_position(7, "NQU6")
    assert gw.working("NQU6") == []
    assert sorted(o["id"] for o in gw.working("ESU6")) == [8, 9]


def test_oco_miss_stop_filled_tp_orphan_is_swept():
    # the OCO sometimes misses: the stop FILLED but the TP was left working. A
    # subsequent close must still cancel the orphaned TP.
    gw = Gateway([_order(1, "NQU6", SL, status=FILLED), _order(2, "NQU6", TP)])
    gw.close_position(7, "NQU6")
    assert gw.working("NQU6") == []


def test_already_cancelled_orders_are_idempotent():
    gw = Gateway([_order(1, "NQU6", SL, status=CANCELLED), _order(2, "NQU6", TP)])
    gw.close_position(7, "NQU6")
    assert gw.working("NQU6") == []


def test_reported_incident_short_close_leaves_no_buystop():
    # the live bug: a +0.63R short close left its protective buy-stop working, which
    # later filled into a naked long. After the fix the buy-stop must be gone.
    gw = Gateway([_order(1, "NQU6", SL)])      # the buy-stop protecting the short
    gw.close_position(7, "NQU6")
    assert gw.working("NQU6") == []


# ── cancel_orders contract: count + best-effort ────────────────────────────

def test_cancel_orders_returns_count():
    gw = Gateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP), _order(9, "ESU6", SL)])
    assert gw.cancel_orders(7, "NQU6") == 2


def test_cancel_continues_after_one_failure():
    # order 1's cancel rejects; order 2 must still be cancelled (best-effort sweep)
    gw = Gateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP)], fail_ids={1})
    gw.close_position(7, "NQU6")
    assert [o["id"] for o in gw.working("NQU6")] == [1]     # only the failed one lingers
    assert next(o for o in gw.orders if o["id"] == 2)["status"] == CANCELLED


def test_close_succeeds_when_searchopen_fails():
    gw = Gateway([_order(1, "NQU6", SL)])
    real_post = gw._post

    def flaky(path, payload, auth=True):
        if path == "/Order/searchOpen":
            raise RuntimeError("gateway hiccup")
        return real_post(path, payload, auth)
    gw._post = flaky
    assert gw.close_position(7, "NQU6").get("success") is True   # close never fails


# ── give-back exit (manage_trail → market close) leaves no orphan ───────────

def _giveback_bars(last_high, last_low, n=14):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC"),
        "open": [90.0] * n, "high": [101.0] * (n - 1) + [last_high],
        "low": [85.0] * (n - 1) + [last_low], "close": [90.0] * n, "volume": [1] * n,
    })


def _run_manage_trail(gw, st, bars):
    return exit_manager.manage_trail(
        tee, types.SimpleNamespace(trail_mult=lambda obs: 1.0),
        gw, 7, "NQU6", 0.25, bars, st, trailing=False)


def test_short_giveback_exit_leaves_no_orphan():
    gw = Gateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP)])
    st = {"sign": -1, "entry": 100.0, "risk": 10.0, "stop": 110.0, "bars_held": 4,
          "mfe": 2.0, "peak_R": 2.0, "trail_ticks": 40, "strategy": None}
    out = _run_manage_trail(gw, st, _giveback_bars(last_high=92.0, last_low=78.0))
    assert out is None and gw.working("NQU6") == []


def test_long_giveback_exit_leaves_no_orphan():
    gw = Gateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP)])
    st = {"sign": 1, "entry": 100.0, "risk": 10.0, "stop": 90.0, "bars_held": 4,
          "mfe": 2.0, "peak_R": 2.0, "trail_ticks": 40, "strategy": None}
    bars = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=14, freq="3min", tz="UTC"),
        "open": [110.0] * 14, "high": [115.0] * 13 + [122.0],
        "low": [105.0] * 13 + [92.0], "close": [110.0] * 13 + [100.0],
        "volume": [1] * 14,
    })
    out = _run_manage_trail(gw, st, bars)
    assert out is None and gw.working("NQU6") == []


# ── flat reconcile sweeps strays and is idempotent ─────────────────────────

def _flat_ctx(gw):
    return types.SimpleNamespace(
        client=gw, account_id=7, contract_id="NQU6", tick_size=0.25,
        tick_value=5.0, log_candles=False, policy=object(), tee=tee,
        strategies=[], trailing=False)


def _flat_bars():
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=20, freq="3min", tz="UTC"),
        "open": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20,
        "close": [100.0] * 20, "volume": [1] * 20,
    })


class FlatGateway(Gateway):
    def open_position(self, account_id, contract_id):
        return None


def test_flat_reconcile_clears_mixed_orphans(monkeypatch):
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    gw = FlatGateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP),
                      _order(3, "NQU6", TRAIL), _order(9, "ESU6", SL)])
    bot.handle_bar(_flat_ctx(gw), _flat_bars(), None)
    assert gw.working("NQU6") == []
    assert [o["id"] for o in gw.working("ESU6")] == [9]      # untouched


def test_flat_reconcile_is_idempotent(monkeypatch):
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    gw = FlatGateway([_order(1, "NQU6", SL), _order(2, "NQU6", TP)])
    ctx, bars = _flat_ctx(gw), _flat_bars()
    bot.handle_bar(ctx, bars, None)
    bot.handle_bar(ctx, bars, None)                          # second flat bar: nothing left
    assert gw.working("NQU6") == []
