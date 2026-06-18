#!/usr/bin/env python3
"""broker.py — TopstepX / ProjectX Gateway broker (implements BrokerClient).

The bot talks to brokers only through the `broker_base.BrokerClient` interface,
so the strategy / exit / bar-loop code is broker-agnostic. To add a broker
(e.g. Rithmic), implement `BrokerClient` in its own module and add a case to
`make_broker()`.
"""
import datetime as dt
from typing import Optional

import pandas as pd
import requests

from broker_base import POSITION_LONG, SIDE, BrokerClient   # re-exported below
from config import API_BASE

# ── ProjectX-specific enums (internal to this broker) ──────────────────
ORDER_TYPE_MARKET = 2
ORDER_TYPE_STOP = 4
ORDER_TYPE_TRAILING_STOP = 5
BRACKET_TYPE_STOP = 4
BRACKET_TYPE_TRAIL = 5
BRACKET_TYPE_LIMIT = 1
ORDER_STATUS_WORKING = 1
UNIT_MINUTE = 2
MIN_BRACKET_TICKS = 4          # TopstepX requires the SL ≥ 4 ticks from the fill

__all__ = ["TopstepXClient", "make_broker", "fetch_contract_specs",
           "SIDE", "POSITION_LONG"]


def _stop_bracket_ticks(side: int, ticks: int) -> int:
    """Signed SL-bracket ticks RELATIVE to the fill: negative for a long (stop
    below), positive for a short (stop above) — TopstepX's convention. Magnitude
    is clamped to the 4-tick broker minimum."""
    mag = max(MIN_BRACKET_TICKS, abs(int(ticks)))
    return -mag if side == SIDE["BUY"] else mag


def _target_bracket_ticks(side: int, ticks: int) -> int:
    """Signed TP-bracket ticks: positive for a long (target above), negative for
    a short (target below)."""
    mag = max(MIN_BRACKET_TICKS, abs(int(ticks)))
    return mag if side == SIDE["BUY"] else -mag


def make_broker() -> BrokerClient:
    """Construct the broker selected by config.BROKER. Add a case here to wire
    up a new broker implementation."""
    import config
    if config.BROKER == "topstepx":
        return TopstepXClient(config.TOPSTEPX_USERNAME, config.TOPSTEPX_API_KEY)
    raise SystemExit(f"unknown broker {config.BROKER!r} (config.BROKER)")


def fetch_contract_specs(symbol: str, live: bool = False):
    """(tick_size, tick_value) for `symbol` from the configured broker API.
    Used by the backtester so contract specs come from the broker, not a
    hard-coded table."""
    b = make_broker()
    b.authenticate()
    return b.get_contract_specs(symbol, live)


class TopstepXClient(BrokerClient):
    """TopstepX / ProjectX Gateway REST broker — a `BrokerClient`."""

    def __init__(self, username: str, api_key: str, base: str = API_BASE):
        self.base = base
        self._username = username
        self._api_key = api_key
        self._http = requests.Session()
        self._token: Optional[str] = None

    def authenticate(self) -> None:
        r = self._post("/Auth/loginKey",
                       {"userName": self._username, "apiKey": self._api_key},
                       auth=False)
        if not r.get("success") or not r.get("token"):
            raise RuntimeError(f"login failed: {r.get('errorMessage', r)}")
        self._token = r["token"]
        self._http.headers["Authorization"] = f"Bearer {self._token}"

    def pick_account(self, selector: str = "") -> dict:
        r = self._post("/Account/search", {"onlyActiveAccounts": True})
        tradable = [a for a in r.get("accounts", []) if a.get("canTrade")]
        if not tradable:
            raise RuntimeError("no tradable account found")
        print("Tradable accounts:")
        for a in tradable:
            print(f"   • {a['name']}  (id={a['id']}, balance=${a.get('balance', '?')})")
        if not selector:
            return tradable[0]                 # default: first tradable
        for a in tradable:                     # match by id OR name
            if str(a["id"]) == str(selector) or a.get("name") == selector:
                return a
        raise RuntimeError(f"account {selector!r} not found among tradable accounts")

    def search_contracts(self, search_text: str, live: bool = False) -> list:
        # /Contract/search — returns up to 20 matching contracts, each with
        # tickSize, tickValue, activeContract, etc.
        r = self._post("/Contract/search",
                       {"searchText": search_text, "live": live})
        return r.get("contracts", [])

    def get_active_contract(self, symbol: str, live: bool = False) -> dict:
        # Contract names look like '<symbol><monthcode><yeardigit>' (e.g.
        # 'NQM6'), so the base symbol is name[:-2]. Letting the broker pick
        # the active month handles contract rolls for free. The returned object
        # carries the authoritative tickSize / tickValue.
        for c in self.search_contracts(symbol, live):
            if c.get("activeContract") and c.get("name", "")[:-2] == symbol:
                return c
        raise RuntimeError(f"no active contract found for {symbol!r}")

    def get_contract_specs(self, symbol: str, live: bool = False):
        """(tick_size, tick_value) for the active contract, from the broker."""
        c = self.get_active_contract(symbol, live)
        return float(c["tickSize"]), float(c["tickValue"])

    def get_bars(self, contract_id: str, minutes: int, limit: int = 300) -> pd.DataFrame:
        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(minutes=minutes * (limit + 2))
        r = self._post("/History/retrieveBars", {
            "contractId": contract_id, "live": False,
            "startTime": start.isoformat(), "endTime": now.isoformat(),
            "unit": UNIT_MINUTE, "unitNumber": minutes,
            "limit": limit, "includePartialBar": False,
        })
        df = pd.DataFrame(r.get("bars", []))
        if df.empty:
            return df
        df = df.rename(columns={"t": "time", "o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)

    def open_position(self, account_id: int, contract_id: str) -> Optional[dict]:
        r = self._post("/Position/searchOpen", {"accountId": account_id})
        for p in r.get("positions", []):
            if p.get("contractId") == contract_id and p.get("size"):
                return p
        return None

    def place_market_with_brackets(self, account_id: int, contract_id: str, *,
                                   side: int, size: int,
                                   stop_ticks: int, target_ticks: int) -> dict:
        # One market entry; the gateway attaches + OCO-links the stop and
        # take-profit (distances are TICKS relative to the fill).
        r = self._post("/Order/place", {
            "accountId": account_id, "contractId": contract_id,
            "type": ORDER_TYPE_MARKET, "side": side, "size": size,
            "stopLossBracket": {"ticks": _stop_bracket_ticks(side, stop_ticks),
                                "type": BRACKET_TYPE_STOP},
            "takeProfitBracket": {"ticks": _target_bracket_ticks(side, target_ticks),
                                  "type": BRACKET_TYPE_LIMIT},
        })
        if not r.get("success"):
            raise RuntimeError(f"order rejected: {r.get('errorMessage', r)}")
        return r

    def place_market_with_stop(self, account_id: int, contract_id: str, *,
                               side: int, size: int, stop_ticks: int) -> dict:
        # Market entry with only a protective stop attached (no take-profit) —
        # the PPO trailing exit manages the stop bar-by-bar from here.
        r = self._post("/Order/place", {
            "accountId": account_id, "contractId": contract_id,
            "type": ORDER_TYPE_MARKET, "side": side, "size": size,
            "stopLossBracket": {"ticks": _stop_bracket_ticks(side, stop_ticks),
                                "type": BRACKET_TYPE_STOP},
        })
        if not r.get("success"):
            raise RuntimeError(f"order rejected: {r.get('errorMessage', r)}")
        return r

    def place_market_with_trail(self, account_id: int, contract_id: str, *,
                                side: int, size: int, trail_ticks: int) -> dict:
        # Market entry with a broker-native TRAILING stop attached: the gateway
        # keeps the stop `trail_ticks` behind the best price automatically. The
        # PPO updates trail_ticks each bar (see modify_trail_price).
        r = self._post("/Order/place", {
            "accountId": account_id, "contractId": contract_id,
            "type": ORDER_TYPE_MARKET, "side": side, "size": size,
            "stopLossBracket": {"ticks": _stop_bracket_ticks(side, trail_ticks),
                                "type": BRACKET_TYPE_TRAIL},
        })
        if not r.get("success"):
            raise RuntimeError(f"order rejected: {r.get('errorMessage', r)}")
        return r

    def working_stop_order(self, account_id: int, contract_id: str) -> Optional[dict]:
        # Find the live protective stop (plain or trailing) for this contract.
        r = self._post("/Order/searchOpen", {"accountId": account_id})
        for o in r.get("orders", []):
            if (o.get("contractId") == contract_id
                    and o.get("type") in (ORDER_TYPE_STOP, ORDER_TYPE_TRAILING_STOP)
                    and o.get("status", ORDER_STATUS_WORKING) == ORDER_STATUS_WORKING):
                return o
        return None

    def modify_stop_price(self, account_id: int, order_id: int,
                          stop_price: float) -> dict:
        r = self._post("/Order/modify", {
            "accountId": account_id, "orderId": order_id,
            "stopPrice": stop_price,
        })
        if not r.get("success"):
            raise RuntimeError(f"stop modify rejected: {r.get('errorMessage', r)}")
        return r

    def modify_trail_price(self, account_id: int, order_id: int,
                           trail_price: float) -> dict:
        # Tighten a native trailing stop's follow distance. NOTE: /Order/modify
        # exposes trailPrice as a DECIMAL price distance (the bracket is created
        # in ticks, but the modify field is a price), so callers pass
        # trail_ticks * tick_size here, not a raw tick count.
        r = self._post("/Order/modify", {
            "accountId": account_id, "orderId": order_id,
            "trailPrice": trail_price,
        })
        if not r.get("success"):
            raise RuntimeError(f"trail modify rejected: {r.get('errorMessage', r)}")
        return r

    def cancel_order(self, account_id: int, order_id: int) -> dict:
        r = self._post("/Order/cancel", {
            "accountId": account_id, "orderId": order_id,
        })
        if not r.get("success"):
            raise RuntimeError(f"cancel rejected: {r.get('errorMessage', r)}")
        return r

    def close_position(self, account_id: int, contract_id: str, price=None) -> dict:
        # Flatten the whole position at market. `price` is ignored — the broker
        # fills at market (it's only a fill hint for the backtest sim).
        r = self._post("/Position/closeContract", {
            "accountId": account_id, "contractId": contract_id,
        })
        if not r.get("success"):
            raise RuntimeError(f"close rejected: {r.get('errorMessage', r)}")
        # A market close does NOT fire a bracket leg, so the OCO won't auto-cancel
        # — sweep and cancel every resting order for this contract so the
        # protective stop/TP can't orphan and later fill into a NAKED position.
        try:
            orders = self._post("/Order/searchOpen",
                                {"accountId": account_id}).get("orders", [])
        except Exception:
            orders = []
        for o in orders:
            if (o.get("contractId") == contract_id
                    and o.get("status", ORDER_STATUS_WORKING) == ORDER_STATUS_WORKING):
                try:
                    self.cancel_order(account_id, o["id"])
                except Exception:
                    pass            # best-effort — never fail the close itself
        return r

    def _post(self, path: str, payload: dict, auth: bool = True) -> dict:
        if auth and not self._token:
            raise RuntimeError("not authenticated — call authenticate() first")
        resp = self._http.post(self.base + path, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
