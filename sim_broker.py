#!/usr/bin/env python3
"""sim_broker.py — a CSV-driven broker that mimics TopstepXClient for backtests.

It implements the order/position methods the bot calls, but fills against
historical bars instead of the API:

  * market entries fill at the signal bar's close
  * a protective stop / trailing stop / fixed-target are simulated bar-by-bar
  * trailing stops follow the best price natively (ratchet-only); the PPO can
    tighten the follow distance via modify_trail_price
  * every closed trade is recorded for the backtest summary

Conservative fills: if a bar's range touches both stop and target, the stop is
assumed first. Open positions at end-of-data are closed at the last close.
"""
from dataclasses import dataclass

from broker import SIDE


@dataclass
class Trade:
    strategy: str
    direction: int          # +1 long / -1 short
    entry_time: object
    entry: float
    exit_time: object
    exit: float
    risk: float
    r: float                # realized R-multiple
    bars_held: int
    reason: str             # "stop" | "target" | "eod"


class SimBroker:
    """Drop-in stand-in for TopstepXClient over a bars DataFrame."""

    def __init__(self, df, tick_size: float):
        self.df = df.reset_index(drop=True)
        self.tick = tick_size
        self.cursor = 0
        self.pos = None        # open-position dict, or None
        self.trades = []

    # ── backtest driver hooks ──────────────────────────────────────────
    def set_bar(self, i: int):
        self.cursor = i

    def process_exits(self):
        """Test the current bar against the working stop/target, close if hit,
        else advance a native trailing stop. Called once per bar before the
        bot acts."""
        if self.pos is None:
            return
        p, i = self.pos, self.cursor
        bar = self.df.iloc[i]
        sign, hi, lo = p["sign"], bar["high"], bar["low"]

        hit_stop = (lo <= p["stop"]) if sign > 0 else (hi >= p["stop"])
        if hit_stop:
            self._close(p["stop"], i, "stop")
            return
        if p.get("target") is not None:
            hit_tp = (hi >= p["target"]) if sign > 0 else (lo <= p["target"])
            if hit_tp:
                self._close(p["target"], i, "target")
                return

        p["bars_held"] += 1
        if p.get("trailing"):          # native trailing stop ratchets to best price
            dist = p["trail_ticks"] * self.tick
            if sign > 0:
                p["best"] = max(p["best"], hi)
                p["stop"] = max(p["stop"], p["best"] - dist)
            else:
                p["best"] = min(p["best"], lo)
                p["stop"] = min(p["stop"], p["best"] + dist)

    def close_open(self):
        """Force-close any open position at the last bar's close (end of data)."""
        if self.pos is not None:
            self._close(float(self.df.iloc[self.cursor]["close"]),
                        self.cursor, "eod")

    def tag_strategy(self, name: str):
        if self.pos is not None and self.pos.get("strategy") is None:
            self.pos["strategy"] = name

    def _close(self, price, i, reason):
        p = self.pos
        r = p["sign"] * (price - p["entry"]) / p["risk"]
        self.trades.append(Trade(
            strategy=p.get("strategy") or "?", direction=p["sign"],
            entry_time=self.df.iloc[p["entry_idx"]]["time"], entry=p["entry"],
            exit_time=self.df.iloc[i]["time"], exit=price, risk=p["risk"],
            r=float(r), bars_held=p["bars_held"], reason=reason))
        self.pos = None

    # ── TopstepXClient-compatible surface ──────────────────────────────
    def open_position(self, account_id, contract_id):
        if self.pos is None:
            return None
        return {"contractId": contract_id, "size": self.pos["size"],
                "averagePrice": self.pos["entry"],
                "type": 1 if self.pos["sign"] > 0 else 2}

    def _enter(self, side, size, stop_ticks, *, trailing, target_ticks=None):
        sign = 1 if side == SIDE["BUY"] else -1
        entry = float(self.df.iloc[self.cursor]["close"])
        risk = stop_ticks * self.tick
        self.pos = {
            "sign": sign, "size": size, "entry": entry, "entry_idx": self.cursor,
            "risk": risk, "stop": entry - sign * risk, "bars_held": 0,
            "best": entry, "trailing": trailing, "trail_ticks": stop_ticks,
            "target": (entry + sign * target_ticks * self.tick
                       if target_ticks else None),
            "strategy": None,
        }

    def place_market_with_stop(self, account_id, contract_id, *, side, size, stop_ticks):
        self._enter(side, size, stop_ticks, trailing=False)
        return {"success": True}

    def place_market_with_trail(self, account_id, contract_id, *, side, size, trail_ticks):
        self._enter(side, size, trail_ticks, trailing=True)
        return {"success": True}

    def place_market_with_brackets(self, account_id, contract_id, *, side, size,
                                   stop_ticks, target_ticks):
        self._enter(side, size, stop_ticks, trailing=False, target_ticks=target_ticks)
        return {"success": True}

    def working_stop_order(self, account_id, contract_id):
        if self.pos is None:
            return None
        return {"id": 1, "stopPrice": self.pos["stop"],
                "type": 5 if self.pos["trailing"] else 4}

    def modify_stop_price(self, account_id, order_id, stop_price):
        if self.pos is not None:
            self.pos["stop"] = stop_price
        return {"success": True}

    def modify_trail_price(self, account_id, order_id, trail_price):
        if self.pos is not None:
            self.pos["trail_ticks"] = max(1, round(trail_price / self.tick))
        return {"success": True}
