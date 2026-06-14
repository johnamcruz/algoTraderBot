#!/usr/bin/env python3
"""
supertrend_ai_bot.py — SuperTrend + Chronos AI bot, with a 2R exit.

A complete, single-file TopstepX bot that ties the SuperTrend+Chronos model
(in this folder) into a live bar-by-bar loop:

    detect a SuperTrend flip  →  ask the AI to grade it  →  if good, trade it

    • SuperTrend flip (ATR-band direction change) = the candidate entry
    • Chronos+XGBoost head grades the flip → proba = P(win)
    • Trade only when proba >= 0.35 (the model's production floor)
    • Stop at the SuperTrend line, take-profit at a fixed 2R

Drop this file in the supertrend_chronos folder (next to predict.py), edit
the two lines below, and run it. That's it.

    pip install -r requirements.txt requests pandas
    python supertrend_ai_bot.py

⚠️  EDUCATIONAL — places LIVE orders. Run it on a practice/evaluation
    account first. NQ 3-min only (that's the model's training scope).
"""

# ══════════════════════════════════════════════════════════════════════
#  EDIT THESE  (get the API key from your TopstepX dashboard — it is an
#  API KEY, not your account password)
TOPSTEPX_USERNAME = "your_login_here"
TOPSTEPX_API_KEY  = "your_api_key_here"
# Which account to trade. Leave "" to use the first tradable account, or
# set your account id or name (the bot prints all of them on startup).
ACCOUNT = ""
# ══════════════════════════════════════════════════════════════════════

import datetime as dt
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

# The model's inference helpers — this file lives beside them.
from predict import chronos_embedding, predict

API_BASE = "https://api.topstepx.com/api"

# Strategy settings
SYMBOL = "NQ"
TIMEFRAME_MIN = 3
SIZE = 1
PROBA_FLOOR = 0.35        # enter only when the AI grades the flip >= this.
                          # 0.35 = model's documented floor (more trades to
                          # learn from). Bump to 0.45 for fewer, higher-
                          # conviction entries (holdout: only ~5/1406 hit 0.50).
RR = 2.0                  # take-profit at 2R (fallback exit if no PPO policy)
ST_PERIOD, ST_MULT = 10, 3.0   # SuperTrend ATR period / multiplier

# PPO trailing exit: when ppo_trail_exit.npz is present, the bot enters with a
# protective stop at the SuperTrend line and trails it each bar from the policy
# (see train_ppo_exit.py). Set USE_PPO_EXIT=False to force the fixed-2R bracket.
USE_PPO_EXIT = True
POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ppo_trail_exit.npz")

# Exit order mechanism (only used when USE_PPO_EXIT is on):
#   True  → enter with a broker-native TRAILING stop that auto-follows price
#           tick-by-tick; each bar the PPO tightens its trail DISTANCE via an
#           order-modify (the policy can only ratchet tighter, never looser).
#   False → enter with a plain STOP; each bar the PPO reprices the stop level.
# Either way the per-bar order-modify is what carries the policy's decision.
USE_TRAILING_STOP = True

# ProjectX enums
ORDER_TYPE_MARKET = 2
ORDER_TYPE_STOP = 4
ORDER_TYPE_TRAILING_STOP = 5
BRACKET_TYPE_STOP = 4
BRACKET_TYPE_TRAIL = 5
BRACKET_TYPE_LIMIT = 1
ORDER_STATUS_WORKING = 1
SIDE = {"BUY": 0, "SELL": 1}
UNIT_MINUTE = 2


# ─────────────────────────── broker client ───────────────────────────

class TopstepXClient:
    """Thin REST wrapper around the TopstepX / ProjectX Gateway API."""

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
        # Show what's available so you can see the ids/names to hard-code.
        print("Tradable accounts:")
        for a in tradable:
            print(f"   • {a['name']}  (id={a['id']}, balance=${a.get('balance', '?')})")
        if not selector:
            return tradable[0]                 # default: first tradable
        for a in tradable:                     # match by id OR name
            if str(a["id"]) == str(selector) or a.get("name") == selector:
                return a
        raise RuntimeError(f"account {selector!r} not found among tradable accounts")

    def get_active_contract(self, symbol: str) -> dict:
        # Contract names look like '<symbol><monthcode><yeardigit>' (e.g.
        # 'NQM6'), so the base symbol is name[:-2]. Letting the broker pick
        # the active month handles contract rolls for free.
        r = self._post("/Contract/available", {"live": False})
        for c in r.get("contracts", []):
            if c.get("activeContract") and c.get("name", "")[:-2] == symbol:
                return c
        raise RuntimeError(f"no active contract found for {symbol!r}")

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
            "stopLossBracket": {"ticks": stop_ticks, "type": BRACKET_TYPE_STOP},
            "takeProfitBracket": {"ticks": target_ticks, "type": BRACKET_TYPE_LIMIT},
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
            "stopLossBracket": {"ticks": stop_ticks, "type": BRACKET_TYPE_STOP},
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
            "stopLossBracket": {"ticks": trail_ticks, "type": BRACKET_TYPE_TRAIL},
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

    def _post(self, path: str, payload: dict, auth: bool = True) -> dict:
        if auth and not self._token:
            raise RuntimeError("not authenticated — call authenticate() first")
        resp = self._http.post(self.base + path, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ──────────────────── indicators: SuperTrend + ADX ────────────────────
# Both are standard public indicators. SuperTrend gives the flip + the stop
# line; ADX feeds two of the model's 78 hand-crafted feature slots.

def _wilder_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(bars: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Returns (line, direction) arrays. direction: +1 bull / -1 bear; a flip
    is where direction[i] != direction[i-1]. line = the trailing stop level."""
    hl2 = (bars["high"] + bars["low"]) / 2
    atr = _wilder_atr(bars, period)
    upper = (hl2 + mult * atr).values
    lower = (hl2 - mult * atr).values
    close = bars["close"].values
    n = len(bars)

    fu, fl = upper.copy(), lower.copy()
    for i in range(1, n):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]

    line = np.zeros(n)
    direction = np.ones(n)
    for i in range(1, n):
        if close[i] > fu[i - 1]:
            direction[i] = 1
        elif close[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return line, direction


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    up, down = h.diff(), -l.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    atr = _wilder_atr(bars, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def build_features(bars: pd.DataFrame) -> np.ndarray:
    """The model wants 78 hand-crafted features. The 76 proprietary ones are
    not shipped, so we leave them NaN (XGBoost handles missing natively) and
    fill only the two public ones: adx and adx_slope. The Chronos embedding
    carries most of the signal; this runs at reduced-but-real accuracy."""
    a = adx(bars)
    feats = np.full(78, np.nan, dtype=np.float32)
    feats[76] = float(a.iloc[-1])                    # adx
    feats[77] = float(a.iloc[-1] - a.iloc[-2])       # adx_slope
    return feats


# ───────────────────── PPO trailing-exit management ─────────────────────

POSITION_LONG = 1                      # ProjectX position.type enum

def _reconstruct_state(client, account_id, contract_id, pos) -> Optional[dict]:
    """Rebuild trail state if the bot starts up already in a position (e.g.
    after a restart): infer side/entry from the position and risk from the
    distance to the working stop. bars_held resets to 0."""
    entry = pos.get("averagePrice")
    sign = 1 if pos.get("type") == POSITION_LONG else -1
    order = client.working_stop_order(account_id, contract_id)
    if entry is None or order is None or order.get("stopPrice") is None:
        return None
    stop = float(order["stopPrice"])
    risk = (float(entry) - stop) * sign
    if risk <= 0:
        return None
    return {"sign": sign, "entry": float(entry), "risk": risk,
            "stop": stop, "bars_held": 0, "mfe": 0.0}


def _exit_obs(tee, st: dict, bars: pd.DataFrame, line) -> np.ndarray:
    """Build the policy's observation for the live open position — identical
    layout to TrailExitSim._obs in trail_exit_env.py."""
    sign, entry, risk = st["sign"], st["entry"], st["risk"]
    atr = float(_wilder_atr(bars, tee.ATR_PERIOD).iloc[-1])
    cur = float(bars["close"].iloc[-1])
    prev = float(bars["close"].iloc[-1 - tee.MOM_LOOKBACK])
    unreal = sign * (cur - entry) / risk
    st["mfe"] = max(st["mfe"], unreal)
    obs = np.array([
        unreal,                                   # unrealized R
        st["mfe"],                                # max favorable excursion
        sign * (cur - st["stop"]) / risk,         # stop distance (R)
        atr / risk,                               # volatility vs initial risk
        st["bars_held"] / tee.MAX_HOLD,           # time in trade
        sign * (cur - prev) / risk,               # recent momentum (R)
        sign * (cur - float(line[-1])) / risk,    # distance from SuperTrend line
    ], dtype=np.float32)
    return np.clip(obs, -tee.OBS_CLIP, tee.OBS_CLIP)


def _manage_trail(tee, policy, client, account_id, contract_id, tick_size,
                  bars, line, st: dict, trailing: bool):
    """One bar of exit management: ask the policy how tight to trail, then push
    that to the broker. In `trailing` mode the order natively follows price and
    we only tighten its follow DISTANCE; otherwise we reprice a plain stop. Both
    only ever ratchet in our favor."""
    st["bars_held"] += 1
    sign, atr = st["sign"], float(_wilder_atr(bars, tee.ATR_PERIOD).iloc[-1])
    cur = float(bars["close"].iloc[-1])
    stamp = bars["time"].iloc[-1].strftime("%H:%M")

    mult = policy.trail_mult(_exit_obs(tee, st, bars, line))   # updates st["mfe"]
    trail_dist = mult * atr                                    # price distance
    order = client.working_stop_order(account_id, contract_id)
    if order is None:
        print(f"   {stamp}  no working stop found — skip trail")
        return

    if trailing:
        # Native trailing stop: only ever tighten the follow distance.
        new_ticks = max(1, round(trail_dist / tick_size))
        cur_ticks = st.get("trail_ticks", new_ticks)
        if new_ticks < cur_ticks:
            # trailPrice is a decimal price distance, not a tick count
            client.modify_trail_price(account_id, order["id"],
                                      new_ticks * tick_size)
            st["trail_ticks"] = new_ticks
            print(f"   {stamp}  trail tightened → {new_ticks}t "
                  f"({mult:.2f}x ATR, {st['bars_held']} bars in)")
        else:
            print(f"   {stamp}  hold trail {cur_ticks}t ({mult:.2f}x ATR)")
        # keep a stop estimate for the observation (broker trails from best price)
        best = st["entry"] + sign * st["mfe"] * st["risk"]
        est = best - sign * st.get("trail_ticks", new_ticks) * tick_size
        st["stop"] = max(st["stop"], est) if sign > 0 else min(st["stop"], est)
    else:
        # Plain stop: reprice the level, favorable ratchet only.
        cand = cur - sign * trail_dist
        new_stop = max(st["stop"], cand) if sign > 0 else min(st["stop"], cand)
        if abs(new_stop - st["stop"]) >= tick_size:
            new_stop = round(new_stop / tick_size) * tick_size
            client.modify_stop_price(account_id, order["id"], new_stop)
            st["stop"] = new_stop
            print(f"   {stamp}  trail → stop {new_stop:.2f} "
                  f"({mult:.2f}x ATR, {st['bars_held']} bars in)")
        else:
            print(f"   {stamp}  hold (stop {st['stop']:.2f}, {mult:.2f}x ATR)")


# ─────────────────────────────── the bot ───────────────────────────────

def run():
    import trail_exit_env as tee     # lazy: avoids the import cycle, numpy-only

    policy = None
    if USE_PPO_EXIT and os.path.exists(POLICY_PATH):
        policy = tee.NumpyMlpPolicy.load(POLICY_PATH)
    trailing = bool(policy) and USE_TRAILING_STOP
    exit_mode = ("PPO native-trail" if trailing else
                 "PPO stop-reprice" if policy else "fixed 2R")

    client = TopstepXClient(
        os.environ.get("TOPSTEPX_USERNAME", TOPSTEPX_USERNAME),
        os.environ.get("TOPSTEPX_API_KEY", TOPSTEPX_API_KEY),
    )
    client.authenticate()
    acct = client.pick_account(os.environ.get("TOPSTEPX_ACCOUNT", ACCOUNT))
    contract = client.get_active_contract(SYMBOL)
    account_id, contract_id = acct["id"], contract["id"]
    tick_size = float(contract["tickSize"])
    print(f"✅ {acct['name']} | {contract_id} | {TIMEFRAME_MIN}-min | "
          f"SuperTrend AI ({exit_mode})")
    print("▶ running — Ctrl-C to stop")

    trade_state = None                # tracks the live trade for the trail

    while True:
        # wait for the next bar close (+2s so the API has published it)
        period = TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)

        try:
            bars = client.get_bars(contract_id, TIMEFRAME_MIN)
            if len(bars) < 150:        # need >=128 closes for Chronos + warmup
                continue
            line, direction = supertrend(bars, ST_PERIOD, ST_MULT)
            stamp = bars["time"].iloc[-1].strftime("%H:%M")
            pos = client.open_position(account_id, contract_id)

            if pos:
                # In a trade. With a policy, trail the stop; otherwise the
                # attached 2R bracket manages the exit on its own.
                if policy is None:
                    continue
                if trade_state is None:
                    trade_state = _reconstruct_state(client, account_id,
                                                     contract_id, pos)
                if trade_state:
                    _manage_trail(tee, policy, client, account_id, contract_id,
                                  tick_size, bars, line, trade_state, trailing)
                continue

            trade_state = None         # flat — clear any stale trail state

            flipped = direction[-1] != direction[-2]
            if not flipped:
                print(f"   {stamp}  no flip")
                continue

            is_long = direction[-1] == 1
            entry = float(bars["close"].iloc[-1])
            stop = float(line[-1])                      # SuperTrend line = stop
            risk = entry - stop if is_long else stop - entry
            if risk <= 0:
                print(f"   {stamp}  flip but stop on wrong side — skip")
                continue

            # Grade the flip with the AI: Chronos embedding + XGBoost heads.
            emb = chronos_embedding(bars["close"].values)
            proba, r_hat = predict(emb, build_features(bars))
            side_txt = "LONG" if is_long else "SHORT"
            print(f"   {stamp}  {side_txt} flip | proba={proba:.3f} r_hat={r_hat:.2f}")

            if proba < PROBA_FLOOR:
                print(f"        ↳ below {PROBA_FLOOR} floor — skip")
                continue

            stop_ticks = max(1, round(risk / tick_size))
            side = SIDE["BUY"] if is_long else SIDE["SELL"]
            if policy is not None:
                # Enter with a protective stop only; the trail takes over next bar.
                trade_state = {"sign": 1 if is_long else -1, "entry": entry,
                               "risk": risk, "stop": stop, "bars_held": 0,
                               "mfe": 0.0, "trail_ticks": stop_ticks}
                if trailing:
                    client.place_market_with_trail(
                        account_id, contract_id, side=side, size=SIZE,
                        trail_ticks=stop_ticks)
                    print(f"🎯 ENTER {side_txt} {SIZE} | native trail {stop_ticks}t | PPO")
                else:
                    client.place_market_with_stop(
                        account_id, contract_id, side=side, size=SIZE,
                        stop_ticks=stop_ticks)
                    print(f"🎯 ENTER {side_txt} {SIZE} | stop {stop_ticks}t | PPO reprice")
            else:
                # Fallback: original fixed-2R bracket.
                target_ticks = max(1, round(RR * risk / tick_size))
                client.place_market_with_brackets(
                    account_id, contract_id, side=side, size=SIZE,
                    stop_ticks=stop_ticks, target_ticks=target_ticks)
                print(f"🎯 ENTER {side_txt} {SIZE} | stop {stop_ticks}t | "
                      f"target {target_ticks}t (2R)")

        except Exception as e:        # keep the loop alive on transient errors
            print(f"⚠️  {e}")


if __name__ == "__main__":
    if TOPSTEPX_USERNAME == "your_login_here":
        raise SystemExit("edit TOPSTEPX_USERNAME / TOPSTEPX_API_KEY at the top first")
    run()
