#!/usr/bin/env python3
"""exit_manager.py — PPO trailing-exit management for an open position.

Strategy-agnostic: the per-bar reference line comes from the strategy that
opened the trade (stored in trade_state). The policy picks a trail tightness;
we push it to the broker as a stop reprice or a native-trail tighten.
"""
import math

import numpy as np
import pandas as pd

import config
import indicators as ind
from broker import POSITION_LONG
from logsetup import get_logger

log = get_logger()


def _snap(price: float, sign: int, tick: float) -> float:
    """Direction-aware tick snap so a rounded stop never lands on the wrong side
    of the market: a long stop sits below price → floor; a short stop above → ceil."""
    return (math.floor(price / tick) * tick if sign > 0
            else math.ceil(price / tick) * tick)


def reconstruct_state(client, account_id, contract_id, pos, strategy) -> dict | None:
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
    return {"sign": sign, "entry": float(entry), "risk": risk, "stop": stop,
            "bars_held": 0, "mfe": 0.0, "peak_R": 0.0, "trail_ticks": None,
            "strategy": strategy}


def stop_fill_exit(st: dict):
    """The broker's RESTING stop filled and flattened us (the bot didn't close the
    trade), so the position is gone before we can manage it. Infer the exit from
    the level the stop order was resting at — that's where it filled — and return
    (exit_price, realized_R). For the initial stop this is −1R; for a trailed stop
    it's the locked-in R."""
    sign, entry, risk, stop = st["sign"], st["entry"], st["risk"], st["stop"]
    return float(stop), float(sign * (stop - entry) / risk)


def exit_obs(tee, st: dict, bars: pd.DataFrame) -> np.ndarray:
    """Policy observation for the live open position — identical layout to
    TrailExitSim._obs in trail_exit_env.py. Strategy-agnostic: it's purely the
    trade's R-state on the standard 0.5×ATR stop, so one policy fits every
    strategy."""
    sign, entry, risk = st["sign"], st["entry"], st["risk"]
    a = float(ind.atr(bars, tee.ATR_PERIOD)[-1])
    cur = float(bars["close"].iloc[-1])
    prev = float(bars["close"].iloc[-1 - tee.MOM_LOOKBACK])
    unreal = sign * (cur - entry) / risk
    st["mfe"] = max(st["mfe"], unreal)
    obs = np.array([
        unreal,                                   # unrealized R
        st["mfe"],                                # max favorable excursion
        sign * (cur - st["stop"]) / risk,         # stop distance (R)
        a / risk,                                 # volatility vs initial risk
        st["bars_held"] / tee.MAX_HOLD,           # time in trade
        sign * (cur - prev) / risk,               # recent momentum (R)
    ], dtype=np.float32)
    return np.clip(obs, -tee.OBS_CLIP, tee.OBS_CLIP)


def manage_trail(tee, policy, client, account_id, contract_id, tick_size,
                 bars, st: dict, trailing: bool):
    """One bar of exit management. Returns the (possibly updated) trade_state, or
    None if the position was closed. Asks the policy how tight to trail, then:
      • anchors the peak to the bar's FAVORABLE extreme (catches intra-bar spikes);
      • if the bar's UNFAVORABLE extreme crossed the trailed stop, ENFORCES it by
        closing at market (the resting broker stop may be stale / wrong-side);
      • otherwise reprices the stop (direction-aware tick snap)."""
    st["bars_held"] += 1
    sign = st["sign"]
    a = float(ind.atr(bars, tee.ATR_PERIOD)[-1])
    bar = bars.iloc[-1]
    cur, hi, lo = float(bar["close"]), float(bar["high"]), float(bar["low"])
    stamp = bars["time"].iloc[-1].strftime("%H:%M")

    # Peak tracks the bar's FAVORABLE extreme (high for long, low for short), so
    # the give-back level reflects the true intra-bar peak, not just the close.
    fav = hi if sign > 0 else lo
    st["peak_R"] = max(st.get("peak_R", 0.0), sign * (fav - st["entry"]) / st["risk"])

    obs = exit_obs(tee, st, bars)                            # obs mfe stays close-based

    # Trail only once the peak reaches ACTIVATE_R; until then hold the initial
    # stop. Either way we fall through to the MAX_HOLD timeout below, so parity
    # with the training sim (TrailExitSim) holds whether or not we've activated.
    if st["peak_R"] < config.ACTIVATE_R:
        log.info("   %s  armed — peak %.2fR < %.1fR, holding initial stop",
                 stamp, st["peak_R"], config.ACTIVATE_R)
    else:
        mult = policy.trail_mult(obs)
        trail_dist = mult * a                                # price distance
        giveback = config.GIVEBACK_R * st["risk"]
        order = client.working_stop_order(account_id, contract_id)
        if order is None:
            log.warning("   %s  no working stop found — skip trail", stamp)
        elif trailing:
            # Native trailing stop: only ever tighten the follow distance, capped
            # at the give-back limit (the broker trails tick-by-tick between bars).
            new_ticks = max(1, round(min(trail_dist, giveback) / tick_size))
            cur_ticks = st.get("trail_ticks") or new_ticks
            if new_ticks < cur_ticks:
                client.modify_trail_price(account_id, order["id"], new_ticks * tick_size)
                st["trail_ticks"] = new_ticks
                log.info("   %s  trail tightened → %dt (%.2fx ATR, %d bars in)",
                         stamp, new_ticks, mult, st["bars_held"])
            else:
                log.info("   %s  hold trail %dt (%.2fx ATR)", stamp, cur_ticks, mult)
            best = st["entry"] + sign * st["peak_R"] * st["risk"]
            est = best - sign * (st.get("trail_ticks") or new_ticks) * tick_size
            st["stop"] = max(st["stop"], est) if sign > 0 else min(st["stop"], est)
        else:
            # Plain stop: PPO trail level, never looser than the give-back cap
            # (peak − GIVEBACK_R), favorable ratchet only, direction-aware snap.
            peak_price = st["entry"] + sign * st["peak_R"] * st["risk"]
            cap = peak_price - sign * giveback
            cand = cur - sign * trail_dist
            tightest = max(cand, cap) if sign > 0 else min(cand, cap)
            new_stop = max(st["stop"], tightest) if sign > 0 else min(st["stop"], tightest)
            new_stop = _snap(new_stop, sign, tick_size)

            # Enforce the trailed SL: if this bar's UNFAVORABLE extreme crossed the
            # floor we just TIGHTENED to (which isn't a resting order yet — the
            # resting stop at the PRIOR level is filled by the broker /
            # SimBroker.process_exits), close at MARKET. The realistic fill is the
            # bar close (cur), NOT the floor — on a fast spike-and-reverse the
            # market is already past the floor. So the sim isn't optimistic and
            # matches the trained TrailExitSim's market-close tier. Checked BEFORE
            # the timeout so a stop hit takes precedence (hit, then `elif` timeout).
            unfav = lo if sign > 0 else hi
            if (unfav <= new_stop) if sign > 0 else (unfav >= new_stop):
                client.close_position(account_id, contract_id, price=cur)
                log.info("   %s  trailed-SL crossed (bar %s=%.2f vs SL %.2f) — market close @ %.2f",
                         stamp, "low" if sign > 0 else "high", unfav, new_stop, cur)
                return None
            if abs(new_stop - st["stop"]) >= tick_size:
                try:
                    client.modify_stop_price(account_id, order["id"], new_stop)
                    st["stop"] = new_stop
                    log.info("   %s  trail → stop %.2f (%.2fx ATR, %d bars in)",
                             stamp, new_stop, mult, st["bars_held"])
                except Exception as e:    # valid-side reject — wick-cross will enforce
                    log.warning("   %s  stop modify rejected (%s) — holding %.2f",
                                stamp, e, st["stop"])
            else:
                log.info("   %s  hold (stop %.2f, %.2fx ATR)", stamp, st["stop"], mult)

    # MAX_HOLD force-exit — the policy observes bars_held/MAX_HOLD and the training
    # sim force-exits at MAX_HOLD, so live MUST too or it rides past the horizon the
    # policy was trained on. Close at market (≈ the bar close, like TrailExitSim).
    if st["bars_held"] >= tee.MAX_HOLD:
        client.close_position(account_id, contract_id, price=cur)
        log.info("   %s  max-hold %d bars reached — closed at market %.2f",
                 stamp, tee.MAX_HOLD, cur)
        return None
    return st
