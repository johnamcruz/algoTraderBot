#!/usr/bin/env python3
"""strategies/jev/memory.py — live track record of every strategy, for Jev.

Jev can't be trained here, so it learns IN CONTEXT: every signal any strategy
fires (taken or not) becomes a "shadow trade" that is followed forward bar by bar
until it hits +TARGET_R, its stop (-1R), or MAX_BARS (marked to market). This is a
fixed proxy for "did this setup work" — the bot's real exit is the PPO trail.
Jev then sees each strategy's recent results — which setups are actually working
in the current market — plus how its own recent picks did. DecisionLog keeps its
last few decisions as worked examples (situation → choice → how a long and a
short would each have turned out).

Strictly causal: a shadow trade enters at its signal bar's close (where a live
entry fills) and is scored only from LATER bars, once they have closed. When a bar straddles stop and target the stop
is assumed first (same convention as the backtester).
"""
from __future__ import annotations

from collections import deque

import config

JEV = "jev"      # key under which Jev's own picks are tracked


class TrackRecord:
    def __init__(self, target_r=None, max_bars=None, keep=None):
        self.target_r = target_r or config.JEV_MEMORY_TARGET_R
        self.max_bars = max_bars or config.JEV_MEMORY_MAX_BARS
        self.keep = keep or config.JEV_MEMORY_KEEP
        self.open = []                      # live shadow trades
        self.done = {}                      # name → deque of realized R
        self.last_time = None               # last bar folded into the record

    def add(self, name, sig, time, atr=None):
        """Start tracking a signal entered at the close of the bar at `time`.
        With `atr`, the timeout scales with the stop's width (a 2-ATR stop gets 4×
        the bars of the standard STOP_ATR one) so wide-stop setups aren't mostly
        marked as timeouts. Returns the shadow trade (its "r" is set once it
        resolves), or None."""
        if sig.risk <= 0:
            return None
        width = sig.risk / (config.STOP_ATR * atr) if atr and atr > 0 else 1.0
        tr = {"name": name, "sign": sig.direction, "entry": sig.entry,
              "risk": sig.risk, "time": time, "bars": 0,
              "max_bars": int(round(self.max_bars * max(1.0, width)))}
        self.open.append(tr)
        return tr

    def update(self, bar):
        """Fold one newly closed bar (a row with time/high/low/close) into every
        open shadow trade that entered before it."""
        t = bar["time"]
        if self.last_time is not None and t <= self.last_time:
            return
        self.last_time = t
        self.open = [tr for tr in self.open
                     if tr["time"] >= t or not self._step(tr, bar)]

    def _step(self, tr, bar) -> bool:
        """Score one bar for one shadow trade; True if it resolved."""
        s, e, r = tr["sign"], tr["entry"], tr["risk"]
        adverse = (bar["low"] - e) * s if s > 0 else (bar["high"] - e) * s
        favor = (bar["high"] - e) * s if s > 0 else (bar["low"] - e) * s
        tr["bars"] += 1
        if adverse <= -r:
            self._close(tr, -1.0)
        elif favor >= self.target_r * r:
            self._close(tr, self.target_r)
        elif tr["bars"] >= tr.get("max_bars", self.max_bars):
            self._close(tr, s * (bar["close"] - e) / r)
        else:
            return False
        return True

    def _close(self, tr, r):
        tr["r"] = float(r)
        self.done.setdefault(tr["name"], deque(maxlen=self.keep)).append(float(r))

    def summary(self, name) -> dict:
        """Recent results for one strategy (or JEV): resolved count, hit rate of
        +TARGET_R, mean R, and the last few outcomes oldest→newest (W/L/F = hit
        target / stopped / timed out)."""
        rs = list(self.done.get(name, ()))
        out = {"resolved": len(rs),
               "pending": sum(1 for t in self.open if t["name"] == name)}
        if rs:
            out["winRate"] = round(sum(r >= self.target_r for r in rs) / len(rs), 2)
            out["avgR"] = round(sum(rs) / len(rs), 2)
            out["last"] = " ".join("W" if r >= self.target_r else "L" if r <= -1
                                   else "F" for r in rs[-8:])
        return out


def outcome(tr, target_r) -> str:
    """A shadow trade's result: "W +2.0R" / "L -1.0R" / "F +0.4R" / "pending"."""
    if tr is None or "r" not in tr:
        return "pending"
    r = tr["r"]
    tag = "W" if r >= target_r else "L" if r <= -1 else "F"
    return f"{tag} {r:+.1f}R"


class DecisionLog:
    """The decider's last few decisions as worked examples: the situation, which
    strategy signals fired, what it chose, and how BOTH a long and a short entered
    at that bar would have turned out — so a `none` shows whether it dodged a
    loser or missed a winner, and an entry shows whether the other side was
    better. Outcomes come from the TrackRecord's shadow trades, so they appear only
    once they have really happened (strictly causal)."""

    def __init__(self, keep=None, target_r=None):
        self.keep = keep or config.JEV_HISTORY_KEEP
        self.target_r = target_r or config.JEV_MEMORY_TARGET_R
        self.entries = deque(maxlen=self.keep)

    def add(self, time, situation: dict, signals: list, sides: dict, choice: str):
        """`signals` = ["bos long", ...] fired this bar; `sides` = {"long": shadow
        trade, "short": shadow trade} for a trade entered at this bar."""
        self.entries.append({"time": time, "situation": situation,
                             "signals": signals, "sides": sides, "choice": choice})

    def render(self, now) -> list:
        """Oldest → newest, relative times only (no dates)."""
        out = []
        for e in self.entries:
            res = {side: outcome(tr, self.target_r) for side, tr in e["sides"].items()}
            if e["choice"] in res:
                result = f"took {e['choice']}: {res[e['choice']]}"
            else:
                result = "stayed flat; " + ", ".join(
                    f"{side} would have been {r}" for side, r in res.items())
            out.append({
                "minutesAgo": int((now - e["time"]).total_seconds() // 60),
                "situation": e["situation"],
                "signalsFired": e["signals"],
                "yourChoice": e["choice"],
                "longOutcome": res.get("long"),
                "shortOutcome": res.get("short"),
                "result": result,
            })
        return out
