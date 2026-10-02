#!/usr/bin/env python3
"""strategies/jev/memory.py — live track record of every strategy, for Jev.

Jev can't be trained here, so it learns IN CONTEXT: every signal any strategy
fires (taken or not) becomes a "shadow trade" that is followed forward bar by bar
until it hits +TARGET_R, its stop (-1R), or MAX_BARS (marked to market). Jev then
sees each strategy's recent results — which setups are actually working in the
current market — plus how its own recent picks did. DecisionLog keeps its last
few decisions as worked examples (situation → choice → how every candidate
turned out).

Strictly causal: a shadow trade is scored only from bars AFTER its signal bar,
and only once those bars have closed. When a bar straddles stop and target the
stop is assumed first (same convention as the backtester).
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

    def add(self, name, sig, time):
        """Start tracking a signal that fired on the bar closing at `time`.
        Returns the shadow trade (its "r" is set once it resolves), or None."""
        if sig.risk <= 0:
            return None
        tr = {"name": name, "sign": sig.direction, "entry": sig.entry,
              "risk": sig.risk, "time": time, "bars": 0}
        self.open.append(tr)
        return tr

    def update(self, bar):
        """Fold one newly closed bar (a row with time/high/low/close) into every
        open shadow trade that fired before it."""
        t = bar["time"]
        if self.last_time is not None and t <= self.last_time:
            return
        self.last_time = t
        still = []
        for tr in self.open:
            if tr["time"] >= t:             # fired on this bar — scored from the next
                still.append(tr)
                continue
            s, e, r = tr["sign"], tr["entry"], tr["risk"]
            adverse = (bar["low"] - e) * s if s > 0 else (bar["high"] - e) * s
            favor = (bar["high"] - e) * s if s > 0 else (bar["low"] - e) * s
            tr["bars"] += 1
            if adverse <= -r:
                self._close(tr, -1.0)
            elif favor >= self.target_r * r:
                self._close(tr, self.target_r)
            elif tr["bars"] >= self.max_bars:
                self._close(tr, s * (bar["close"] - e) / r)
            else:
                still.append(tr)
        self.open = still

    def _close(self, tr, r):
        tr["r"] = float(r)
        self.done.setdefault(tr["name"], deque(maxlen=self.keep)).append(float(r))

    def summary(self, name) -> dict:
        """Recent results for one strategy (or JEV): resolved count, hit rate of
        +TARGET_R, mean R, and the last few outcomes oldest→newest (W/L/F = hit
        target / stopped / flat at timeout)."""
        rs = list(self.done.get(name, ()))
        out = {"resolved": len(rs),
               "pending": sum(1 for t in self.open if t["name"] == name)}
        if rs:
            out["winRate"] = round(sum(r >= self.target_r for r in rs) / len(rs), 2)
            out["avgR"] = round(sum(rs) / len(rs), 2)
            out["last"] = " ".join("W" if r >= self.target_r else "L" if r <= -1
                                   else "F" for r in rs[-8:])
        return out


def _outcome(tr, target_r) -> str:
    """A shadow trade's result: "W +2.0R" / "L -1.0R" / "F +0.4R" / "pending"."""
    if tr is None or "r" not in tr:
        return "pending"
    r = tr["r"]
    tag = "W" if r >= target_r else "L" if r <= -1 else "F"
    return f"{tag} {r:+.1f}R"


class DecisionLog:
    """The decider's last few decisions as worked examples: the situation, the
    candidates, what it chose, and how EVERY candidate actually turned out —
    so a `none` shows whether skipping avoided a loser or missed a winner.
    Outcomes come from the TrackRecord's shadow trades, so they appear only once
    they have really happened (strictly causal)."""

    def __init__(self, keep=None, target_r=None):
        self.keep = keep or config.JEV_HISTORY_KEEP
        self.target_r = target_r or config.JEV_MEMORY_TARGET_R
        self.entries = deque(maxlen=self.keep)

    def add(self, time, snapshot: dict, candidates: dict, choice: str):
        """`candidates` = {name: (side, modelWinProb, shadow trade or None)}."""
        self.entries.append({"time": time, "snapshot": snapshot,
                             "candidates": candidates, "choice": choice})

    def render(self, now) -> list:
        """Oldest → newest, relative times only (no dates)."""
        out = []
        for e in self.entries:
            cands = {n: {"side": side, "modelWinProb": p,
                         "outcome": _outcome(tr, self.target_r)}
                     for n, (side, p, tr) in e["candidates"].items()}
            out.append({
                "minutesAgo": int((now - e["time"]).total_seconds() // 60),
                "situation": e["snapshot"],
                "candidates": cands,
                "yourChoice": e["choice"],
                "result": (f"took {e['choice']}: {cands[e['choice']]['outcome']}"
                           if e["choice"] in cands else
                           "took none; skipped: " + ", ".join(
                               f"{n} {c['outcome']}" for n, c in cands.items())),
            })
        return out
