#!/usr/bin/env python3
"""strategies/jev/strategy.py — a reasoning model decides which entry to take.

The decider (config.JEV_DECIDER — Jev by default, any model behind the Decider
interface) chooses among the entries the other strategies (supertrend, ema,
keltner, bos, orb, cisd_ote) fire, using everything they know:

  detect()  folds every newly closed bar into the track record (each strategy's
            signals followed to +2R / stop — see memory.py) and runs every
            sub-strategy's detect() on the current bar. Nothing fired → no
            decision this bar.
  grade()   grades each fired signal with its own Chronos+XGBoost model (on the
            shared per-bar embedding), builds the state (EVERY strategy's view,
            live signal and track record + market context + the decider's own
            recent picks) and asks the decider which fired entry to take, or none.
            The pick's own Signal (side, entry, stop) becomes the trade.
  accepts() the decision IS the entry — no PROBA_FLOOR. A pick enters; none, or
            a failed call, stays flat.

proba on the Signal = the decider's probability for its pick (informational).
While a pick is live the instance's `name` reads `jev:<strategy>`, so the bot's
logs and the backtest breakdown show which entry was taken. Entry, sizing,
broker and the PPO exit are the bot's own — nothing decider-specific.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import config
from logsetup import get_logger
from strategies.base import Strategy, embed_context
from strategies.jev.deciders import DeciderError, make_decider
from strategies.jev.memory import JEV, TrackRecord
from strategies.jev.state import build_question, build_state

log = get_logger()

MIN_BARS = 60           # history a sub-strategy needs before its detect() is useful


class JevStrategy(Strategy):
    name = "jev"
    model_filename = ""             # no bundle of its own — uses the sub-strategies'

    def __init__(self, subs=None, decider=None, memory=None):
        super().__init__()
        self._subs = subs
        self._decider = decider
        self.memory = memory or TrackRecord()
        self._fired_now = []        # [(strategy, Signal)] fired on the current bar
        self._decided = False       # did the decider choose long/short this bar

    @property
    def subs(self):
        """Strategies Jev listens to (config.JEV_STRATEGIES, default: all) — only
        those with a model for the active timeframe."""
        if self._subs is None:
            import strategies       # late: the package registers this class
            names = config.JEV_STRATEGIES or [
                n for n in strategies.REGISTRY if n != JevStrategy.name]
            self._subs = [s for s in (strategies.REGISTRY[n]() for n in names)
                          if s.has_model()]
        return self._subs

    @property
    def decider(self):
        if self._decider is None:
            self._decider = make_decider()
            log.info("jev decider: %s", self._decider.spec)
        return self._decider

    def has_model(self) -> bool:
        return bool(self.subs)

    def model_path(self) -> str:
        return config.MODELS_DIR    # only used in make_strategies' "no model" error

    def accepts(self, sig) -> bool:
        return self._decided        # the decision IS the entry — no floor

    @property
    def skip_reason(self) -> str:
        return "jev: no pick"

    # base's indicator-flip hooks don't apply — detect()/grade() are overridden
    def _fired(self, bars):
        return None

    def _hand_features(self, bars, i, direction):
        raise NotImplementedError("JevStrategy grades via its sub-strategies")

    # ── track record ────────────────────────────────────────────────────────
    def _scan(self, bars):
        """Fold every not-yet-seen closed bar into the track record: resolve open
        shadow trades on it, then record any signals that fired on it. Covers the
        startup bootstrap (last JEV_MEMORY_BOOT_BARS) and bars missed while the
        bot was in a position (detect only runs when flat). Returns the signals
        fired on the CURRENT (last) bar."""
        times = bars["time"]
        n = len(bars)
        last = self.memory.last_time
        if last is None:
            start = max(MIN_BARS, n - config.JEV_MEMORY_BOOT_BARS)
        else:
            newer = np.nonzero((times > last).to_numpy())[0]
            start = max(int(newer[0]), MIN_BARS) if len(newer) else n
        fired = []
        for k in range(start, n):
            view = bars if k == n - 1 else bars.iloc[:k + 1]
            self.memory.update(bars.iloc[k])
            fired = [(s, sig) for s in self.subs if (sig := s.detect(view))]
            for s, sig in fired:
                self.memory.add(s.name, sig, times.iloc[k])
        if start >= n:              # current bar already folded (re-called) — just detect
            fired = [(s, sig) for s in self.subs if (sig := s.detect(bars))]
        return fired

    # ── per-bar decision ────────────────────────────────────────────────────
    def detect(self, bars):
        self.name = JevStrategy.name
        self._decided = False
        self._fired_now = self._scan(bars)
        if not self._fired_now:
            return None
        # placeholder — grade() asks the decider which fired entry to take
        return dataclasses.replace(self._fired_now[0][1], strategy=self.name)

    def grade(self, bars, sig, emb=None):
        if emb is None:
            emb = embed_context(bars, len(bars) - 1)
        for s, sub in self._fired_now:
            sub.proba, sub.r_hat = s.grade(bars, sub, emb=emb)
        fired = {s.name: sub for s, sub in self._fired_now}
        stamp = bars["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        sigs = " ".join(f"{n}({'L' if x.direction > 0 else 'S'} {x.proba:.2f})"
                        for n, x in fired.items())

        try:
            ans = self.decider.decide(
                build_state(bars, self.subs, fired, self.memory),
                build_question(fired, [s.name for s in self.subs]))
        except DeciderError as e:
            log.warning("⚠️  jev %s | %s | no decision, staying flat: %s", stamp, sigs, e)
            return 0.0, 0.0

        pick = ans["choice"]
        p = float(ans["probabilities"].get(pick, 0.0))
        log.info("jev %s | %s → %s (p=%.2f)%s", stamp, sigs, pick, p,
                 f" | {ans['reason']}" if ans.get("reason") else "")
        chosen = fired.get(pick)
        if chosen is None:          # none (or an option that wasn't offered)
            return 0.0, 0.0

        for f in dataclasses.fields(chosen):
            setattr(sig, f.name, getattr(chosen, f.name))
        sig.proba = p
        self.name = sig.strategy = f"{JevStrategy.name}:{pick}"
        self._decided = True
        self.memory.add(JEV, chosen, bars["time"].iloc[-1])
        return p, chosen.r_hat
