#!/usr/bin/env python3
"""strategies/jev/strategy.py — a reasoning model IS the strategy.

On every flat bar the decider (config.JEV_DECIDER — Jev by default, any model
behind the Decider interface) decides long / short / none BY ITSELF. The six
mechanical strategies (supertrend, ema, keltner, bos, orb, cisd_ote) never trade;
they are its context:

  detect()  once per new bar: folds every newly closed bar into the track record
            (each strategy's signals followed to +2R / stop — see memory.py) and
            runs every strategy's detect() so their signals are known. A bar
            already decided is never decided again.
  grade()   grades each fired signal with its own Chronos+XGBoost model (on the
            shared per-bar embedding), builds the consolidated state (market
            context; every strategy's view, signal and track record; its own
            recent picks and last JEV_HISTORY_KEEP decisions) and asks the
            decider: long, short, or none.
  accepts() the decision IS the entry — no PROBA_FLOOR. long/short enters at the
            close with the bot's standard STOP_ATR × ATR stop (what the PPO exit
            is trained on); none, a failed call, or (live) a decision slower than
            JEV_MAX_DECISION_SEC stays flat.

proba on the Signal = the decider's probability for its choice (informational).
While a trade is live the instance's `name` reads `jev:long` / `jev:short`.
Entry, sizing, broker and the PPO exit are the bot's own.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

import config
import indicators as ind
from logsetup import get_logger
from strategies.base import Signal, Strategy, embed_context
from strategies.jev.deciders import DeciderError, make_decider
from strategies.jev.memory import JEV, DecisionLog, TrackRecord
from strategies.jev.state import (LONG, NONE, SHORT, build_question, build_state,
                                  situation)

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
        self.history = DecisionLog()   # its last few decisions + how they turned out
        self._fired_now = []        # [(strategy, Signal)] fired on the current bar
        self._decided = False       # did the decider choose long/short this bar
        self._last_bar = None       # time of the last bar handled by detect()
        self._t0 = 0.0              # when this bar's work started (decision deadline)

    @property
    def subs(self):
        """Strategies whose context Jev gets (config.JEV_STRATEGIES, default:
        all) — only those with a model for the active timeframe."""
        if self._subs is None:
            import strategies       # late: the package registers this class
            names = [n for n in (config.JEV_STRATEGIES or strategies.REGISTRY)
                     if n != JevStrategy.name]
            unknown = [n for n in names if n not in strategies.REGISTRY]
            if unknown:
                raise SystemExit(f"JEV_STRATEGIES has unknown strategies {unknown} "
                                 f"(have {list(strategies.REGISTRY)})")
            self._subs = [s for s in (strategies.REGISTRY[n]() for n in names)
                          if s.has_model()]
        return self._subs

    @property
    def decider(self):
        if self._decider is None:
            self._decider = make_decider()
            log.info("jev decider: %s", self._decider.spec)
        return self._decider

    def prepare(self):
        """Fail fast at startup — not on the first signal mid-session — if the
        decider is misconfigured (unknown backend, missing key or package)."""
        self.decider.check()

    def reset(self):
        """New contract (roll): old-contract shadow trades would be scored against
        new-contract prices, so start the record and decision log over."""
        self.memory = TrackRecord()
        self.history = DecisionLog()
        self._last_bar = None

    def has_model(self) -> bool:
        return bool(self.subs)

    def model_path(self) -> str:
        return config.MODELS_DIR    # only used in make_strategies' "no model" error

    def accepts(self, sig) -> bool:
        return self._decided        # the decision IS the entry — no floor

    @property
    def skip_reason(self) -> str:
        return "jev: none"

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
        closes = bars["close"].to_numpy(float)
        atr = ind.atr(bars, config.ATR_P)
        n = len(bars)
        last = self.memory.last_time
        if last is None:
            start = max(MIN_BARS, n - config.JEV_MEMORY_BOOT_BARS)
        else:
            newer = np.nonzero((times > last).to_numpy())[0]
            start = int(newer[0]) if len(newer) else n
        fired = []
        for k in range(start, n):
            self.memory.update(bars.iloc[k])          # every unseen bar is scored
            if k < MIN_BARS:
                continue                              # too early to detect on
            view = bars if k == n - 1 else bars.iloc[:k + 1]
            fired = [(s, sig) for s in self.subs if (sig := s.detect(view))]
            # shadow-trade at the price the bot would actually fill (this bar's
            # close) with the signal's stop distance — cisd_ote's signal entry is
            # the bar's open, but a live entry fills at the close
            for s, sig in fired:
                self.memory.add(s.name, dataclasses.replace(sig, entry=float(closes[k])),
                                times.iloc[k], atr=float(atr[k]))
        return fired if start <= n - 1 else []

    # ── per-bar decision ────────────────────────────────────────────────────
    def detect(self, bars):
        self.name = JevStrategy.name
        self._decided = False
        self._t0 = time.monotonic()
        now = bars["time"].iloc[-1]
        if self._last_bar is not None and now <= self._last_bar:
            return None             # this bar was already decided — never re-ask
        self._last_bar = now
        self._fired_now = self._scan(bars)
        if config.JEV_ONLY_ON_SIGNALS and not self._fired_now:
            return None
        i = len(bars) - 1
        a = float(ind.atr(bars, config.ATR_P)[i])
        if not np.isfinite(a) or a <= 0:
            return None
        entry = float(bars["close"].iloc[i])
        risk = config.STOP_ATR * a
        # placeholder (long) — grade() asks the decider and sets the real side
        return Signal(self.name, 1, entry, entry - risk, risk, i, now)

    def grade(self, bars, sig, emb=None):
        if self._fired_now and emb is None:
            emb = embed_context(bars, len(bars) - 1)
        for s, sub in self._fired_now:
            sub.proba, sub.r_hat = s.grade(bars, sub, emb=emb)
        fired = {s.name: sub for s, sub in self._fired_now}
        now = bars["time"].iloc[-1]
        stamp = now.strftime("%Y-%m-%d %H:%M")
        signals = [f"{n} {'long' if x.direction > 0 else 'short'}"
                   for n, x in fired.items()]
        seen = ", ".join(signals) or "no signals"

        state = build_state(bars, self.subs, fired, self.memory,
                            history=self.history.render(now))
        try:
            ans = self.decider.decide(state, build_question([s.name for s in self.subs]))
        except DeciderError as e:
            log.warning("⚠️  jev %s | %s | no decision, staying flat: %s",
                        stamp, seen, e)
            return 0.0, 0.0

        choice = ans["choice"]
        p = ans["probabilities"].get(choice, 0.0)
        log.info("jev %s | %s → %s (p=%.2f)%s", stamp, seen, choice, p,
                 f" | {ans['reason']}" if ans.get("reason") else "")

        # what a long and a short entered here would each do — the worked example
        # for later decisions, and (for an entry) its own track record
        atr = float(ind.atr(bars, config.ATR_P)[-1])
        sides = {side: self.memory.add(
                     f"_{side}", dataclasses.replace(sig, direction=d,
                                                     stop=sig.entry - d * sig.risk),
                     now, atr=atr)
                 for side, d in ((LONG, 1), (SHORT, -1))}
        if choice in (LONG, SHORT) or signals:     # skip quiet no-signal flat bars
            self.history.add(now, situation(state), signals, sides, choice)

        if choice not in (LONG, SHORT):
            return 0.0, 0.0
        elapsed = time.monotonic() - self._t0
        if config.LIVE and elapsed > config.JEV_MAX_DECISION_SEC:
            log.warning("⚠️  jev %s | decision took %.0fs (> %ss) — price has moved "
                        "on, skipping the entry", stamp, elapsed,
                        config.JEV_MAX_DECISION_SEC)
            return 0.0, 0.0

        d = 1 if choice == LONG else -1
        sig.direction, sig.stop = d, sig.entry - d * sig.risk
        sig.proba = p
        self.name = sig.strategy = f"{JevStrategy.name}:{choice}"
        self._decided = True
        self.memory.add(JEV, sig, now, atr=atr)
        return p, 0.0
