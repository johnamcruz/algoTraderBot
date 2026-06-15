#!/usr/bin/env python3
"""strategies — pluggable entry strategies, one model each.

    from strategies import make_strategies
    for s in make_strategies(["supertrend", "ema"]):
        sig = s.detect(bars)            # mechanical entry on the last bar
        if sig: proba, r_hat = s.grade(bars, sig)
"""
import config
from strategies.base import Signal, Strategy
from strategies.bos import BosStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.keltner import KeltnerAdxStrategy
from strategies.supertrend import SuperTrendStrategy

REGISTRY = {
    SuperTrendStrategy.name: SuperTrendStrategy,
    EmaCrossStrategy.name: EmaCrossStrategy,
    KeltnerAdxStrategy.name: KeltnerAdxStrategy,
    BosStrategy.name: BosStrategy,
}

__all__ = ["Signal", "Strategy", "SuperTrendStrategy", "EmaCrossStrategy",
           "KeltnerAdxStrategy", "BosStrategy", "REGISTRY", "make_strategies"]


def make_strategies(active=None):
    """Instantiate the active strategies (default: config.ACTIVE_STRATEGIES)."""
    active = active if active is not None else config.ACTIVE_STRATEGIES
    out = []
    for name in active:
        if name not in REGISTRY:
            raise ValueError(f"unknown strategy {name!r} (have {list(REGISTRY)})")
        out.append(REGISTRY[name]())
    return out
