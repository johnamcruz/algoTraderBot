#!/usr/bin/env python3
"""strategies — pluggable entry strategies, one model each.

    from strategies import make_strategies
    for s in make_strategies(["supertrend", "ema"]):
        sig = s.detect(bars)            # mechanical entry on the last bar
        if sig: proba, r_hat = s.grade(bars, sig)
"""
import os

import config
from strategies.base import Signal, Strategy, embed_context
from strategies.bos import BosStrategy
from strategies.cisd_ote import CisdOteStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.keltner import KeltnerAdxStrategy
from strategies.jev import JevStrategy
from strategies.orb import OrbStrategy
from strategies.supertrend import SuperTrendStrategy

REGISTRY = {
    SuperTrendStrategy.name: SuperTrendStrategy,
    EmaCrossStrategy.name: EmaCrossStrategy,
    KeltnerAdxStrategy.name: KeltnerAdxStrategy,
    BosStrategy.name: BosStrategy,
    OrbStrategy.name: OrbStrategy,
    CisdOteStrategy.name: CisdOteStrategy,
    JevStrategy.name: JevStrategy,     # meta: Jev picks among the others
}

__all__ = ["Signal", "Strategy", "embed_context", "SuperTrendStrategy",
           "EmaCrossStrategy", "KeltnerAdxStrategy", "BosStrategy",
           "OrbStrategy", "CisdOteStrategy", "JevStrategy", "REGISTRY", "make_strategies"]


def available_for_timeframe():
    """Strategy names that have a model for the active timeframe (config.TIMEFRAME_MIN)."""
    return [n for n, cls in REGISTRY.items() if cls().has_model()]


def make_strategies(active=None):
    """Instantiate the active strategies (default: config.ACTIVE_STRATEGIES).

    Only strategies with a model for the active timeframe are allowed — there's no
    cross-timeframe fallback (a 3-min model is not run on 1-min bars). Requesting
    one without a matching-timeframe model is a hard error."""
    active = active if active is not None else config.ACTIVE_STRATEGIES
    if JevStrategy.name in active and len(active) > 1:
        raise SystemExit("jev already uses every strategy's context and makes the "
                         "decision itself — run it on its own (--strategy jev)")
    out = []
    for name in active:
        if name not in REGISTRY:
            raise ValueError(f"unknown strategy {name!r} (have {list(REGISTRY)})")
        s = REGISTRY[name]()
        if not s.has_model():
            raise SystemExit(
                f"strategy {name!r} has no {config.TIMEFRAME_MIN}-min model "
                f"({os.path.basename(s.model_path())} not found). "
                f"Available for {config.TIMEFRAME_MIN}-min: "
                f"{available_for_timeframe() or 'none'}")
        out.append(s)
    return out
