"""strategies/jev — a reasoning model is the strategy.

On every flat bar the decider (config.JEV_DECIDER: TypeSafe Jev by default, or any
model behind strategies.jev.deciders.Decider) reads every other strategy's view,
live signals and track record and decides long / short / none.

    python bot.py --strategy jev        # opt-in, never the default
"""
from strategies.jev.strategy import JevStrategy

__all__ = ["JevStrategy"]
