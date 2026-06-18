"""Shared test setup: put the repo root on sys.path and restore mutable config
globals around each test (the bot keeps settings in module-level config vars)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

_RESTORE = ["SIZE", "RISK_PER_TRADE", "MAX_CONTRACTS", "PROBA_FLOOR",
            "ACTIVATE_R", "GIVEBACK_R", "ACTIVE_STRATEGIES", "SYMBOL",
            "RR", "USE_PPO_EXIT", "ORB_ADX_GATE", "ORB_CLOSE_MIN",
            "ADX_GATE", "KC_ADX_THRESH", "STOP_ATR", "ATR_P"]


@pytest.fixture(autouse=True)
def restore_config():
    saved = {k: getattr(config, k) for k in _RESTORE}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
