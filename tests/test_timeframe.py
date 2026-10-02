"""Timeframe support (--timeframe). The default is 3-min; another timeframe loads
the matching data file and the matching `_<tf>min` model, and a strategy without a
model for that timeframe is rejected (no cross-timeframe fallback — a 3-min model
is never run on 1-min bars)."""
import os

import pytest

import backtest
import config
import strategies
from strategies.supertrend import SuperTrendStrategy
from strategies.ema_cross import EmaCrossStrategy


# ── model path resolves per timeframe ──────────────────────────────────────

def test_default_3min_uses_plain_filename():
    config.TIMEFRAME_MIN = 3
    assert SuperTrendStrategy().model_path().endswith("supertrend_chronos.joblib")


def test_1min_uses_suffixed_filename():
    config.TIMEFRAME_MIN = 1
    assert SuperTrendStrategy().model_path().endswith("supertrend_chronos_1min.joblib")
    assert "_1min" not in EmaCrossStrategy().model_path() or \
        EmaCrossStrategy().model_path().endswith("ema_cross_chronos_1min.joblib")


# ── has_model reflects what's shipped ──────────────────────────────────────

def test_supertrend_has_1min_model_others_do_not():
    config.TIMEFRAME_MIN = 1
    assert SuperTrendStrategy().has_model() is True            # we shipped the 1-min bundle
    assert EmaCrossStrategy().has_model() is False             # only 3-min exists


def test_all_strategies_have_3min_models():
    config.TIMEFRAME_MIN = 3
    for cls in strategies.REGISTRY.values():
        assert cls().has_model(), f"{cls.name} missing its 3-min model"


# ── make_strategies gates by timeframe ─────────────────────────────────────

def test_1min_allows_supertrend_only():
    config.TIMEFRAME_MIN = 1
    # jev is a meta-strategy: available whenever any sub-strategy has a model
    assert strategies.available_for_timeframe() == ["supertrend", "jev"]
    assert [s.name for s in strategies.make_strategies(["jev"])[0].subs] == ["supertrend"]
    built = strategies.make_strategies(["supertrend"])
    assert len(built) == 1 and built[0].name == "supertrend"


def test_1min_rejects_strategy_without_model():
    config.TIMEFRAME_MIN = 1
    with pytest.raises(SystemExit) as e:
        strategies.make_strategies(["ema"])
    assert "no 1-min model" in str(e.value) and "supertrend" in str(e.value)


def test_3min_allows_all():
    config.TIMEFRAME_MIN = 3
    built = strategies.make_strategies(["supertrend", "ema", "bos", "keltner", "orb"])
    assert [s.name for s in built] == ["supertrend", "ema", "bos", "keltner", "orb"]


# ── backtest loads the timeframe-matching data file ────────────────────────

def test_backtest_loads_timeframe_specific_csv():
    config.TIMEFRAME_MIN = 5            # no NQ_5min.csv shipped → clear error naming it
    with pytest.raises(SystemExit) as e:
        backtest._load("NQ", None)
    assert "NQ_5min.csv" in str(e.value)
