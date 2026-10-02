#!/usr/bin/env python3
"""strategies/base.py — the generic Strategy interface.

A Strategy (a) DETECTS its mechanical entry on the latest closed bar and
(b) GRADES it with its own pre-trained Chronos+XGBoost model (a joblib bundle
from futures_foundation). Concrete strategies live in sibling files and inherit
from `Strategy`; the bot can run one or several at once.

Trade definition matches the models' training: entry ≈ next-bar fill, stop =
STOP_ATR × ATR(ATR_P). The model only learns SELECTION; direction is mechanical.

Inference per signal bar i:  X = concat([embed_256, hand]) → heads
  embed  = futures_foundation.foundation.embed_bars(closes, [i])   (subprocess)
  hand   = 76 FFM features (live, in the models' parquet column order) + the
           strategy's public handcrafts (adx/adx_slope, or the 5 EMA features)
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import config
import indicators as ind

# FFM feature columns in the EXACT order the models were trained on (extracted
# from the training parquet). Live values are placed by name into this order;
# any column the current library doesn't produce stays NaN (XGBoost handles it).
with open(config.FFM_COLUMNS_PATH) as _f:
    FFM_COLS = json.load(_f)


@dataclass
class Signal:
    strategy: str
    direction: int          # +1 long / -1 short
    entry: float            # signal-bar close (≈ live fill)
    stop: float             # protective stop price
    risk: float             # |entry - stop| in price (= STOP_ATR × ATR)
    bar_index: int
    bar_time: object
    proba: float = 0.0
    r_hat: float = 0.0


def embed_context(bars: pd.DataFrame, i: int) -> np.ndarray:
    """Chronos context embedding (1, 256) for the window ending at bar i. All
    strategies firing on the same bar share this same context, so it is computed
    once per bar and reused across strategies. Routed through the warm embedding
    worker (model loaded once per session)."""
    import embedder
    return embedder.embed_bars(bars["close"].to_numpy(float), [i], ctx=config.CTX)


def ffm_block(bars: pd.DataFrame, i: int) -> np.ndarray:
    """76 FFM features at bar i, in the models' parquet column order. Computed
    live via futures_foundation.derive_features; absent columns → NaN."""
    from futures_foundation.features import derive_features

    df = bars.rename(columns={"time": "datetime"})
    # micros (MNQ…) derive features under their parent instrument (NQ…)
    feats = derive_features(df, instrument=config.base_symbol(config.SYMBOL),
                            atr_period=config.ATR_P)
    row = feats.iloc[i]
    cols = feats.columns
    out = np.full(len(FFM_COLS), np.nan, dtype=np.float32)
    for k, name in enumerate(FFM_COLS):
        if name in cols:
            val = row[name]
            if pd.notna(val):          # leave NaN for absent/NA (XGBoost handles it)
                out[k] = val
    return out


def adx_pair(bars: pd.DataFrame, i: int):
    """(adx, adx_slope) at bar i — the public handcrafts both models share."""
    a = ind.adx(bars, config.ADX_P)
    adx_i = float(a[i]) if np.isfinite(a[i]) else 0.0
    k = config.ADX_SLOPE
    if i >= k and np.isfinite(a[i]) and np.isfinite(a[i - k]):
        slope = float(a[i] - a[i - k])
    else:
        slope = 0.0
    return adx_i, slope


class Strategy(ABC):
    """Generic strategy: detect a mechanical entry, then grade it with a model."""

    name: str = "strategy"
    model_filename: str = ""

    def __init__(self):
        self._bundle = None                 # lazy joblib load

    # ── signal detection (subclass-specific) ───────────────────────────
    @abstractmethod
    def _fired(self, bars: pd.DataFrame) -> Optional[int]:
        """Return the trade direction (+1/-1) if the last closed bar is an
        entry for this strategy, else None."""

    @abstractmethod
    def _hand_features(self, bars: pd.DataFrame, i: int, direction: int) -> np.ndarray:
        """The strategy's hand-crafted feature vector at bar i (FFM + handcrafts)."""

    # ── shared entry construction ──────────────────────────────────────
    def detect(self, bars: pd.DataFrame) -> Optional[Signal]:
        d = self._fired(bars)
        if d is None:
            return None
        i = len(bars) - 1
        a = float(ind.atr(bars, config.ATR_P)[i])
        if not np.isfinite(a) or a <= 0:
            return None
        entry = float(bars["close"].iloc[i])
        risk = config.STOP_ATR * a                  # stop distance in price
        stop = entry - d * risk
        return Signal(self.name, d, entry, stop, risk, i, bars["time"].iloc[i])

    # ── grading (shared) ───────────────────────────────────────────────
    def grade(self, bars: pd.DataFrame, sig: Signal, emb=None):
        """(proba, r_hat) from this strategy's model for the detected signal.

        `emb` is the Chronos context embedding; strategies firing on the same bar
        share the SAME context, so the caller computes it once (embed_context)
        and passes it in — one Chronos pass per bar, not per strategy."""
        if emb is None:
            emb = embed_context(bars, sig.bar_index)
        hand = self._hand_features(bars, sig.bar_index, sig.direction).reshape(1, -1)
        X = np.concatenate([emb, hand], axis=1).astype(np.float32)

        bundle = self._load_bundle()
        proba = float(bundle["signal_head"].predict_proba(X)[0, 1])
        risk_head = bundle.get("risk_head")
        r_hat = float(risk_head.predict(X)[0]) if risk_head is not None else 0.0
        return proba, r_hat

    def accepts(self, sig: Signal) -> bool:
        """Whether to take a graded signal: its model proba clears PROBA_FLOOR."""
        return sig.proba >= config.PROBA_FLOOR

    @property
    def skip_reason(self) -> str:
        return f"<{config.PROBA_FLOOR}"

    def model_path(self) -> str:
        """The model bundle for the active timeframe. The trained default (3-min)
        uses the plain filename; any other timeframe REQUIRES a `_<tf>min` variant
        (e.g. supertrend_chronos_1min.joblib) — no cross-timeframe fallback, so a
        strategy without a matching-timeframe model is simply unavailable."""
        fn = self.model_filename
        if config.TIMEFRAME_MIN != config.TRAINED_TIMEFRAME_MIN:
            base, ext = os.path.splitext(fn)
            return os.path.join(config.MODELS_DIR,
                                f"{base}_{config.TIMEFRAME_MIN}min{ext}")
        return os.path.join(config.MODELS_DIR, fn)

    def has_model(self) -> bool:
        """Whether this strategy has a model for the active timeframe."""
        return os.path.exists(self.model_path())

    def _load_bundle(self) -> dict:
        if self._bundle is None:
            self._bundle = load_bundle(self.model_path())
        return self._bundle


# Module paths the shipped bundles were pickled under before futures_foundation
# renamed chronos → pipeline (and dropped its own compat aliases, 2026-06-23).
_LEGACY_PATHS = ("futures_foundation.chronos", "pipelines", "pipelines.chronos")


def load_bundle(path: str) -> dict:
    """joblib.load a model bundle, mapping the legacy module paths onto
    futures_foundation.pipeline for THIS load only — the aliases are removed
    afterwards so they never shadow another package of the same name."""
    import sys
    try:
        import futures_foundation.pipeline as pipe
        import futures_foundation.pipeline.head_xgb as head_xgb
    except ModuleNotFoundError:          # pre-rename library: old paths are real
        import futures_foundation.chronos  # noqa: F401
        return joblib.load(path)
    alias = {}
    for legacy in _LEGACY_PATHS:
        alias[legacy] = pipe
        alias[f"{legacy}.head_xgb"] = head_xgb
    added = [k for k in alias if k not in sys.modules]
    sys.modules.update({k: alias[k] for k in added})
    try:
        return joblib.load(path)
    finally:
        for k in added:
            sys.modules.pop(k, None)
