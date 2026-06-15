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


def ffm_block(bars: pd.DataFrame, i: int) -> np.ndarray:
    """76 FFM features at bar i, in the models' parquet column order. Computed
    live via futures_foundation.derive_features; absent columns → NaN."""
    from futures_foundation.features import derive_features

    df = bars.rename(columns={"time": "datetime"})
    feats = derive_features(df, instrument=config.SYMBOL, atr_period=config.ATR_P)
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
    def reference_line(self, bars: pd.DataFrame) -> np.ndarray:
        """Per-bar reference line the PPO exit measures distance from
        (SuperTrend line / slow EMA)."""

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
    def grade(self, bars: pd.DataFrame, sig: Signal):
        """(proba, r_hat) from this strategy's model for the detected signal."""
        from futures_foundation import foundation

        i = sig.bar_index
        emb = foundation.embed_bars(
            bars["close"].to_numpy(float), [i], ctx=config.CTX)      # (1, 256)
        hand = self._hand_features(bars, i, sig.direction).reshape(1, -1)
        X = np.concatenate([emb, hand], axis=1).astype(np.float32)

        bundle = self._load_bundle()
        proba = float(bundle["signal_head"].predict_proba(X)[0, 1])
        risk_head = bundle.get("risk_head")
        r_hat = float(risk_head.predict(X)[0]) if risk_head is not None else 0.0
        return proba, r_hat

    def _load_bundle(self) -> dict:
        if self._bundle is None:
            # Importing the chronos subpackage installs the legacy 'pipelines.chronos'
            # pickle-compat alias so older bundles unpickle without their origin repo.
            import futures_foundation.chronos  # noqa: F401
            self._bundle = joblib.load(
                os.path.join(config.MODELS_DIR, self.model_filename))
        return self._bundle
