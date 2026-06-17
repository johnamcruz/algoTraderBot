#!/usr/bin/env python3
"""strategies/orb.py — Opening Range Breakout (15-min ORB) strategy.

The opening range is the high/low of the first 5 bars (15 min) from the 09:30 ET
session open; a close breaking beyond it (ADX-gated) is the mechanical entry —
long above the range high, short below the range low. The ORBChronos model grades
whether the breakout rides or fails. Hand features = 76 FFM +
[or_size, breakout_ext, session_gap, approach_pos, or_vol_ratio, adx/100,
adx_slope/100] (83 total → feat_dim 339).
"""
import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block


class OrbStrategy(Strategy):
    name = "orb"
    model_filename = "orb_chronos.joblib"

    def _fired(self, bars):
        c = bars["close"].to_numpy(float)
        oh, ol = ind.opening_range(bars, config.ORB_BARS, config.ORB_OPEN_MIN,
                                   config.ORB_TZ)
        a = ind.adx(bars, config.ADX_P)
        i = len(c) - 1
        if config.ORB_ADX_GATE and (not np.isfinite(a[i]) or a[i] < config.ORB_ADX_GATE):
            return None                                  # not a trending regime
        if not (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])):
            return None                                  # range not active yet
        if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
            return 1                                     # break above the range high
        if c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
            return -1                                    # break below the range low
        return None

    def _hand_features(self, bars, i, direction):
        c = bars["close"].to_numpy(float)
        v = bars["volume"].to_numpy(float)
        oh, ol = ind.opening_range(bars, config.ORB_BARS, config.ORB_OPEN_MIN,
                                   config.ORB_TZ)
        sess_open, prior_close, or_avg_vol = ind.orb_extras(
            bars, config.ORB_BARS, config.ORB_OPEN_MIN, config.ORB_TZ)
        atr_i = float(ind.atr(bars, config.ATR_P)[i])
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        d = direction

        def g(x):
            return float(x) if np.isfinite(x) else 0.0

        level = oh[i] if d == 1 else ol[i]
        or_mid = 0.5 * (oh[i] + ol[i])
        half = 0.5 * (oh[i] - ol[i])
        ov = or_avg_vol[i]
        with np.errstate(invalid="ignore"):
            or_size = (oh[i] - ol[i]) / a
            breakout_ext = (c[i] - level) / a * d
            session_gap = (sess_open[i] - prior_close[i]) / a * d
            approach_pos = (c[i - 1] - or_mid) / half * d
            or_vol_ratio = (v[i] / ov) if (np.isfinite(ov) and ov > 0) else np.nan
        adx_i, adx_slope = adx_pair(bars, i)
        hand = [g(or_size), g(breakout_ext), g(session_gap), g(approach_pos),
                g(or_vol_ratio), adx_i / 100.0, adx_slope / 100.0]
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, hand]).astype(np.float32)
