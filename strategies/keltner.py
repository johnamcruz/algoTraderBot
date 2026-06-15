#!/usr/bin/env python3
"""strategies/keltner.py — Keltner-channel breakout strategy.

A close breaking out of the Keltner channel (EMA ± mult×ATR), gated by ADX ≥
threshold, is the mechanical entry; the KeltnerADXChronos model grades whether
the breakout rides or fails. Hand features = 76 FFM +
[kc_pos, mid_slope, adx/100, adx_slope/100], signed by direction
(80 total → feat_dim 336).
"""
import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block


class KeltnerAdxStrategy(Strategy):
    name = "keltner"
    model_filename = "keltner_adx_chronos.joblib"

    def _fired(self, bars):
        c = bars["close"].to_numpy(float)
        up, _mid, lo = ind.keltner_channel(bars, config.KC_LEN, config.KC_MULT,
                                            config.KC_ATR_P)
        a = ind.adx(bars, config.ADX_P)
        i = len(c) - 1
        if not np.isfinite(a[i]) or a[i] < config.KC_ADX_THRESH:
            return None                                  # not a trending regime
        if not (np.isfinite(up[i - 1]) and np.isfinite(lo[i - 1])):
            return None
        if c[i - 1] <= up[i - 1] and c[i] > up[i]:
            return 1                                     # break above upper band
        if c[i - 1] >= lo[i - 1] and c[i] < lo[i]:
            return -1                                    # break below lower band
        return None

    def reference_line(self, bars):
        _up, mid, _lo = ind.keltner_channel(bars, config.KC_LEN, config.KC_MULT,
                                            config.KC_ATR_P)
        return mid                                       # channel centre (EMA)

    def _hand_features(self, bars, i, direction):
        c = bars["close"].to_numpy(float)
        _up, mid, _lo = ind.keltner_channel(bars, config.KC_LEN, config.KC_MULT,
                                            config.KC_ATR_P)
        atr_i = float(ind.atr(bars, config.ATR_P)[i])
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        d = direction

        def g(x):
            return float(x) if np.isfinite(x) else 0.0

        with np.errstate(invalid="ignore"):
            kc_pos = (c[i] - mid[i]) / (config.KC_MULT * a)
            k = config.KC_MID_SLOPE_K
            mid_slope = ((mid[i] - mid[i - k]) / a) if i - k >= 0 else np.nan
        adx_i, adx_slope = adx_pair(bars, i)
        hand = [g(kc_pos) * d, g(mid_slope) * d, adx_i / 100.0, adx_slope / 100.0]
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, hand]).astype(np.float32)
