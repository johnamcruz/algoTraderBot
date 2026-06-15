#!/usr/bin/env python3
"""strategies/bos.py — Break of Structure (BOS) strategy.

A close beyond the most recent CONFIRMED swing (fractal) is the mechanical
entry — long above the last swing high, short below the last swing low; the
BOSChronos model grades whether the break rides or fails. Hand features =
76 FFM + [bos_ext, swing_range, bos_age, adx/100, adx_slope/100]
(81 total → feat_dim 337).
"""
import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block


class BosStrategy(Strategy):
    name = "bos"
    model_filename = "bos_chronos.joblib"

    def _fired(self, bars):
        c = bars["close"].to_numpy(float)
        sh, sl, _shi, _sli = ind.causal_swings(bars, config.SWING_K)
        i = len(c) - 1
        # long: close crosses above the last confirmed swing high
        if (np.isfinite(sh[i]) and np.isfinite(sh[i - 1])
                and c[i] > sh[i] and c[i - 1] <= sh[i - 1]):
            return 1
        # short: close crosses below the last confirmed swing low
        if (np.isfinite(sl[i]) and np.isfinite(sl[i - 1])
                and c[i] < sl[i] and c[i - 1] >= sl[i - 1]):
            return -1
        return None

    def reference_line(self, bars):
        # BOS has no continuous line; use a slow EMA as the trend baseline the
        # PPO exit measures extension from.
        return ind.ema(bars["close"].to_numpy(float), config.EMA_SLOW)

    def _hand_features(self, bars, i, direction):
        c = bars["close"].to_numpy(float)
        sh, sl, shi, sli = ind.causal_swings(bars, config.SWING_K)
        atr_i = float(ind.atr(bars, config.ATR_P)[i])
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        d = direction

        def g(x):
            return float(x) if np.isfinite(x) else 0.0

        level = sh[i] if d == 1 else sl[i]
        sw_idx = shi[i] if d == 1 else sli[i]
        with np.errstate(invalid="ignore"):
            bos_ext = (c[i] - level) / a * d              # break strength (signed)
            swing_range = (sh[i] - sl[i]) / a             # structure leg size
        bos_age = np.log1p(max(i - int(sw_idx), 0)) if sw_idx >= 0 else 0.0
        adx_i, adx_slope = adx_pair(bars, i)
        hand = [g(bos_ext), g(swing_range), bos_age,
                adx_i / 100.0, adx_slope / 100.0]
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, hand]).astype(np.float32)
