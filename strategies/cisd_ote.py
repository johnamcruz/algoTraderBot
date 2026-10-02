#!/usr/bin/env python3
"""strategies/cisd_ote.py — CISD+OTE meta-label strategy (cisd_ote_chronos.joblib).

Mechanical ICT/SMC entry: a CISD displacement on the 12-min zone timeframe defines
an OTE fib zone; price pulling back into the zone fires a candidate (bull zone →
long, bear zone → short) with a pivot-based stop. The model grades selection only.

Detection + the 11 OTE/HTF geometry features are a VERBATIM port of the training
pipeline (see cisd_ote_detect.py), so live == training. Unlike the other
strategies, this one:
  • has its OWN stop (the zone's pivot/origin), not 0.5×ATR — so it overrides
    `detect()` to build the Signal from the zone, and
  • embeds at the SIGNAL bar (exec_idx−1), not the bar loop's last bar — so it
    overrides `grade()` to embed at sig.bar_index instead of the shared embedding.

Live caveat: training enters at the next bar's OPEN (exec_idx); the bot fills at
market on the signal-bar close, so there's up to one bar of entry slippage — the
graded features still use the training entry, only the fill basis differs.
"""
import numpy as np
import pandas as pd

import config
import indicators as ind
from strategies import cisd_ote_detect as cod
from strategies.base import Signal, Strategy, embed_context, ffm_block


class CisdOteStrategy(Strategy):
    name = "cisd_ote"
    model_filename = "cisd_ote_chronos.joblib"

    def __init__(self):
        super().__init__()
        self._ctx = None        # per-detect cache for _hand_features (set in detect)
        self.last_pipeline = None   # last detect()'s pipeline output (read by jev)

    # not used — detect() is overridden (zone-based, not a last-bar indicator flip)
    def _fired(self, bars):
        return None

    # ── per-bar pipeline (verbatim detection) → current-bar signal ──────────
    def _pipeline(self, bars: pd.DataFrame):
        df3 = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["time"], utc=True)))
        df3 = df3[["open", "high", "low", "close", "volume"]]
        atr = np.asarray(ind.atr(bars, cod.ATR_P), dtype=np.float64)
        hour = df3.index.tz_convert("America/New_York").hour.to_numpy()

        # HTF trend (causal): resample to HTF_TF, 9/21 EMA-cross sign + strength,
        # forward-fill the last CLOSED HTF bar onto each base bar (no lookahead).
        _hc = df3["close"].resample(cod.HTF_TF, label="right", closed="left").last().dropna()
        _ef = _hc.ewm(span=9, adjust=False).mean()
        _es = _hc.ewm(span=21, adjust=False).mean()
        htf_sign = pd.Series(np.sign((_ef - _es).to_numpy()), index=_hc.index
                             ).reindex(df3.index, method="ffill").to_numpy()
        htf_str = pd.Series(((_ef - _es) / _es).to_numpy(), index=_hc.index
                            ).reindex(df3.index, method="ffill").to_numpy()

        df_zone = cod.resample(df3, cod.CISD_TF)
        df_sig = cod.detect_zone_signals(df_zone, cod.PARAMS)
        smask = np.ones(len(df3), dtype=bool)            # SESSION='all' → no gate
        recs = cod.extract_signals(df3, df_sig, smask)
        return atr, hour, htf_sign, htf_str, recs

    def detect(self, bars: pd.DataFrame):
        self.last_pipeline = None
        if len(bars) < cod.CTX + cod.ATR_P + 5:
            return None
        atr, hour, htf_sign, htf_str, recs = self.last_pipeline = self._pipeline(bars)
        last = len(bars) - 1
        # the signal whose entry executes on the just-closed bar (sig_bar = last−1)
        rec = next((r for r in reversed(recs) if r["exec_idx"] == last), None)
        if rec is None:
            return None
        sb = rec["sig_bar"]
        a = atr[sb]
        if not (np.isfinite(a) and a > 0):
            return None
        d = 1 if rec["is_long"] else -1
        entry, sl = rec["entry"], rec["sl"]
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        self._ctx = {"rec": rec, "atr": atr, "hour": hour,
                     "htf_sign": htf_sign, "htf_str": htf_str}
        return Signal(self.name, d, float(entry), float(sl), float(risk),
                      sb, bars["time"].iloc[sb])

    # ── 87-dim hand features: 76 FFM (parquet order) + 11 OTE/HTF geom ──────
    def _hand_features(self, bars, i, direction):
        c = self._ctx
        rec, sb = c["rec"], i
        ff = ffm_block(bars, sb)                          # 76 FFM at the signal bar
        a = c["atr"][sb]
        a = a if (np.isfinite(a) and a > 0) else 1e-6
        zh = rec["fib_top"] - rec["fib_bot"]
        zh_s = zh if zh > 0 else 1e-6
        geom = np.zeros(cod.N_GEOM, dtype=np.float32)
        geom[0] = 1.0 if rec["is_long"] else -1.0
        geom[1] = np.clip(zh / a, 0, 10)
        geom[2] = np.clip(rec["zone_age"] / 40.0, 0, 1)
        geom[3] = 1.0 if rec["had_sweep"] else 0.0
        geom[4] = np.clip(rec["disp"], 0, 5)
        geom[5] = (rec["entry"] - rec["fib_bot"]) / zh_s
        geom[6] = np.clip(abs(rec["entry"] - rec["sl"]) / a, 0, 20)
        geom[7] = c["hour"][sb] / 24.0
        dir_ = 1.0 if rec["is_long"] else -1.0
        hs, ht = c["htf_sign"][sb], c["htf_str"][sb]
        geom[8] = (hs if np.isfinite(hs) else 0.0) * dir_
        geom[9] = (ht if np.isfinite(ht) else 0.0) * dir_
        h_ = c["hour"][sb]
        sess_id = (2 if 8 <= h_ < 17 else 1 if 3 <= h_ < 8
                   else 0 if (h_ >= 18 or h_ < 3) else 3)
        geom[10] = sess_id / 3.0
        out = np.concatenate([ff, geom]).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # embed at the SIGNAL bar (sig.bar_index), not the loop's shared last-bar emb
    def grade(self, bars, sig, emb=None):
        return super().grade(bars, sig, emb=None)
