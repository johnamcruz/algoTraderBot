#!/usr/bin/env python3
"""precompute_proba.py — grade every SuperTrend flip with the entry model.

train_ppo_exit.py trains the PPO exit only on flips the bot would actually take
(proba >= floor). This computes that proba for every flip once and caches it.

Pipeline (all via the public futures_foundation library + the shipped joblib):
    embeddings = foundation.embed_bars(closes, flip_indices)   # batched, causal
    hand       = 76 FFM (derive_features, parquet order) + adx + adx_slope
    proba      = supertrend_chronos.joblib signal_head.predict_proba

Cached to proba_cache.npz keyed on the data file + flip set, so re-runs are
instant. Standalone:  python precompute_proba.py
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import config
import indicators as ind
from trail_exit_env import build_arrays, build_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data", "NQ_3min.csv")
CACHE = os.path.join(HERE, "proba_cache.npz")
SUPERTREND_MODEL = os.path.join(config.MODELS_DIR, "supertrend_chronos.joblib")

with open(config.FFM_COLUMNS_PATH) as _f:
    _FFM_COLS = json.load(_f)


def _ffm_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_bars, 76) FFM features for the whole frame, in parquet column order.
    Computed once via derive_features; absent columns stay NaN."""
    from futures_foundation.features import derive_features

    feats = derive_features(df.rename(columns={"datetime": "datetime"}),
                            instrument=config.SYMBOL, atr_period=config.ATR_P)
    out = np.full((len(df), len(_FFM_COLS)), np.nan, dtype=np.float32)
    for k, name in enumerate(_FFM_COLS):
        if name in feats.columns:       # coerce nullable/NA → NaN
            out[:, k] = pd.to_numeric(feats[name], errors="coerce").to_numpy(np.float32)
    return out


def _hand_all(df: pd.DataFrame, flip_idx: np.ndarray) -> np.ndarray:
    """(N, 78) SuperTrend hand features = 76 FFM + adx + adx_slope per flip."""
    bars = df.rename(columns={"datetime": "time"})
    ffm = _ffm_matrix(df)
    a = ind.adx(bars, config.ADX_P)
    k = config.ADX_SLOPE
    out = np.empty((len(flip_idx), len(_FFM_COLS) + 2), dtype=np.float32)
    for r, i in enumerate(flip_idx):
        adx_i = a[i] if np.isfinite(a[i]) else 0.0
        slope = a[i] - a[i - k] if i >= k and np.isfinite(a[i]) \
            and np.isfinite(a[i - k]) else 0.0
        out[r, :-2] = ffm[i]
        out[r, -2] = adx_i
        out[r, -1] = slope
    return out


def _key(csv_path, flip_idx):
    h = hashlib.sha1()
    h.update(str(int(os.path.getmtime(csv_path))).encode())
    h.update(flip_idx.tobytes())
    return h.hexdigest()


def read_cache(df, catalog, csv_path=DATA_CSV, cache=CACHE):
    """Numpy-only cache read (no xgboost/torch import). Returns the cached proba
    aligned to `catalog`, or None if absent / stale. Used by train_ppo_exit so
    its torch process never loads xgboost (they segfault together on macOS)."""
    if not os.path.exists(cache):
        return None
    z = np.load(cache, allow_pickle=True)
    return z["proba"] if str(z["key"]) == _key(csv_path, catalog[:, 0]) else None


def grade_in_subprocess(csv_path=DATA_CSV, rows=None):
    """Run this module as a child process to populate the proba cache with
    xgboost — keeping the caller (which may hold torch/SB3) xgboost-free."""
    import subprocess
    import sys
    cmd = [sys.executable, os.path.abspath(__file__), "--csv", csv_path]
    if rows:
        cmd += ["--rows", str(rows)]
    subprocess.run(cmd, check=True)


def proba_for_catalog(df, catalog, csv_path=DATA_CSV, cache=CACHE):
    """proba aligned 1:1 with `catalog` rows. Cached on (data mtime, flips)."""
    import futures_foundation.chronos  # noqa: F401  (pipelines.chronos shim)
    import joblib
    from futures_foundation import foundation

    flip_idx = catalog[:, 0]
    key = _key(csv_path, flip_idx)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            return z["proba"]

    print(f"▶ grading {len(flip_idx)} flips (batched embeddings)…")
    closes = df["close"].to_numpy(float)
    emb = foundation.embed_bars(closes, list(flip_idx), ctx=config.CTX)  # (N,256)
    hand = _hand_all(df, flip_idx)                                       # (N,78)
    X = np.concatenate([emb, hand], axis=1).astype(np.float32)
    bundle = joblib.load(SUPERTREND_MODEL)
    proba = bundle["signal_head"].predict_proba(X)[:, 1].astype(np.float32)

    np.savez(cache, key=key, proba=proba)
    print(f"✅ cached proba → {os.path.relpath(cache, HERE)}")
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DATA_CSV)
    ap.add_argument("--rows", type=int, default=0,
                    help="grade only the first N bars (matches train --quick)")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if args.rows:
        df = df.iloc[:args.rows].reset_index(drop=True)
    catalog = build_catalog(build_arrays(df))
    proba = proba_for_catalog(df, catalog, args.csv)
    for thr in (0.35, 0.45, 0.50):
        print(f"  proba >= {thr:.2f}:  {(proba >= thr).sum():5d} / {len(proba)} flips")


if __name__ == "__main__":
    main()
