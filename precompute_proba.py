#!/usr/bin/env python3
"""precompute_proba.py — grade every SuperTrend flip with the entry model.

The live bot only enters a flip when the Chronos+XGBoost head gives
``proba >= 0.35``. To train the trailing exit on the *same* trades the bot
actually takes, we need that proba for every flip in the history.

Calling predict.chronos_embedding() once per flip is far too slow (each call
spawns a fresh torch subprocess that reloads the model). Instead this script:

    stage 1 (torch only, one subprocess):  batch-embed all flip windows
    stage 2 (xgboost, main process):       run the two heads -> proba per flip

Results are cached to proba_cache.npz keyed on the data file + flip set, so a
re-run is instant. Used by train_ppo_exit.py; can also be run standalone:

    python precompute_proba.py            # grade data/NQ_3min.csv flips
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from predict import CTX_WINDOW, EMBED_DIM, HAND_DIM
from trail_exit_env import build_arrays, build_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data", "NQ_3min.csv")
CACHE = os.path.join(HERE, "proba_cache.npz")


# ── stage 1: batched Chronos embeddings (torch-only subprocess) ────────

def _embed_worker(windows_path, out_path):
    """Runs in a torch-only subprocess: [N,128] log-close windows -> [N,256]
    masked-mean-pooled embeddings, exactly matching predict._embed_in_this_process
    but batched over all flips at once."""
    import torch
    from chronos import BaseChronosPipeline

    ctx = torch.tensor(np.load(windows_path), dtype=torch.float32)   # [N,128]
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32)
    model = getattr(pipe, "inner_model", None) or pipe.model

    out = np.empty((ctx.shape[0], EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, ctx.shape[0], 512):                        # chunk
            chunk = ctx[s:s + 512]
            h, _ls, _emb, mask = model.encode(context=chunk)
            w = mask.unsqueeze(-1).to(h.dtype)
            emb = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            out[s:s + chunk.shape[0]] = emb.numpy().astype(np.float32)
            print(f"  embedded {s + chunk.shape[0]}/{ctx.shape[0]}", flush=True)
    np.save(out_path, out)


def _embed_all(closes, flip_idx):
    """Build one [N,128] log-window matrix and embed it in a subprocess."""
    windows = np.stack([
        np.log(closes[i - CTX_WINDOW + 1: i + 1]) for i in flip_idx
    ]).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        in_p, out_p = os.path.join(tmp, "w.npy"), os.path.join(tmp, "e.npy")
        np.save(in_p, windows)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--_embed-worker",
             in_p, out_p])
        if r.returncode != 0:
            raise SystemExit("embedding subprocess failed")
        return np.load(out_p)


# ── stage 2: hand features + the two XGBoost heads ─────────────────────

def _hand_features(df, flip_idx):
    """Per-flip 78-vector: 76 proprietary slots NaN (as live), then adx and
    adx_slope at the flip bar — identical to supertrend_ai_bot.build_features."""
    from supertrend_ai_bot import adx
    a = adx(df).to_numpy(dtype=np.float32)
    feats = np.full((len(flip_idx), HAND_DIM), np.nan, dtype=np.float32)
    for k, i in enumerate(flip_idx):
        feats[k, 76] = a[i]
        feats[k, 77] = a[i] - a[i - 1]
    return feats


def _proba_from_heads(emb, hand):
    import xgboost as xgb
    hand = np.nan_to_num(hand, nan=np.nan, posinf=0.0, neginf=0.0)
    X = np.concatenate([emb, hand], axis=1).astype(np.float32)
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(HERE, "signal_head.json"))
    return clf.predict_proba(X)[:, 1].astype(np.float32)


# ── cache + public entry point ─────────────────────────────────────────

def _key(csv_path, flip_idx):
    h = hashlib.sha1()
    h.update(str(int(os.path.getmtime(csv_path))).encode())
    h.update(flip_idx.tobytes())
    return h.hexdigest()


def proba_for_catalog(df, catalog, csv_path=DATA_CSV, cache=CACHE):
    """proba aligned 1:1 with `catalog` rows. Cached on (data mtime, flips)."""
    flip_idx = catalog[:, 0]
    key = _key(csv_path, flip_idx)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            return z["proba"]

    closes = df["close"].to_numpy(dtype=np.float64)
    print(f"▶ grading {len(flip_idx)} flips (batched embeddings)…")
    emb = _embed_all(closes, flip_idx)
    hand = _hand_features(df, flip_idx)
    proba = _proba_from_heads(emb, hand)
    np.savez(cache, key=key, proba=proba)
    print(f"✅ cached proba → {os.path.relpath(cache, HERE)}")
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DATA_CSV)
    ap.add_argument("--_embed-worker", nargs=2, metavar=("IN", "OUT"),
                    dest="embed_worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.embed_worker:
        _embed_worker(*args.embed_worker)
        return

    df = pd.read_csv(args.csv)
    catalog = build_catalog(build_arrays(df))
    proba = proba_for_catalog(df, catalog, args.csv)
    for thr in (0.35, 0.45, 0.50):
        print(f"  proba >= {thr:.2f}:  {(proba >= thr).sum():5d} / {len(proba)} flips")


if __name__ == "__main__":
    main()
