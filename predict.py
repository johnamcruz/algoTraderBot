#!/usr/bin/env python3
"""SuperTrend+Chronos signal head — standalone inference.

Grades a SuperTrend flip event on NQ futures (3-min bars):
    proba  = P(win) for the flip            (0..1)
    r_hat  = predicted peak R-multiple      (0..15)

Quick start (verifies your install end-to-end with synthetic data):
    python predict.py --demo

Real usage:
    python predict.py --closes closes.csv --features features.csv

  closes.csv   one closing price per line, >= 128 rows, oldest -> newest
               (3-min NQ bars ending AT the signal bar)
  features.csv 78 values, one per line (see README: 76 FFM features,
               then adx, then adx_slope). Missing values may be left
               as 'nan' — XGBoost handles NaN natively.

Files signal_head.json / risk_head.json / metadata.json must sit next
to this script.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CTX_WINDOW = 128      # closes fed to Chronos
EMBED_DIM = 256       # chronos-bolt-tiny d_model
HAND_DIM = 78         # 76 FFM + adx + adx_slope


def _embed_in_this_process(closes):
    """256-d embedding of the last 128 LOG closes via amazon/chronos-bolt-tiny.

    Masked-mean pool of the encoder hidden states — exactly how the heads
    were trained. The checkpoint (~45 MB) downloads automatically from
    HuggingFace on first run and is cached locally after that.

    NOTE: imports torch — call only in a process that will NOT also load
    xgboost (they bundle conflicting OpenMP runtimes; loading both in one
    process segfaults on macOS). Use chronos_embedding() instead, which
    isolates this in a subprocess automatically.
    """
    import torch
    from chronos import BaseChronosPipeline

    closes = np.asarray(closes, dtype=np.float32)
    if len(closes) < CTX_WINDOW:
        raise SystemExit(f"need >= {CTX_WINDOW} closes, got {len(closes)}")
    ctx = torch.tensor(np.log(closes[-CTX_WINDOW:])).unsqueeze(0)  # [1, 128]

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32)
    model = getattr(pipe, "inner_model", None) or pipe.model
    with torch.no_grad():
        h, _ls, _emb, mask = model.encode(context=ctx)
        w = mask.unsqueeze(-1).to(h.dtype)
        emb = (h * w).sum(1) / w.sum(1).clamp(min=1.0)   # masked mean pool
    return emb.squeeze(0).numpy().astype(np.float32)    # (256,)


def chronos_embedding(closes):
    """Compute the embedding in an isolated subprocess (torch-only), so the
    calling process stays xgboost-safe. Same output as
    _embed_in_this_process — this is just process isolation."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        in_p = os.path.join(tmp, "closes.npy")
        out_p = os.path.join(tmp, "emb.npy")
        np.save(in_p, np.asarray(closes, dtype=np.float32))
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--_embed-worker", in_p, out_p],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"embedding subprocess failed:\n{r.stderr[-2000:]}")
        return np.load(out_p)


def predict(emb_256, hand_78):
    """(proba, r_hat) from the two XGBoost heads."""
    import xgboost as xgb

    emb = np.asarray(emb_256, dtype=np.float32)
    hand = np.asarray(hand_78, dtype=np.float32)
    if emb.shape != (EMBED_DIM,):
        raise SystemExit(f"embedding must be ({EMBED_DIM},), got {emb.shape}")
    if hand.shape != (HAND_DIM,):
        raise SystemExit(f"features must be ({HAND_DIM},), got {hand.shape}")

    # NaN passes through (XGBoost handles it); only +/-inf is zeroed.
    hand = np.nan_to_num(hand, nan=np.nan, posinf=0.0, neginf=0.0)
    X = np.concatenate([emb, hand]).reshape(1, -1).astype(np.float32)

    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(HERE, "signal_head.json"))
    proba = float(clf.predict_proba(X)[0, 1])

    reg = xgb.XGBRegressor()
    reg.load_model(os.path.join(HERE, "risk_head.json"))
    # risk head was trained on log1p(R); invert + clip to plausible range.
    r_hat = float(np.clip(np.expm1(reg.predict(X)[0]), 0.0, 15.0))
    return proba, r_hat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data to verify the install")
    ap.add_argument("--closes", help="CSV/txt: one close per line, >=128 rows")
    ap.add_argument("--features", help="CSV/txt: 78 feature values, one per line")
    ap.add_argument("--_embed-worker", nargs=2, metavar=("IN", "OUT"),
                    dest="embed_worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.embed_worker:
        # internal: torch-only subprocess mode (no xgboost in this process)
        in_p, out_p = args.embed_worker
        np.save(out_p, _embed_in_this_process(np.load(in_p)))
        return

    if args.demo:
        rng = np.random.RandomState(0)
        closes = 20000 + np.cumsum(rng.randn(CTX_WINDOW)) * 5  # random walk
        hand = np.full(HAND_DIM, np.nan, dtype=np.float32)     # all-missing OK
        print("demo: synthetic random-walk closes + all-NaN features")
    elif args.closes and args.features:
        closes = np.loadtxt(args.closes, dtype=np.float32).ravel()
        hand = np.loadtxt(args.features, dtype=np.float32).ravel()
    else:
        ap.print_help()
        sys.exit(1)

    emb = chronos_embedding(closes)
    proba, r_hat = predict(emb, hand)

    meta = json.load(open(os.path.join(HERE, "metadata.json")))
    print(f"proba = {proba:.4f}   (P(win) — production used a 0.35 floor)")
    print(f"r_hat = {r_hat:.3f}   (predicted peak R; TP rule was "
          f"clip(0.8*r_hat, 1.5, 8.0))")
    print(f"[model trained {meta['training_metadata']['train_date']}, "
          f"{meta['training_metadata']['n_train_signals']} signals]")


if __name__ == "__main__":
    main()
