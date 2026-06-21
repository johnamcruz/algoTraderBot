#!/usr/bin/env python3
"""optimize_exit.py — Optuna search for the best PPO trailing-exit config.

The exit shape is governed by three knobs (config.py):
    ACTIVATE_R  hold the initial stop until the peak reaches this many R
    GIVEBACK_R  once trailing, never sit more than this many R below the peak
    STOP_ATR    initial stop = STOP_ATR × ATR(ATR_P)

The PPO trail collapses to this give-back cap (the policy ≈ the best constant
trail), so we can score a config WITHOUT retraining: replay the give-back sim
(TrailExitSim, the exact training=live exit) over a VALIDATION slice and pick the
config with the best expectancy, then report the winner on a held-out TEST slice
so the choice isn't overfit to one window. Multi-ticker for more data.

    python optimize_exit.py --timeframe 1 --tickers NQ ES RTY YM GC --trials 200

Plug the printed config into config.py, then retrain: train_ppo_exit --timeframe N.
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
import trail_exit_env as tee
from trail_exit_env import TRAIL_MULTS, TrailExitSim, build_arrays, build_catalog

HERE = os.path.dirname(os.path.abspath(__file__))


# ── data prep (per ticker) ─────────────────────────────────────────────────

def _load_ticker(symbol, tf, proba_floor):
    csv = os.path.join(HERE, "data", f"{symbol}_{tf}min.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"no data file: {csv} (copy it from the FFM data dir)")
    df = pd.read_csv(csv)
    arr = build_arrays(df)
    catalog = build_catalog(arr)
    if proba_floor > 0:                       # only the flips the bot would enter
        import precompute_proba as pp
        config.TIMEFRAME_MIN, config.SYMBOL = tf, symbol   # pick model + cache
        proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            pp.grade_in_subprocess(csv)
            proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            raise SystemExit(f"proba grading failed for {symbol}")
        catalog = catalog[proba >= proba_floor]
    return arr, catalog, len(df)


def _split(catalog, n_bars, val_lo=0.60, val_hi=0.80):
    """Time-ordered split: validation = bars [val_lo, val_hi), test = [val_hi, 1].
    The early 60% is unused here (no model is trained — the config is the only
    'parameter', and it's chosen on val, judged on test)."""
    idx = catalog[:, 0]
    val = catalog[(idx >= val_lo * n_bars) & (idx < val_hi * n_bars)]
    test = catalog[idx >= val_hi * n_bars]
    return val, test


# ── give-back replay (no PPO training needed) ──────────────────────────────

def _realized_R(arr, catalog, action):
    sim = TrailExitSim(arr)
    out = np.empty(len(catalog), dtype=np.float64)
    for r, (entry_idx, sign) in enumerate(catalog):
        sim.reset(int(entry_idx), int(sign))
        done = False
        while not done:
            _o, _r, done, info = sim.step(action)
        out[r] = info["realized_R"]
    return out


def _pooled_R(datasets, which, action):
    parts = [_realized_R(arr, cats[which], action) for arr, cats in datasets]
    return np.concatenate(parts) if parts else np.empty(0)


def _metrics(R):
    if len(R) == 0:
        return dict(meanR=0.0, wr=0.0, pf=0.0, sumR=0.0, n=0)
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    return dict(meanR=float(R.mean()), wr=float((R > 0).mean()),
                pf=(float(wins / losses) if losses > 0 else float("inf")),
                sumR=float(R.sum()), n=int(len(R)))


def _score(datasets, which, scan_mults, objective):
    """Best-achievable metric for the current tee.* config over `which` slice.
    scan_mults: if True, take the best over all trail mults (when the cap doesn't
    bind); else use a single representative mult (the cap usually dominates)."""
    actions = range(len(TRAIL_MULTS)) if scan_mults else [1]   # [1] = 1.0×ATR
    best = None
    for a in actions:
        m = _metrics(_pooled_R(datasets, which, a))
        v = m[objective]
        if best is None or v > best:
            best = v
    return best


# ── main ───────────────────────────────────────────────────────────────────

def main():
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", type=int, default=config.TRAINED_TIMEFRAME_MIN)
    ap.add_argument("--tickers", nargs="+", default=["NQ"],
                    help="symbols to pool for more data (default NQ)")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--proba-floor", type=float, default=0.35,
                    help="grade entries and keep proba>=this (0 = all flips, no grading)")
    ap.add_argument("--objective", default="meanR",
                    choices=["meanR", "pf", "sumR"])
    ap.add_argument("--scan-mults", action="store_true",
                    help="evaluate every trail mult (slower; only matters when the "
                         "give-back cap doesn't bind)")
    ap.add_argument("--progress", action="store_true",
                    help="show Optuna's live progress bar (off by default — noisy in logs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # one line per improvement, not per trial

    print(f"▶ loading {args.tickers} @ {args.timeframe}-min "
          f"(proba≥{args.proba_floor})…")
    datasets = []
    for sym in args.tickers:
        arr, catalog, n_bars = _load_ticker(sym, args.timeframe, args.proba_floor)
        val, test = _split(catalog, n_bars)
        datasets.append((arr, {"val": val, "test": test}))
        print(f"   {sym}: {len(catalog)} flips | {len(val)} val | {len(test)} test")

    def objective(trial):
        tee.ACTIVATE_R = trial.suggest_float("ACTIVATE_R", 0.5, 4.0, step=0.25)
        tee.GIVEBACK_R = trial.suggest_float("GIVEBACK_R", 0.25, 2.0, step=0.25)
        tee.STOP_ATR = trial.suggest_float("STOP_ATR", 0.3, 1.2, step=0.1)
        return _score(datasets, "val", args.scan_mults, args.objective)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=args.progress)

    best = study.best_params
    # evaluate the WINNER and the current baseline on the held-out TEST slice
    def test_metrics(activate, giveback, stop_atr):
        tee.ACTIVATE_R, tee.GIVEBACK_R, tee.STOP_ATR = activate, giveback, stop_atr
        return _metrics(_pooled_R(datasets, "test",
                                  1 if not args.scan_mults else 1))

    base = test_metrics(config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR)
    won = test_metrics(best["ACTIVATE_R"], best["GIVEBACK_R"], best["STOP_ATR"])

    def _row(tag, cfg, m):
        print(f"   {tag:<10} ACTIVATE_R={cfg[0]:.2f} GIVEBACK_R={cfg[1]:.2f} "
              f"STOP_ATR={cfg[2]:.2f} | meanR={m['meanR']:+.3f} WR={m['wr']:5.1%} "
              f"PF={m['pf']:.2f} sumR={m['sumR']:+.1f} n={m['n']}")

    print(f"\n── best val {args.objective}={study.best_value:+.3f} → TEST (held out) ──")
    _row("baseline", (config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR), base)
    _row("optuna", (best["ACTIVATE_R"], best["GIVEBACK_R"], best["STOP_ATR"]), won)

    improved = won[args.objective] > base[args.objective]
    print(f"\n{'✅ improves' if improved else '⚠️  no improvement on'} test "
          f"{args.objective}. Plug into config.py:")
    print(f"    ACTIVATE_R = {best['ACTIVATE_R']:.2f}")
    print(f"    GIVEBACK_R = {best['GIVEBACK_R']:.2f}")
    print(f"    STOP_ATR   = {best['STOP_ATR']:.2f}")
    print(f"then: python train_ppo_exit.py --timeframe {args.timeframe}")


if __name__ == "__main__":
    main()
